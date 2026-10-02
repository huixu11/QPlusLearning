"""CPU checks for kernel admission, binding verification and pinned installation."""
import functools
import hashlib
import io
import json
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from optimized_training import install_kernels, kernel_lock, selected_implementation, validate_platform, track_kernel_calls, verify_bindings


class OptimizedTrainingTests(unittest.TestCase):
    def test_fused_adamw_dispatch_is_observed_without_profiler(self):
        import torch
        original = torch._fused_adamw_
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=0.1, fused=True)
        with track_kernel_calls([("optimizer", torch, "_fused_adamw_")]) as calls:
            for _ in range(2):
                parameter.grad = torch.ones_like(parameter)
                optimizer.step()
                optimizer.zero_grad()
            self.assertEqual(calls["optimizer"], 2)
        self.assertIs(torch._fused_adamw_, original)
        self.assertLess(parameter.item(), 1.0)

    def test_kernel_probe_restores_functions_after_failure(self):
        def fail():
            raise ValueError("kernel failure")
        owner = SimpleNamespace(kernel=fail)
        with self.assertRaisesRegex(ValueError, "kernel failure"):
            with track_kernel_calls([("kernel", owner, "kernel")]) as calls:
                owner.kernel()
        self.assertEqual(calls["kernel"], 1)
        self.assertIs(owner.kernel, fail)
        with self.assertRaises(AttributeError):
            with track_kernel_calls([("kernel", owner, "kernel"), ("missing", owner, "missing")]):
                pass
        self.assertIs(owner.kernel, fail)

    def test_watchdog_reports_stacks_then_cancels_on_exit(self):
        code = """import time
from optimized_training import ProgressWatchdog
with ProgressWatchdog(seconds=0.04) as phase:
    phase('test forward')
    time.sleep(0.12)
print('AFTER_WATCHDOG', flush=True)
time.sleep(0.12)
"""
        result = subprocess.run([sys.executable, "-u", "-c", code], cwd=Path(__file__).parent,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout)
        before, after = result.stdout.split("AFTER_WATCHDOG", 1)
        self.assertIn('"phase": "test forward"', before)
        self.assertIn("Timeout", before)
        self.assertIn('File "<string>"', before)
        self.assertNotIn("Timeout", after)

    def test_binding_only_report_does_not_claim_cuda_training(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch("optimized_training.require_optimized_bindings", return_value={"kernels": {}, "versions": {}}), \
                patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.is_bf16_supported", return_value=True):
            report = Path(folder) / "check.json"
            verify_bindings(report)
            result = json.loads(report.read_text())
        self.assertEqual(result["result"], "bindings_verified")
        self.assertEqual(result["cuda_loss_backward"], "not_run")
        self.assertEqual(result["optimizer_steps"], 0)

    def test_platform_rejects_incompatible_wheel_and_non_bf16_gpu(self):
        info = {**kernel_lock()["platform"], "bf16": True}
        validate_platform(info)
        for key, value in [("python", "3.12"), ("torch", "2.9.0"), ("cuda", "12.6"),
                           ("cxx11_abi", False), ("machine", "aarch64"), ("bf16", False)]:
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate_platform({**info, key: value})

    def test_binding_is_captured_function_not_deceptive_wrapped_name(self):
        def fallback():
            return "slow"
        def optimized():
            return "fast"
        def decorate(implementation):
            @functools.wraps(fallback)
            def wrapped():
                return implementation()
            return wrapped
        self.assertIs(selected_implementation(decorate(optimized)), optimized)
        self.assertIs(selected_implementation(decorate(fallback)), fallback)
        with self.assertRaisesRegex(RuntimeError, "Cannot verify"):
            selected_implementation(fallback)

    def test_installer_verifies_bytes_without_replacing_locked_dependencies(self):
        content = b"verified test wheel"
        spec = {"filename": "example.whl", "url": "https://example.test/pinned.whl", "version": "1",
                "sha256": hashlib.sha256(content).hexdigest()}
        lock = {"platform": kernel_lock()["platform"], "packages": {"example": spec}}
        with tempfile.TemporaryDirectory() as folder, patch("optimized_training.kernel_lock", return_value=lock):
            with patch("optimized_training.subprocess.check_output", return_value=json.dumps({**lock["platform"], "bf16": True})), \
                    patch("optimized_training.subprocess.run") as run, \
                    patch("optimized_training.urlopen", return_value=io.BytesIO(content)) as fetch:
                install_kernels("python", folder, folder)
                self.assertEqual(fetch.call_count, 1)
                command = run.call_args.args[0]
                self.assertIn("--no-deps", command)
                self.assertEqual(Path(command[-1]).read_bytes(), content)
            with patch("optimized_training.subprocess.check_output", return_value=json.dumps({**lock["platform"], "bf16": True})), \
                    patch("optimized_training.subprocess.run"), patch("optimized_training.urlopen") as fetch:
                install_kernels("python", folder, folder)
                fetch.assert_not_called()

    def test_installer_refuses_corrupt_download_before_install(self):
        lock = {"platform": kernel_lock()["platform"], "packages": {"example": {
            "filename": "bad.whl", "url": "https://example.test/pinned.whl", "version": "1", "sha256": "0" * 64}}}
        with tempfile.TemporaryDirectory() as folder, patch("optimized_training.kernel_lock", return_value=lock), \
                patch("optimized_training.subprocess.check_output", return_value=json.dumps({**lock["platform"], "bf16": True})), \
                patch("optimized_training.subprocess.run") as run, \
                patch("optimized_training.urlopen", return_value=io.BytesIO(b"bad")):
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                install_kernels("python", folder, folder)
            run.assert_not_called()
