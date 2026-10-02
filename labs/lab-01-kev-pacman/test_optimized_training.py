"""CPU checks for kernel admission, binding verification and pinned installation."""
import functools
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from optimized_training import install_kernels, kernel_lock, selected_implementation, validate_platform


class OptimizedTrainingTests(unittest.TestCase):
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
