"""CPU checks for the local notebook's isolated, pinned training environment."""
import copy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from cloud_runtime import BASE_MODEL, BASE_REVISION, CODE_REVISION, CloudRuntime, GPU_PREFLIGHT
from lab_runtime import DEFAULT_TRAINING_ENVIRONMENT, LAB_EXECUTION, LabRuntime
from planner_data import COUNTS, RECIPE


ROOT = Path(__file__).resolve().parent
GPU = {"name": "NVIDIA RTX A6000", "memory_gib": 48, "compute_capability": [8, 6],
       "torch": "2.8.0+cu128", "cuda": "12.8", "bf16_supported": True,
       "kernel_backward": "passed"}
META = {"base": BASE_MODEL, "base_revision": BASE_REVISION, "lora": 16,
        "head_dim": 256, "option_isolation": False, "special_embeddings": False,
        "weights_dtype": "fp32", "weights": "lora"}


def runtime_in(folder, **kwargs):
    return LabRuntime(Path(folder) / "runtime", training_environment=Path(folder) / "envs/training", **kwargs)


def checkpoint_files(folder):
    folder.mkdir(parents=True)
    for name in ("head.pt", "adapter_config.json", "adapter_model.safetensors"):
        (folder / name).write_bytes(b"fixture")


def command_flags(command):
    return dict(zip(command[3::2], command[4::2]))


class LabRuntimeTests(unittest.TestCase):
    def test_default_and_environment_override_keep_jupyter_separate(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"QPLUS_TRAINING_ENV": DEFAULT_TRAINING_ENVIRONMENT}):
            runtime = LabRuntime(Path(folder) / "runtime")
            self.assertEqual(str(runtime.training_environment), DEFAULT_TRAINING_ENVIRONMENT)
            self.assertEqual(runtime.python, runtime.training_environment / "bin/python")
            self.assertTrue(runtime.allow_other_gpu)
            self.assertEqual(runtime.training_profile, "memory_safe")
            self.assertEqual(runtime.initial_execution, LAB_EXECUTION)
            self.assertEqual(runtime.pacman_execution, LAB_EXECUTION)
            target = Path(folder) / "custom-training"
            with patch.dict(os.environ, {"QPLUS_TRAINING_ENV": str(target), "UV_PROJECT_ENVIRONMENT": "unrelated"}):
                overridden = LabRuntime(Path(folder) / "runtime")
                env = overridden.environment()
                self.assertEqual(overridden.training_environment, target)
                self.assertEqual(env["UV_PROJECT_ENVIRONMENT"], str(target))
                self.assertEqual(os.environ["UV_PROJECT_ENVIRONMENT"], "unrelated")

    def test_execution_overrides_are_copied_and_preserve_effective_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            choice = {"batch": 2, "accum": 4, "row_budget": 1024}
            runtime = runtime_in(folder, initial_execution=choice, pacman_execution=choice)
            choice["batch"] = 8
            self.assertEqual(runtime.initial_execution["batch"], 2)
            self.assertEqual(runtime.pacman_execution["batch"], 2)
            original = copy.deepcopy(RECIPE)
            selected = runtime.selected_pacman_recipe(RECIPE)
            self.assertEqual(selected, {**RECIPE, "batch": 2, "accum": 4, "row_budget": 1024})
            selected["lr"] = "different"
            self.assertEqual(RECIPE, original)
            for bad in ({"batch": 1, "accum": 4, "row_budget": 2048},
                        {"batch": True, "accum": 8, "row_budget": 2048},
                        {"batch": 1, "accum": 8, "row_budget": -1},
                        {"batch": 1, "accum": 8}):
                for name in ("initial_execution", "pacman_execution"):
                    with self.subTest(name=name, bad=bad), self.assertRaises(ValueError):
                        runtime_in(folder, **{name: bad})

    def test_setup_syncs_external_python313_with_frozen_pins_and_verified_kernels(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = runtime_in(folder)
            runtime.repo.mkdir(parents=True)
            with patch.dict(os.environ, {"UV_PROJECT_ENVIRONMENT": "unrelated"}), \
                    patch("pacman_lab.ensure_node"), patch("lab_runtime.shutil.which", return_value="uv"), \
                    patch("lab_runtime.subprocess.run") as run, \
                    patch("lab_runtime.subprocess.check_output", side_effect=[CODE_REVISION + "\n", json.dumps(GPU)]) as output, \
                    patch("lab_runtime.install_kernels") as kernels, \
                    patch("lab_runtime.ensure_native_kernels", return_value={"status": "already_verified"}) as native, \
                    patch("lab_runtime.print"):
                runtime.setup()
                sync = next(call for call in run.call_args_list if call.args[0][:2] == ["uv", "sync"])
                self.assertEqual(sync.args[0], ["uv", "sync", "--frozen", "--no-dev", "--extra", "serve", "--python", "3.13"])
                self.assertEqual(sync.kwargs["cwd"], runtime.repo)
                self.assertTrue(sync.kwargs["check"])
                self.assertEqual(sync.kwargs["env"]["UV_PROJECT_ENVIRONMENT"], str(runtime.training_environment))
                self.assertEqual(output.call_args.args[0], [str(runtime.python), "-c", GPU_PREFLIGHT])
                self.assertEqual(output.call_args.kwargs["env"]["UV_PROJECT_ENVIRONMENT"], str(runtime.training_environment))
                self.assertEqual(os.environ["UV_PROJECT_ENVIRONMENT"], "unrelated")
                kernels.assert_called_once_with(runtime.python, runtime.repo, runtime.workspace)
                native.assert_called_once_with(runtime.python, runtime.repo, runtime.workspace, runtime.environment())
            self.assertEqual(json.loads(runtime.environment_receipt.read_text()), runtime._ownership())
            self.assertEqual(json.loads((runtime.workspace / "runtime-preflight.json").read_text()), GPU)
            self.assertEqual(json.loads((runtime.workspace / "lab-native-kernels.json").read_text()),
                             {"status": "already_verified"})
            self.assertEqual(runtime.dtype, "bf16")

    def test_interrupted_owned_install_can_retry_without_process_environment_change(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = runtime_in(folder)
            runtime.repo.mkdir(parents=True)
            fail = True
            def run(command, **kwargs):
                nonlocal fail
                if command[:2] == ["uv", "sync"]:
                    runtime.training_environment.mkdir(parents=True, exist_ok=True)
                    (runtime.training_environment / "partial-install").write_text("preserve")
                    if fail:
                        fail = False
                        raise subprocess.CalledProcessError(1, command)
            with patch.dict(os.environ, {"UV_PROJECT_ENVIRONMENT": "original"}), \
                    patch("pacman_lab.ensure_node"), patch("lab_runtime.shutil.which", return_value="uv"), \
                    patch("lab_runtime.subprocess.run", side_effect=run), \
                    patch("lab_runtime.subprocess.check_output", side_effect=[CODE_REVISION, CODE_REVISION, json.dumps(GPU)]), \
                    patch("lab_runtime.install_kernels"), \
                    patch("lab_runtime.ensure_native_kernels", return_value={"status": "already_verified"}), \
                    patch("lab_runtime.print"):
                with self.assertRaises(subprocess.CalledProcessError):
                    runtime.setup()
                self.assertEqual(os.environ["UV_PROJECT_ENVIRONMENT"], "original")
                self.assertEqual((runtime.training_environment / "partial-install").read_text(), "preserve")
                runtime.setup()
                self.assertEqual(os.environ["UV_PROJECT_ENVIRONMENT"], "original")
            recreated = runtime_in(folder)
            self.assertEqual(recreated.training_environment, runtime.training_environment)

    def test_toolkit_selection_reuses_jupyter_and_honors_explicit_override(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            kernel, default, separate, explicit = [root / name for name in
                                                   ("jupyter", "default-jupyter", "toolkit", "explicit")]
            for prefix in (kernel, default, separate, explicit):
                (prefix / "bin").mkdir(parents=True)
                (prefix / "bin/nvcc").write_text("fixture")
            with patch("lab_runtime.sys.prefix", str(kernel)), \
                    patch("lab_runtime.DEFAULT_JUPYTER_ENVIRONMENT", str(default)), \
                    patch("lab_runtime.DEFAULT_CUDA_TOOLKIT", str(separate)), \
                    patch.dict(os.environ, {"QPLUS_CUDA_HOME": "", "CUDA_HOME": "/old/cuda"}):
                runtime = runtime_in(folder)
                for expected in (kernel, default, separate):
                    env = runtime.environment()
                    self.assertEqual(env["CUDA_HOME"], str(expected))
                    self.assertEqual(env["PATH"].split(os.pathsep)[0], str(expected / "bin"))
                    self.assertEqual(env["VIRTUAL_ENV"], str(runtime.training_environment))
                    self.assertEqual(os.environ["CUDA_HOME"], "/old/cuda")
                    (expected / "bin/nvcc").unlink()
                self.assertEqual(runtime.environment()["CUDA_HOME"], "/old/cuda")
                with patch.dict(os.environ, {"QPLUS_CUDA_HOME": str(explicit)}):
                    self.assertEqual(runtime.environment()["CUDA_HOME"], str(explicit))
                with patch.dict(os.environ, {"QPLUS_CUDA_HOME": "relative"}):
                    with self.assertRaisesRegex(ValueError, "absolute CUDA Toolkit"):
                        runtime.environment()

    def test_native_import_failure_prevents_setup_readiness_and_clears_training_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = runtime_in(folder)
            runtime.repo.mkdir(parents=True)
            runtime.training_preflight = {"result": "passed", "cuda_loss_backward": "passed", "optimizer_steps": 2}
            with patch("pacman_lab.ensure_node"), patch("lab_runtime.shutil.which", return_value="uv"), \
                    patch("lab_runtime.subprocess.run"), \
                    patch("lab_runtime.subprocess.check_output", side_effect=[CODE_REVISION, json.dumps(GPU)]), \
                    patch("lab_runtime.install_kernels"), \
                    patch("lab_runtime.ensure_native_kernels", side_effect=RuntimeError("native import failed")):
                with self.assertRaisesRegex(RuntimeError, "native import failed"):
                    runtime.setup()
            self.assertIsNone(runtime.training_preflight)
            self.assertIsNone(runtime.native_training_kernels)
            self.assertFalse((runtime.workspace / "runtime-preflight.json").exists())

    def test_setup_rejects_wrong_revision_before_adopting_training_prefix(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = runtime_in(folder)
            runtime.repo.mkdir(parents=True)
            with patch("pacman_lab.ensure_node"), patch("lab_runtime.shutil.which", return_value="uv"), \
                    patch("lab_runtime.subprocess.run") as run, \
                    patch("lab_runtime.subprocess.check_output", return_value="another-revision"), \
                    patch("lab_runtime.install_kernels") as kernels:
                with self.assertRaisesRegex(RuntimeError, "differs from the lab pin"):
                    runtime.setup()
                self.assertFalse(any(call.args[0][:2] == ["uv", "sync"] for call in run.call_args_list))
                kernels.assert_not_called()
            self.assertFalse(runtime.environment_receipt.exists())

    def test_cuda_bf16_gate_is_not_bypassed_for_a6000(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = runtime_in(folder)
            runtime.repo.mkdir(parents=True)
            with patch("pacman_lab.ensure_node"), patch("lab_runtime.shutil.which", return_value="uv"), \
                    patch("lab_runtime.subprocess.run"), \
                    patch("lab_runtime.subprocess.check_output", side_effect=[CODE_REVISION, json.dumps({**GPU, "bf16_supported": False})]), \
                    patch("lab_runtime.install_kernels") as kernels:
                with self.assertRaisesRegex(RuntimeError, "must support BF16"):
                    runtime.setup()
                kernels.assert_not_called()

    def test_training_prefix_cannot_overlap_kernel_base_conda_or_source_checkout(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            kernel = root / "conda_envs/jupyter"
            base = root / "base-python"
            with patch("lab_runtime.sys.prefix", str(kernel)), patch("lab_runtime.sys.base_prefix", str(base)), \
                    patch.dict(os.environ, {"CONDA_PREFIX": str(root / "conda-env")}):
                protected = (kernel, kernel / "child", kernel.parent, base, base / "child",
                             root / "conda-env", root / "conda-env/child", root / "runtime/kev",
                             root / "runtime/kev/child", root / "runtime", ROOT)
                for target in protected:
                    with self.subTest(target=target), self.assertRaisesRegex(ValueError, "overlaps"):
                        LabRuntime(root / "runtime", training_environment=target)
                sibling = LabRuntime(root / "runtime", training_environment=root / "conda_envs/training")
                self.assertEqual(sibling.training_environment, root / "conda_envs/training")

    def test_conda_metadata_and_unknown_prefix_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, inside in (("conda", "conda-meta"), ("unknown", "keep.txt")):
                target = root / name
                target.mkdir()
                (target / inside).write_text("preserve")
                with self.subTest(name=name), self.assertRaises(ValueError):
                    LabRuntime(root / "runtime", training_environment=target)
                self.assertEqual((target / inside).read_text(), "preserve")
                self.assertFalse(target.with_name("." + target.name + ".qplus-kev-runtime.json").exists())
            nested = root / "conda/nested"
            with self.assertRaisesRegex(ValueError, "Conda environment"):
                LabRuntime(root / "runtime", training_environment=nested)
            with self.assertRaisesRegex(ValueError, "absolute"):
                LabRuntime(root / "runtime", training_environment="relative-venv")

    def test_empty_git_placeholder_is_ignored_but_real_checkout_is_protected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkout = root / "checkout"
            source = checkout / "labs/source"
            source.mkdir(parents=True)
            metadata = checkout / ".git"
            metadata.mkdir()
            target = checkout / "envs/training"
            with patch.dict(os.environ, {"QPLUS_LAB_SOURCE": str(source)}):
                runtime = LabRuntime(root / "runtime", training_environment=target)
                self.assertEqual(runtime.training_environment, target)
                (metadata / "HEAD").write_text("ref: refs/heads/main\n")
                with self.assertRaisesRegex(ValueError, "overlaps"):
                    LabRuntime(root / "runtime", training_environment=target)
                shutil.rmtree(metadata)
                linked_metadata = root / "linked-metadata"
                linked_metadata.mkdir()
                (linked_metadata / "HEAD").write_text("ref: refs/heads/main\n")
                metadata.write_text("gitdir: ../linked-metadata\n")
                with self.assertRaisesRegex(ValueError, "overlaps"):
                    LabRuntime(root / "runtime", training_environment=target)

    def test_guard_is_rechecked_before_setup_and_invalid_receipt_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = runtime_in(folder)
            runtime.environment_receipt.parent.mkdir(parents=True)
            runtime.environment_receipt.write_text("unrelated receipt")
            with patch("lab_runtime.subprocess.run") as launch:
                with self.assertRaisesRegex(ValueError, "ownership receipt is invalid"):
                    runtime.setup()
                launch.assert_not_called()
            self.assertEqual(runtime.environment_receipt.read_text(), "unrelated receipt")
        with tempfile.TemporaryDirectory() as folder:
            runtime = runtime_in(folder)
            runtime.training_environment.mkdir(parents=True)
            (runtime.training_environment / "keep").write_text("unrelated")
            with patch("lab_runtime.subprocess.run") as launch:
                with self.assertRaisesRegex(ValueError, "not owned"):
                    runtime.setup()
                launch.assert_not_called()

    def test_initial_and_pacman_full_commands_keep_pins_counts_and_evidence_recipe(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = runtime_in(folder)
            (runtime.repo / "experiments").mkdir(parents=True)
            shutil.copy2(ROOT / "vendor/kev/experiments/q35-4b-s23.json",
                         runtime.repo / "experiments/q35-4b-s23.json")
            initial = command_flags(runtime.pretraining_command("initial"))
            self.assertEqual(initial["--base"], BASE_MODEL)
            self.assertEqual(initial["--base_revision"], BASE_REVISION)
            self.assertEqual(initial["--epochs"], "2")
            self.assertNotIn("--max_steps", initial)
            self.assertNotIn("--init_from", initial)
            parent = runtime.workspace / "skills"
            checkpoint_files(parent)
            original = copy.deepcopy(RECIPE)
            with patch("cloud_runtime.subprocess.check_output", return_value=json.dumps(META)):
                pacman = command_flags(runtime.finetuning_command("train.jsonl", "pacman", parent))
            selected = runtime.selected_pacman_recipe(RECIPE)
            for key, value in selected.items():
                self.assertEqual(pacman["--" + key], str(value))
            for flags in (initial, pacman):
                self.assertEqual(flags["--batch"], "1")
                self.assertEqual(flags["--accum"], "8")
                self.assertEqual(flags["--row_budget"], "2048")
                self.assertEqual(flags["--checkpointing"], "1")
            self.assertEqual(pacman["--max_steps"], "0")
            self.assertEqual(pacman["--epochs"], "1")
            self.assertEqual(pacman["--replay"], "2000")
            self.assertEqual(pacman["--max_state"], "4096")
            requests = COUNTS["train"] + int(pacman["--replay"])
            self.assertEqual(requests, 6096)
            self.assertEqual(math.ceil(requests / (int(pacman["--batch"]) * int(pacman["--accum"]))), 762)
            self.assertEqual(RECIPE, original)
            self.assertEqual(json.loads(json.dumps(selected)), selected)

    def test_cloud_runtime_original_environment_and_recipe_stay_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = CloudRuntime(folder)
            self.assertFalse(runtime.allow_other_gpu)
            self.assertIsNone(runtime.initial_execution)
            self.assertEqual(runtime.python, runtime.repo / ".venv/bin/python")
            self.assertEqual((RECIPE["batch"], RECIPE["accum"], RECIPE["row_budget"]), (4, 2, 0))
            with patch.dict(os.environ, {"UV_PROJECT_ENVIRONMENT": "original"}):
                self.assertEqual(runtime.environment()["UV_PROJECT_ENVIRONMENT"], "original")

    def test_lab_training_requires_real_current_kernel_check(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = runtime_in(folder)
            with patch.object(CloudRuntime, "prepare_training", return_value={"audit": "fixture"}) as prepare:
                self.assertEqual(runtime.prepare_training(), {"audit": "fixture"})
                prepare.assert_called_once_with(run_training_preflight=True)
                with self.assertRaisesRegex(ValueError, "RUN_TRAINING_PREFLIGHT=True"):
                    runtime.prepare_training(False)
            with patch("lab_runtime.stream_training") as launch:
                for checked in (None, {"result": "bindings_verified", "optimizer_steps": 0},
                                {"result": "passed", "cuda_loss_backward": "not_run", "optimizer_steps": 2},
                                {"result": "passed", "cuda_loss_backward": "passed", "optimizer_steps": 1}):
                    runtime.training_preflight = checked
                    with self.subTest(checked=checked), self.assertRaisesRegex(RuntimeError, "real optimized CUDA"):
                        runtime._train(["fixture-python", "-m", "kev.train"], Path(folder)/"out", 180, "documents")
                launch.assert_not_called()

    def test_stage_memory_failure_blocks_full_training_and_clears_old_admission(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = runtime_in(folder)
            runtime.training_preflight = {"result": "passed", "cuda_loss_backward": "passed", "optimizer_steps": 2}
            runtime.memory_preflights["documents"] = {"result": "passed", "optimizer_steps": 2}
            with patch.object(runtime, "stop") as stop, \
                    patch("lab_runtime.stream_training", side_effect=RuntimeError("CUDA out of memory")), \
                    patch.object(CloudRuntime, "_train") as full:
                with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
                    runtime._train([str(runtime.python), "-m", "kev.train", "--max_state", "7552"],
                                   Path(folder)/"out", 180, "documents")
                stop.assert_called_once()
                full.assert_not_called()
            self.assertNotIn("documents", runtime.memory_preflights)
            self.assertFalse((Path(folder)/"out").exists())

    def test_stage_probe_uses_actual_command_environment_and_separate_diagnostics(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = runtime_in(folder)
            runtime.training_preflight = {"result": "passed", "cuda_loss_backward": "passed", "optimizer_steps": 2}
            command = [str(runtime.python), "-m", "kev.train", "--data", "reviewed.jsonl", "--replay", "2000",
                       "--out", str(Path(folder)/"out"), "--batch", "1", "--accum", "8"]
            measured = {"result": "passed", "optimizer_steps": 2, "peak_memory_gib": 30}
            order = []
            def probe(launch, **kwargs):
                order.append("probe")
                self.assertEqual(launch[launch.index("--")+1:], command[3:])
                self.assertEqual(kwargs["cwd"], runtime.repo)
                self.assertEqual(kwargs["env"]["UV_PROJECT_ENVIRONMENT"], str(runtime.training_environment))
                self.assertFalse(kwargs["require_metrics"])
                report = Path(launch[launch.index("--report")+1])
                self.assertTrue(report.is_relative_to(runtime.workspace/"logs/pacman"))
                report.write_text(json.dumps(measured))
            with patch.object(runtime, "stop", side_effect=lambda: order.append("stop")), \
                    patch("lab_runtime.stream_training", side_effect=probe), \
                    patch.object(CloudRuntime, "_train", side_effect=lambda *args, **kwargs: order.append("full") or "fixture-result") as full:
                self.assertEqual(runtime._train(command, Path(folder)/"out", 180, "pacman", resume=True), "fixture-result")
                full.assert_called_once_with(command, Path(folder)/"out", 180, "pacman", resume=True,
                                             allow_execution_change=False, resume_from=None)
            self.assertEqual(order, ["stop", "probe", "full"])
            self.assertEqual(runtime.memory_preflights["pacman"], measured)


if __name__ == "__main__":
    unittest.main()
