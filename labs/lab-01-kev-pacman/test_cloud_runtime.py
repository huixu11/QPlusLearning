"""Check hardware admission and the published-to-domain checkpoint transition."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from cloud_runtime import CloudRuntime, validate_gpu
from training_stages import specifications

ROOT = Path(__file__).resolve().parent


class CloudRuntimeTests(unittest.TestCase):
    def test_hardware_contract_and_explicit_fallback(self):
        info = {"name": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "memory_gib": 96,
                "compute_capability": [12, 0], "torch": "2.8.0+cu128", "cuda": "12.8",
                "bf16_supported": True, "kernel_backward": "passed"}
        self.assertEqual(validate_gpu(info), "bf16")
        t4 = {**info, "name": "Tesla T4", "compute_capability": [7, 5], "bf16_supported": False}
        with self.assertRaisesRegex(RuntimeError, "received Tesla T4"):
            validate_gpu(t4)
        self.assertEqual(validate_gpu(t4, allow_other_gpu=True), "fp32")
        p100 = {**t4, "name": "Tesla P100", "compute_capability": [6, 0]}
        with self.assertRaisesRegex(RuntimeError, "P100/Pascal"):
            validate_gpu(p100, allow_other_gpu=True)
        with self.assertRaisesRegex(RuntimeError, "pinned PyTorch"):
            validate_gpu({**info, "cuda": "12.6"})
        with self.assertRaisesRegex(RuntimeError, "must support BF16"):
            validate_gpu({**info, "bf16_supported": False})

    def test_initial_command_matches_selected_published_experiment(self):
        recipe_file = ROOT / "vendor/kev/experiments/q35-08b.json"
        recipe = json.loads(recipe_file.read_text())[2]
        with tempfile.TemporaryDirectory() as folder:
            runtime = CloudRuntime(folder, training_profile="published_reference")
            (runtime.repo / "experiments").mkdir(parents=True)
            shutil.copy2(recipe_file, runtime.repo / "experiments/q35-08b.json")
            command = runtime.pretraining_command(Path(folder) / "initial")
            flags = dict(zip(command[3::2], command[4::2]))
            for key, value in recipe.items():
                self.assertEqual(flags["--" + key], str(value))
            self.assertEqual(flags["--suite"], "evals/v7/decision-v7")
            self.assertEqual(flags["--lora"], "16")
            self.assertEqual(flags["--weights_dtype"], "fp32")
            self.assertNotIn("--init_from", flags)
            self.assertNotIn("--max_steps", flags)
            runtime.dtype = "fp32"
            with self.assertRaisesRegex(RuntimeError, "requires a BF16"):
                runtime.pretraining_command("unused")

    def test_domain_command_uses_learner_checkpoint_and_legal_action_set(self):
        meta = {"base": "Qwen/Qwen3.5-0.8B-Base", "base_revision": "base-pin", "lora": 16,
                "head_dim": 256, "option_isolation": False, "special_embeddings": False,
                "weights_dtype": "fp32", "weights": "lora"}
        with tempfile.TemporaryDirectory() as folder:
            runtime = CloudRuntime(folder)
            initial = Path(folder) / "initial"
            with patch("cloud_runtime.subprocess.check_output", return_value=json.dumps(meta)) as read_meta:
                command = runtime.training_command("reviewed.jsonl", "candidate", init_from=initial)
            self.assertEqual(read_meta.call_args.args[0][-1], str(initial))
        flags = dict(zip(command[3::2], command[4::2]))
        self.assertEqual(flags["--init_from"], str(initial))
        self.assertEqual(flags["--lr"], "2e-5")
        self.assertEqual(flags["--epochs"], "2")
        self.assertEqual(flags["--batch"], "1")
        self.assertEqual(flags["--accum"], "8")
        self.assertEqual(flags["--max_steps"], "0")
        self.assertEqual(flags["--dtype"], "bf16")
        for key in ["--p_none", "--p_none_distract", "--p_distract"]:
            self.assertEqual(flags[key], "0")

    def test_intermediate_stages_keep_replay_and_distinct_parent_checkpoints(self):
        meta = {"base": "Qwen/Qwen3.5-0.8B-Base", "base_revision": "base-pin", "lora": 16,
                "head_dim": 256, "option_isolation": False, "special_embeddings": False,
                "weights_dtype": "fp32", "weights": "lora"}
        with tempfile.TemporaryDirectory() as folder:
            runtime = CloudRuntime(folder, training_profile="published_reference")
            for stage, parent in [("dates", "initial"), ("documents_skills", "dates")]:
                with patch("cloud_runtime.subprocess.check_output", return_value=json.dumps(meta)):
                    command = runtime.intermediate_command(stage, "new-checkpoint", Path(folder) / parent)
                flags = dict(zip(command[3::2], command[4::2]))
                for key, value in specifications()[stage]["args"].items():
                    self.assertEqual(flags["--" + key], str(value))
                self.assertEqual(flags["--init_from"], str(Path(folder) / parent))
                self.assertEqual(flags["--suite"], "evals/v7/decision-v7")
                self.assertEqual(flags["--p_none_pair"], "0.25")
                self.assertEqual(flags["--p_none"], "0.1")
                self.assertEqual(flags["--max_steps"], "0")
                if stage == "dates":
                    self.assertEqual(flags["--checkpointing"], "0")
                    self.assertEqual(flags["--max_state"], "384")

    def test_memory_profile_preserves_effective_batch_and_curriculum(self):
        recipe_file = ROOT / "vendor/kev/experiments/q35-08b.json"
        meta = {"base": "Qwen/Qwen3.5-0.8B-Base", "base_revision": "base-pin", "lora": 16,
                "head_dim": 256, "option_isolation": False, "special_embeddings": False,
                "weights_dtype": "fp32", "weights": "lora"}
        with tempfile.TemporaryDirectory() as folder:
            runtime = CloudRuntime(folder)
            (runtime.repo / "experiments").mkdir(parents=True)
            shutil.copy2(recipe_file, runtime.repo / "experiments/q35-08b.json")
            commands = [runtime.pretraining_command("initial")]
            with patch("cloud_runtime.subprocess.check_output", return_value=json.dumps(meta)):
                commands += [runtime.intermediate_command(s, s, "parent") for s in specifications()]
                commands += [runtime.training_command("reviewed.jsonl", "domain", "parent")]
            for command in commands:
                flags = dict(zip(command[3::2], command[4::2]))
                self.assertEqual(int(flags["--batch"]) * int(flags["--accum"]), 8)
                self.assertEqual(flags["--batch"], "1")
                self.assertEqual(flags["--checkpointing"], "1")
                self.assertEqual(flags["--row_budget"], "2048")
            initial = dict(zip(commands[0][3::2], commands[0][4::2]))
            self.assertEqual(initial["--epochs"], "2")
            self.assertEqual(initial["--p_none_pair"], "0.25")
            self.assertEqual(runtime.selected_recipe({"lr": 1e-4, "batch": 8})["lr"], 1e-4)
            self.assertEqual(runtime.environment()["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")

    def test_old_failed_run_cannot_be_resumed_without_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = CloudRuntime(folder)
            output = Path(folder) / "old"
            output.mkdir()
            (output / "training_config.json").write_text('{}')
            with self.assertRaisesRegex(RuntimeError, "No resumable LoRA snapshot"):
                runtime._train([], output, 90, "initial", resume=True)
            with self.assertRaisesRegex(RuntimeError, "Choose a new"):
                runtime._train([], output, 90, "initial")


if __name__ == "__main__":
    unittest.main()
