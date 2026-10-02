"""Check hardware admission and the published-to-domain checkpoint transition."""
import ast
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from cloud_runtime import CloudRuntime, validate_gpu, BASE_MODEL, BASE_REVISION, select_initial_benchmark
from training_stages import specifications

ROOT = Path(__file__).resolve().parent


class CloudRuntimeTests(unittest.TestCase):
    def test_notebook_initial_resume_explicitly_uses_4x2_after_prior_selection(self):
        notebook = json.loads((ROOT / 'notebooks/pacman_kev_lab.ipynb').read_text())
        source = next(''.join(cell['source']) for cell in notebook['cells']
                      if cell['cell_type'] == 'code' and ''.join(cell['source']).startswith('INITIAL ='))
        tree = ast.parse(source)
        for statement in tree.body:
            if isinstance(statement, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id in {'RESUME_INITIAL', 'ALLOW_INITIAL_EXECUTION_CHANGE'}
                    for target in statement.targets):
                statement.value = ast.copy_location(ast.Constant(True), statement.value)
        code = compile(tree, '<notebook initial resume>', 'exec')
        for previous in (None, {'batch': 1, 'accum': 8, 'row_budget': 2048},
                         {'batch': 2, 'accum': 4, 'row_budget': 4096}):
            with self.subTest(previous=previous), tempfile.TemporaryDirectory() as folder:
                runtime = CloudRuntime(folder, initial_execution=previous)
                (runtime.repo / 'experiments').mkdir(parents=True)
                shutil.copy2(ROOT / 'vendor/kev/experiments/q35-4b-s23.json',
                             runtime.repo / 'experiments/q35-4b-s23.json')
                output = Path(folder) / 'checkpoints/kev-4b-initial'
                namespace = {'runtime': runtime, 'CHECKPOINT_ROOT': output.parent, 'STAGE_OWNERS': {},
                             'json': json, 'print': lambda *args, **kwargs: None,
                             'inspect_checkpoint': lambda *args, **kwargs: ({}, {})}
                with patch.object(runtime, '_train', return_value=output) as train:
                    exec(code, namespace)
                command = train.call_args.args[0]
                flags = dict(zip(command[3::2], command[4::2]))
                self.assertEqual([flags['--' + key] for key in ('batch', 'accum', 'row_budget')], ['4', '2', '0'])
                self.assertEqual(flags['--out'], str(output.resolve()))
                self.assertEqual(flags['--checkpointing'], '1')
                self.assertEqual(flags['--lr'], '5e-05')
                self.assertTrue(train.call_args.kwargs['resume'])
                self.assertTrue(train.call_args.kwargs['allow_execution_change'])
                self.assertEqual(runtime.selected_initial_recipe({})['batch'], 4)
                self.assertEqual(runtime.selected_recipe({})['batch'], 1)

    def test_initial_execution_changes_only_batching_and_keeps_published_training_recipe(self):
        recipe_file = ROOT / 'vendor/kev/experiments/q35-4b-s23.json'
        with tempfile.TemporaryDirectory() as folder:
            runtime = CloudRuntime(folder, initial_execution={'batch': 4, 'accum': 2, 'row_budget': 8192})
            (runtime.repo / 'experiments').mkdir(parents=True)
            shutil.copy2(recipe_file, runtime.repo / 'experiments/q35-4b-s23.json')
            flags = dict(zip(runtime.pretraining_command('initial')[3::2], runtime.pretraining_command('initial')[4::2]))
            self.assertEqual([flags['--' + key] for key in ('batch', 'accum', 'row_budget')], ['4', '2', '8192'])
            self.assertEqual(flags['--lr'], '5e-05')
            self.assertEqual(flags['--epochs'], '2')
            self.assertEqual(flags['--checkpointing'], '1')
            self.assertEqual(runtime.selected_initial_recipe({'batch': 4, 'accum': 2})['row_budget'], 8192)
            self.assertEqual(runtime.selected_recipe({'batch': 2, 'accum': 4})['batch'], 1)
            for settings in ({'batch': 4, 'accum': 4, 'row_budget': 0}, {'batch': 4, 'accum': 2, 'row_budget': -1},
                             {'batch': 4, 'accum': 2, 'row_budget': 0, 'lr': 0.01}):
                with self.assertRaises(ValueError):
                    CloudRuntime(folder, initial_execution=settings)
            with self.assertRaisesRegex(ValueError, 'restricted'):
                runtime.pretrain('initial', allow_execution_change=True)

    def test_benchmark_excludes_warmup_keeps_production_files_and_continues_after_failure(self):
        recipe_file = ROOT / 'vendor/kev/experiments/q35-4b-s23.json'
        with tempfile.TemporaryDirectory() as folder, patch('cloud_runtime.print'):
            initial_execution = {'batch': 1, 'accum': 8, 'row_budget': 2048}
            runtime = CloudRuntime(folder, initial_execution=initial_execution)
            runtime.gpu = {'memory_gib': 95}
            runtime.training_preflight = {'result': 'bindings_verified'}
            (runtime.repo / 'experiments').mkdir(parents=True)
            shutil.copy2(recipe_file, runtime.repo / 'experiments/q35-4b-s23.json')
            production = Path(folder) / 'checkpoints' / 'kev-4b-initial'
            production.mkdir(parents=True)
            (production / 'marker').write_text('preserve')
            def trial(command, output, timeout, stage):
                self.assertEqual(stage, 'benchmark-initial')
                self.assertNotEqual(output, production)
                self.assertIn('--max_steps', command)
                if output.name == 'published-4x2':
                    raise RuntimeError('simulated OOM')
                output.mkdir()
                seconds, peak = {'memory-1x8': (2, 18), 'bounded-2x4': (0.9, 30), 'bounded-4x2': (0.8, 78)}[output.name]
                (output / 'training_metrics.json').write_text(json.dumps({
                    'optimizer_steps': 18, 'step_seconds': [99] * 8 + [seconds] * 10, 'peak_device_bytes': peak * 2**30}))
            with patch.object(runtime, '_train', side_effect=trial) as train:
                report = runtime.benchmark_initial(steps=18, warmup_steps=8)
            self.assertEqual(train.call_count, 4)
            self.assertEqual(report['selected_name'], 'bounded-2x4')
            self.assertAlmostEqual(report['cases'][0]['mean_step_seconds'], 2)
            self.assertEqual(report['cases'][-1]['status'], 'failed')
            self.assertEqual(runtime.initial_execution, initial_execution)
            self.assertEqual((production / 'marker').read_text(), 'preserve')
            self.assertEqual(len(list((Path(folder) / 'benchmarks').glob('initial/*/report.json'))), 1)

    def test_benchmark_selection_requires_baseline_speed_gain_and_memory_headroom(self):
        baseline = {'name': 'memory-1x8', 'status': 'completed', 'mean_step_seconds': 2, 'peak_memory_gib': 18}
        faster = {'name': 'candidate', 'status': 'completed', 'mean_step_seconds': 1, 'peak_memory_gib': 60}
        self.assertIs(select_initial_benchmark([baseline, faster], 95), faster)
        self.assertIs(select_initial_benchmark([baseline, {**faster, 'mean_step_seconds': 1.95}], 95), baseline)
        self.assertIs(select_initial_benchmark([baseline, {**faster, 'peak_memory_gib': 80}], 95), baseline)
        self.assertIsNone(select_initial_benchmark([faster], 95))
        self.assertIsNone(select_initial_benchmark([{**baseline, 'peak_memory_gib': 90}], 95))

    def test_prepare_uses_fast_bindings_by_default_and_training_check_only_on_request(self):
        for full_check in (False, True):
            with self.subTest(full_check=full_check), tempfile.TemporaryDirectory() as folder:
                runtime = CloudRuntime(folder)
                def audit(command, **kwargs):
                    Path(command[-1]).write_text(json.dumps({"suite_records": 12576}))
                def check(command, **kwargs):
                    report = Path(command[command.index("--report") + 1])
                    result = "bindings_verified" if "--bindings_only" in command else "passed"
                    report.write_text(json.dumps({"result": result}))
                with patch("cloud_runtime.subprocess.run", side_effect=audit), \
                        patch("cloud_runtime.stream_training", side_effect=check) as stream:
                    if full_check:
                        result = runtime.prepare_training(run_training_preflight=True)
                    else:
                        result = runtime.prepare_training()
                command = stream.call_args.args[0]
                self.assertEqual("--bindings_only" in command, not full_check)
                self.assertEqual(stream.call_args.kwargs["timeout_seconds"], 1200 if full_check else 120)
                self.assertFalse(stream.call_args.kwargs["require_metrics"])
                self.assertEqual(result["suite_records"], 12576)
                self.assertEqual(runtime.training_preflight["result"], "passed" if full_check else "bindings_verified")

    def test_real_training_accepts_verified_bindings_and_records_unrun_separate_check(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = CloudRuntime(folder)
            output = Path(folder) / "initial"
            command = ["python", "-m", "kev.train", "--out", str(output)]
            with self.assertRaisesRegex(RuntimeError, "optimized bindings are required"):
                runtime._train(command, output, 90, "initial")
            runtime.training_preflight = {"result": "bindings_verified", "cuda_loss_backward": "not_run", "optimizer_steps": 0}
            (Path(folder) / "optimized-training-preflight.json").write_text(json.dumps(runtime.training_preflight))
            def train(*args, **kwargs):
                output.mkdir()
                (output / "training_metrics.json").write_text(json.dumps({"optimizer_steps": 1}))
            with patch("cloud_runtime.stream_training", side_effect=train):
                runtime._train(command, output, 90, "initial")
            evidence = json.loads((output / "run-evidence.json").read_text())
            self.assertEqual(evidence["optimized_training_preflight"]["cuda_loss_backward"], "not_run")
            self.assertEqual(evidence["optimized_training_preflight"]["optimizer_steps"], 0)

    def test_hardware_contract_and_explicit_fallback(self):
        info = {"name": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "memory_gib": 96,
                "compute_capability": [12, 0], "torch": "2.8.0+cu128", "cuda": "12.8",
                "bf16_supported": True, "kernel_backward": "passed"}
        self.assertEqual(validate_gpu(info), "bf16")
        t4 = {**info, "name": "Tesla T4", "compute_capability": [7, 5], "bf16_supported": False}
        with self.assertRaisesRegex(RuntimeError, "received Tesla T4"):
            validate_gpu(t4)
        with self.assertRaisesRegex(RuntimeError, "must support BF16"):
            validate_gpu(t4, allow_other_gpu=True)
        p100 = {**t4, "name": "Tesla P100", "compute_capability": [6, 0]}
        with self.assertRaisesRegex(RuntimeError, "P100/Pascal"):
            validate_gpu(p100, allow_other_gpu=True)
        with self.assertRaisesRegex(RuntimeError, "pinned PyTorch"):
            validate_gpu({**info, "cuda": "12.6"})
        with self.assertRaisesRegex(RuntimeError, "must support BF16"):
            validate_gpu({**info, "bf16_supported": False})

    def test_initial_command_matches_selected_published_experiment(self):
        recipe_file = ROOT / "vendor/kev/experiments/q35-4b-s23.json"
        recipe = json.loads(recipe_file.read_text())[0]
        with tempfile.TemporaryDirectory() as folder:
            runtime = CloudRuntime(folder, training_profile="published_reference")
            (runtime.repo / "experiments").mkdir(parents=True)
            shutil.copy2(recipe_file, runtime.repo / "experiments/q35-4b-s23.json")
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
        meta = {"base": BASE_MODEL, "base_revision": BASE_REVISION, "lora": 16,
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
        meta = {"base": BASE_MODEL, "base_revision": BASE_REVISION, "lora": 16,
                "head_dim": 256, "option_isolation": False, "special_embeddings": False,
                "weights_dtype": "fp32", "weights": "lora"}
        with tempfile.TemporaryDirectory() as folder:
            runtime = CloudRuntime(folder, training_profile="published_reference")
            for stage, parent in [("dates", "initial"), ("documents", "dates"), ("skills", "documents")]:
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
                    self.assertEqual(flags["--checkpointing"], "1")
                    self.assertEqual(flags["--max_state"], "384")

    def test_memory_profile_preserves_effective_batch_and_curriculum(self):
        recipe_file = ROOT / "vendor/kev/experiments/q35-4b-s23.json"
        meta = {"base": BASE_MODEL, "base_revision": BASE_REVISION, "lora": 16,
                "head_dim": 256, "option_isolation": False, "special_embeddings": False,
                "weights_dtype": "fp32", "weights": "lora"}
        with tempfile.TemporaryDirectory() as folder:
            runtime = CloudRuntime(folder)
            (runtime.repo / "experiments").mkdir(parents=True)
            shutil.copy2(recipe_file, runtime.repo / "experiments/q35-4b-s23.json")
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

    def test_domain_rejects_incompatible_backbone(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = CloudRuntime(folder)
            with patch("cloud_runtime.subprocess.check_output", return_value=json.dumps({
                    "base": "Qwen/Qwen3.5-0.8B-Base", "base_revision": "old"})):
                with self.assertRaisesRegex(RuntimeError, "0.8B adapters are incompatible"):
                    runtime.training_command("data.jsonl", "out", "old-parent")

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
