"""Colab RTX PRO 6000 Blackwell helpers for an interactive Kev notebook.

Setup runs before the 90-minute session. GPU training duration needs a preflight.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
from urllib.error import URLError
from urllib.request import urlopen

from training_monitor import stream_training
from training_stages import checkpoint_fingerprint, specifications
from lora_recovery import latest_snapshot
from optimized_training import install_kernels, file_hash

CODE_REVISION = "84847f0a883d900f7de5b7a57eaa341ca7f9a6b4"
MODEL_RUN = "jaredpalmer/kev-4b@6cfce5c2fa4b4bd64026336ab649c5ca78857d52"
TARGET_GPU = "RTX PRO 6000 Blackwell"
BASE_MODEL = "Qwen/Qwen3.5-4B-Base"
BASE_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
TRAINING_SUITE = "evals/v7/decision-v7"
MEMORY_OVERRIDES = {"batch": 1, "accum": 8, "checkpointing": 1, "row_budget": 2048}

GPU_PREFLIGHT = """
import importlib.util, json, torch
if not torch.cuda.is_available():
    raise RuntimeError('Select a GPU in Runtime > Change runtime type, then reconnect.')
props = torch.cuda.get_device_properties(0)
bf16 = torch.cuda.is_bf16_supported()
# Exercise CUDA kernels and a backward pass in the installed Kev environment.
dtype = torch.bfloat16 if bf16 else torch.float32
x = torch.randn(32, 32, device='cuda', dtype=dtype, requires_grad=True)
(x @ x.T).float().square().mean().backward()
torch.cuda.synchronize()
print(json.dumps({'name': props.name, 'memory_gib': props.total_memory / 2**30,
                  'compute_capability': [props.major, props.minor],
                  'torch': torch.__version__, 'cuda': torch.version.cuda,
                  'bf16_supported': bf16, 'kernel_backward': 'passed',
                  'optional_packages_before_overlay': {name: importlib.util.find_spec(name) is not None
                                                for name in ['causal_conv1d', 'fla']}}))
"""


def validate_gpu(info, allow_other_gpu=False):
    """Select precision only after checking the classroom hardware/software contract."""
    target = TARGET_GPU.lower() in info["name"].lower()
    if not target and not allow_other_gpu:
        raise RuntimeError(f"This lab targets {TARGET_GPU}; received {info['name']}. "
                           "Select the target GPU if available, or use the instructor's validated fallback.")
    # The pinned PyTorch 2.8 CUDA 12.8 wheel does not support Pascal/P100.
    if tuple(info["compute_capability"]) < (7, 0):
        raise RuntimeError("This environment does not support P100/Pascal GPUs. Use the target GPU or a validated BF16-capable fallback.")
    if not info["torch"].startswith("2.8.0") or tuple(map(int, info["cuda"].split(".")[:2])) < (12, 8):
        raise RuntimeError("The lab requires its pinned PyTorch 2.8.0 / CUDA 12.8 environment. Re-run setup.")
    if info["kernel_backward"] != "passed":
        raise RuntimeError("CUDA forward/backward preflight did not pass.")
    if not info["bf16_supported"]:
        raise RuntimeError("Optimized training must support BF16. Check the installed CUDA environment.")
    return "bf16"


class CloudRuntime:
    def __init__(self, workspace, allow_other_gpu=False, training_profile="memory_safe"):
        if training_profile not in {"memory_safe", "published_reference"}:
            raise ValueError("Choose memory_safe or published_reference training_profile")
        self.workspace = Path(workspace).resolve()
        self.repo = self.workspace / "kev"
        self.python = self.repo / ".venv" / "bin" / "python"
        self.process = None
        self.log = None
        self.allow_other_gpu = allow_other_gpu
        self.gpu = None
        self.dtype = "bf16"
        self.training_profile = training_profile
        self.training_preflight = None

    def selected_recipe(self, recipe):
        return {**recipe, **(MEMORY_OVERRIDES if self.training_profile == "memory_safe" else {})}

    def _profile(self, command):
        if self.training_profile == "memory_safe":
            for key, value in MEMORY_OVERRIDES.items():
                flag = "--" + key
                if flag in command:
                    command[command.index(flag) + 1] = str(value)
                else:
                    command += [flag, str(value)]
        return command

    def setup(self):
        self.workspace.mkdir(parents=True, exist_ok=True)
        subprocess.run(["nvidia-smi"], check=True)
        if not shutil.which("uv"):
            subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)
        if not self.repo.exists():
            subprocess.run(["git", "clone", "https://github.com/jaredpalmer/kev.git", str(self.repo)], check=True)
            subprocess.run(["git", "checkout", CODE_REVISION], cwd=self.repo, check=True)
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.repo, text=True).strip()
        if actual != CODE_REVISION:
            raise RuntimeError("Existing Kev checkout differs from the lab pin. Use a fresh workspace.")
        subprocess.run(["uv", "sync", "--frozen", "--no-dev", "--extra", "serve", "--python", "3.13"], cwd=self.repo, check=True)
        self.gpu = json.loads(subprocess.check_output([str(self.python), "-c", GPU_PREFLIGHT], cwd=self.repo, text=True))
        self.dtype = validate_gpu(self.gpu, self.allow_other_gpu)
        install_kernels(self.python, self.repo, self.workspace)
        (self.workspace / "runtime-preflight.json").write_text(json.dumps(self.gpu, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(self.gpu, indent=2))
        print(f"Prepared pinned Kev environment using {self.dtype}. Download weights before class.")

    def environment(self):
        env = dict(os.environ)
        # BF16 on the target or another explicitly validated compatible GPU.
        env.update(KEV_DTYPE=self.dtype, KEV_BACKEND="torch", KEV_FUSED="0", KEV_CUDA_GRAPHS="0", PYTHONUNBUFFERED="1",
                   USE_HUB_KERNELS="0", LAB_REQUIRE_OPTIMIZED_KERNELS="1", LAB_FUSED_ADAMW="1")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        # Recovery settings belong to one launch, never inherit another run's.
        for name in ("LAB_RECOVERY_ROOT", "LAB_RESUME_FROM", "LAB_SAVE_STEPS", "LAB_SAVE_SECONDS"):
            env.pop(name, None)
        env.pop("KEV_API_KEY", None)  # this server binds only to notebook-local loopback
        return env

    def prepare_training(self, run_training_preflight=False):
        """Fetch verified training data/base weights and inspect exactly what can train."""
        self.training_preflight = None
        audit_path = self.workspace / "trainable-parameters.json"
        self.stop()
        # A helper refresh in the same connected runtime need not reinstall the
        # environment. Keep the GPU receipt from its completed setup.
        gpu_receipt = self.workspace / "runtime-preflight.json"
        if self.gpu is None and gpu_receipt.is_file():
            self.gpu = json.loads(gpu_receipt.read_text())
            self.dtype = validate_gpu(self.gpu, self.allow_other_gpu)
        code = """
import json, sys
from kev.suite import load_split
from kev.model import DecisionModel, load_tokenizer
rows = load_split(sys.argv[1], 'train')
tok = load_tokenizer(sys.argv[2], revision=sys.argv[3])
model = DecisionModel(sys.argv[2], tok, 'cpu', lora=16, revision=sys.argv[3], lora_targets='all')
trainable = {n:p.numel() for n,p in model.named_parameters() if p.requires_grad}
assert trainable and any(n.startswith('head.') for n in trainable)
assert all(n.startswith('head.') or 'lora_A' in n or 'lora_B' in n for n in trainable)
assert any('lora_A' in n for n in trainable)
report = {'base':sys.argv[2], 'base_revision':sys.argv[3], 'suite_records':len(rows),
          'trainable_parameters':trainable, 'original_base_matrices_frozen':True,
          'effective_encoder_features_fixed':False, 'lora_rank':16, 'lora_alpha':32}
with open(sys.argv[4], 'w') as out: json.dump(report, out, indent=2)
print('Prepared', len(rows), 'verified training records and base weights. LoRA/head audit saved.')
"""
        subprocess.run([str(self.python), "-c", code, TRAINING_SUITE, BASE_MODEL, BASE_REVISION, str(audit_path)],
                       cwd=self.repo, env=self.environment(), check=True)
        report = self.workspace / "optimized-training-preflight.json"
        log_dir = self.workspace / "logs" / "kernel-preflight" / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
        command = [str(self.python), "-u", str(Path(__file__).with_name("optimized_training.py")),
                   "--base", BASE_MODEL, "--revision", BASE_REVISION,
                   "--suite", TRAINING_SUITE, "--report", str(report)]
        if run_training_preflight:
            print("Running optional two-record Kev loss/backward check (no CUDA profiler)", flush=True)
        else:
            command.append("--bindings_only")
            print("Checking optimized bindings; skipping separate two-record training preflight", flush=True)
        stream_training(command,
                        cwd=self.repo, env=self.environment(), log_dir=log_dir,
                        stage="kernel-preflight", timeout_seconds=1200 if run_training_preflight else 120, require_metrics=False)
        self.training_preflight = json.loads(report.read_text())
        return json.loads(audit_path.read_text())

    def pretraining_command(self, output, steps=0):
        if steps < 0:
            raise ValueError("steps must be nonnegative; 0 runs the complete two-epoch recipe")
        recipe = json.loads((self.repo / "experiments/q35-4b-s23.json").read_text())[0]
        assert recipe["base"] == BASE_MODEL and recipe["base_revision"] == BASE_REVISION
        if self.dtype != "bf16":
            raise RuntimeError("The published initial recipe requires a BF16-capable GPU.")
        command = [str(self.python), "-m", "kev.train", "--suite", TRAINING_SUITE,
                   "--out", str(Path(output).resolve()), "--device", "cuda",
                   "--lora", "16", "--lora_targets", "all", "--head_dim", "256",
                   "--weights_dtype", "fp32"]
        for key, value in recipe.items():
            command += ["--" + key, str(value)]
        if steps:
            command += ["--max_steps", str(steps)]
        return self._profile(command)

    def pretrain(self, output, steps=0, timeout_minutes=180, resume=False):
        """Published decision-v7 base stage, without --init_from. Full run is prework."""
        return self._train(self.pretraining_command(output, steps), output, timeout_minutes, "initial", resume=resume)

    def prepare_intermediate_data(self):
        code = "import sys; sys.path.insert(0, sys.argv[1]); from training_stages import prepare_data; prepare_data(sys.argv[2])"
        subprocess.run([str(self.python), "-c", code, str(Path(__file__).parent), str(self.repo)],
                       cwd=self.repo, env=self.environment(), check=True)

    def intermediate_command(self, stage, output, init_from):
        if self.dtype != "bf16":
            raise RuntimeError("Published intermediate stages require a BF16-capable GPU.")
        spec = specifications()[stage]
        command = self.training_command(self.repo / spec["data"], output, init_from=init_from)
        recipe = spec["args"]
        for key, value in recipe.items():
            flag = "--" + key
            if flag in command:
                command[command.index(flag) + 1] = str(value)
            else:
                command += [flag, str(value)]
        # Restore the generic dates-stage defaults, rather than inheriting the
        # task-specific Pac-Man augmentation/length/checkpointing settings.
        if stage == "dates":
            for key, value in {"checkpointing": 1, "max_state": 384,
                               "p_none": 0.1, "p_none_distract": 0.12, "p_distract": 0.15}.items():
                command[command.index("--" + key) + 1] = str(value)
        command += ["--suite", TRAINING_SUITE]
        return self._profile(command)

    def intermediate(self, stage, output, init_from, timeout_minutes=180, resume=False):
        command = self.intermediate_command(stage, output, init_from)
        return self._train(command, output, timeout_minutes, stage, resume=resume)

    def start(self, run=MODEL_RUN):
        self.stop()
        self.log = open(self.workspace / "server.log", "w", encoding="utf-8")
        self.process = subprocess.Popen([str(self.python), "-m", "kev.serve", "--run", str(run), "--host", "127.0.0.1", "--port", "8009"], cwd=self.repo, env=self.environment(), stdout=self.log, stderr=subprocess.STDOUT)
        os.environ["KEV_BASE_URL"] = "http://127.0.0.1:8009"
        os.environ.pop("KEV_API_KEY", None)
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("Kev stopped while loading. Inspect server.log in the notebook workspace.")
            try:
                with urlopen("http://127.0.0.1:8009/v1/models", timeout=3) as response:
                    models = json.load(response)
                (self.workspace / "models.json").write_text(json.dumps(models, indent=2) + "\n", encoding="utf-8")
                print("Kev ready at notebook-local localhost:8009")
                return models
            except (URLError, TimeoutError):
                time.sleep(3)
        self.stop()
        raise TimeoutError("Model startup exceeded 15 minutes. Inspect server.log and connection status.")

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None

    def training_command(self, training_data, output, init_from, steps=0, max_state=2048):
        # Read architecture from the checkpoint rather than guessing compatible flags.
        code = "import json,sys; from kev.checkpoint import Checkpoint; m=Checkpoint(sys.argv[1]).meta; print(json.dumps({k:getattr(m,k) for k in ['base','base_revision','lora','head_dim','option_isolation','special_embeddings','weights_dtype','weights']}))"
        meta = json.loads(subprocess.check_output([str(self.python), "-c", code, str(init_from)], cwd=self.repo, env=self.environment(), text=True))
        if meta["base"] != BASE_MODEL or meta["base_revision"] != BASE_REVISION:
            raise RuntimeError("This Kev-4B lab requires a checkpoint from its pinned 4B backbone. Start fresh; 0.8B adapters are incompatible.")
        if meta["weights"] != "lora":
            raise RuntimeError("This classroom exercise expects the pinned small LoRA checkpoint.")
        command = [str(self.python), "-m", "kev.train", "--init_from", str(init_from), "--data", str(Path(training_data).resolve()), "--out", str(Path(output).resolve()), "--base", meta["base"], "--lora", str(meta["lora"]), "--head_dim", str(meta["head_dim"]), "--option_isolation", str(int(meta["option_isolation"])), "--special_embeddings", str(int(meta["special_embeddings"])), "--weights_dtype", meta["weights_dtype"], "--dtype", self.dtype, "--device", "cuda", "--epochs", "2", "--lr", "2e-5", "--batch", "1", "--accum", "8", "--max_steps", str(steps), "--max_state", str(max_state), "--checkpointing", "1", "--p_none", "0", "--p_none_distract", "0", "--p_distract", "0", "--seed", "7"]
        command[command.index("--max_state") + 1] = str(max_state)
        command[command.index("--dtype") + 1] = self.dtype
        # The pinned release targets all modules (including hybrid projections).
        # warm_start checks that its adapter tensors match the training model.
        command += ["--lora_targets", "all"]
        if meta["base_revision"]:
            command += ["--base_revision", meta["base_revision"]]
        return self._profile(command)

    def finetune(self, training_data, output, init_from, steps=0, resume=False):
        command = self.training_command(training_data, output, init_from, steps)
        return self._train(command, output, 20, "pacman", resume=resume)

    def _train(self, command, output, timeout_minutes, stage, resume=False):
        output = Path(output).resolve()
        recovery_root = Path(str(output) + "-recovery")
        snapshot = latest_snapshot(recovery_root) if resume else None
        if resume and (snapshot is None or not output.is_dir()):
            raise RuntimeError("No resumable LoRA snapshot for this output. Older failed runs saved logs/config only; choose a new output.")
        if (output.exists() or recovery_root.exists()) and not resume:
            raise RuntimeError("Choose a new checkpoint output directory, or resume=True for a run with recovery snapshots.")
        if resume and (output / "run-evidence.json").exists():
            raise RuntimeError("This checkpoint is already complete; use it as the next stage's input.")
        report = self.workspace / "optimized-training-preflight.json"
        if self.training_preflight is None or self.training_preflight.get("result") not in {"passed", "bindings_verified"}:
            raise RuntimeError("Run setup and prepare_training first; pinned optimized bindings are required.")
        self.stop()  # release the inference model's GPU allocation before training
        started = time.perf_counter()
        log_dir = self.workspace / "logs" / stage / (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
        # The pinned trainer is instrumented in memory; native --resume is
        # full-weight-only, so the lab restores LoRA state through guarded hooks.
        observer = Path(__file__).with_name("training_monitor.py")
        launch = [command[0], "-u", str(observer), *command[3:]]
        env = self.environment()
        env.update(LAB_RECOVERY_ROOT=str(recovery_root), LAB_SAVE_STEPS="100", LAB_SAVE_SECONDS="300")
        if snapshot:
            env["LAB_RESUME_FROM"] = str(snapshot)
        try:
            stream_training(launch, cwd=self.repo, env=env, log_dir=log_dir,
                            stage=stage, timeout_seconds=timeout_minutes * 60)
        except (Exception, KeyboardInterrupt):
            saved = latest_snapshot(recovery_root)
            if saved:
                info = json.loads((saved / "complete.json").read_text())
                print(f"Recovery available at step {info['step']}/{info['steps']}: {saved}. "
                      "Rerun with resume=True and identical arguments. Preserve this directory before disconnecting.", flush=True)
            else:
                print("No learned checkpoint saved yet. Logs: " + str(log_dir), flush=True)
            raise
        evidence = {"stage": stage, "command": command, "elapsed_seconds_including_load_save": time.perf_counter() - started,
                    "gpu": self.gpu, "code_revision": CODE_REVISION, "observer_command": launch,
                    "training_logs": str(log_dir), "telemetry": "one sample per optimizer step",
                    "training_profile": self.training_profile, "recovery_root": str(recovery_root),
                    "resumed_from": str(snapshot) if snapshot else None,
                    "optimized_training_preflight": self.training_preflight,
                    "optimized_training_preflight_sha256": file_hash(report)}
        if "--init_from" in command:
            evidence["parent_checkpoint_sha256"] = checkpoint_fingerprint(command[command.index("--init_from") + 1])
        if "--data" in command:
            import hashlib
            evidence["training_data_sha256"] = hashlib.sha256(Path(command[command.index("--data") + 1]).read_bytes()).hexdigest()
        (output / "run-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        metrics = json.loads((output / "training_metrics.json").read_text())
        if metrics["optimizer_steps"] < 1:
            raise RuntimeError("Training completed without a positive optimizer step.")
        print(f"{stage} training finished in {evidence['elapsed_seconds_including_load_save'] / 60:.1f} minutes")
        return output
