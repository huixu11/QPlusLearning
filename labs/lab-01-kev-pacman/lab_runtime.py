"""Local-Jupyter runtime for a BF16-capable laboratory GPU.

Keep the pinned classroom helpers unchanged.  Jupyter runs in its own Conda
environment; Kev's frozen Python 3.13 environment is a separately owned uv venv.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from cloud_runtime import (
    CODE_REVISION, GPU_PREFLIGHT, CloudRuntime, validate_gpu,
    validate_initial_execution,
)
from optimized_training import install_kernels
from lab_native_kernels import ensure_native_kernels
from training_monitor import stream_training


DEFAULT_TRAINING_ENVIRONMENT = "/chronos_data/conda_envs/qplus-kev-training-py313"
DEFAULT_JUPYTER_ENVIRONMENT = "/chronos_data/conda_envs/qplus-jupyter-py313"
DEFAULT_CUDA_TOOLKIT = "/chronos_data/conda_envs/qplus-cuda-toolkit-12.8"
LAB_EXECUTION = {"batch": 1, "accum": 8, "row_budget": 2048}
ENVIRONMENT_FORMAT = "qplus-kev-uv-environment-v1"


def _overlap(first, second):
    return first.is_relative_to(second) or second.is_relative_to(first)


def _checkout_root(source):
    """Find real Git metadata; empty sandbox .git placeholders are not roots."""
    for parent in (source, *source.parents):
        metadata = parent / ".git"
        if metadata.is_dir() and (metadata / "HEAD").is_file():
            return parent
        if metadata.is_file():
            try:
                pointer = metadata.read_text().strip()
                if pointer.startswith("gitdir:"):
                    directory = Path(pointer[len("gitdir:"):].strip())
                    directory = directory if directory.is_absolute() else parent / directory
                    if (directory / "HEAD").is_file():
                        return parent
            except (OSError, ValueError):
                pass
    return source


class LabRuntime(CloudRuntime):
    """Use one selected GPU and preserve the curriculum's effective batch of 8.

    ``training_environment`` must name a dedicated uv environment, never the
    active Jupyter/Conda environment. Set QPLUS_TRAINING_ENV to change its default.
    Both execution overrides retain effective batch 8 and are independent copies.
    """

    def __init__(self, workspace, training_environment=None, allow_other_gpu=True,
                 training_profile="memory_safe", initial_execution=None,
                 pacman_execution=None):
        initial = LAB_EXECUTION if initial_execution is None else initial_execution
        super().__init__(workspace, allow_other_gpu=allow_other_gpu,
                         training_profile=training_profile, initial_execution=initial)
        self.pacman_execution = validate_initial_execution(
            LAB_EXECUTION if pacman_execution is None else pacman_execution)
        target = Path(training_environment if training_environment is not None else
                      os.environ.get("QPLUS_TRAINING_ENV", DEFAULT_TRAINING_ENVIRONMENT)).expanduser()
        if not target.is_absolute():
            raise ValueError("training_environment must be an absolute, dedicated uv environment path")
        self.training_environment = target.resolve()
        self.python = self.training_environment / "bin" / "python"
        self.environment_receipt = self.training_environment.with_name(
            "." + self.training_environment.name + ".qplus-kev-runtime.json")
        self.memory_preflights = {}
        self.native_training_kernels = None
        self._guard_training_environment()

    def _ownership(self):
        return {"format": ENVIRONMENT_FORMAT,
                "training_environment": str(self.training_environment),
                "code_revision": CODE_REVISION}

    def _guard_training_environment(self):
        """Refuse uv exact-sync against a kernel, Conda env, checkout or unknown files."""
        target = self.training_environment
        protected = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(), self.repo}
        if os.environ.get("CONDA_PREFIX"):
            protected.add(Path(os.environ["CONDA_PREFIX"]).resolve())
        sources = [Path(__file__).resolve().parent]
        if os.environ.get("QPLUS_LAB_SOURCE"):
            sources.append(Path(os.environ["QPLUS_LAB_SOURCE"]).expanduser().resolve())
        for source in sources:
            protected.add(_checkout_root(source))
        if any(_overlap(target, path) for path in protected):
            raise ValueError("Training environment overlaps the active Python/Conda environment or a source checkout; "
                             "choose a separate uv prefix")
        if any((path / "conda-meta").exists() for path in (target, *target.parents)):
            raise ValueError("Training prefix is inside an existing Conda environment; "
                             "uv exact sync must not modify a Conda environment")
        if target.exists() and not target.is_dir():
            raise ValueError("Training prefix already exists and is not a directory")
        owned = False
        if self.environment_receipt.exists():
            try:
                owned = json.loads(self.environment_receipt.read_text()) == self._ownership()
            except (OSError, ValueError):
                pass
            if not owned:
                raise ValueError("Training environment ownership receipt is invalid; choose a fresh uv prefix")
        if target.is_dir() and any(target.iterdir()) and not owned:
            raise ValueError("Training prefix is populated but is not owned by this lab runtime; "
                             "choose a fresh uv prefix and preserve the existing directory")

    def _claim_training_environment(self):
        self._guard_training_environment()
        self.training_environment.parent.mkdir(parents=True, exist_ok=True)
        if not self.environment_receipt.exists():
            # A sidecar does not populate uv's new venv directory. Write it before
            # sync so an interrupted installation can be retried without adopting
            # unrelated existing environments.
            with self.environment_receipt.open("x", encoding="utf-8") as receipt:
                receipt.write(json.dumps(self._ownership(), indent=2) + "\n")

    def environment(self):
        env = super().environment()
        env["UV_PROJECT_ENVIRONMENT"] = str(self.training_environment)
        env["VIRTUAL_ENV"] = str(self.training_environment)
        override = env.get("QPLUS_CUDA_HOME")
        if override:
            toolkit = Path(override).expanduser()
            if not toolkit.is_absolute():
                raise ValueError("QPLUS_CUDA_HOME must be an absolute CUDA Toolkit path")
        else:
            # Reuse the user's existing Jupyter Conda Toolkit before considering
            # an optional separate compiler prefix. Do not create another env.
            toolkit = next((path for path in (
                Path(sys.prefix), Path(DEFAULT_JUPYTER_ENVIRONMENT), Path(DEFAULT_CUDA_TOOLKIT)
            ) if (path / "bin/nvcc").is_file()), None)
        if toolkit is not None:
            env["CUDA_HOME"] = str(toolkit)
            env["PATH"] = str(toolkit / "bin") + os.pathsep + env.get("PATH", "")
        return env

    def setup(self):
        """Install the same frozen Kev, CUDA and checksum-verified kernel pins.

        The explicit subprocess environment avoids changing the notebook's
        process environment, even if installation or the CUDA check fails.
        """
        self._guard_training_environment()
        self.training_preflight = None
        self.native_training_kernels = None
        self.memory_preflights.clear()
        self.workspace.mkdir(parents=True, exist_ok=True)
        from pacman_lab import ensure_node
        ensure_node(self.workspace)
        subprocess.run(["nvidia-smi"], check=True)
        if not shutil.which("uv"):
            subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)
        if not self.repo.exists():
            subprocess.run(["git", "clone", "https://github.com/jaredpalmer/kev.git", str(self.repo)], check=True)
            subprocess.run(["git", "checkout", CODE_REVISION], cwd=self.repo, check=True)
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.repo, text=True).strip()
        if actual != CODE_REVISION:
            raise RuntimeError("Existing Kev checkout differs from the lab pin. Use a fresh workspace.")
        self._claim_training_environment()
        env = self.environment()
        subprocess.run(["uv", "sync", "--frozen", "--no-dev", "--extra", "serve", "--python", "3.13"],
                       cwd=self.repo, env=env, check=True)
        self.gpu = json.loads(subprocess.check_output([str(self.python), "-c", GPU_PREFLIGHT],
                                                     cwd=self.repo, env=env, text=True))
        self.dtype = validate_gpu(self.gpu, self.allow_other_gpu)
        install_kernels(self.python, self.repo, self.workspace)
        self.native_training_kernels = ensure_native_kernels(
            self.python, self.repo, self.workspace, env)
        (self.workspace / "lab-native-kernels.json").write_text(
            json.dumps(self.native_training_kernels, indent=2) + "\n", encoding="utf-8")
        (self.workspace / "runtime-preflight.json").write_text(
            json.dumps(self.gpu, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(self.gpu, indent=2))
        print(f"Prepared pinned Python 3.13 Kev environment at {self.training_environment} using {self.dtype}.")

    def prepare_training(self, run_training_preflight=True):
        """Require the real kernel smoke check before a lab training launch."""
        if run_training_preflight is not True:
            raise ValueError("The A6000 notebook requires RUN_TRAINING_PREFLIGHT=True; "
                             "bindings alone do not test CUDA loss/backward/optimizer")
        return super().prepare_training(run_training_preflight=True)

    def _train(self, command, output, timeout_minutes, stage, resume=False,
               allow_execution_change=False, resume_from=None):
        checked = self.training_preflight or {}
        if (checked.get("result") != "passed" or checked.get("cuda_loss_backward") != "passed"
                or checked.get("optimizer_steps", 0) < 2):
            raise RuntimeError("Run setup and prepare_training(True) in this kernel before training; "
                               "a real optimized CUDA training check is required")
        # Release the serving model before a separate, disposable training probe.
        # The probe uses this stage's actual data/architecture/execution arguments,
        # and never writes a learned curriculum checkpoint.
        self.stop()
        log_dir = self.workspace / "logs" / stage / (
            "memory-check-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
        report = log_dir / "memory-preflight.json"
        helper = Path(__file__).with_name("lab_memory_check.py")
        launch = [str(self.python), "-u", str(helper), "--report", str(report), "--", *command[3:]]
        self.memory_preflights.pop(stage, None)
        print(f"Checking {stage} long-sample training memory before the full stage. "
              "This is a measured probe, not a guarantee for every later batch.", flush=True)
        stream_training(launch, cwd=self.repo, env=self.environment(), log_dir=log_dir,
                        stage=stage + "-memory-check", timeout_seconds=3600, require_metrics=False)
        measured = json.loads(report.read_text())
        if measured.get("result") != "passed" or measured.get("optimizer_steps") != 2:
            raise RuntimeError(f"Stage memory check did not pass; preserve its diagnostics: {report}")
        self.memory_preflights[stage] = measured
        print(f"{stage} measured memory report: {report}", flush=True)
        return super()._train(command, output, timeout_minutes, stage, resume=resume,
                              allow_execution_change=allow_execution_change, resume_from=resume_from)

    def selected_pacman_recipe(self, recipe):
        """Copy the pinned recipe and apply only this lab's execution override."""
        return {**recipe, **validate_initial_execution(self.pacman_execution)}

    def finetuning_command(self, training_data, output, init_from, steps=0):
        from planner_data import RECIPE
        command = super().finetuning_command(training_data, output, init_from, steps)
        for key, value in self.selected_pacman_recipe(RECIPE).items():
            flag = "--" + key
            if flag in command:
                command[command.index(flag) + 1] = str(value)
            else:
                command += [flag, str(value)]
        return command
