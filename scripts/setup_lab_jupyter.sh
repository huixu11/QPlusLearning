#!/usr/bin/env bash
# Run on the lab machine after logging in from your own laptop.
set -euo pipefail

fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }

export QPLUS_GPU="${QPLUS_GPU:-1}"
export QPLUS_JUPYTER_ENV="${QPLUS_JUPYTER_ENV:-/chronos_data/conda_envs/qplus-jupyter-py313}"
export QPLUS_TRAINING_ENV="${QPLUS_TRAINING_ENV:-/chronos_data/conda_envs/qplus-kev-training-py313}"
export QPLUS_LAB_SOURCE="${QPLUS_LAB_SOURCE:-/chronos_data/huixu/QPlusLearning/labs/lab-01-kev-pacman}"
export QPLUS_LAB_DIR="${QPLUS_LAB_DIR:-/chronos_data/huixu/qpluslearning-runtime-a6000}"
export QPLUS_BACKUP_DIR="${QPLUS_BACKUP_DIR:-$QPLUS_LAB_DIR/backups}"

require_absolute_paths() {
    local name value
    for name in "$@"; do
        value="${!name}"
        [[ "$value" == /* && "$value" != *'~'* ]] || fail "$name must start with / and contain no literal ~; use an absolute path."
    done
}
require_absolute_paths QPLUS_JUPYTER_ENV QPLUS_TRAINING_ENV QPLUS_LAB_SOURCE QPLUS_LAB_DIR QPLUS_BACKUP_DIR

[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || fail 'The pinned training wheels require Linux x86_64.'
[[ "$QPLUS_GPU" =~ ^[0-9]+$ ]] || fail 'QPLUS_GPU must be one physical GPU index, for example 1.'
for required in conda git nvidia-smi; do
    command -v "$required" >/dev/null 2>&1 || fail "Missing command: $required"
done
[[ -f "$QPLUS_LAB_SOURCE/lab_runtime.py" && -f "$QPLUS_LAB_SOURCE/notebooks/pacman_kev_lab_local.ipynb" ]] || fail 'Use the updated lab checkout containing lab_runtime.py and pacman_kev_lab_local.ipynb.'
git -C "$QPLUS_LAB_SOURCE" rev-parse --show-toplevel >/dev/null
nvidia-smi -i "$QPLUS_GPU" --query-gpu=index,name,memory.free,memory.total --format=csv

# Validate destinations before conda/uv can write into them. The trainer's
# adjacent ownership receipt is made by LabRuntime, never by this script.
conda_base="$(conda info --base)"
[[ -x "$conda_base/bin/python" ]] || fail 'Cannot find the base Conda Python.'
"$conda_base/bin/python" - <<'PY'
import json
import os
from pathlib import Path
import subprocess

names = ("QPLUS_JUPYTER_ENV", "QPLUS_TRAINING_ENV", "QPLUS_LAB_SOURCE", "QPLUS_LAB_DIR", "QPLUS_BACKUP_DIR")
paths = {}
for name in names:
    path = Path(os.environ[name])
    if not path.is_absolute():
        raise SystemExit(f"{name} must be an absolute path")
    paths[name] = path.resolve()
kernel, trainer = paths["QPLUS_JUPYTER_ENV"], paths["QPLUS_TRAINING_ENV"]
checkout = Path(subprocess.check_output(
    ["git", "-C", str(paths["QPLUS_LAB_SOURCE"]), "rev-parse", "--show-toplevel"],
    text=True).strip()).resolve()
for other in (kernel, checkout, paths["QPLUS_LAB_DIR"]):
    if trainer == other or trainer in other.parents or other in trainer.parents:
        raise SystemExit("The training environment must be separate from the Jupyter environment, source checkout and runtime directory")
for ancestor in (trainer, *trainer.parents):
    if (ancestor / "conda-meta").exists():
        raise SystemExit("The training prefix must be a separate uv environment, outside every Conda environment")
if trainer.exists():
    if not trainer.is_dir():
        raise SystemExit("The existing training prefix is not a directory")
    if any(trainer.iterdir()):
        receipt = trainer.with_name("." + trainer.name + ".qplus-kev-runtime.json")
        try:
            owner = json.loads(receipt.read_text())
        except (OSError, ValueError):
            raise SystemExit("Refusing an unknown populated training prefix; choose a fresh QPLUS_TRAINING_ENV")
        if owner.get("format") != "qplus-kev-uv-environment-v1" or owner.get("training_environment") != str(trainer):
            raise SystemExit("Training prefix ownership does not match; choose a fresh QPLUS_TRAINING_ENV")
PY

for parent in "$(dirname "$QPLUS_JUPYTER_ENV")" "$(dirname "$QPLUS_TRAINING_ENV")"; do
    mkdir -p "$parent"
    [[ -w "$parent" ]] || fail "No write permission for environment parent: $parent"
done
mkdir -p "$QPLUS_LAB_DIR" "$QPLUS_BACKUP_DIR"
[[ -w "$QPLUS_LAB_DIR" && -w "$QPLUS_BACKUP_DIR" ]] || fail 'Runtime and backup directories must be writable.'

export UV_CACHE_DIR="${UV_CACHE_DIR:-$QPLUS_LAB_DIR/cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$QPLUS_LAB_DIR/cache/python}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$QPLUS_LAB_DIR/cache/pip}"
export HF_HOME="${HF_HOME:-$QPLUS_LAB_DIR/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$QPLUS_LAB_DIR/cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$QPLUS_LAB_DIR/cache/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$QPLUS_LAB_DIR/cache/cuda}"
require_absolute_paths UV_CACHE_DIR UV_PYTHON_INSTALL_DIR PIP_CACHE_DIR HF_HOME TORCH_HOME TRITON_CACHE_DIR CUDA_CACHE_PATH
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

if [[ -e "$QPLUS_JUPYTER_ENV" ]]; then
    [[ -d "$QPLUS_JUPYTER_ENV/conda-meta" && -x "$QPLUS_JUPYTER_ENV/bin/python" ]] || fail 'Existing Jupyter prefix is not a Conda environment; it has been left unchanged.'
    "$QPLUS_JUPYTER_ENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 13), "Existing environment must use Python 3.13; choose a fresh prefix instead of changing it"'
else
    conda create --yes --prefix "$QPLUS_JUPYTER_ENV" python=3.13 pip
fi
conda run --no-capture-output --prefix "$QPLUS_JUPYTER_ENV" \
    python -m pip install uv jupyterlab ipykernel tensorboard==2.20.0
conda run --no-capture-output --prefix "$QPLUS_JUPYTER_ENV" \
    python -m ipykernel install --sys-prefix \
    --name qplus-a6000-py313 --display-name 'QPlusLearning (A6000, Python 3.13)'
conda run --no-capture-output --prefix "$QPLUS_JUPYTER_ENV" python -m pip check

printf '\nJupyter environment is ready. The notebook will install the separate pinned trainer.\n'
printf 'Run the following commands in this lab terminal to start Jupyter:\n\n'
for name in QPLUS_GPU QPLUS_JUPYTER_ENV QPLUS_TRAINING_ENV QPLUS_LAB_SOURCE QPLUS_LAB_DIR QPLUS_BACKUP_DIR UV_CACHE_DIR UV_PYTHON_INSTALL_DIR PIP_CACHE_DIR HF_HOME TORCH_HOME TRITON_CACHE_DIR CUDA_CACHE_PATH; do
    printf 'export %s=%q\n' "$name" "${!name}"
done
printf 'export CUDA_VISIBLE_DEVICES="$QPLUS_GPU"\n'
printf 'conda run --no-capture-output --prefix "$QPLUS_JUPYTER_ENV" jupyter lab --no-browser --ip=127.0.0.1 --port=8888 --ServerApp.port_retries=0 --ServerApp.root_dir=%q\n' "$(git -C "$QPLUS_LAB_SOURCE" rev-parse --show-toplevel)"
