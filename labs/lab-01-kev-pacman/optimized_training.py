"""Pinned CUDA training kernels and a real Kev loss/backward preflight.

Install an additive, hash-verified wheel overlay; keep Kev's uv.lock intact.
Inference-only Kev fusion and CUDA graphs are not used for training.
"""
import argparse
from contextlib import contextmanager
import faulthandler
import functools
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent


class ProgressWatchdog:
    """Print Python stacks after a phase stops progressing; never kill the run."""
    def __init__(self, seconds=60):
        self.seconds = seconds

    def __enter__(self):
        faulthandler.enable()
        self.touch()
        return self

    def touch(self):
        faulthandler.cancel_dump_traceback_later()
        faulthandler.dump_traceback_later(self.seconds, repeat=True)

    def __call__(self, message):
        print("LAB_PHASE " + json.dumps({"phase": message}), flush=True)
        self.touch()

    def __exit__(self, *exception):
        faulthandler.cancel_dump_traceback_later()


@contextmanager
def track_kernel_calls(bindings):
    """Count actual function dispatch without collecting a CPU/CUDA trace."""
    originals, calls = [], {}
    def wrap(name, function):
        @functools.wraps(function)
        def counted(*args, **kwargs):
            calls[name] += 1
            return function(*args, **kwargs)
        return counted
    try:
        for name, owner, attribute in bindings:
            function = getattr(owner, attribute)
            calls[name] = 0
            originals.append((owner, attribute, function))
            setattr(owner, attribute, wrap(name, function))
        yield calls
    finally:
        for owner, attribute, function in reversed(originals):
            setattr(owner, attribute, function)


def kernel_lock():
    return json.loads((ROOT / "training-kernels.json").read_text())


def validate_platform(info):
    expected = kernel_lock()["platform"]
    for name, value in expected.items():
        if info.get(name) != value:
            raise RuntimeError(f"Optimized kernel wheel needs {name}={value!r}; got {info.get(name)!r}")
    if not info.get("bf16"):
        raise RuntimeError("Optimized training requires a BF16-capable CUDA GPU. Select the lab target.")


def install_kernels(python, repo, workspace):
    code = """import json, platform, sys, torch
print(json.dumps({'python':f'{sys.version_info.major}.{sys.version_info.minor}',
                  'system':platform.system(), 'machine':platform.machine(),
                  'torch':torch.__version__.split('+')[0], 'cuda':torch.version.cuda,
                  'cxx11_abi':torch._C._GLIBCXX_USE_CXX11_ABI,
                  'bf16':torch.cuda.is_available() and torch.cuda.is_bf16_supported()}))
"""
    validate_platform(json.loads(subprocess.check_output([str(python), "-c", code], cwd=repo, text=True)))
    cache = Path(workspace) / "kernel-wheels"
    cache.mkdir(parents=True, exist_ok=True)
    wheels = []
    for name, spec in kernel_lock()["packages"].items():
        path = cache / spec["filename"]
        if not path.is_file() or file_hash(path) != spec["sha256"]:
            temporary = path.with_suffix(".download")
            print(f"Downloading pinned training kernel: {name} {spec['version']}", flush=True)
            with urlopen(spec["url"], timeout=60) as source, temporary.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
            if file_hash(temporary) != spec["sha256"]:
                raise ValueError(f"Kernel wheel checksum mismatch: {name}")
            temporary.replace(path)
        wheels.append(str(path))
    # --no-deps prevents pip from replacing Torch/CUDA/Transformers in Kev's lock.
    subprocess.run(["uv", "pip", "install", "--python", str(python), "--no-deps", *wheels], cwd=repo, check=True)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_implementation(function):
    """Read Transformers 5.17's captured implementation, not its wrapped name."""
    visited = set()
    while function is not None and id(function) not in visited:
        visited.add(id(function))
        if inspect.isfunction(function):
            implementation = inspect.getclosurevars(function).nonlocals.get("implementation")
            if implementation is not None:
                return implementation
        function = getattr(function, "__wrapped__", None)
    raise RuntimeError("Cannot verify Transformers' selected kernel; review the pinned integration.")


def require_optimized_bindings():
    from importlib.metadata import version
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    kernels = {}
    for name, function, package in [("delta_rule", qwen.torch_chunk_gated_delta_rule, "fla."),
                                    ("convolution", qwen.causal_conv1d_fn, "causal_conv1d.")]:
        implementation = selected_implementation(function)
        module = implementation.__module__
        if not module.startswith(package):
            raise RuntimeError(f"{name} selected {module}: reference fallback forbidden. Re-run optimized setup.")
        kernels[name] = module + "." + implementation.__name__
    versions = {name: version(name) for name in [*kernel_lock()["packages"], "torch", "triton", "transformers", "peft"]}
    for name, spec in kernel_lock()["packages"].items():
        if versions[name].split("+")[0] != spec["version"]:
            raise RuntimeError(f"Unpinned training kernel: {name} {versions[name]}")
    return {"kernels": kernels, "versions": versions}


def verify_bindings(report):
    """Fast admission only: do not claim that a CUDA training step was tested."""
    import torch
    checked = require_optimized_bindings()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required for optimized training")
    checked.update(result="bindings_verified", optimizer_steps=0, cuda_loss_backward="not_run",
                   scope="Pinned optimized bindings and CUDA/BF16 availability only; no model training executed")
    Path(report).write_text(json.dumps(checked, indent=2) + "\n")
    print("Optimized bindings verified; separate two-record training check skipped. "
          "The first real training step exercises loss/backward/optimizer.", flush=True)


def verify_training(base, revision, suite, report):
    with ProgressWatchdog() as phase:
        phase("importing pinned Kev/Transformers training stack")
        _verify_training(base, revision, suite, report, phase)


def _verify_training(base, revision, suite, report, phase):
    import sys
    import torch
    from kev.model import DecisionModel, load_tokenizer
    from kev.suite import load_split
    from kev.train import parse_args, encode_batch, row_passes, batch_loss
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    checked = require_optimized_bindings()
    print("Selected optimized training kernels: " + json.dumps(checked), flush=True)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required for the training preflight")
    # Reuse Kev's parser/encoder/augmentation/objective and the actual pinned base.
    sys.argv = ["kernel-preflight", "--suite", suite, "--out", str(Path(report).with_suffix(".unused")),
                "--base", base, "--base_revision", revision, "--device", "cuda", "--dtype", "bf16",
                "--lora", "16", "--lora_targets", "all", "--p_none_pair", "0.25",
                "--checkpointing", "1", "--batch", "1", "--accum", "8", "--row_budget", "2048", "--seed", "2"]
    args = parse_args()
    torch.manual_seed(args.seed)
    print("Loading the actual Kev backbone/LoRA/head for CUDA training verification", flush=True)
    phase("loading tokenizer and actual backbone/LoRA/head")
    tok = load_tokenizer(base, revision=revision)
    model = DecisionModel(base, tok, "cuda", lora=16, revision=revision, lora_targets="all")
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.lm.config.use_cache = False
    if model.lm.config._attn_implementation != "sdpa":
        raise RuntimeError("Kev CUDA full attention must use PyTorch SDPA")
    model.train()
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4, fused=True)
    requests = load_split(suite, "train")[:2]
    if len(requests) != 2:
        raise RuntimeError("Preflight needs two verified suite records")
    losses = []
    began = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    bindings = [("delta_rule", qwen, "torch_chunk_gated_delta_rule"),
                ("convolution", qwen, "causal_conv1d_fn"),
                ("fused_adamw", torch, "_fused_adamw_")]
    with track_kernel_calls(bindings) as calls:
        for index, request in enumerate(requests, 1):
            record_started = time.perf_counter()
            print(f"CUDA training preflight {index}/2: FLA/conv compilation, loss, backward, fused AdamW", flush=True)
            phase(f"check {index}/2: encoding record")
            batch = encode_batch(model, tok, args, [request], 0)
            loss_sum = 0
            for number, part in enumerate(row_passes(batch, args.row_budget, args.shared_prefix), 1):
                phase(f"check {index}/2, pass {number}: forward/loss; may compile new Triton kernels")
                loss, _ = batch_loss(model, args, part, "cuda", {}, None, torch.autocast("cuda", dtype=torch.bfloat16))
                phase(f"check {index}/2, pass {number}: backward; may compile new Triton kernels")
                loss.backward()
                loss_sum += float(loss.detach())
            phase(f"check {index}/2: checking finite LoRA/head gradients")
            names = [name for name, p in model.named_parameters() if p.grad is not None]
            if not any(name.startswith("head.") for name in names) or not any("lora_B" in name for name in names):
                raise RuntimeError("Preflight did not backpropagate through the LoRA and pointer head")
            finite = torch.stack([torch.isfinite(p.grad).all() for p in model.trainable_parameters() if p.grad is not None]).all()
            if not finite.item():
                raise RuntimeError("Non-finite gradients in optimized training preflight")
            phase(f"check {index}/2: clipping gradients and fused AdamW step")
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            opt.zero_grad()
            torch.cuda.synchronize()
            losses.append(loss_sum)
            phase(f"check {index}/2 completed in {time.perf_counter() - record_started:.1f}s; loss {loss_sum:.4f}")
        torch.cuda.synchronize()
    phase("both optimizer checks complete; writing verification report")
    if not all(calls.values()) or calls["fused_adamw"] < 2:
        raise RuntimeError(f"Preflight did not execute all optimized kernels: {calls}")
    checked.update(result="passed", optimizer="torch.optim.AdamW(fused=True)", attention="sdpa",
                   kernel_calls=calls, cuda_loss_backward="passed",
                   dtype="bf16 autocast / fp32 parameters", checkpointing=True,
                   optimizer_steps=2, losses=losses, peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
                   seconds_including_kernel_compilation=time.perf_counter() - began,
                   scope="Two actual decision-v7 records: loss/backward/optimizer; not a full-stage VRAM or timing result")
    Path(report).write_text(json.dumps(checked, indent=2) + "\n")
    print("Optimized Kev loss/backward preflight passed. Report: " + str(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--bindings_only", action="store_true")
    for name in ("base", "revision", "suite"):
        parser.add_argument("--" + name)
    cli = parser.parse_args()
    if cli.bindings_only:
        verify_bindings(cli.report)
    else:
        if not all((cli.base, cli.revision, cli.suite)):
            parser.error("--base, --revision and --suite are required for the two-record training check")
        verify_training(cli.base, cli.revision, cli.suite, cli.report)
