"""Atomic LoRA recovery snapshots, including optimizer, schedule and RNG state.

Kev's native --resume handles full-weight runs only. This adapter saves at an
optimizer boundary and restores the pinned LoRA trainer's own counters.
"""
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import uuid

COUNTERS = ("step", "seen", "tokens_seen", "peak_mem", "optimizer_seconds", "step_seconds",
            "elapsed", "start_epoch", "start_mb", "grad_norms", "run")
EXECUTION_FLAGS = {"batch", "accum", "row_budget"}


def execution_resume_position(saved, state, allow_change):
    """Map a plain single-GPU plan at an optimizer boundary to its next record."""
    old, new = saved["args"], vars(state["a"])
    changed = {key for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
    position = dict(saved["position"])
    if not changed:
        return position, None
    if not allow_change or changed - EXECUTION_FLAGS:
        raise ValueError("Resume with the same training arguments and output directory; "
                         "an explicit execution change permits only batch, accum and row_budget")
    if state.get("world", 1) != 1 or any(old.get(key, 0) or new.get(key, 0)
                                        for key in ("full_ft", "length_sort", "pass_tokens_max")):
        raise ValueError("Execution changes require a plain single-GPU LoRA microbatch plan")
    if any(not isinstance(args.get(key), int) or isinstance(args[key], bool) or args[key] < 1
           for args in (old, new) for key in ("batch", "accum")):
        raise ValueError("Execution batch and accumulation must be positive integers")
    if old["batch"] * old["accum"] != new["batch"] * new["accum"]:
        raise ValueError("Execution changes must preserve the effective batch")
    if new.get("row_budget", 0) < 0:
        raise ValueError("Row budget must be nonnegative")
    count = len(state["reqs"])
    old_mb = position["start_mb"]
    old_batches = math.ceil(count / old["batch"])
    if not 0 <= old_mb <= old_batches or (old_mb != old_batches and old_mb % old["accum"]):
        raise ValueError("Recovery position is not an optimizer boundary")
    offset = min(old_mb * old["batch"], count)
    if offset < count and offset % new["batch"]:
        raise ValueError("Execution change would repeat or skip source records")
    position["start_mb"] = math.ceil(offset / new["batch"])
    change = {"step": position["step"], "epoch": position["start_epoch"], "next_source_record_offset": offset,
              "previous": {key: old.get(key, 0) for key in sorted(EXECUTION_FLAGS)},
              "selected": {key: new.get(key, 0) for key in sorted(EXECUTION_FLAGS)}}
    return position, change


def latest_snapshot(root):
    root = Path(root).resolve()
    pointer = root / "latest.json"
    if not pointer.is_file():
        return None
    info = json.loads(pointer.read_text())
    folder = (root / info["directory"]).resolve()
    if not folder.is_relative_to(root) or not (folder / "complete.json").is_file():
        raise ValueError("Invalid/incomplete recovery snapshot")
    return folder


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_copy(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


class LoRARecovery:
    def __init__(self, root=None, resume_from=None, every_steps=100, every_seconds=300, allow_execution_change=False):
        self.root = Path(root) if root else None
        self.resume_from = Path(resume_from) if resume_from else None
        self.every_steps, self.every_seconds = every_steps, every_seconds
        if every_steps < 1 or every_seconds <= 0:
            raise ValueError("Recovery intervals must be positive")
        self.saved_at = time.monotonic()
        self.backend = None
        self.allow_execution_change = allow_execution_change
        self.execution_history = []

    @property
    def resuming(self):
        return self.resume_from is not None

    @classmethod
    def from_env(cls):
        return cls(os.environ.get("LAB_RECOVERY_ROOT"), os.environ.get("LAB_RESUME_FROM"),
                   int(os.environ.get("LAB_SAVE_STEPS", "100")), float(os.environ.get("LAB_SAVE_SECONDS", "300")),
                   allow_execution_change=os.environ.get("LAB_ALLOW_EXECUTION_CHANGE") == "1")

    def due(self, step):
        return self.root is not None and (step == 1 or step % self.every_steps == 0
                                         or time.monotonic() - self.saved_at >= self.every_seconds)

    def save_if_due(self, state, export=None):
        if not self.due(state["step"]) and not (self.root and state["step"] == state["steps"]):
            return None
        import torch
        if getattr(state["a"], "full_ft", 0):
            raise ValueError("Lab recovery supports LoRA training only")
        self.root.mkdir(parents=True, exist_ok=True)
        name = f"step-{state['step']:07d}-{uuid.uuid4().hex[:8]}"
        temporary = self.root / (".writing-" + name)
        temporary.mkdir()
        weights = {name: _cpu_copy(parameter) for name, parameter in state["model"].named_parameters()
                   if parameter.requires_grad}
        if not weights:
            raise ValueError("No trainable checkpoint weights")
        counters = {"step": state["step"], "seen": state["seen"], "tokens_seen": state["tokens_seen"],
                    "peak_mem": state["peak_mem"], "optimizer_seconds": state["optimizer_seconds"],
                    "step_seconds": list(state["step_seconds"]), "elapsed": time.time() - state["t0"],
                    "start_epoch": state["ep"], "start_mb": state["mb"] + 1,
                    "grad_norms": _cpu_copy(state["grad_norms"]), "run": dict(state["run"])}
        saved = {"version": 1, "args": vars(state["a"]).copy(), "steps": state["steps"],
                 "init_source": state.get("init_source"), "training_backend": self.backend,
                 "suite_sha256": state["suite_hash"], "data_sha256": _hash_file(state["a"].data) if state["a"].data else None,
                 "parameters": weights, "optimizer": _cpu_copy(state["opt"].state_dict()),
                 "scheduler": state["sched"].state_dict(), "position": counters,
                 "execution_history": self.execution_history,
                 "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if state["dev"] == "cuda" else None}
        torch.save(saved, temporary / "recovery.pt")
        if export is not None:
            export(temporary / "checkpoint", state)
        else:
            from kev.train import finish_checkpoint
            checkpoint = temporary / "checkpoint"
            state["model"].lm.save_pretrained(checkpoint)
            meta = dataclasses.replace(state["meta"], head=_cpu_copy(state["model"].head.state_dict()),
                                       extra={"args": vars(state["a"]), "suite_sha256": state["suite_hash"],
                                              "init_source": state["init_source"], "snapshot": {"step": state["step"], "steps": state["steps"]}})
            finish_checkpoint(checkpoint, meta, state["tok"])
        receipt = {"step": state["step"], "steps": state["steps"], "directory": name,
                   "execution_history": self.execution_history,
                   "recovery_sha256": _hash_file(temporary / "recovery.pt")}
        (temporary / "complete.json").write_text(json.dumps(receipt, indent=2) + "\n")
        destination = self.root / name
        temporary.rename(destination)
        pointer = self.root / (".latest-" + uuid.uuid4().hex + ".json")
        pointer.write_text(json.dumps(receipt, indent=2) + "\n")
        os.replace(pointer, self.root / "latest.json")
        # Retain the two most recent committed snapshots. Staging files are never
        # used as recovery points and are left for explicit inspection on failure.
        completed = sorted((path for path in self.root.glob("step-*") if (path / "complete.json").is_file()),
                           key=lambda path: (json.loads((path / "complete.json").read_text())["step"], path.stat().st_mtime))
        for old in completed[:-2]:
            if old != destination:
                shutil.rmtree(old)
        self.saved_at = time.monotonic()
        print(f"Recovery saved at optimizer step {state['step']}: {destination}", flush=True)
        return destination

    def restore(self, state):
        if not self.resuming:
            return None
        import torch
        from collections import Counter
        receipt = json.loads((self.resume_from / "complete.json").read_text())
        path = self.resume_from / "recovery.pt"
        if _hash_file(path) != receipt["recovery_sha256"]:
            raise ValueError("Recovery file checksum mismatch")
        # This is a locally generated trusted optimizer/RNG file, not a Hub model.
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["version"] != 1 or saved["steps"] != state["steps"]:
            raise ValueError("Resume with the same training arguments and output directory")
        position, execution_change = execution_resume_position(saved, state, self.allow_execution_change)
        data_hash = _hash_file(state["a"].data) if state["a"].data else None
        if saved["suite_sha256"] != state["suite_hash"] or saved["data_sha256"] != data_hash:
            raise ValueError("Recovery training data differs")
        if saved["init_source"] != state.get("init_source"):
            raise ValueError("Recovery parent checkpoint differs")
        if saved["training_backend"] != self.backend:
            raise ValueError("Recovery training backend differs; keep the same optimized setup")
        trainable = {name: parameter for name, parameter in state["model"].named_parameters() if parameter.requires_grad}
        if set(trainable) != set(saved["parameters"]):
            raise ValueError("Recovery trainable parameters differ")
        with torch.no_grad():
            for name, parameter in trainable.items():
                parameter.copy_(saved["parameters"][name].to(parameter.device))
        state["opt"].load_state_dict(saved["optimizer"])
        state["sched"].load_state_dict(saved["scheduler"])
        torch.set_rng_state(saved["torch_rng"])
        random.setstate(saved["python_rng"])
        if saved["cuda_rng"] is not None:
            if state["dev"] != "cuda":
                raise ValueError("CUDA checkpoint requires a CUDA runtime")
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        self.execution_history = list(saved.get("execution_history", []))
        if execution_change:
            self.execution_history.append(execution_change)
        position["run"] = Counter(position["run"])
        print("LAB_RESUMED " + json.dumps({"step": position["step"], "steps": saved["steps"],
                                           "execution_change": execution_change}), flush=True)
        return tuple(position[key] for key in COUNTERS)
