"""Atomic LoRA recovery snapshots, including optimizer, schedule and RNG state.

Kev's native --resume handles full-weight runs only. This adapter saves at an
optimizer boundary and restores the pinned LoRA trainer's own counters.
"""
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import uuid

COUNTERS = ("step", "seen", "tokens_seen", "peak_mem", "optimizer_seconds", "step_seconds",
            "elapsed", "start_epoch", "start_mb", "grad_norms", "run")


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
    def __init__(self, root=None, resume_from=None, every_steps=100, every_seconds=300):
        self.root = Path(root) if root else None
        self.resume_from = Path(resume_from) if resume_from else None
        self.every_steps, self.every_seconds = every_steps, every_seconds
        if every_steps < 1 or every_seconds <= 0:
            raise ValueError("Recovery intervals must be positive")
        self.saved_at = time.monotonic()
        self.backend = None

    @property
    def resuming(self):
        return self.resume_from is not None

    @classmethod
    def from_env(cls):
        return cls(os.environ.get("LAB_RECOVERY_ROOT"), os.environ.get("LAB_RESUME_FROM"),
                   int(os.environ.get("LAB_SAVE_STEPS", "100")), float(os.environ.get("LAB_SAVE_SECONDS", "300")))

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
        if saved["version"] != 1 or saved["args"] != vars(state["a"]) or saved["steps"] != state["steps"]:
            raise ValueError("Resume with the same training arguments and output directory")
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
        position = saved["position"]
        position["run"] = Counter(position["run"])
        print("LAB_RESUMED " + json.dumps({"step": position["step"], "steps": saved["steps"]}), flush=True)
        return tuple(position[key] for key in COUNTERS)
