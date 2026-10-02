"""Instrument the pinned Kev trainer with telemetry and LoRA recovery hooks."""
import ast
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
import types

TRAIN_SHA256 = "de0b0971defd79c053d56f8e8b17b0501b2e264e93a66a8d5493f1b1ba839416"
PREFIX = "LAB_METRIC "


class StepObserver:
    def __init__(self):
        self.previous_ce = self.previous_n = 0

    def before_batch(self, state):
        from kev.model import rows_of
        lengths = [len(prefix) + len(row["ids"])
                   for variant in state["part"]
                   for prefix, _, rows in [rows_of(variant.enc)] for row in rows]
        info = {"next_step": state["step"] + 1, "epoch": state["ep"], "microbatch": state["mb"],
                "record_ids": [r.get("_meta", {}).get("id") for r in state["chunk"]],
                "physical_rows": len(lengths), "max_row_tokens": max(lengths, default=0),
                "padded_tokens": len(lengths) * max(lengths, default=0), **self.memory(state)}
        print("LAB_BATCH " + json.dumps(info), flush=True)

    @staticmethod
    def memory(state):
        if state.get("dev") != "cuda":
            return {}
        import torch
        free, _ = torch.cuda.mem_get_info()
        return {"allocated_memory_gib": torch.cuda.memory_allocated() / 2**30,
                "reserved_memory_gib": torch.cuda.memory_reserved() / 2**30,
                "free_memory_gib": free / 2**30}

    def __call__(self, state):
        step, run = state["step"], state["run"]
        # Kev resets its aggregate after steps 10, 20, ... . Difference the
        # accumulator to obtain this optimizer step's record-weighted CE.
        if (step - 1) % 10 == 0:
            self.previous_ce = self.previous_n = 0
        ce, count = run["ce"], run["n"]
        loss = (ce - self.previous_ce) / (count - self.previous_n)
        self.previous_ce, self.previous_n = ce, count
        metrics = {"step": step, "steps": state["steps"],
                   "epoch": state["ep"] + (state["mb"] + 1) / len(state["plan"]),
                   "loss_cross_entropy": loss,
                   "learning_rate_next_step": state["opt"].param_groups[0]["lr"],
                   "grad_norm_before_clip": state["norm"],
                   "step_seconds": state["step_seconds"][-1],
                   "records_seen": state["seen"],
                   "peak_memory_gib": state["peak_mem"] / 2**30, **self.memory(state)}
        if not all(math.isfinite(float(value)) for value in metrics.values()):
            raise ValueError("Non-finite training telemetry")
        print(PREFIX + json.dumps(metrics, allow_nan=False), flush=True)


def instrument_trainer(source, filename, with_recovery=False, *, return_tree=False):
    """Guard every insertion site; the checkout and loss/optimizer stay intact."""
    target = ast.dump(ast.parse("step % 10 == 0", mode="eval").body)
    existing = ast.dump(ast.parse('Path(a.out).exists() and not a.resume and os.environ.get("RANK", "0") == "0"', mode="eval").body)

    class InsertObserver(ast.NodeTransformer):
        count = 0
        guards = {"existing": 0, "mkdir": 0, "restore": 0, "batch": 0, "optimizer": 0}

        def visit_If(self, node):
            self.generic_visit(node)
            if with_recovery and ast.dump(node.test) == existing:
                node.test = ast.BoolOp(op=ast.And(), values=[node.test, ast.parse("not _lab_recovery.resuming", mode="eval").body])
                self.guards["existing"] += 1
            if ast.dump(node.test) != target:
                return node
            self.count += 1
            observer = ast.parse("_lab_observe(locals())").body[0]
            result = [ast.copy_location(observer, node), node]
            if with_recovery:
                result += ast.parse("_lab_recovery.save_if_due(locals())").body
            return result

        def visit_Call(self, node):
            self.generic_visit(node)
            if with_recovery and ast.unparse(node.func) == "torch.optim.AdamW":
                node.keywords.append(ast.keyword(arg="fused", value=ast.parse(
                    'dev == "cuda" and os.environ.get("LAB_FUSED_ADAMW") == "1"', mode="eval").body))
                self.guards["optimizer"] += 1
            if with_recovery and ast.unparse(node.func) == "out_dir.mkdir":
                keyword = next(k for k in node.keywords if k.arg == "exist_ok")
                keyword.value = ast.parse("bool(a.resume) or _lab_recovery.resuming", mode="eval").body
                self.guards["mkdir"] += 1
            return node

        def visit_Expr(self, node):
            self.generic_visit(node)
            if with_recovery and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "model.train":
                self.guards["restore"] += 1
                restore = ast.parse("""_lab_position = _lab_recovery.restore(locals())
if _lab_position is not None:
    step, seen, tokens_seen, peak_mem, optimizer_seconds, step_seconds, elapsed, start_epoch, start_mb, grad_norms, run = _lab_position
    if step == steps:
        start_epoch = a.epochs  # Only finish/export a final-step recovery; no extra optimizer step.
    _lab_observe.previous_ce, _lab_observe.previous_n = run['ce'], run['n']
""").body
                return [*restore, node]
            return node

        def visit_Assign(self, node):
            self.generic_visit(node)
            if with_recovery and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "batch_loss":
                self.guards["batch"] += 1
                return [ast.parse("_lab_observe.before_batch(locals())").body[0], node]
            return node

    visitor = InsertObserver()
    tree = visitor.visit(ast.parse(source, filename=filename))
    if visitor.count != 1:
        raise RuntimeError("Pinned Kev logging site changed; review the observer before training.")
    if with_recovery and any(count != 1 for count in visitor.guards.values()):
        raise RuntimeError(f"Pinned Kev recovery sites changed: {visitor.guards}")
    tree = ast.fix_missing_locations(tree)
    return tree if return_tree else compile(tree, filename, "exec")


class TensorBoardWriter:
    def __init__(self, directory):
        try:
            from tensorboard.summary.writer.event_file_writer import EventFileWriter
            from tensorboard.compat.proto.event_pb2 import Event
            from tensorboard.compat.proto.summary_pb2 import Summary
        except ImportError as error:
            raise RuntimeError("Install tensorboard==2.20.0 in the notebook kernel using the setup cell.") from error
        self.Event, self.Summary = Event, Summary
        self.writer = EventFileWriter(str(directory), flush_secs=5)

    def write(self, metrics):
        values = [self.Summary.Value(tag="train/" + key, simple_value=float(value))
                  for key, value in metrics.items() if key not in {"step", "steps"}]
        values.append(self.Summary.Value(tag="train/progress", simple_value=metrics["step"] / metrics["steps"]))
        self.writer.add_event(self.Event(wall_time=time.time(), step=metrics["step"],
                                        summary=self.Summary(value=values)))

    def close(self):
        self.writer.flush()
        self.writer.close()


def _stop_process(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def stream_training(command, *, cwd, env, log_dir, stage, timeout_seconds, require_metrics=True):
    """Relay subprocess output in Colab and keep logs even on failure/timeout."""
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=False)
    writer = TensorBoardWriter(directory / "tensorboard")
    started = time.monotonic()
    last_heartbeat, latest, rows, process = started, None, 0, None
    resumed, last_batch = None, None
    status = "failed"
    messages = queue.Queue()

    def pump():
        try:
            for line in process.stdout:
                messages.put(line)
        finally:
            messages.put(None)

    print(f"[{stage}] Starting. Loading the model/preparing data before optimizer step 1. Logs: {directory}", flush=True)
    try:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                   bufsize=1, start_new_session=True)
        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        with (directory / "stdout.log").open("w", encoding="utf-8") as log, \
             (directory / "metrics.jsonl").open("w", encoding="utf-8") as jsonl, \
             (directory / "batches.jsonl").open("w", encoding="utf-8") as batches, \
             (directory / "metrics.csv").open("w", newline="", encoding="utf-8") as csvfile:
            table = None
            while True:
                elapsed = time.monotonic() - started
                if elapsed > timeout_seconds:
                    status = "timed_out"
                    raise TimeoutError(f"{stage} exceeded {timeout_seconds / 60:g} minutes; logs kept at {directory}")
                try:
                    line = messages.get(timeout=min(1, max(0.01, timeout_seconds - elapsed)))
                except queue.Empty:
                    line = ""
                if line is None:
                    break
                if line:
                    log.write(line)
                    log.flush()
                    if line.startswith(PREFIX):
                        latest = json.loads(line[len(PREFIX):])
                        if not all(math.isfinite(float(value)) for value in latest.values()):
                            raise ValueError("Non-finite training telemetry")
                        if table is None:
                            table = csv.DictWriter(csvfile, fieldnames=list(latest))
                            table.writeheader()
                        table.writerow(latest)
                        csvfile.flush()
                        jsonl.write(json.dumps(latest) + "\n")
                        jsonl.flush()
                        writer.write(latest)
                        rows += 1
                        if latest["step"] == 1 or latest["step"] % 10 == 0 or latest["step"] == latest["steps"]:
                            print(f"[{stage}] epoch {latest['epoch']:.3f} | step {latest['step']}/{latest['steps']} | "
                                  f"CE {latest['loss_cross_entropy']:.4f} | next lr {latest['learning_rate_next_step']:.3g} | "
                                  f"grad {latest['grad_norm_before_clip']:.3f} | peak {latest['peak_memory_gib']:.2f} GiB | "
                                  f"step {latest['step_seconds']:.2f}s", flush=True)
                    elif line.startswith("LAB_BATCH "):
                        last_batch = json.loads(line[len("LAB_BATCH "):])
                        batches.write(json.dumps(last_batch) + "\n")
                        batches.flush()
                    elif line.startswith("LAB_RESUMED "):
                        resumed = json.loads(line[len("LAB_RESUMED "):])
                        print(f"[{stage}] Resumed at step {resumed['step']}/{resumed['steps']}", flush=True)
                    else:
                        print(line, end="", flush=True)
                if time.monotonic() - last_heartbeat >= 15:
                    phase = f"last completed step {latest['step']}/{latest['steps']}" if latest else "loading model/preparing data; no optimizer step yet"
                    print(f"[{stage}] {elapsed / 60:.1f} min elapsed; {phase}", flush=True)
                    last_heartbeat = time.monotonic()
            remaining = max(0.01, timeout_seconds - (time.monotonic() - started))
            try:
                result = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                status = "timed_out"
                raise TimeoutError(f"{stage} timed out; logs kept at {directory}") from None
            if result:
                if last_batch:
                    print("Last attempted batch: " + json.dumps(last_batch), flush=True)
                raise subprocess.CalledProcessError(result, command)
            if require_metrics and rows == 0 and not (resumed and resumed["step"] == resumed["steps"]):
                raise RuntimeError(f"No optimizer-step metrics received; inspect {directory / 'stdout.log'}")
            status = "completed"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        if process is not None:
            _stop_process(process)
            reader.join(timeout=5)
            process.stdout.close()
        writer.close()
        (directory / "status.json").write_text(json.dumps({"stage": stage, "status": status,
                                                          "logged_steps": rows, "resumed": resumed,
                                                          "last_batch": last_batch, "elapsed_seconds": time.monotonic() - started}, indent=2) + "\n")
    return directory


def main():
    from lora_recovery import LoRARecovery
    path = Path(importlib.util.find_spec("kev.train").origin)
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != TRAIN_SHA256:
        raise RuntimeError("Kev trainer differs from the pinned source; refusing to instrument it.")
    module = types.ModuleType("kev.train")
    recovery = LoRARecovery.from_env()
    if os.environ.get("LAB_REQUIRE_OPTIMIZED_KERNELS") == "1":
        from optimized_training import require_optimized_bindings
        recovery.backend = {**require_optimized_bindings(), "fused_adamw": os.environ.get("LAB_FUSED_ADAMW") == "1"}
        print("Verified training backend: " + json.dumps(recovery.backend), flush=True)
    module.__dict__.update(__package__="kev", __file__=str(path), _lab_observe=StepObserver(),
                           _lab_recovery=recovery)
    sys.modules["kev.train"] = module
    exec(instrument_trainer(source.decode("utf-8"), str(path), with_recovery=True), module.__dict__)
    module.main()


if __name__ == "__main__":
    main()
