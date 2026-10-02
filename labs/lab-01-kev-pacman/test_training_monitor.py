"""Exercise live telemetry, event files, process failures and cancellation on CPU."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from training_monitor import PREFIX, TRAIN_SHA256, StepObserver, instrument_trainer, stream_training


class MonitorTests(unittest.TestCase):
    def test_step_loss_survives_upstream_ten_step_counter_reset(self):
        observer, stream = StepObserver(), io.StringIO()
        opt = type("Optimizer", (), {"param_groups": [{"lr": 0.001}]})()
        ce = count = 0
        with contextlib.redirect_stdout(stream):
            for step in range(1, 13):
                ce += step * 2
                count += 2
                observer({"step": step, "steps": 12, "run": {"ce": ce, "n": count},
                          "ep": 0, "mb": step - 1, "plan": range(12), "opt": opt,
                          "norm": 0.5, "step_seconds": [0.1], "seen": step * 2, "peak_mem": 2**30})
                if step % 10 == 0:
                    ce = count = 0
        rows = [json.loads(line[len(PREFIX):]) for line in stream.getvalue().splitlines()]
        self.assertEqual([row["loss_cross_entropy"] for row in rows], list(range(1, 13)))

    def test_observer_insertion_preserves_computation_and_refuses_unknown_site(self):
        pinned = Path(__file__).parent / "vendor/kev/kev/train.py"
        self.assertEqual(hashlib.sha256(pinned.read_bytes()).hexdigest(), TRAIN_SHA256)
        instrument_trainer(pinned.read_text(), str(pinned))
        source = "step = 1\nresult = 7 * 9\nif step % 10 == 0:\n    result += 1\n"
        seen = []
        scope = {"_lab_observe": lambda state: seen.append(state["result"])}
        exec(instrument_trainer(source, "toy.py"), scope)
        self.assertEqual(scope["result"], 63)
        self.assertEqual(seen, [63])
        with self.assertRaisesRegex(RuntimeError, "logging site changed"):
            instrument_trainer("print('no observer site')", "toy.py")

    def test_stream_writes_real_tensorboard_events_and_failure_logs(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        row = {"step": 1, "steps": 1, "epoch": 1, "loss_cross_entropy": 0.25,
               "learning_rate_next_step": 0.0001, "grad_norm_before_clip": 0.8,
               "step_seconds": 0.01, "records_seen": 8, "peak_memory_gib": 1.5}
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()) as console:
            code = "print('model loaded', flush=True); print(" + repr(PREFIX + json.dumps(row)) + ", flush=True)"
            directory = Path(folder) / "success"
            stream_training([sys.executable, "-u", "-c", code], cwd=folder, env=os.environ.copy(),
                            log_dir=directory, stage="test", timeout_seconds=10)
            self.assertIn("model loaded", console.getvalue())
            self.assertIn("step 1/1", console.getvalue())
            self.assertEqual(json.loads((directory / "status.json").read_text())["status"], "completed")
            self.assertEqual(json.loads((directory / "metrics.jsonl").read_text())["loss_cross_entropy"], 0.25)
            events = EventAccumulator(str(directory / "tensorboard")).Reload()
            self.assertAlmostEqual(events.Scalars("train/loss_cross_entropy")[0].value, 0.25)
            with self.assertRaises(subprocess.CalledProcessError):
                stream_training([sys.executable, "-u", "-c", "print('diagnostic', flush=True); raise SystemExit(7)"],
                                cwd=folder, env=os.environ.copy(), log_dir=Path(folder) / "failed",
                                stage="test", timeout_seconds=10)
            self.assertIn("diagnostic", (Path(folder) / "failed/stdout.log").read_text())
            self.assertEqual(json.loads((Path(folder) / "failed/status.json").read_text())["status"], "failed")

    def test_silent_process_timeout_keeps_logs_and_terminates(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            directory = Path(folder) / "timeout"
            with self.assertRaises(TimeoutError):
                stream_training([sys.executable, "-u", "-c", "import time; print('loading', flush=True); time.sleep(20)"],
                                cwd=folder, env=os.environ.copy(), log_dir=directory, stage="test", timeout_seconds=0.3)
            status = json.loads((directory / "status.json").read_text())
            self.assertEqual(status["status"], "timed_out")
            self.assertLess(status["elapsed_seconds"], 5)
