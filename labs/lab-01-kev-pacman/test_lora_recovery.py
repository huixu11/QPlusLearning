"""CPU evidence: resume restores weights, Adam moments, scheduler and dropout RNG."""
from collections import Counter
import ast
import contextlib
import io
import json
import math
from pathlib import Path
import random
import tempfile
import time
from types import SimpleNamespace
import unittest

import torch

from lora_recovery import COUNTERS, LoRARecovery, latest_snapshot
from training_monitor import StepObserver, instrument_trainer


def make_state():
    torch.manual_seed(19)
    random.seed(23)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.3), torch.nn.Linear(4, 1))
    for parameter in model[0].parameters():
        parameter.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.01, total_steps=6, pct_start=0.4)
    return {"model": model, "opt": opt, "sched": sched, "a": SimpleNamespace(out="run", data=None, full_ft=0),
            "steps": 6, "suite_hash": "same-suite", "dev": "cpu", "step": 0, "seen": 0,
            "tokens_seen": 0, "peak_mem": 0, "optimizer_seconds": 0, "step_seconds": [],
            "elapsed": 0, "start_epoch": 0, "start_mb": 0, "ep": 0, "mb": 0,
            "grad_norms": [[], []], "run": Counter(), "t0": time.time()}


def advance(state, until):
    losses = []
    while state["step"] < until:
        step = state["step"]
        state["ep"], state["mb"] = divmod(step, 3)
        prediction = state["model"](torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16)
        loss = (prediction - random.random()).square().mean()
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(state["model"].parameters(), 1.0))
        state["opt"].step()
        state["sched"].step()
        state["opt"].zero_grad()
        state["step"] += 1
        state["seen"] += 8
        state["tokens_seen"] += 64
        state["step_seconds"].append(0.1)
        state["grad_norms"][state["ep"]].append(norm)
        state["run"].update({"ce": float(loss.detach()), "n": 1})
        losses.append(float(loss.detach()))
    return losses


def export(folder, state):
    folder.mkdir()
    (folder / "adapter_model.safetensors").write_bytes(b"test adapter")
    (folder / "head.pt").write_bytes(b"test head")


class RecoveryTests(unittest.TestCase):
    def test_execution_change_in_actual_loop_preserves_records_optimizer_and_schedule(self):
        source = Path(__file__).parent / 'vendor/kev/kev/train.py'
        tree = instrument_trainer(source.read_text(), str(source), with_recovery=True, return_tree=True)
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        begin = next(i for i, node in enumerate(main.body) if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == '_lab_position' for target in node.targets))
        end = next(i for i in range(begin, len(main.body)) if isinstance(main.body[i], ast.For))
        loop = compile(ast.Module(body=main.body[begin:end + 1], type_ignores=[]), str(source), 'exec')
        plan_functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                          and node.name in {'accumulation_records', 'microbatch_plan'}]
        plan = compile(ast.Module(body=plan_functions, type_ignores=[]), str(source), 'exec')
        class Manager(LoRARecovery):
            stop_at = None
            def save_if_due(self, state):
                result = super().save_if_due(state, export)
                if state['step'] == self.stop_at:
                    raise InterruptedError('simulated process loss')
                return result
        def scope(manager, batch):
            state = make_state()
            state['model'][1].p = 0  # Grouping can change dropout draws; test data position/gradient equivalence independently.
            state['a'] = SimpleNamespace(out='run', data=None, full_ft=0, epochs=2, batch=batch, accum=8 // batch,
                                        length_sort=0, none_pair_max_state=None, pass_tokens_max=0, row_budget=2048,
                                        shared_prefix=0, stop_after=0, save_every_steps=0, save_every_minutes=0)
            model = state['model']
            model.trainable_parameters = lambda: [p for p in model.parameters() if p.requires_grad]
            observer = StepObserver()
            observer.before_batch = lambda state: None
            order = []
            def encode(model, tok, args, chunk, *unused):
                order.extend(r['id'] for r in chunk)
                return [SimpleNamespace(row=r, share=1, tokens=4) for r in chunk]
            def loss_fn(model, args, part, *unused):
                losses = []
                for variant in part:
                    x = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16 + variant.row['id'] / 100
                    losses.append((model(x) - 0.25).square().mean())
                loss = sum(losses)
                return loss, Counter(ce=float(loss.detach()))
            state.update(torch=torch, time=time, math=math, Counter=Counter, MAX_GRAD_NORM=1.0,
                         _lab_recovery=manager, _lab_observe=observer, rank=0, world=1,
                         rng=random.Random(29), reqs=[{'id': i} for i in range(19)],
                         state_tokens=None, tok=None, anchors={}, anchor_sources=None, autocast=None,
                         snapshots=None, full_ft=SimpleNamespace(rank_share=lambda reqs, *unused: reqs), resume_seconds=[],
                         encode_batch=encode, row_passes=lambda batch, *unused: [batch], batch_loss=loss_fn,
                         allocated_bytes=lambda dev: 0, sync=lambda dev: None, consumed_order=order)
            exec(plan, state)
            return state
        for batch in (2, 4):
            for boundary in (2, 3, 5, 6):
                with self.subTest(batch=batch, boundary=boundary), tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
                    baseline = scope(Manager(), 1)
                    exec(loop, baseline)
                    manager = Manager(folder, every_steps=1)
                    manager.stop_at = boundary
                    interrupted = scope(manager, 1)
                    with self.assertRaises(InterruptedError):
                        exec(loop, interrupted)
                    manager = Manager(folder, resume_from=latest_snapshot(folder), allow_execution_change=True)
                    resumed = scope(manager, batch)
                    exec(loop, resumed)
                    self.assertEqual(resumed['step'], 6)
                    self.assertEqual(interrupted['consumed_order'] + resumed['consumed_order'], baseline['consumed_order'])
                    self.assertEqual(resumed['seen'], baseline['seen'])
                    self.assertEqual(resumed['sched'].state_dict(), baseline['sched'].state_dict())
                    for key, value in baseline['model'].state_dict().items():
                        torch.testing.assert_close(value, resumed['model'].state_dict()[key], rtol=1e-6, atol=1e-7)
                    for key, expected in baseline['opt'].state_dict()['state'].items():
                        for name, value in expected.items():
                            torch.testing.assert_close(value, resumed['opt'].state_dict()['state'][key][name], rtol=1e-6, atol=1e-7)
                    self.assertEqual(manager.execution_history[-1]['step'], boundary)
                    if boundary < 6:
                        receipt = json.loads((latest_snapshot(folder) / 'complete.json').read_text())
                        self.assertEqual(receipt['execution_history'], manager.execution_history)

    def test_execution_change_requires_opt_in_same_recipe_and_valid_boundary(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            state = make_state()
            state['a'] = SimpleNamespace(out='run', data=None, full_ft=0, batch=1, accum=8, row_budget=2048, lr=0.01)
            advance(state, 1)
            state['mb'] = 7
            snapshot = LoRARecovery(folder).save_if_due(state, export)
            def restored():
                value = make_state()
                value['a'] = SimpleNamespace(out='run', data=None, full_ft=0, batch=4, accum=2, row_budget=8192, lr=0.01)
                value['reqs'] = [{}] * 19
                return value
            with self.assertRaisesRegex(ValueError, 'explicit execution change'):
                LoRARecovery(resume_from=snapshot).restore(restored())
            manager = LoRARecovery(resume_from=snapshot, allow_execution_change=True)
            for key, value, message in [('lr', 0.02, 'same training arguments'), ('accum', 4, 'effective batch')]:
                changed = restored()
                setattr(changed['a'], key, value)
                with self.assertRaisesRegex(ValueError, message):
                    manager.restore(changed)
            changed = restored()
            changed['a'].length_sort = 1
            with self.assertRaisesRegex(ValueError, 'same training arguments'):
                manager.restore(changed)
            payload = torch.load(snapshot / 'recovery.pt', weights_only=False)
            payload['position']['start_mb'] = 7
            from lora_recovery import execution_resume_position
            with self.assertRaisesRegex(ValueError, 'optimizer boundary'):
                execution_resume_position(payload, restored(), True)

    def test_pinned_training_loop_resume_replays_shuffle_without_repeating_steps(self):
        source = Path(__file__).parent / 'vendor/kev/kev/train.py'
        tree = instrument_trainer(source.read_text(), str(source), with_recovery=True, return_tree=True)
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        begin = next(i for i, node in enumerate(main.body) if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == '_lab_position' for target in node.targets))
        end = next(i for i in range(begin, len(main.body)) if isinstance(main.body[i], ast.For))
        loop = compile(ast.Module(body=main.body[begin:end + 1], type_ignores=[]), str(source), 'exec')

        class Manager(LoRARecovery):
            stop_at = None
            def save_if_due(self, state):
                result = super().save_if_due(state, export)
                if state['step'] == self.stop_at:
                    raise InterruptedError('simulated process loss')
                return result

        def scope(manager):
            state = make_state()
            state['a'] = SimpleNamespace(out='run', data=None, full_ft=0, epochs=2,
                                        none_pair_max_state=None, pass_tokens_max=0, row_budget=0,
                                        shared_prefix=0, stop_after=0, save_every_steps=0, save_every_minutes=0)
            model = state['model']
            model.trainable_parameters = lambda: [p for p in model.parameters() if p.requires_grad]
            observer = StepObserver()
            observer.before_batch = lambda state: None  # Tiny encodings have no real tokenizer.
            def loss_fn(model, args, part, *unused):
                x = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16 + part[0].row['id'] / 10
                loss = (model(x) - 0.25).square().mean()
                return loss, Counter(ce=float(loss.detach()))
            state.update(torch=torch, time=time, Counter=Counter, MAX_GRAD_NORM=1.0,
                         _lab_recovery=manager, _lab_observe=observer, rank=0, world=1,
                         rng=random.Random(29), reqs=[{'id': i} for i in range(3)],
                         state_tokens=None, tok=None, anchors={}, anchor_sources=None, autocast=None,
                         snapshots=None, full_ft=None, resume_seconds=[],
                         microbatch_plan=lambda reqs, *unused: [([r], 1, True) for r in reqs],
                         encode_batch=lambda model, tok, args, chunk, *unused: [SimpleNamespace(row=r, share=1, tokens=4) for r in chunk],
                         row_passes=lambda batch, *unused: [batch], batch_loss=loss_fn,
                         allocated_bytes=lambda dev: 0, sync=lambda dev: None)
            return state

        for boundary in (2, 3, 6):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
                baseline = scope(Manager())
                exec(loop, baseline)
                manager = Manager(folder, every_steps=1)
                manager.stop_at = boundary
                interrupted = scope(manager)
                with self.assertRaises(InterruptedError):
                    exec(loop, interrupted)
                resumed = scope(Manager(folder, resume_from=latest_snapshot(folder)))
                exec(loop, resumed)
                self.assertEqual(resumed['step'], 6)
                self.assertEqual(resumed['seen'], baseline['seen'])
                for name, value in baseline['model'].state_dict().items():
                    self.assertTrue(torch.equal(value, resumed['model'].state_dict()[name]), name)

    def test_resume_matches_uninterrupted_dropout_training(self):
        for boundary in (2, 3):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
                baseline = make_state()
                expected_losses = advance(baseline, 6)
                interrupted = make_state()
                advance(interrupted, boundary)
                manager = LoRARecovery(folder, every_steps=1)
                snapshot = manager.save_if_due(interrupted, export)
                payload = torch.load(snapshot / "recovery.pt", weights_only=False)
                self.assertEqual(set(payload["parameters"]), {"2.weight", "2.bias"})
                resumed = make_state()
                position = LoRARecovery(folder, resume_from=latest_snapshot(folder)).restore(resumed)
                resumed.update(zip(COUNTERS, position))
                self.assertEqual((resumed["start_epoch"], resumed["start_mb"]), (0, boundary))
                self.assertEqual(advance(resumed, 6), expected_losses[boundary:])
                for key, value in baseline["model"].state_dict().items():
                    self.assertTrue(torch.equal(value, resumed["model"].state_dict()[key]), key)
                self.assertEqual(baseline["sched"].state_dict(), resumed["sched"].state_dict())
                self.assertEqual(baseline["seen"], resumed["seen"])
                self.assertEqual(baseline["grad_norms"], resumed["grad_norms"])
                for key, expected in baseline["opt"].state_dict()["state"].items():
                    for name, value in expected.items():
                        self.assertTrue(torch.equal(value, resumed["opt"].state_dict()["state"][key][name]))

    def test_incomplete_save_never_replaces_latest_and_keeps_two_complete(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            state, manager = make_state(), LoRARecovery(folder, every_steps=1)
            advance(state, 1)
            first = manager.save_if_due(state, export)
            advance(state, 2)
            def failure(*args):
                raise OSError("disk full")
            with self.assertRaisesRegex(OSError, "disk full"):
                manager.save_if_due(state, failure)
            self.assertEqual(latest_snapshot(folder), first.resolve())
            manager.save_if_due(state, export)
            advance(state, 3)
            third = manager.save_if_due(state, export)
            self.assertFalse(first.exists())
            self.assertEqual(latest_snapshot(folder), third.resolve())
            self.assertEqual(len(list(Path(folder).glob("step-*"))), 2)

    def test_argument_and_checksum_changes_refuse_resume(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            state = make_state()
            advance(state, 1)
            snapshot = LoRARecovery(folder).save_if_due(state, export)
            restored = make_state()
            restored["a"].out = "another-output"
            with self.assertRaisesRegex(ValueError, "same training arguments"):
                LoRARecovery(resume_from=snapshot).restore(restored)
            (snapshot / "recovery.pt").write_bytes(b"truncated")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                LoRARecovery(resume_from=snapshot).restore(make_state())


if __name__ == "__main__":
    unittest.main()
