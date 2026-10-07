"""CPU tests for token-cost selection; these do not claim CUDA training coverage."""
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from lab_memory_check import (loss_normalizer, main, pass_description, probe_arguments,
                              select_costliest, validate_probe_recipe)


ROOT = Path(__file__).resolve().parent


def train_packing_functions():
    """Execute the unchanged pinned trainer's pure token packing functions on shapes."""
    tree = ast.parse((ROOT / "vendor/kev/kev/train.py").read_text())
    wanted = {"pass_tokens", "question_parts", "row_passes"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace = {"shape": lambda encoding: (encoding["state"], encoding["branches"])}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "<pinned Kev token packing>", "exec"), namespace)
    return namespace


PACKING = train_packing_functions()


def request(name, state, branches, later_state=None):
    return {"_meta": {"id": name, "source": "fixture"}, "state_tokens": state,
            "branches": branches, "later_state": later_state}


def encode_fixture(source, epoch, forced, budget):
    """Numeric encodings isolate packing/selection from tokenizer and GPU execution."""
    state = source["later_state"] if epoch and source["later_state"] is not None else source["state_tokens"]
    encodings = [{"state": state, "branches": source["branches"]}]
    if forced:
        encodings += [{"state": state, "branches": [max(source["branches"]) + 200]}] * 2
    variants = []
    for encoding in encodings:
        for questions in PACKING["question_parts"](encoding, budget, False):
            variants.append(SimpleNamespace(enc={"state": state, "branches": [encoding["branches"][i] for i in questions]},
                                            share=len(questions) / len(encoding["branches"]), permuted=None))
    return variants


def choose(requests, budget, epochs=1, forced=None):
    return select_costliest(requests, epochs,
                            lambda source, epoch, force: encode_fixture(source, epoch, force, budget),
                            lambda batch: [pass_description(part, PACKING["shape"], PACKING["pass_tokens"], False)
                                           for part in PACKING["row_passes"](batch, budget, False)],
                            forced)


class LabMemoryCheckTests(unittest.TestCase):
    def test_multi_question_records_are_ranked_by_actual_rows_after_budget_splitting(self):
        long_single = request("single", 900, [100])
        many_questions = request("many", 600, [100, 100, 100])
        selected = choose([long_single, many_questions], 2048)
        self.assertIs(selected["request"], many_questions)
        self.assertEqual(selected["cost_tokens"], 1400)
        self.assertEqual(selected["encoded_variants"], 2)
        self.assertEqual(selected["variant_share"], 1)
        costliest_pass = max(selected["passes"], key=lambda part: part["cost_tokens"])
        self.assertEqual(costliest_pass["question_rows"], 2)
        self.assertEqual(costliest_pass["variants"], 1)
        self.assertEqual(costliest_pass["row_tokens"], [700, 700])

    def test_single_long_row_can_exceed_row_budget_and_is_never_silently_truncated(self):
        long_single = request("long-unbreakable-row", 3000, [100])
        selected = choose([request("short", 300, [100] * 5), long_single], 2048)
        self.assertIs(selected["request"], long_single)
        self.assertEqual(selected["cost_tokens"], 3100)
        self.assertGreater(selected["cost_tokens"], 2048)
        self.assertEqual(selected["passes"][0]["question_rows"], 1)

    def test_forced_eligible_none_pair_siblings_participate_in_real_padded_cost(self):
        eligible = request("eligible", 600, [100])
        other = request("other", 1500, [100])
        selected = choose([eligible, other], 0, forced={id(eligible)})
        self.assertIs(selected["request"], eligible)
        self.assertTrue(selected["forced_none_pair"])
        self.assertEqual(selected["cost_tokens"], 2700)
        self.assertEqual(selected["passes"][0]["question_rows"], 3)
        self.assertEqual(selected["variant_share"], 3)
        normal = choose([eligible, other], 0)
        self.assertIs(normal["request"], other)
        self.assertFalse(normal["forced_none_pair"])

    def test_all_configured_epochs_are_scanned_without_changing_requests(self):
        later = request("later", 200, [100], later_state=2800)
        selected = choose([request("constant", 1000, [100]), later], 2048, epochs=2)
        self.assertEqual(selected["epoch"], 1)
        self.assertEqual(selected["cost_tokens"], 2900)
        self.assertEqual(selected["cases_scanned"], 4)
        self.assertEqual(later["state_tokens"], 200)

    def test_loss_normalizer_matches_kev_variant_shares_and_eight_record_accumulation(self):
        main = request("many-questions", 600, [100] * 5)
        split = encode_fixture(main, 0, False, 2048)
        self.assertGreater(len(split), 1)
        self.assertEqual(loss_normalizer(split, 8), 8)
        siblings = encode_fixture(main, 0, True, 2048)
        self.assertGreater(len(siblings), 3)
        self.assertEqual(loss_normalizer(siblings, 8), 24)
        with self.assertRaises(ValueError):
            loss_normalizer([], 8)

    def test_permuted_forward_cost_is_included_in_memory_proxy(self):
        variant = SimpleNamespace(enc={"state": 200, "branches": [100]},
                                  permuted=({"state": 200, "branches": [120]}, []), share=1)
        detail = pass_description([variant], PACKING["shape"], PACKING["pass_tokens"], False)
        self.assertEqual(detail["padded_tokens"], 300)
        self.assertEqual(detail["permutation_padded_tokens"], 320)
        self.assertEqual(detail["cost_tokens"], 620)

    def test_encoder_failure_is_not_skipped_when_selecting_memory_case(self):
        def broken(source, epoch, forced):
            raise ValueError("strict encoder context overflow")
        with self.assertRaisesRegex(ValueError, "context overflow"):
            select_costliest([request("invalid", 100, [100])], 1, broken, lambda batch: [])
        with self.assertRaisesRegex(ValueError, "No training requests"):
            choose([], 2048)

    def test_only_output_path_is_replaced_for_parse_args_then_original_is_retained(self):
        flags = ["--seed", "7", "--data", "train.jsonl", "--init_from", "skills", "--out", "existing-pacman",
                 "--batch", "1", "--accum", "8", "--max_steps", "0"]
        changed, original = probe_arguments(flags, "/tmp/unused-checkpoint")
        self.assertEqual(original, "existing-pacman")
        self.assertEqual(changed[changed.index("--out") + 1], "/tmp/unused-checkpoint")
        self.assertEqual(flags[flags.index("--out") + 1], "existing-pacman")
        self.assertEqual(changed[:changed.index("--out")], flags[:flags.index("--out")])
        self.assertEqual(changed[changed.index("--out") + 2:], flags[flags.index("--out") + 2:])
        combined, original = probe_arguments(["--out=existing"], "/tmp/unused")
        self.assertEqual(combined, ["--out=/tmp/unused"])
        self.assertEqual(original, "existing")

    def test_unsupported_training_plans_fail_explicitly(self):
        allowed = {"full_ft": 0, "batch": 1, "accum": 8, "device": "cuda", "dtype": "bf16",
                   "checkpointing": 1, "length_sort": 0, "pass_tokens_max": 0}
        validate_probe_recipe(SimpleNamespace(**allowed))
        for name, value in (("full_ft", 1), ("batch", 2), ("accum", 4), ("device", "cpu"),
                            ("dtype", "fp32"), ("checkpointing", 0), ("length_sort", 1)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_probe_recipe(SimpleNamespace(**{**allowed, name: value}))

    def test_failure_replaces_stale_passed_report_without_writing_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / "memory-check.json"
            checkpoint = Path(folder) / "checkpoint"
            report.write_text(json.dumps({"result": "passed"}))
            with patch("lab_memory_check.verify_stage", side_effect=ValueError("CUDA unavailable")):
                with self.assertRaisesRegex(ValueError, "CUDA unavailable"):
                    main(["--report", str(report), "--", "--out", str(checkpoint), "--seed", "7"])
            receipt = json.loads(report.read_text())
            self.assertEqual(receipt["result"], "failed")
            self.assertIn("CUDA unavailable", receipt["error"])
            self.assertIn("not proof", receipt["scope"])
            self.assertFalse(checkpoint.exists())


if __name__ == "__main__":
    unittest.main()
