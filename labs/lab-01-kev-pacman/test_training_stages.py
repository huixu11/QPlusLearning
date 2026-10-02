"""Check verified training-only concatenation and parent checkpoint provenance."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from training_stages import assemble_joint, checkpoint_fingerprint, inspect_checkpoint, restore_checkpoint


class StageTests(unittest.TestCase):
    def test_joint_requires_exact_train_parts_hashes_and_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder)
            parts, content = [], b""
            for number in range(3):
                chunk = (json.dumps({"state": number}) + "\n").encode()
                suite = f"evals/part-{number}"
                (repo / suite).mkdir(parents=True)
                (repo / suite / "train.jsonl").write_bytes(chunk)
                parts.append({"suite": suite, "records": 1, "sha256": hashlib.sha256(chunk).hexdigest()})
                content += chunk
            expected = {"data": "evals/joint/train.jsonl", "records": 3,
                        "sha256": hashlib.sha256(content).hexdigest(), "parts": parts}
            self.assertEqual(assemble_joint(repo, expected).read_bytes(), content)
            (repo / expected["data"]).unlink()
            (repo / parts[1]["suite"] / "train.jsonl").write_text('{"tampered":true}\n')
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                assemble_joint(repo, expected)

    def test_checkpoint_requires_full_training_and_matching_parent(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parent = root / "parent"
            parent.mkdir()
            (parent / "head.pt").write_bytes(b"head")
            (parent / "adapter_model.safetensors").write_bytes(b"adapter")
            child = root / "child"
            child.mkdir()
            (child / "training_config.json").write_text(json.dumps({"args": {"epochs": 1, "batch": 4, "accum": 2}, "init_source": {"resolved": "old-runtime-path"}}))
            (child / "training_metrics.json").write_text(json.dumps({"optimizer_steps": 2, "records_seen": 22,
                                                                     "requested_records": 16, "peak_device_bytes": 0}))
            (child / "run-evidence.json").write_text(json.dumps({"stage": "dates", "parent_checkpoint_sha256": checkpoint_fingerprint(parent)}))
            inspect_checkpoint(child, stage="dates", owner="learner", parent=parent, recipe={"epochs": 1})
            metrics = child / "training_metrics.json"
            complete = json.loads(metrics.read_text())
            metrics.write_text(json.dumps({**complete, "optimizer_steps": 1}))
            with self.assertRaisesRegex(ValueError, "full training stage"):
                inspect_checkpoint(child, stage="dates", owner="learner", parent=parent)
            metrics.write_text(json.dumps(complete))
            (parent / "head.pt").write_bytes(b"different parent")
            with self.assertRaisesRegex(ValueError, "parent checkpoint"):
                inspect_checkpoint(child, stage="dates", owner="learner", parent=parent)

    def test_archive_rejects_traversal_before_extracting(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "bad.zip"
            with zipfile.ZipFile(archive, "w") as saved:
                saved.writestr("../escape", "bad")
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                restore_checkpoint(archive, Path(folder) / "checkpoint")
            self.assertFalse((Path(folder) / "escape").exists())
