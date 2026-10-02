"""Verified stage data and separate checkpoint backups for the teaching notebook."""
import hashlib
import json
import math
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent


def specifications():
    return json.loads((ROOT / "training-stages.json").read_text())


def verify_data(path, expected):
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected["sha256"]:
        raise ValueError(f"Training data checksum mismatch: {path}")
    rows = [json.loads(line) for line in content.decode("utf-8").split("\n") if line.strip()]
    if len(rows) != expected["records"]:
        raise ValueError(f"Training data record count mismatch: {path}")
    return rows


def assemble_joint(repo, expected):
    """Rebuild the documented partition order, admitting only verified train files."""
    repo = Path(repo)
    target = repo / expected["data"]
    if target.exists():
        verify_data(target, expected)
        return target
    contents = []
    for part in expected["parts"]:
        path = repo / part["suite"] / "train.jsonl"
        verify_data(path, part)
        contents.append(path.read_bytes())
    content = b"".join(contents)
    if hashlib.sha256(content).hexdigest() != expected["sha256"]:
        raise ValueError("Concatenated training file differs from the course-pinned checksum.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    verify_data(target, expected)
    return target


def prepare_data(repo):
    # Called inside the pinned Kev environment, rather than the notebook kernel.
    from kev.suite import load_split
    spec = specifications()
    dates = Path(repo) / spec["dates"]["data"]
    verify_data(dates, spec["dates"])
    print("Dates/missing-evidence data verified: 1,425 records", flush=True)
    documents = spec["documents"]
    load_split(str(Path(repo) / Path(documents["data"]).parent), "train")
    verify_data(Path(repo) / documents["data"], documents)
    print("Documents data verified: 5,219 records", flush=True)
    skills = spec["skills"]
    if not (Path(repo) / skills["data"]).exists():
        for part in skills["parts"]:
            load_split(str(Path(repo) / part["suite"]), "train")
            print(f"Verified training partition: {part['suite']}", flush=True)
    assemble_joint(repo, skills)
    print("Skills/devtools data verified: 11,320 records (course-pinned concatenation)", flush=True)
    return {name: {"records": stage["records"], "replay": stage["replay"], "sha256": stage["sha256"]}
            for name, stage in spec.items()}


def checkpoint_fingerprint(folder):
    folder = Path(folder)
    files = [folder / "head.pt", *sorted(folder.glob("adapter_model.*"))]
    if not all(file.is_file() for file in files) or len(files) < 2:
        raise ValueError("Expected a LoRA adapter and pointer-head checkpoint")
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode())
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def inspect_checkpoint(folder, *, stage, owner, parent=None, recipe=None, data_spec=None):
    folder = Path(folder)
    config = json.loads((folder / "training_config.json").read_text())
    metrics = json.loads((folder / "training_metrics.json").read_text())
    evidence = json.loads((folder / "run-evidence.json").read_text())
    if evidence["stage"] != stage:
        raise ValueError("Restored checkpoint belongs to a different training stage")
    if recipe and any(config["args"].get(key) != value for key, value in recipe.items()):
        raise ValueError("Checkpoint configuration differs from the selected published recipe")
    if data_spec and evidence.get("training_data_sha256") != data_spec["sha256"]:
        raise ValueError("Checkpoint was trained on different stage data")
    # Kev's records_seen includes augmentation siblings; requested_records counts
    # source requests. None minimal pairs make these totals legitimately differ.
    args = config["args"]
    epochs = args["epochs"]
    effective_batch = args["batch"] * args["accum"] * metrics.get("world_size", 1)
    expected_steps = epochs * math.ceil(metrics["requested_records"] / epochs / effective_batch)
    if (metrics["optimizer_steps"] != expected_steps or metrics["records_seen"] < metrics["requested_records"]
            or metrics.get("truncated_records", 0) or metrics.get("rejected_records", 0)
            or args.get("max_steps", 0)):
        raise ValueError("Complete the full training stage before continuing")
    if parent is None:
        if config["init_source"] is not None:
            raise ValueError("Initial training must start with fresh LoRA and head")
    elif evidence.get("parent_checkpoint_sha256") != checkpoint_fingerprint(parent):
        raise ValueError("This stage was not trained from the selected parent checkpoint")
    if owner not in {"learner", "instructor"}:
        raise ValueError("Declare learner or instructor checkpoint ownership")
    print(f"{stage}: {owner}; {metrics['optimizer_steps']} optimizer steps; "
          f"{metrics['records_seen']} records seen; peak {metrics['peak_device_bytes'] / 2**30:.2f} GiB")
    return config, metrics


def restore_checkpoint(archive, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Restore into a new checkpoint directory")
    with zipfile.ZipFile(archive) as saved:
        for member in saved.infolist():
            if not (destination / member.filename).resolve().is_relative_to(destination):
                raise ValueError("Unsafe checkpoint archive path")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Checkpoint archive contains a symbolic link")
        destination.mkdir(parents=True)
        saved.extractall(destination)
    return destination


def backup_checkpoint(folder, archive):
    folder = Path(folder)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as saved:
        for file in sorted(folder.rglob("*")):
            if file.is_file():
                saved.write(file, file.relative_to(folder))
    return Path(archive)
