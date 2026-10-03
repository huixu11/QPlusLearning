"""Verified checkpoint archives for a mounted Google Drive directory."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import uuid
import zipfile


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name('.writing-' + uuid.uuid4().hex + '.json')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def save_backup(output, snapshot, backup_root, workspace, training_data=None):
    output, workspace = Path(output).resolve(), Path(workspace).resolve()
    snapshot = Path(snapshot).resolve() if snapshot else None
    recovery_root = Path(str(output) + '-recovery')
    if not output.is_dir():
        raise ValueError('Checkpoint output is missing; cannot back it up')
    if snapshot is not None:
        if snapshot.parent != recovery_root or not (snapshot / 'complete.json').is_file():
            raise ValueError('Back up only a complete snapshot for this output')
        recovery = json.loads((snapshot / 'complete.json').read_text())
        if file_hash(snapshot / 'recovery.pt') != recovery['recovery_sha256']:
            raise ValueError('Recovery checksum mismatch; Drive backup not published')
        step = recovery['step']
    elif (output / 'run-evidence.json').is_file():
        step = json.loads((output / 'training_metrics.json').read_text())['optimizer_steps']
    else:
        raise ValueError('No learned recovery snapshot or completed checkpoint to back up')
    data = Path(training_data).resolve() if training_data else None
    if data is not None and (not data.is_relative_to(workspace) or not data.is_file()):
        raise ValueError('Training data backup must be a file in the lab workspace')
    folder = Path(backup_root).resolve() / output.name
    folder.mkdir(parents=True, exist_ok=True)
    name = f'backup-{time.time_ns()}-{uuid.uuid4().hex[:8]}.zip'
    print(f'Drive backup starting at optimizer step {step}: {folder}', flush=True)
    with tempfile.TemporaryDirectory(prefix='kev-checkpoint-backup-') as temporary:
        archive = Path(temporary) / name
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_STORED) as saved:
            for directory in [output, *([snapshot] if snapshot else [])]:
                for file in sorted(directory.rglob('*')):
                    if file.is_symlink():
                        raise ValueError('Checkpoint backup contains a symbolic link')
                    if file.is_file():
                        saved.write(file, file.relative_to(output.parent))
            if snapshot is not None:
                saved.writestr(recovery_root.name + '/latest.json', json.dumps(recovery) + '\n')
            if data is not None:
                saved.write(data, 'inputs/training.jsonl')
        checksum = file_hash(archive)
        uploading = folder / ('.uploading-' + name)
        shutil.copyfile(archive, uploading)
        if file_hash(uploading) != checksum:
            raise ValueError('Drive archive checksum mismatch; previous backup retained')
        os.replace(uploading, folder / name)
    receipt = {'version': 1, 'archive': name, 'sha256': checksum, 'created_ns': time.time_ns(),
               'output_path': str(output), 'workspace_path': str(workspace), 'step': step,
               'snapshot': snapshot.name if snapshot else None,
               'completed_stage': (output / 'run-evidence.json').is_file(),
               'training_data': {'path': str(data), 'sha256': file_hash(data)} if data else None}
    atomic_json(folder / (name + '.json'), receipt)
    atomic_json(folder / 'latest.json', receipt)
    # Keep the two most recently published backups, including an older selected
    # checkpoint continued as a new attempt. Ignore interrupted uploads.
    previous = sorted(folder.glob('backup-*.zip.json'),
                      key=lambda p: json.loads(p.read_text())['created_ns'])
    for metadata in previous[:-2]:
        old = json.loads(metadata.read_text())
        archive = folder / old['archive']
        if archive.parent != folder:
            raise ValueError('Invalid backup archive name')
        archive.unlink(missing_ok=True)
        metadata.unlink()
    print(f'Drive backup complete at optimizer step {step}: {folder / name}', flush=True)
    return receipt


def restore_backup(output, backup_root, workspace):
    output, workspace = Path(output).resolve(), Path(workspace).resolve()
    recovery_root = Path(str(output) + '-recovery')
    folder = Path(backup_root).resolve() / output.name
    pointer = folder / 'latest.json'
    if not pointer.is_file() or output.exists() or recovery_root.exists():
        return None  # Existing local training is never replaced.
    receipt = json.loads(pointer.read_text())
    if receipt['version'] != 1 or receipt['output_path'] != str(output) or receipt['workspace_path'] != str(workspace):
        raise ValueError('Restore this backup at its original lab workspace and output paths')
    archive = folder / receipt['archive']
    print(f'Drive restore starting at optimizer step {receipt["step"]}: {archive}', flush=True)
    if archive.parent != folder or file_hash(archive) != receipt['sha256']:
        raise ValueError('Drive backup archive checksum mismatch')
    data_info = receipt.get('training_data')
    data = Path(data_info['path']).resolve() if data_info else None
    if data is not None and (not data.is_relative_to(workspace) or
                            (data.exists() and file_hash(data) != data_info['sha256'])):
        raise ValueError('Restore training data at its original path without replacing changed labels')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.restoring-', dir=output.parent) as temporary:
        staging = Path(temporary)
        with zipfile.ZipFile(archive) as saved:
            allowed = {output.name, recovery_root.name, 'inputs'}
            for member in saved.infolist():
                target = (staging / member.filename).resolve()
                if not target.is_relative_to(staging.resolve()) or Path(member.filename).parts[0] not in allowed:
                    raise ValueError('Unsafe checkpoint archive path')
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Checkpoint archive contains a symbolic link')
            saved.extractall(staging)
        if not (staging / output.name).is_dir():
            raise ValueError('Backup is missing the checkpoint output')
        if receipt['snapshot'] is not None:
            from lora_recovery import latest_snapshot
            snapshot = latest_snapshot(staging / recovery_root.name)
            if snapshot is None or snapshot.name != receipt['snapshot']:
                raise ValueError('Backup is missing its selected recovery snapshot')
            saved_receipt = json.loads((snapshot / 'complete.json').read_text())
            if file_hash(snapshot / 'recovery.pt') != saved_receipt['recovery_sha256']:
                raise ValueError('Restored recovery checksum mismatch')
        if data is not None:
            source = staging / 'inputs/training.jsonl'
            if file_hash(source) != data_info['sha256']:
                raise ValueError('Restored training data checksum mismatch')
            if not data.exists():
                data.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, data)
        (staging / output.name).rename(output)
        if (staging / recovery_root.name).is_dir():
            (staging / recovery_root.name).rename(recovery_root)
    print(f'Restored Drive backup at optimizer step {receipt["step"]}: {output}', flush=True)
    return receipt
