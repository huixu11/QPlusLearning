"""Check notebook syntax and pinned helper hashes without a GPU or model download."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "labs/lab-01-kev-pacman"


def check_notebook(filename, local=False):
    notebook = json.loads((LAB / "notebooks" / filename).read_text())
    assert notebook["nbformat"] == 4
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids))
    code_cells = 0
    files = None
    revision = None
    runtime_sha = None
    memory_sha = None
    native_sha = None
    local_files = None
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        code_cells += 1
        assert cell["outputs"] == [] and cell["execution_count"] is None
        tree = ast.parse("".join(cell["source"]))
        for statement in tree.body:
            if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "FILES"
                for target in statement.targets
            ):
                files = ast.literal_eval(statement.value)
            if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "COURSE_REVISION"
                for target in statement.targets
            ):
                revision = ast.literal_eval(statement.value)
            if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "LOCAL_RUNTIME_SHA256"
                for target in statement.targets
            ):
                runtime_sha = ast.literal_eval(statement.value)
            if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "LOCAL_MEMORY_CHECK_SHA256"
                for target in statement.targets
            ):
                memory_sha = ast.literal_eval(statement.value)
            if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "LOCAL_NATIVE_KERNELS_SHA256"
                for target in statement.targets
            ):
                native_sha = ast.literal_eval(statement.value)
            if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "LOCAL_FILES"
                for target in statement.targets
            ):
                local_files = statement.value
    assert files, "Missing notebook source manifest"
    lock = json.loads((LAB / "notebook-source.json").read_text())
    assert revision == lock["revision"] and files == lock["files"]
    assert len(revision) == 40
    for name, expected in files.items():
        assert hashlib.sha256((LAB / name).read_bytes()).hexdigest() == expected, f"Stale helper: {name}"
    if local:
        assert runtime_sha == hashlib.sha256((LAB / 'lab_runtime.py').read_bytes()).hexdigest(), 'Stale lab runtime'
        assert memory_sha == hashlib.sha256((LAB / 'lab_memory_check.py').read_bytes()).hexdigest(), 'Stale lab memory check'
        assert native_sha == hashlib.sha256((LAB / 'lab_native_kernels.py').read_bytes()).hexdigest(), 'Stale lab native kernels'
        assert isinstance(local_files, ast.Dict), 'Missing local helper manifest'
        constants = {'LOCAL_RUNTIME_SHA256': runtime_sha,
                     'LOCAL_MEMORY_CHECK_SHA256': memory_sha,
                     'LOCAL_NATIVE_KERNELS_SHA256': native_sha}
        manifest = {}
        for key, value in zip(local_files.keys, local_files.values):
            assert isinstance(value, ast.Name), 'Unexpected local helper hash expression'
            if key is None:
                assert value.id == 'FILES', 'Unexpected local helper manifest expansion'
                manifest.update(files)
            else:
                assert value.id in constants, 'Unpinned local helper hash'
                manifest[ast.literal_eval(key)] = constants[value.id]
        assert manifest == {**files, 'lab_runtime.py': runtime_sha,
                            'lab_memory_check.py': memory_sha,
                            'lab_native_kernels.py': native_sha}, 'Incomplete local helper manifest'
        assert notebook['metadata']['kernelspec']['name'] == 'qplus-a6000-py313'
    print(f"{filename}: checked {code_cells} code cells and {len(files)} pinned helper files" +
          (' plus three lab runtime helpers.' if local else '.'))


def main():
    check_notebook('pacman_kev_lab.ipynb')
    check_notebook('pacman_kev_lab_local.ipynb', local=True)


if __name__ == "__main__":
    main()
