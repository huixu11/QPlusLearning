"""Check notebook syntax and bundled sources without a GPU or model download."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "labs/lab-01-kev-pacman"


def main():
    notebook = json.loads((LAB / "notebooks/pacman_kev_lab.ipynb").read_text())
    assert notebook["nbformat"] == 4
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids))
    code_cells = 0
    bundled = None
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        code_cells += 1
        assert cell["outputs"] == [] and cell["execution_count"] is None
        tree = ast.parse("".join(cell["source"]))
        for statement in tree.body:
            if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "BUNDLED_FILES"
                for target in statement.targets
            ):
                bundled = ast.literal_eval(statement.value)
    assert bundled, "Missing notebook bundle"
    for name, content in bundled.items():
        source = LAB / ("vendor/jev-pacman/" + name.removeprefix("community/")
                        if name.startswith("community/") else name)
        assert content == source.read_text(), f"Stale bundled source: {name}"
    print(f"Checked {code_cells} code cells and {len(bundled)} bundled files.")


if __name__ == "__main__":
    main()
