"""Execute the local bootstrap without GPU/model downloads and preserve user data."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / 'notebooks/pacman_kev_lab_local.ipynb'


class LocalNotebookTests(unittest.TestCase):
    def setUp(self):
        self.notebook = json.loads(NOTEBOOK.read_text())
        self.cells = {cell['id']: ''.join(cell['source']) for cell in self.notebook['cells']}

    def run_bootstrap(self, directory, source=ROOT):
        # A subprocess isolates the notebook's sys.path/module-refresh side effects.
        program = '''
import hashlib, json
from pathlib import Path
code = json.loads(Path(%r).read_text())['cells'][3]['source']
state = {}
exec(''.join(code), state)
runtime = state['runtime']
assert runtime.allow_other_gpu
assert runtime.initial_execution == {'batch':1,'accum':8,'row_budget':2048}
assert runtime.python == Path(__import__('os').environ['QPLUS_TRAINING_ENV'])/'bin/python'
assert runtime.environment()['CUDA_VISIBLE_DEVICES'] == '1'
assert runtime.environment()['UV_PROJECT_ENVIRONMENT'] == str(runtime.python.parent.parent)
assert state['manifest']['expected_optimizer_steps'] == 762
assert 'ARCADE_SRCDOC' not in state['GAME']
for name in ('lab_runtime.py', 'lab_memory_check.py', 'lab_native_kernels.py'):
    copied = state['LAB_DIR']/name
    assert hashlib.sha256(copied.read_bytes()).hexdigest() == state['LOCAL_FILES'][name]
native = __import__('lab_native_kernels')
assert Path(native.__file__).resolve() == state['LAB_DIR']/'lab_native_kernels.py'
# An updated bootstrap must discard the already-imported native helper too.
stale_native_function = object()
native.ensure_native_kernels = stale_native_function
keep = state['CHECKPOINT_ROOT']/'keep-head.bin'
keep.parent.mkdir(parents=True,exist_ok=True)
keep.write_bytes(b'existing checkpoint')
reviewed = state['LAB_DIR']/'data/pacman-planner-v1-train-reviewed.jsonl'
reviewed.write_bytes(b'existing learner labels')
exec(''.join(code), state)
assert keep.read_bytes() == b'existing checkpoint'
assert reviewed.read_bytes() == b'existing learner labels'
assert __import__('lab_native_kernels').ensure_native_kernels is not stale_native_function
assert state['runtime'].selected_pacman_recipe(state['RECIPE'])['batch'] == 1
assert state['RECIPE']['batch'] == 4
print('Local bootstrap, native assets, full dataset and repeat preservation passed.')
''' % str(NOTEBOOK)
        env = dict(os.environ, QPLUS_GPU='1', QPLUS_LAB_SOURCE=str(source),
                   QPLUS_LAB_DIR=str(Path(directory)/'runtime'),
                   QPLUS_TRAINING_ENV=str(Path(directory)/'envs/training'))
        return subprocess.run([sys.executable, '-c', program], env=env,
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)

    def test_real_local_bootstrap_preserves_checkpoints_and_reviewed_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_bootstrap(directory)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('repeat preservation passed', result.stdout)

    def test_missing_checkout_fails_before_gpu_or_dependency_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_bootstrap(directory, Path(directory)/'missing-source')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Missing local lab helper', result.stderr)

    def test_changed_native_helper_fails_checksum_before_gpu_or_dependency_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source'
            source.mkdir()
            colab = json.loads((ROOT/'notebooks/pacman_kev_lab.ipynb').read_text())
            # Keep the real course assets, but present one changed lab helper.
            for node in ast.parse(''.join(colab['cells'][3]['source'])).body:
                if isinstance(node, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == 'FILES'
                        for target in node.targets):
                    files = ast.literal_eval(node.value)
                    break
            for name in [*files, 'lab_runtime.py', 'lab_memory_check.py']:
                target = source/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(ROOT/name)
            (source/'lab_native_kernels.py').write_text('changed helper, never imported\n')
            result = self.run_bootstrap(directory, source)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Unexpected source: lab_native_kernels.py', result.stderr)

    def test_recipe_assertions_use_selected_execution_without_changing_pins(self):
        colab = json.loads((ROOT/'notebooks/pacman_kev_lab.ipynb').read_text())
        colab_code = ''.join(colab['cells'][3]['source'])
        def literal(code, name):
            for node in ast.parse(code).body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                    return ast.literal_eval(node.value)
            raise AssertionError(name)
        for name in ('COURSE_REVISION', 'FILES'):
            self.assertEqual(literal(self.cells['cell-03'], name), literal(colab_code, name))
        self.assertIn('runtime.selected_pacman_recipe(RECIPE)', self.cells['cell-36'])
        self.assertIn('for key, value in PACMAN_RECIPE.items()', self.cells['cell-36'])
        self.assertIn('comparison[\'lab_execution\']', self.cells['cell-38'])
        self.assertTrue(literal(self.cells['cell-08'], 'RUN_TRAINING_PREFLIGHT'))
        self.assertFalse(literal(self.cells['cell-05'], 'SHOW_TENSORBOARD'))
        self.assertFalse(literal(self.cells['cell-10'], 'SAVE_TO_DRIVE'))


if __name__ == '__main__':
    unittest.main()
