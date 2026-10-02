"""Rebuild the stage-by-stage Kev notebook and browser game."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "vendor/jev-pacman"
HELPERS = ["api_client.py", "pacman_lab.py", "cloud_runtime.py", "training_monitor.py",
           "training_stages.py", "training-stages.json", "games/player-controller.js",
           "vendor/jev-pacman/index.html", "vendor/jev-pacman/LICENSE"]


def source_lock(pin=False):
    path = ROOT / "notebook-source.json"
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in HELPERS}
    if pin:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        # A notebook must only fetch reviewed, committed sources.
        for name in HELPERS:
            content = subprocess.check_output(["git", "show", f"{revision}:labs/lab-01-kev-pacman/{name}"], cwd=ROOT)
            if hashlib.sha256(content).hexdigest() != hashes[name]:
                raise ValueError(f"Commit helper changes before pinning the notebook: {name}")
        path.write_text(json.dumps({"repository": "yxc20089/QPlusLearning", "revision": revision,
                                    "files": hashes}, indent=2) + "\n")
    lock = json.loads(path.read_text())
    if lock["files"] != hashes:
        raise ValueError("Helper sources changed; commit them and rebuild with --pin-source")
    return lock


def notebook():
    cells = []
    def add(kind, content):
        cell = {"id": f"cell-{len(cells):02d}", "cell_type": kind, "metadata": {}, "source": content.strip().splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    md = lambda text: add("markdown", text)
    code = lambda text: add("code", text)
    md("""# Lab 1 — Train a Decision Model to Play Pac-Man

Follow Kev's published **LoRA plus pointer-head** architecture. First train a fresh decision model from `Qwen/Qwen3.5-0.8B-Base` on the frozen `decision-v7` training suite. Continue through separate dates/missing-evidence and documents/skills stages, then fine-tune your own adapter and head on Pac-Man labels. Original base matrices stay frozen, and LoRA changes the encoder's effective features. This is initial decision-model training over an already pretrained LLM.

**You train Pac-Man. Ghosts follow deterministic game code.** Start with the community browser game's maze and renderer, then use Kev to select the player's legal moves. This adaptation changes the original community demo, where Jev controls the ghosts.

0–10 compare architectures and play; 10–20 inspect initial training; 20–30 prepare player labels; **30–60 mandatory fine-tuning**; 60–80 compare; 80–90 debrief. Installation, downloads and the three general-decision training stages are prework, so we can keep the published recipe and the 90-minute class.

The target runtime is **Colab with one NVIDIA RTX PRO 6000 Blackwell GPU**. The full Server Edition has 96 GB VRAM. Check the actual allocation in the prework cell. The notebook uses BF16 inference and training autocast, with FP32 stored backbone, adapter and head parameters in the initial recipe. Colab does not guarantee this GPU, including on paid plans. Arrange access before class and record actual runtime cost. All stages still need an instructor GPU preflight.

The notebook separates **Stage 1: initial decision training → Stage 2: dates/missing evidence → Stage 3: documents/skills → Stage 4: Pac-Man fine-tuning**. Each saves its own checkpoint and training logs. Calibration is a separate probability-fitting procedure and is not applied here; these fresh runs do not inherit the released model's benchmark scores or fitted temperature. [Recipe and compute](https://github.com/jaredpalmer/kev/blob/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4/docs/model-cards/kev-0.8b.md#training-procedure).""")

    md("""## Game provenance

Game source: [codaaiteam/jev-pacman](https://github.com/codaaiteam/jev-pacman), MIT, pinned `8446fe74690cd61909bda91acfadbccb0f02b422`. Model source: [jaredpalmer/kev](https://github.com/jaredpalmer/kev). The model receives structured state, not screenshots. No ROM or paid API key is required for the notebook path.""")
    md("""## Prework: prepare the runtime

Save your own copy of this notebook in Colab. Open **Runtime > Change runtime type**, select the RTX PRO 6000 Blackwell option if your account offers it, then connect. Run the prework cells before class. Confirm the GPU name rather than relying on a menu label. A different allocation requires an instructor-approved, timed fallback.

Setup creates a separate Python 3.13 environment using Kev's locked dependencies, including PyTorch 2.8.0 with CUDA 12.8. It checks GPU identity, BF16 support and a CUDA forward/backward pass in that environment, then writes `runtime-preflight.json`. The notebook kernel only runs the teaching helpers and Colab callback. Complete installation, data/model downloads and all three general stages before class.

The target GPU is an assumption for this session, not a free-compute promise. [Colab availability](https://research.google.com/colaboratory/faq.html), [NVIDIA GPU specifications](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/), [PyTorch Blackwell support](https://pytorch.org/blog/pytorch-2-7/).""")
    lock = source_lock()
    code("""from pathlib import Path
from urllib.request import urlopen
import hashlib, json, os, sys
LAB_DIR = Path.cwd() / 'pacman-kev-lab'
LAB_DIR.mkdir(exist_ok=True)
COURSE_REVISION = """ + repr(lock['revision']) + """
FILES = """ + repr(lock['files']) + """
for name, expected_sha256 in FILES.items():
    target = LAB_DIR / name
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected_sha256:
        continue
    url = f'https://raw.githubusercontent.com/yxc20089/QPlusLearning/{COURSE_REVISION}/labs/lab-01-kev-pacman/{name}'
    with urlopen(url, timeout=60) as response:
        content = response.read()
    assert hashlib.sha256(content).hexdigest() == expected_sha256, f'Unexpected source: {name}'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    print('Downloaded verified helper:', name, flush=True)
sys.path.insert(0, str(LAB_DIR))
from cloud_runtime import CloudRuntime
from training_stages import specifications, inspect_checkpoint, restore_checkpoint, backup_checkpoint
from pacman_lab import *
from api_client import call, distribution
SOURCE_HTML = (LAB_DIR / 'vendor/jev-pacman/index.html').read_text()
MAZE = maze_from_html(SOURCE_HTML)
runtime = CloudRuntime(LAB_DIR)
manifest = make_data(SOURCE_HTML, LAB_DIR / 'data')
GAME = notebook_game(SOURCE_HTML, (LAB_DIR / 'games/player-controller.js').read_text(), (LAB_DIR / 'vendor/jev-pacman/LICENSE').read_text())
STAGE_OWNERS = {}
print('Prepared player controller and disjoint synthetic snapshots.')
""")
    md("""Setup fetches nine small files from a specific course commit and verifies every SHA-256. Their readable source is in GitHub. The previous embedded source dictionary was a portability mechanism; it is not model input or training data. This notebook now needs network access to fetch helpers on first use.

## Training monitor: open TensorBoard before running any stage

Colab supports TensorBoard inside the notebook. Every stage writes a separate run below `logs/`: per-optimizer-step cross-entropy, next-step learning rate, gradient norm before clipping, epoch, step time, records seen and peak allocated GPU memory. These are training curves; no validation loss or accuracy is invented. Raw subprocess output, JSONL and CSV are also saved, including failed attempts.

The monitor checks the exact Kev trainer checksum and adds one logging call in memory. It leaves the upstream checkout and optimizer calculations intact. Console progress appears at the first step, every ten steps and the last step; a 15-second heartbeat also covers loading and data preparation. [TensorBoard in Colab](https://www.tensorflow.org/tensorboard/tensorboard_in_notebooks).""")
    code("""import subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', 'tensorboard==2.20.0'])
LOG_DIR = LAB_DIR / 'logs'
LOG_DIR.mkdir(exist_ok=True)
SHOW_TENSORBOARD = True
if SHOW_TENSORBOARD:
    from IPython import get_ipython
    ip = get_ipython()
    if ip is not None:
        ip.run_line_magic('load_ext', 'tensorboard')
        ip.run_line_magic('tensorboard', f'--logdir "{LOG_DIR}" --reload_interval 5')
    else:
        print('Outside a notebook, run: tensorboard --logdir', LOG_DIR)
""")
    code("""runtime.setup()
audit = runtime.prepare_training()
print('Training suite records:', audit['suite_records'])
print('Trainable parameters:', sum(audit['trainable_parameters'].values()))
""")
    md("""## Stage 1 — Initial decision training (prework)

Only this stage starts fresh LoRA adapters and a pointer head over the pretrained Qwen base. It uses the selected published `experiments/q35-08b.json` seed-2 recipe: all 12,576 `decision-v7` training records, two epochs, batch 8, accumulation 1, learning rate 1e-4, rank 16 / alpha 32, a 256-dimensional head and BF16 autocast over FP32 weights. Option shuffling, none/distractor insertion and 25% none minimal pairs remain active. There is no `--init_from`.

The initial cell performs this stage only. It streams progress and writes TensorBoard curves. The full run is prework with a configurable 90-minute attempt cap, not a promised duration. To restore a completed initial checkpoint, upload its ZIP, set `RESTORE_ARCHIVE` and declare learner/instructor ownership. Use a new output directory for each attempt.""")
    code("""INITIAL = LAB_DIR / 'checkpoints/decision-v7-initial'
print('Published command:', runtime.pretraining_command(INITIAL), flush=True)
RESTORE_ARCHIVE = None  # Example: '/content/pacman-initial-checkpoint.zip'
RESTORED_CHECKPOINT_OWNER = 'learner'  # 'instructor' for a supplied fallback
if RESTORE_ARCHIVE is None:
    runtime.pretrain(INITIAL, steps=0)
    STAGE_OWNERS['initial'] = 'learner'
else:
    restore_checkpoint(RESTORE_ARCHIVE, INITIAL)
    STAGE_OWNERS['initial'] = RESTORED_CHECKPOINT_OWNER
published = json.loads((runtime.repo / 'experiments/q35-08b.json').read_text())[2]
initial_config, initial_metrics = inspect_checkpoint(INITIAL, stage='initial', owner=STAGE_OWNERS['initial'], recipe=published)
""")
    code("""PREWORK_ARCHIVE = backup_checkpoint(INITIAL, LAB_DIR.parent / 'pacman-initial-checkpoint.zip')
try:
    from google.colab import files
except ImportError:
    print('Save initial checkpoint:', PREWORK_ARCHIVE)
else:
    files.download(str(PREWORK_ARCHIVE))
""")
    md("""## Prepare the intermediate-stage data (prework)

Verify the dates/missing-evidence JSONL and reconstruct the published documents/skills training concatenation from the pinned `documents-v1`, `hard-v1` and `devtools-v1` training partitions. Every partition and the final joint file must match its published checksum and count. Replay always samples `decision-v7` **train**. Development/test rows do not enter any training stage.""")
    code("""runtime.prepare_intermediate_data()
STAGES = specifications()
print({name: {'new_records': stage['records'], 'replay_records': stage['replay']} for name, stage in STAGES.items()})
""")
    md("""## Stage 2 — Dates and missing evidence (prework)

Warm-start from your **Stage 1** checkpoint. Use 1,425 generated records (900 date-policy cases, 255 missing-fact cases and 270 intact controls) plus 2,000 replayed `decision-v7` training records. Published settings: one epoch, learning rate 4e-5, batch 8, accumulation 1, BF16, seed 1 and 25% none minimal pairs. This is a separate run and a separate checkpoint. Its 90-minute attempt cap is a scheduling limit pending GPU measurements.""")
    code("""DATES = LAB_DIR / 'checkpoints/dates-missing-evidence'
print('Dates command:', runtime.intermediate_command('dates', DATES, INITIAL), flush=True)
RESTORE_DATES_ARCHIVE = None
RESTORED_DATES_OWNER = 'learner'
if RESTORE_DATES_ARCHIVE is None:
    runtime.intermediate('dates', DATES, init_from=INITIAL)
    STAGE_OWNERS['dates'] = 'learner'
else:
    restore_checkpoint(RESTORE_DATES_ARCHIVE, DATES)
    STAGE_OWNERS['dates'] = RESTORED_DATES_OWNER
dates_config, dates_metrics = inspect_checkpoint(DATES, stage='dates', owner=STAGE_OWNERS['dates'], parent=INITIAL, recipe=STAGES['dates']['args'], data_spec=STAGES['dates'])
""")
    code("""DATES_ARCHIVE = backup_checkpoint(DATES, LAB_DIR.parent / 'pacman-dates-checkpoint.zip')
try:
    from google.colab import files
except ImportError:
    print('Save dates checkpoint:', DATES_ARCHIVE)
else:
    files.download(str(DATES_ARCHIVE))
""")
    md("""## Stage 3 — Documents and skills (prework)

Warm-start from your **Stage 2** checkpoint. Train on 5,219 consumer-finance complaint records, 6,000 generated skill records and 5,320 developer-tooling records (16,539 new records), plus 6,000 replayed `decision-v7` training records. Follow the released configuration: one epoch, learning rate 2e-5, batch 4, accumulation 2, BF16, 7,552-token state budget, gradient checkpointing, seed 1 and 25% none minimal pairs. This yields 2,818 optimizer steps if every published record is admitted.

After this separate run, the documents/skills checkpoint is the general-decision baseline for class. Its attempt cap is configurable, initially 90 minutes. Complete all three general stages before the 90-minute class. We have not timed them on the target GPU.""")
    code("""SKILLS = LAB_DIR / 'checkpoints/documents-skills'
print('Documents/skills command:', runtime.intermediate_command('documents_skills', SKILLS, DATES), flush=True)
RESTORE_SKILLS_ARCHIVE = None
RESTORED_SKILLS_OWNER = 'learner'
if RESTORE_SKILLS_ARCHIVE is None:
    runtime.intermediate('documents_skills', SKILLS, init_from=DATES)
    STAGE_OWNERS['documents_skills'] = 'learner'
else:
    restore_checkpoint(RESTORE_SKILLS_ARCHIVE, SKILLS)
    STAGE_OWNERS['documents_skills'] = RESTORED_SKILLS_OWNER
skills_config, skills_metrics = inspect_checkpoint(SKILLS, stage='documents_skills', owner=STAGE_OWNERS['documents_skills'], parent=DATES, recipe=STAGES['documents_skills']['args'], data_spec=STAGES['documents_skills'])
""")
    code("""SKILLS_ARCHIVE = backup_checkpoint(SKILLS, LAB_DIR.parent / 'pacman-skills-checkpoint.zip')
try:
    from google.colab import files
except ImportError:
    print('Save documents/skills checkpoint:', SKILLS_ARCHIVE)
else:
    files.download(str(SKILLS_ARCHIVE))
GENERAL = SKILLS
stage_metrics = {'initial': initial_metrics, 'dates': dates_metrics, 'documents_skills': skills_metrics}
models = runtime.start(GENERAL)
print(json.dumps(models, indent=2))
""")
    md("""## Calibration is separate from training

Kev's release fitted one probability temperature after its three training stages. We do not copy that fitted value into freshly trained checkpoints. This notebook keeps their own raw probabilities. A workload calibration experiment needs suitable held-out labels and is outside the mandatory 30-minute Pac-Man fine-tuning block. It does not update LoRA/head weights or change the top-ranked action.""")
    md("""## CP0: play, then give Kev the controls (0–10 minutes)

Try **Human** mode with arrows/WASD. Restart, select **Kev**, and watch its choices. The ghosts move after every second player turn. The whole simulation waits for each model answer; wall-clock survival is therefore not a fair skill metric. Pause the board before running training cells.

Colab supplies the notebook callback below. On Kaggle, use the structured decision cell and Python rollouts instead. API errors pause visibly; the game does not replace failed player decisions with a hidden rules controller.""")
    code("""from IPython.display import display, HTML, JSON
bridge = NotebookBridge(LAB_DIR / 'results/player-trace.jsonl')
try:
    from google.colab import output
except ImportError:
    print('Kaggle/local: continue with the decision and rollout cells below.')
else:
    output.register_callback('pacman.decide', lambda state: JSON(bridge.decide(state)))
    display(HTML(GAME))
""")
    md("""## CP1: inspect the training stages and a player decision (10–20 minutes)

Inspect `initial_config`, `dates_config`, `skills_config`, `stage_metrics` and `trainable-parameters.json`. Open TensorBoard and compare the three separate runs. Find the fresh pointer head, trainable LoRA parameters and fixed original base matrices. Explain why gradients still travel through the encoder. The three general stages learn typed decisions, dates/evidence and documents/skills; none has seen Pac-Man labels.

Predict the move first. Identify the player, two ghosts, remaining dots and legal options. Explain why a direction through a wall never appears. The game consumes `answers.move.choice`; inspect probabilities by direction name. Confidence is not the probability of clearing the maze.""")
    code("""state = initial_state(MAZE)
request = body(state)
print('Player:', state['player'], '\\nGhosts:', state['ghosts'])
print('Options:', request['questions']['move']['criteria'])
response, elapsed = call('/v1/systemone', request)
distribution(response['answers']['move'], request['questions']['move']['criteria'])
print(json.dumps(response['answers'], indent=2), '\\nHTTP ms:', round(elapsed))
""")
    md("""## CP2: inspect and edit labels (20–30 minutes)

Starter data has **64 training, 16 development, 16 evaluation** snapshots. Labels come from a stated heuristic: avoid immediate capture, approach a dot by maze distance, prefer distance from ghosts, then use a fixed tie order. These are synthetic labels, not recorded human expertise.

Inspect three training boards and enter your own legal move labels in `EDITS` before checking the teacher. Keep all edits in the training partition. Fix the training file and one candidate before opening evaluation data. Positions are disjoint across splits, but every split uses the same maze.""")
    code("""training = [json.loads(line) for line in (LAB_DIR / 'data/pacman-train.jsonl').read_text().splitlines()]
for row in training[:3]:
    print(row['_meta']['id'], json.dumps(row['state'], indent=2))
    print('Legal:', list(row['questions']['move']['criteria']))
EDITS = {}  # Example after inspecting a board: {'board-000': 'left'}
assert set(EDITS).issubset({r['_meta']['id'] for r in training}), 'Unknown training board ID'
for row in training:
    if row['_meta']['id'] in EDITS:
        label = EDITS[row['_meta']['id']]
        assert label in row['questions']['move']['criteria'], 'Choose a legal move'
        row['questions']['move'].update(label=label, src='learner_annotation')
        row['_meta']['label_source'] = 'learner annotation'
training_file = LAB_DIR / 'data/pacman-train-reviewed.jsonl'
training_file.write_text(''.join(json.dumps(row) + '\\n' for row in training))
print('Teacher labels for inspected boards:', [(r['_meta']['id'], r['questions']['move']['label']) for r in training[:3]])
print('Split sizes:', manifest['counts'])
before_dev = evaluate(LAB_DIR / 'data/pacman-development.jsonl')
print('Development accuracy:', before_dev['accuracy'])
""")
    md("""## Stage 4 / CP3: fine-tune Pac-Man decisions (30–60 minutes)

30-minute block: 5 minutes inspect a labelled request and the loss; up to 20 minutes fine-tune for **two complete epochs**; 5 minutes inspect and save the checkpoint. This is supervised imitation. Kev updates its LoRA adapter and pointer head while the original base matrices remain frozen; the adapter changes the encoder's effective features.

Warm-start from **your Stage 3 documents/skills checkpoint**, using Kev's documented custom-data settings: learning rate 2e-5, batch 1, accumulation 8, BF16 autocast and gradient checkpointing. With 64 accepted rows, this gives 16 optimizer steps. The Pac-Man adaptation disables none/distractor insertion because the only valid outputs are legal player moves. Option shuffling remains active. This is a documented task-specific departure from the generic initial recipe.

The helper reads architecture from the checkpoint and stops inference to free GPU memory. Use a new checkpoint directory. A timed GPU preflight is still required; successful execution or improved play has not been established by this draft.""")
    code("""print(json.dumps(training[0], indent=2))
CHECKPOINT = LAB_DIR / 'checkpoints/pacman-finetuned'
checkpoint = runtime.finetune(training_file, CHECKPOINT, init_from=GENERAL, steps=0)
metrics = json.loads((checkpoint / 'training_metrics.json').read_text())
print(metrics)
assert metrics['optimizer_steps'] > 0, 'Training must update the model'
assert metrics['records_seen'] == metrics['requested_records'], 'Complete both epochs'
""")
    md("""## CP4: before/after on the same decisions (60–72 minutes)

The candidate is now fixed. Score both models on the same 16 evaluation snapshots and save predictions by ID. Report accuracy against the teacher and immediate captures after the selected moves. Sixteen boards on one maze do not establish general game skill. Fine-tuning may leave answers unchanged or make them worse.""")
    code("""runtime.start(GENERAL)
before = evaluate(LAB_DIR / 'data/pacman-evaluation.jsonl')
before_run = rollout(MAZE, max_turns=24)
runtime.start(checkpoint)
after = evaluate(LAB_DIR / 'data/pacman-evaluation.jsonl')
after_run = rollout(MAZE, max_turns=24)
comparison = {'base_model': audit['base'], 'base_revision': audit['base_revision'], 'baseline_checkpoint': str(GENERAL), 'checkpoint_owners': STAGE_OWNERS, 'fine_tuned_checkpoint': str(checkpoint), 'before': before, 'after': after, 'rollouts': {'general': before_run, 'fine_tuned': after_run}, 'general_training_stages': stage_metrics, 'pacman_training': metrics, 'runtime': runtime.gpu}
(LAB_DIR / 'comparison.json').write_text(json.dumps(comparison, indent=2))
for name, result in [('general', before), ('fine_tuned', after)]:
    print(name, 'accuracy', result['accuracy'], 'immediate captures', result['caught_next_turn'])
""")
    md("""## CP4 continued: watch Pac-Man play (72–80 minutes)

Repeat the browser cell, choose **Kev**, and start a new game with the fine-tuned checkpoint active. Both Python rollouts above use the same starting board, ghost code and 24-turn cap. Compare dots and turns; these two trajectories are an illustration, not a win-rate estimate.

The starter teacher is also a useful rules baseline. It is deliberately simple and can get stuck. The model learns from structured state; there is no screenshot encoder or frame-by-frame RL training in this exercise.""")
    code("""for name, episode in [('general', before_run), ('fine_tuned', after_run)]:
    print(name, {k:episode[k] for k in ['turns', 'dots_collected', 'outcome']})
# Optional rules baseline using the same game mechanics:
def rule_predict(request):
    chosen = teacher(request['state'])
    keys = request['questions']['move']['criteria']
    return {'answers': {'move': {'choice':chosen, 'probabilities':{k:float(k==chosen) for k in keys}}}}
rule_run = rollout(MAZE, predict=rule_predict, max_turns=24)
print('rules', {k:rule_run[k] for k in ['turns','dots_collected','outcome']})
""")
    md("""## CP5: explain and submit (80–90 minutes)

Explain one changed move and one confident mistake. Trace `kev/api.py:to_record`, `kev/model.py:encode` / `PointerHead`, and `kev/train.py`. The lecture's CLM uses separate state/action representations and InfoNCE; Kev scores option-token representations with a pointer head and supervised cross-entropy. They are related decision systems with different training architectures.

Submit the executed notebook, reviewed training JSONL, `comparison.json`, all four small adapter/head checkpoints, all stage configurations/metrics, training logs and TensorBoard events and `runtime-preflight.json`. Record the Colab compute units consumed and elapsed GPU time from your session. Save outputs before the temporary runtime disconnects, then stop the server. On Colab, run the download cell.""")
    code("""runtime.stop()
archive = str(LAB_DIR.parent / 'pacman-lab-submission.zip')
# Export small adapters/heads and results, without foundation weights or packages.
import zipfile
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as out:
    for file in [LAB_DIR/'comparison.json', training_file, LAB_DIR/'runtime-preflight.json', LAB_DIR/'trainable-parameters.json']:
        out.write(file, file.relative_to(LAB_DIR))
    for folder in [INITIAL, DATES, SKILLS, checkpoint, LOG_DIR]:
        for file in folder.rglob('*'):
            if file.is_file():
                out.write(file, file.relative_to(LAB_DIR))
try:
    from google.colab import files
except ImportError:
    print('Download:', archive)
else:
    files.download(archive)
""")
    result = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}, "colab": {"name": "pacman_kev_lab.ipynb"}}, "nbformat": 4, "nbformat_minor": 5}
    (ROOT / 'notebooks/pacman_kev_lab.ipynb').write_text(json.dumps(result, indent=2) + '\n')



def game():
    from pacman_lab import notebook_game
    html = notebook_game((SOURCE / 'index.html').read_text(),
                         (ROOT / 'games/player-controller.js').read_text(),
                         (SOURCE / 'LICENSE').read_text())
    (ROOT / 'games/pacman.html').write_text(html)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pin-source', action='store_true', help='pin committed helpers before rebuilding')
    args = parser.parse_args()
    if args.pin_source:
        source_lock(pin=True)
    notebook()
    game()
    print('Built Lab 1 notebook and Pac-Man browser game.')
