"""Rebuild the self-contained Kev training notebook and browser game."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "vendor/jev-pacman"


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

Follow Kev's published **LoRA plus pointer-head** architecture. First train a fresh decision model from `Qwen/Qwen3.5-0.8B-Base` on the frozen `decision-v7` training suite. Then fine-tune your own adapter and head on Pac-Man labels. Original base matrices stay frozen, and LoRA changes the encoder's effective features. This is initial decision-model training over an already pretrained LLM.

**You train Pac-Man. Ghosts follow deterministic game code.** Start with the community browser game's maze and renderer, then use Kev to select the player's legal moves. This adaptation changes the original community demo, where Jev controls the ghosts.

0–10 compare architectures and play; 10–20 inspect initial training; 20–30 prepare player labels; **30–60 mandatory fine-tuning**; 60–80 compare; 80–90 debrief. Installation, downloads and the complete initial training run are prework, so we can keep the published recipe and the 90-minute class.

The target runtime is **Colab with one NVIDIA RTX PRO 6000 Blackwell GPU**. The full Server Edition has 96 GB VRAM. Check the actual allocation in the prework cell. The notebook uses BF16 inference and training autocast, with FP32 stored backbone, adapter and head parameters in the initial recipe. Colab does not guarantee this GPU, including on paid plans. Arrange access before class and record actual runtime cost. Both stages still need an instructor GPU preflight.

We reproduce the published **base stage**, then add our own Pac-Man domain adaptation. The released Kev-0.8B also has dates, documents/skills and calibration stages. This lab does not reproduce those stages or inherit the release's accuracy/calibration claims. [Recipe and compute](https://github.com/jaredpalmer/kev/blob/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4/docs/model-cards/kev-0.8b.md#training-procedure).""")

    md("""## Game provenance

Game source: [codaaiteam/jev-pacman](https://github.com/codaaiteam/jev-pacman), MIT, pinned `8446fe74690cd61909bda91acfadbccb0f02b422`. Model source: [jaredpalmer/kev](https://github.com/jaredpalmer/kev). The model receives structured state, not screenshots. No ROM or paid API key is required for the notebook path.""")
    md("""## Prework: prepare the runtime

Save your own copy of this notebook in Colab. Open **Runtime > Change runtime type**, select the RTX PRO 6000 Blackwell option if your account offers it, then connect. Run the prework cells before class. Confirm the GPU name rather than relying on a menu label. A different allocation requires an instructor-approved, timed fallback.

Setup creates a separate Python 3.13 environment using Kev's locked dependencies, including PyTorch 2.8.0 with CUDA 12.8. It checks GPU identity, BF16 support and a CUDA forward/backward pass in that environment, then writes `runtime-preflight.json`. The notebook kernel only runs the teaching helpers and Colab callback. Complete installation and model downloads before class.

The target GPU is an assumption for this session, not a free-compute promise. [Colab availability](https://research.google.com/colaboratory/faq.html), [NVIDIA GPU specifications](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/), [PyTorch Blackwell support](https://pytorch.org/blog/pytorch-2-7/).""")
    files = {name: (ROOT / name).read_text() for name in ["api_client.py", "pacman_lab.py", "cloud_runtime.py", "games/player-controller.js"]}
    files.update({"community/index.html": (SOURCE / "index.html").read_text(), "community/LICENSE": (SOURCE / "LICENSE").read_text()})
    code("""from pathlib import Path
import sys, json, os
LAB_DIR = Path.cwd() / 'pacman-kev-lab'
LAB_DIR.mkdir(exist_ok=True)
BUNDLED_FILES = """ + repr(files) + """
for name, content in BUNDLED_FILES.items():
    target = LAB_DIR / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding='utf-8')
sys.path.insert(0, str(LAB_DIR))
from cloud_runtime import CloudRuntime, MODEL_RUN
from pacman_lab import *
from api_client import call, distribution
SOURCE_HTML = (LAB_DIR / 'community/index.html').read_text()
MAZE = maze_from_html(SOURCE_HTML)
runtime = CloudRuntime(LAB_DIR)
manifest = make_data(SOURCE_HTML, LAB_DIR / 'data')
GAME = notebook_game(SOURCE_HTML, (LAB_DIR / 'games/player-controller.js').read_text(), (LAB_DIR / 'community/LICENSE').read_text())
print('Prepared player controller and disjoint synthetic snapshots.')
""")
    code("""runtime.setup()
audit = runtime.prepare_training()
print('Training suite records:', audit['suite_records'])
print('Trainable parameters:', sum(audit['trainable_parameters'].values()))
""")
    md("""## Prework: initial decision training from the base

This cell uses the unchanged upstream `kev.train` module and the selected published experiment (`experiments/q35-08b.json`, seed 2). It uses all **12,576 decision-v7 training records**, two epochs, batch 8, accumulation 1, LoRA rank 16 / alpha 32, a 256-dimensional pointer head, learning rate 1e-4, OneCycleLR, AdamW and BF16 autocast over FP32 weights. It preserves option shuffling, none-of-the-above/distractor augmentation and 25% none minimal pairs. It passes **no `--init_from`** and never uses development or test data for training.

The published H100 initial run took about 20 minutes. We have not timed this Colab GPU. Run it before class and keep its checkpoint. The helper caps the attempt at 90 minutes. Use a new directory for a new attempt. A supplied instructor checkpoint is a fallback and must be labelled as such.

The next cell defaults to full initial training. To restore saved prework after a disconnect, upload its ZIP through Colab's Files sidebar and set `RESTORE_ARCHIVE` to its path. Declare whether the checkpoint is yours or the instructor's. The following cell saves a backup ZIP.""")
    code("""INITIAL = LAB_DIR / 'checkpoints/decision-v7-initial'
print('Published command:', runtime.pretraining_command(INITIAL))
RESTORE_ARCHIVE = None  # Example: '/content/pacman-initial-checkpoint.zip'
RESTORED_CHECKPOINT_OWNER = 'learner'  # Set to 'instructor' for a supplied fallback
import zipfile
if RESTORE_ARCHIVE is None:
    initial_checkpoint = runtime.pretrain(INITIAL, steps=0)
    initial_origin = 'learner trained in this session'
else:
    INITIAL.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(RESTORE_ARCHIVE) as saved:
        assert all((INITIAL / member.filename).resolve().is_relative_to(INITIAL.resolve()) for member in saved.infolist())
        saved.extractall(INITIAL)
    initial_checkpoint = INITIAL
    initial_origin = RESTORED_CHECKPOINT_OWNER + ' restored checkpoint'
initial_config = json.loads((INITIAL / 'training_config.json').read_text())
initial_metrics = json.loads((INITIAL / 'training_metrics.json').read_text())
assert initial_config['init_source'] is None, 'Initial training must start from the base'
published = json.loads((runtime.repo / 'experiments/q35-08b.json').read_text())[2]
assert all(initial_config['args'][key] == value for key,value in published.items()), 'Restore the matching published base-stage checkpoint'
assert initial_metrics['optimizer_steps'] > 0
assert initial_metrics['records_seen'] == initial_metrics['requested_records'], 'Complete both epochs'
models = runtime.start(INITIAL)
print(json.dumps(models, indent=2))
""")
    code("""PREWORK_ARCHIVE = LAB_DIR.parent / 'pacman-initial-checkpoint.zip'
with zipfile.ZipFile(PREWORK_ARCHIVE, 'w', zipfile.ZIP_DEFLATED) as saved:
    for file in INITIAL.rglob('*'):
        if file.is_file():
            saved.write(file, file.relative_to(INITIAL))
try:
    from google.colab import files
except ImportError:
    print('Save your initial checkpoint:', PREWORK_ARCHIVE)
else:
    files.download(str(PREWORK_ARCHIVE))
""")
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
    md("""## CP1: inspect initial training and a player decision (10–20 minutes)

Inspect `initial_config`, `initial_metrics` and `trainable-parameters.json`. Find the fresh pointer head, trainable LoRA parameters and fixed original base matrices. Explain why gradients still travel through the encoder. Initial training learns general typed decisions; it has not yet seen Pac-Man labels.

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
    md("""## CP3: fine-tune Pac-Man decisions (30–60 minutes)

30-minute block: 5 minutes inspect a labelled request and the loss; up to 20 minutes fine-tune for **two complete epochs**; 5 minutes inspect and save the checkpoint. This is supervised imitation. Kev updates its LoRA adapter and pointer head while the original base matrices remain frozen; the adapter changes the encoder's effective features.

Warm-start from **your initial decision-v7 checkpoint**, using Kev's documented custom-data settings: learning rate 2e-5, batch 1, accumulation 8, BF16 autocast and gradient checkpointing. With 64 accepted rows, this gives 16 optimizer steps. The Pac-Man adaptation disables none/distractor insertion because the only valid outputs are legal player moves. Option shuffling remains active. This is a documented task-specific departure from the generic initial recipe.

The helper reads architecture from the checkpoint and stops inference to free GPU memory. Use a new checkpoint directory. A timed GPU preflight is still required; successful execution or improved play has not been established by this draft.""")
    code("""print(json.dumps(training[0], indent=2))
CHECKPOINT = LAB_DIR / 'checkpoints/pacman-finetuned'
checkpoint = runtime.finetune(training_file, CHECKPOINT, init_from=INITIAL, steps=0)
metrics = json.loads((checkpoint / 'training_metrics.json').read_text())
print(metrics)
assert metrics['optimizer_steps'] > 0, 'Training must update the model'
assert metrics['records_seen'] == metrics['requested_records'], 'Complete both epochs'
""")
    md("""## CP4: before/after on the same decisions (60–72 minutes)

The candidate is now fixed. Score both models on the same 16 evaluation snapshots and save predictions by ID. Report accuracy against the teacher and immediate captures after the selected moves. Sixteen boards on one maze do not establish general game skill. Fine-tuning may leave answers unchanged or make them worse.""")
    code("""runtime.start(INITIAL)
before = evaluate(LAB_DIR / 'data/pacman-evaluation.jsonl')
before_run = rollout(MAZE, max_turns=24)
runtime.start(checkpoint)
after = evaluate(LAB_DIR / 'data/pacman-evaluation.jsonl')
after_run = rollout(MAZE, max_turns=24)
comparison = {'base_model': audit['base'], 'base_revision': audit['base_revision'], 'initial_checkpoint': str(INITIAL), 'initial_origin': initial_origin, 'fine_tuned_checkpoint': str(checkpoint), 'before': before, 'after': after, 'rollouts': {'initial': before_run, 'fine_tuned': after_run}, 'initial_training': initial_metrics, 'pacman_training': metrics, 'runtime': runtime.gpu}
(LAB_DIR / 'comparison.json').write_text(json.dumps(comparison, indent=2))
for name, result in [('initial', before), ('fine_tuned', after)]:
    print(name, 'accuracy', result['accuracy'], 'immediate captures', result['caught_next_turn'])
""")
    md("""## CP4 continued: watch Pac-Man play (72–80 minutes)

Repeat the browser cell, choose **Kev**, and start a new game with the fine-tuned checkpoint active. Both Python rollouts above use the same starting board, ghost code and 24-turn cap. Compare dots and turns; these two trajectories are an illustration, not a win-rate estimate.

The starter teacher is also a useful rules baseline. It is deliberately simple and can get stuck. The model learns from structured state; there is no screenshot encoder or frame-by-frame RL training in this exercise.""")
    code("""for name, episode in [('initial', before_run), ('fine_tuned', after_run)]:
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

Submit the executed notebook, reviewed training JSONL, `comparison.json`, both small adapter/head checkpoints, both training configurations/metrics and `runtime-preflight.json`. Record the Colab compute units consumed and elapsed GPU time from your session. Save outputs before the temporary runtime disconnects, then stop the server. On Colab, run the download cell.""")
    code("""runtime.stop()
archive = str(LAB_DIR.parent / 'pacman-lab-submission.zip')
# Export small adapters/heads and results, without foundation weights or packages.
import zipfile
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as out:
    for file in [LAB_DIR/'comparison.json', training_file, LAB_DIR/'runtime-preflight.json', LAB_DIR/'trainable-parameters.json']:
        out.write(file, file.relative_to(LAB_DIR))
    for folder in [INITIAL, checkpoint]:
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
    notebook()
    game()
    print('Built Lab 1 notebook and Pac-Man browser game.')
