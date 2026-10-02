"""Rebuild the stage-by-stage Kev notebook and browser game."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "vendor/jev-pacman"
HELPERS = ["api_client.py", "pacman_lab.py", "cloud_runtime.py", "training_monitor.py",
           "lora_recovery.py", "optimized_training.py", "training-kernels.json", "training_stages.py", "training-stages.json", "games/player-controller.js",
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

Follow Kev's published **LoRA plus pointer-head** architecture. First train a fresh decision model from `Qwen/Qwen3.5-4B-Base` on the frozen `decision-v7` training suite. Continue through separate dates/missing-evidence, documents, and skills/devtools stages, then fine-tune your own adapter and head on Pac-Man labels. Original base matrices stay frozen, and LoRA changes the encoder's effective features. This is initial decision-model training over an already pretrained LLM.

**You train Pac-Man. Ghosts follow deterministic game code.** Start with the community browser game's maze and renderer, then use Kev to select the player's legal moves. This adaptation changes the original community demo, where Jev controls the ghosts.

0–10 compare architectures and play; 10–20 inspect initial training; 20–30 prepare player labels; **30–60 mandatory fine-tuning**; 60–80 compare; 80–90 debrief. Installation, downloads and the four general-decision training stages are prework, so we can keep the published recipe and the 90-minute class.

The target runtime is **Colab with one NVIDIA RTX PRO 6000 Blackwell GPU**. The full Server Edition has 96 GB VRAM. Check the actual allocation in the prework cell. The notebook uses BF16 inference and training autocast, with FP32 stored backbone, adapter and head parameters in the initial recipe. Colab does not guarantee this GPU, including on paid plans. Arrange access before class and record actual runtime cost. All stages still need an instructor GPU preflight.

The notebook separates **Stage 1: initial decision training → Stage 2: dates/missing evidence → Stage 3: documents → Stage 4: skills/devtools → Stage 5: Pac-Man fine-tuning**. Each saves its own checkpoint and training logs. Calibration is a separate probability-fitting procedure and is not applied here; these fresh runs do not inherit the released model's benchmark scores or fitted temperature. [Recipe and compute](https://github.com/jaredpalmer/kev/blob/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4/docs/model-cards/kev-4b.md#training-procedure).""")

    md("""## Game provenance

Game source: [codaaiteam/jev-pacman](https://github.com/codaaiteam/jev-pacman), MIT, pinned `8446fe74690cd61909bda91acfadbccb0f02b422`. Model source: [jaredpalmer/kev](https://github.com/jaredpalmer/kev). The model receives structured state, not screenshots. No ROM or paid API key is required for the notebook path.""")
    md("""## Prework: prepare the runtime

Save your own copy of this notebook in Colab. Open **Runtime > Change runtime type**, select the RTX PRO 6000 Blackwell option if your account offers it, then connect. Run the prework cells before class. Confirm the GPU name rather than relying on a menu label. A different allocation requires an instructor-approved, timed fallback.

Setup creates a separate Python 3.13 environment using Kev's locked dependencies, including PyTorch 2.8.0 with CUDA 12.8. It checks GPU identity, BF16 support and a CUDA forward/backward pass in that environment, then writes `runtime-preflight.json`. The notebook kernel only runs the teaching helpers and Colab callback. Complete installation, data/model downloads and all four general stages before class.

The target GPU is an assumption for this session, not a free-compute promise. [Colab availability](https://research.google.com/colaboratory/faq.html), [NVIDIA GPU specifications](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/), [PyTorch Blackwell support](https://pytorch.org/blog/pytorch-2-7/).""")
    lock = source_lock()
    code("""from pathlib import Path
from urllib.request import urlopen
import hashlib, json, os, sys
TRAINING_PROFILE = 'memory_safe'  # 'published_reference' reproduces the original execution flags
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
# Refresh helpers when upgrading an existing notebook; keep checkpoints/logs.
if 'runtime' in globals():
    runtime.stop()
import importlib
importlib.invalidate_caches()
for name in ['cloud_runtime', 'training_monitor', 'lora_recovery', 'optimized_training', 'training_stages', 'pacman_lab', 'api_client']:
    sys.modules.pop(name, None)
from cloud_runtime import CloudRuntime
from training_stages import specifications, inspect_checkpoint, restore_checkpoint, backup_checkpoint
from pacman_lab import *
from api_client import call, distribution
SOURCE_HTML = (LAB_DIR / 'vendor/jev-pacman/index.html').read_text()
MAZE = maze_from_html(SOURCE_HTML)
runtime = CloudRuntime(LAB_DIR, training_profile=TRAINING_PROFILE)
manifest = make_data(SOURCE_HTML, LAB_DIR / 'data')
GAME = notebook_game(SOURCE_HTML, (LAB_DIR / 'games/player-controller.js').read_text(), (LAB_DIR / 'vendor/jev-pacman/LICENSE').read_text())
STAGE_OWNERS = {}
print('Prepared player controller and disjoint synthetic snapshots.')
""")
    md("""Setup fetches twelve small files from a specific course commit and verifies every SHA-256. Their readable source is in GitHub. The previous embedded source dictionary was a portability mechanism; it is not model input or training data. This notebook now needs network access to fetch helpers on first use.

## Training monitor: open TensorBoard before running any stage

Colab supports TensorBoard inside the notebook. Every stage writes a separate run below `logs/`: per-optimizer-step cross-entropy, next-step learning rate, gradient norm before clipping, epoch, step time, records seen, peak/live allocated memory, reserved memory and free VRAM. These are training curves; no validation loss or accuracy is invented. Raw subprocess output, JSONL and CSV are also saved, including failed attempts. `batches.jsonl` records question-row counts, longest padded rows and record IDs before each forward pass, including the batch that fails.

The monitor checks the exact Kev trainer checksum and instruments telemetry plus LoRA save/restore hooks in memory. It leaves the upstream checkout intact. Console progress appears at the first step, every ten steps and the last step; a 15-second heartbeat also covers loading and data preparation. [TensorBoard in Colab](https://www.tensorflow.org/tensorboard/tensorboard_in_notebooks).""")
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
print('Optimized training verification:', json.dumps(runtime.training_preflight, indent=2))
""")
    md("""## Required optimized training stack

Setup extends Kev's frozen environment with hash-pinned wheels: **Flash Linear Attention / fla-core 0.5.2**, **causal-conv1d 1.7.0** (Python 3.13, Torch 2.8, CUDA 12, Linux x86_64, CXX11 ABI), einops 0.8.1 and Ninja 1.13.0. Torch 2.8.0 / CUDA 12.8, Triton 3.4.0, Transformers 5.17 and PEFT remain at Kev's locked versions. Dependencies are installed without replacing that stack. SDPA handles full attention; FLA's Triton kernels handle DeltaNet; the CUDA convolution handles its short convolution. Training uses fused AdamW, BF16 autocast, gradient checkpointing, gradient accumulation and bounded row passes. The LoRA/head recipe and curriculum remain in Kev's trainer.

Before any long run, a separate process loads the actual pinned base/adapter/head and performs Kev loss, backward and fused AdamW on two verified decision-v7 records. It checks finite adapter/head gradients, records selected function bindings and profiled optimizer/SDPA operators, and writes `optimized-training-preflight.json`. Imports or reference-kernel fallback **fail setup**; each training subprocess verifies bindings again. The notebook reports encoding, forward, backward, gradient checks, completed optimizer checks and profiler finalization separately, with 15-second heartbeats and a 20-minute preflight cap. This cap is a timeout, not an expected duration. Two records are a compatibility test, not proof that the full curriculum fits VRAM or meets a timing target. This CUDA test must pass on your allocation; it has not been run by the course author.

Fused optimizer/kernel arithmetic can differ numerically from the reference implementation. The published-reference profile retains upstream training flags; both profiles now require optimized kernels. Optional inference fusion and serving CUDA graphs remain disabled for training because they do not implement this backward path. [FLA release](https://github.com/fla-org/flash-linear-attention/releases/tag/v0.5.2), [causal-conv1d release](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.7.0).""")
    md("""## Memory profile and recovery storage

A learner's original **0.8B** batch-8 run failed at step 2,073/3,144 with 89.90 GiB actively allocated on a 94.97 GiB GPU. Startup warnings confirmed `causal_conv1d` and `flash-linear-attention` were missing, so Qwen used reference PyTorch kernels with substantial training intermediates. The peak reached 91.91 GiB by step 280 and then stayed flat through step 2,070; that cumulative maximum alone cannot diagnose a leak. [Qwen kernel documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5#usage-tips-and-notes). Record counts are not physical batch sizes: records can contain multiple question rows and none-pair siblings.

The default **`memory_safe`** profile uses **batch 1 × accumulation 8**, gradient checkpointing and Kev's `row_budget=2048` for all five stages. It preserves the data, epochs, optimizer-step count, architecture, learning-rate schedule and augmentation settings. Microbatching, row grouping and dropout execution differ; this is not an exact numerical reproduction. The published execution flags remain available as `TRAINING_PROFILE='published_reference'`. The smaller profile has not yet passed a GPU run. A single question longer than the row budget still runs intact; records are not silently truncated or dropped.

Save at optimizer step 1, every 100 steps or five minutes (checked at optimizer boundaries), and the final step. Keep the latest two complete snapshots. Each contains the LoRA adapter/head, optimizer, scheduler, RNG and progress counters. Kev's native `--resume` supports full-weight runs only; the lab adds a separate LoRA recovery implementation. After a failure, set the affected stage's `RESUME_*` flag to `True`, keep the same profile, arguments, input files and output path, then rerun that stage. Do not rerun completed earlier stages. A snapshot can resume training or supply an intermediate model; it is not a completed curriculum stage.

**The older notebook saved LoRA weights only at the end. Its failed step-2,073 run has logs/configuration but no automatic learned checkpoint.** New output names below preserve that failed directory. Optional Drive storage keeps new checkpoints and recovery snapshots across runtime loss. Local `/content` files are temporary; export before disconnecting. Resume needs the original absolute paths. Restore a recovery ZIP into a new checkpoint root at that same path.""")
    code("""SAVE_TO_DRIVE = False  # Set True before training to persist checkpoints and recovery
RESTORE_RECOVERY_ARCHIVE = None  # ZIP from the inspection/export cell; restore into an absent root
CHECKPOINT_ROOT = LAB_DIR / 'checkpoints'
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    CHECKPOINT_ROOT = Path('/content/drive/MyDrive/QPlusLearning/lab-01-kev-pacman/checkpoints')
if RESTORE_RECOVERY_ARCHIVE is not None:
    restore_checkpoint(RESTORE_RECOVERY_ARCHIVE, CHECKPOINT_ROOT)
else:
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
print('Training profile:', runtime.training_profile, 'overrides:', runtime.selected_recipe({}))
print('Checkpoints and recovery:', CHECKPOINT_ROOT)
""")
    md("""### Inspect a failed run / export recovery

This cell is safe to rerun after a training exception. It checks files rather than assuming a checkpoint exists. Select the failed stage's output in `INSPECT_OUTPUT`; choose the original `decision-v7-initial` path to inspect the earlier OOM. Set `EXPORT_RECOVERY=True` to download the output plus its recovery snapshots. Download logs separately before disconnecting. A ZIP containing only configuration files cannot recover model weights.""")
    code("""INSPECT_OUTPUT = CHECKPOINT_ROOT / 'kev-4b-initial'
EXPORT_RECOVERY = False
EXPORT_LOGS = False
def show_training_state(output):
    output = Path(output)
    weights = [output / 'head.pt', *output.glob('adapter_model.*')]
    print('Output:', output, 'exists:', output.exists())
    print('Weight files:', {p.name: p.stat().st_size for p in weights if p.is_file()})
    print('Completed metrics:', (output / 'training_metrics.json').is_file())
    pointer = Path(str(output) + '-recovery') / 'latest.json'
    print('Latest recovery:', json.loads(pointer.read_text()) if pointer.is_file() else 'none')
show_training_state(LAB_DIR / 'checkpoints/decision-v7-initial')
show_training_state(INSPECT_OUTPUT)
archives = []
if EXPORT_RECOVERY:
    import zipfile
    archive = LAB_DIR.parent / 'pacman-stage-recovery.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as saved:
        for folder in [INSPECT_OUTPUT, Path(str(INSPECT_OUTPUT) + '-recovery')]:
            for file in folder.rglob('*'):
                if file.is_file():
                    saved.write(file, file.relative_to(INSPECT_OUTPUT.parent))
    archives.append(archive)
if EXPORT_LOGS:
    archives.append(backup_checkpoint(LOG_DIR, LAB_DIR.parent / 'pacman-training-logs.zip'))
for archive in archives:
    try:
        from google.colab import files
    except ImportError:
        print('Save:', archive)
    else:
        files.download(str(archive))
""")
    md("""## Stage 1 — Initial decision training (prework)

Only this stage starts fresh LoRA adapters and a pointer head over the pretrained Qwen base. It uses the selected published `experiments/q35-4b-s23.json` seed-2 recipe: all 12,576 `decision-v7` training records, two epochs, batch 4, accumulation 2, gradient checkpointing, learning rate 5e-5, rank 16 / alpha 32, a 256-dimensional head and BF16 autocast over FP32 weights. Option shuffling, none/distractor insertion and 25% none minimal pairs remain active. There is no `--init_from`.

The initial cell performs this stage only. The published batch 4 × accumulation 2 is replaced by batch 1 × accumulation 8 under the default memory profile, with checkpointing and bounded row passes. It streams progress, saves recovery snapshots and writes TensorBoard curves. The full run is prework with a configurable 180-minute attempt cap, not a promised duration. To restore a completed initial checkpoint, upload its ZIP, set `RESTORE_ARCHIVE` and declare learner/instructor ownership. Use `RESUME_INITIAL=True` only when a new-style recovery snapshot exists.""")
    code("""INITIAL = CHECKPOINT_ROOT / 'kev-4b-initial'
print('Selected command:', runtime.pretraining_command(INITIAL), flush=True)
RESUME_INITIAL = False  # True after interruption of this output, with recovery snapshots
RESTORE_ARCHIVE = None  # Example: '/content/pacman-initial-checkpoint.zip'
RESTORED_CHECKPOINT_OWNER = 'learner'  # 'instructor' for a supplied fallback
if RESTORE_ARCHIVE is None:
    runtime.pretrain(INITIAL, steps=0, resume=RESUME_INITIAL)
    STAGE_OWNERS['initial'] = 'learner'
else:
    restore_checkpoint(RESTORE_ARCHIVE, INITIAL)
    STAGE_OWNERS['initial'] = RESTORED_CHECKPOINT_OWNER
published = json.loads((runtime.repo / 'experiments/q35-4b-s23.json').read_text())[0]
initial_config, initial_metrics = inspect_checkpoint(INITIAL, stage='initial', owner=STAGE_OWNERS['initial'], recipe=runtime.selected_recipe(published))
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

Verify dates/missing-evidence JSONL and the separate `documents-v1` train partition. Reconstruct skills/devtools by concatenating verified `hard-v1` train then `devtools-v1` train. Each input matches its published checksum and count. The course pins the combined checksum; the unavailable historical round-10 concatenation is not claimed byte-identical. Replay always samples `decision-v7` **train**. Development/test rows do not enter any training stage.""")
    code("""runtime.prepare_intermediate_data()
STAGES = specifications()
print({name: {'new_records': stage['records'], 'replay_records': stage['replay']} for name, stage in STAGES.items()})
""")
    md("""## Stage 2 — Dates and missing evidence (prework)

Warm-start from your **Stage 1** checkpoint. Use 1,425 generated records (900 date-policy cases, 255 missing-fact cases and 270 intact controls) plus 2,000 replayed `decision-v7` training records. Published settings: one epoch, learning rate 2e-5, batch 4, accumulation 2, gradient checkpointing, BF16, seed 1 and 25% none minimal pairs. This is a separate run and a separate checkpoint. Its 180-minute attempt cap is a scheduling limit pending GPU measurements.""")
    code("""DATES = CHECKPOINT_ROOT / 'kev-4b-dates'
RESUME_DATES = False
print('Dates command:', runtime.intermediate_command('dates', DATES, INITIAL), flush=True)
RESTORE_DATES_ARCHIVE = None
RESTORED_DATES_OWNER = 'learner'
if RESTORE_DATES_ARCHIVE is None:
    runtime.intermediate('dates', DATES, init_from=INITIAL, resume=RESUME_DATES)
    STAGE_OWNERS['dates'] = 'learner'
else:
    restore_checkpoint(RESTORE_DATES_ARCHIVE, DATES)
    STAGE_OWNERS['dates'] = RESTORED_DATES_OWNER
dates_config, dates_metrics = inspect_checkpoint(DATES, stage='dates', owner=STAGE_OWNERS['dates'], parent=INITIAL, recipe=runtime.selected_recipe(STAGES['dates']['args']), data_spec=STAGES['dates'])
""")
    code("""DATES_ARCHIVE = backup_checkpoint(DATES, LAB_DIR.parent / 'pacman-dates-checkpoint.zip')
try:
    from google.colab import files
except ImportError:
    print('Save dates checkpoint:', DATES_ARCHIVE)
else:
    files.download(str(DATES_ARCHIVE))
""")
    md('## Stage 3 — Documents (prework)\n\nWarm-start from **Stage 2**. Use 5,219 consumer-finance complaint records plus 2,000 replayed decision-v7 training records. Published settings: one epoch, learning rate 2e-5, batch 2 × accumulation 4, BF16, FP32 stored weights, state budget 7,552, gradient checkpointing, seed 2 and 25% none minimal pairs. Expect 903 optimizer steps. The memory profile uses batch 1 × accumulation 8 with bounded row passes. Save this checkpoint before continuing; documents and skills are separate in Kev-4B.')
    code("""DOCUMENTS = CHECKPOINT_ROOT / 'kev-4b-documents'
RESUME_DOCUMENTS = False
print('Documents command:', runtime.intermediate_command('documents', DOCUMENTS, DATES), flush=True)
RESTORE_DOCUMENTS_ARCHIVE = None
RESTORED_DOCUMENTS_OWNER = 'learner'
if RESTORE_DOCUMENTS_ARCHIVE is None:
    runtime.intermediate('documents', DOCUMENTS, init_from=DATES, resume=RESUME_DOCUMENTS)
    STAGE_OWNERS['documents'] = 'learner'
else:
    restore_checkpoint(RESTORE_DOCUMENTS_ARCHIVE, DOCUMENTS)
    STAGE_OWNERS['documents'] = RESTORED_DOCUMENTS_OWNER
documents_config, documents_metrics = inspect_checkpoint(DOCUMENTS, stage='documents', owner=STAGE_OWNERS['documents'], parent=DATES, recipe=runtime.selected_recipe(STAGES['documents']['args']), data_spec=STAGES['documents'])
""")
    code("""DOCUMENTS_ARCHIVE = backup_checkpoint(DOCUMENTS, LAB_DIR.parent / 'pacman-documents-checkpoint.zip')
try:
    from google.colab import files
except ImportError:
    print('Save documents checkpoint:', DOCUMENTS_ARCHIVE)
else:
    files.download(str(DOCUMENTS_ARCHIVE))
""")
    md("""## Stage 4 — Skills and developer tools (prework)

Warm-start from **Stage 3 documents**. Use 6,000 generated skill records plus 5,320 developer-tooling records, with 4,000 replayed decision-v7 records. Published settings: one epoch, learning rate 2e-5, batch 2 × accumulation 4, BF16, FP32 stored weights, state budget 7,552, gradient checkpointing, seed 1 and 25% none minimal pairs. Expect 1,915 optimizer steps. The memory profile uses batch 1 × accumulation 8. This checkpoint is the class baseline. Complete all four general stages before class; their configurable 180-minute attempt caps are scheduling limits, not timing estimates.

A 4B model still uses hybrid DeltaNet/full attention and needs optimized kernels. Start fresh from the pinned 4B base; 0.8B adapters cannot initialize this curriculum. Published H100/H200 timings do not predict this GPU's duration.""")
    code("""SKILLS = CHECKPOINT_ROOT / 'kev-4b-skills'
RESUME_SKILLS = False
print('Skills and developer tools command:', runtime.intermediate_command('skills', SKILLS, DOCUMENTS), flush=True)
RESTORE_SKILLS_ARCHIVE = None
RESTORED_SKILLS_OWNER = 'learner'
if RESTORE_SKILLS_ARCHIVE is None:
    runtime.intermediate('skills', SKILLS, init_from=DOCUMENTS, resume=RESUME_SKILLS)
    STAGE_OWNERS['skills'] = 'learner'
else:
    restore_checkpoint(RESTORE_SKILLS_ARCHIVE, SKILLS)
    STAGE_OWNERS['skills'] = RESTORED_SKILLS_OWNER
skills_config, skills_metrics = inspect_checkpoint(SKILLS, stage='skills', owner=STAGE_OWNERS['skills'], parent=DOCUMENTS, recipe=runtime.selected_recipe(STAGES['skills']['args']), data_spec=STAGES['skills'])
""")
    code("""SKILLS_ARCHIVE = backup_checkpoint(SKILLS, LAB_DIR.parent / 'pacman-skills-checkpoint.zip')
try:
    from google.colab import files
except ImportError:
    print('Save skills checkpoint:', SKILLS_ARCHIVE)
else:
    files.download(str(SKILLS_ARCHIVE))
GENERAL = SKILLS
stage_metrics = {'initial': initial_metrics, 'dates': dates_metrics, 'documents': documents_metrics, 'skills': skills_metrics}
models = runtime.start(GENERAL)
print(json.dumps(models, indent=2))
""")
    md("""## Calibration is separate from training

Kev's release fitted one probability temperature after its four training stages. We do not copy that fitted value into freshly trained checkpoints. This notebook keeps their own raw probabilities. A workload calibration experiment needs suitable held-out labels and is outside the mandatory 30-minute Pac-Man fine-tuning block. It does not update LoRA/head weights or change the top-ranked action.""")
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

Inspect `initial_config`, `dates_config`, `documents_config`, `skills_config`, `stage_metrics` and `trainable-parameters.json`. Open TensorBoard and compare the four separate runs. Find the fresh pointer head, trainable LoRA parameters and fixed original base matrices. Explain why gradients still travel through the encoder. The four general stages learn typed decisions, dates/evidence, documents, and skills/devtools; none has seen Pac-Man labels.

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
    md("""## Stage 5 / CP3: fine-tune Pac-Man decisions (30–60 minutes)

30-minute block: 5 minutes inspect a labelled request and the loss; up to 20 minutes fine-tune for **two complete epochs**; 5 minutes inspect and save the checkpoint. This is supervised imitation. Kev updates its LoRA adapter and pointer head while the original base matrices remain frozen; the adapter changes the encoder's effective features.

Warm-start from **your Stage 4 skills/devtools checkpoint**, using Kev's documented custom-data settings: learning rate 2e-5, batch 1, accumulation 8, BF16 autocast and gradient checkpointing. With 64 accepted rows, this gives 16 optimizer steps. The Pac-Man adaptation disables none/distractor insertion because the only valid outputs are legal player moves. Option shuffling remains active. This is a documented task-specific departure from the generic initial recipe.

The helper reads architecture from the checkpoint and stops inference to free GPU memory. Use a new checkpoint directory. A timed GPU preflight is still required; successful execution or improved play has not been established by this draft.""")
    code("""print(json.dumps(training[0], indent=2))
CHECKPOINT = CHECKPOINT_ROOT / 'kev-4b-pacman'
RESUME_PACMAN = False
checkpoint = runtime.finetune(training_file, CHECKPOINT, init_from=GENERAL, steps=0, resume=RESUME_PACMAN)
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

Submit the executed notebook, reviewed training JSONL, `comparison.json`, all five small adapter/head checkpoints, all stage configurations/metrics, training logs and TensorBoard events and `runtime-preflight.json`. Record the Colab compute units consumed and elapsed GPU time from your session. Save outputs before the temporary runtime disconnects, then stop the server. On Colab, run the download cell.""")
    code("""runtime.stop()
archive = str(LAB_DIR.parent / 'pacman-lab-submission.zip')
# Export small adapters/heads and results, without foundation weights or packages.
import zipfile
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as out:
    for file in [LAB_DIR/'comparison.json', training_file, LAB_DIR/'runtime-preflight.json', LAB_DIR/'optimized-training-preflight.json', LAB_DIR/'trainable-parameters.json']:
        out.write(file, file.relative_to(LAB_DIR))
    for folder in [INITIAL, DATES, DOCUMENTS, SKILLS, checkpoint]:
        for file in folder.rglob('*'):
            if file.is_file():
                out.write(file, Path('checkpoints') / folder.name / file.relative_to(folder))
    for file in LOG_DIR.rglob('*'):
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
