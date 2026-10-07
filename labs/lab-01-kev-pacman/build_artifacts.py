"""Rebuild the stage-by-stage Kev notebook and browser game."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HELPERS = ["api_client.py", "pacman_lab.py", "cloud_runtime.py", "training_monitor.py",
           "lora_recovery.py", "checkpoint_backup.py", "optimized_training.py", "training-kernels.json", "training_stages.py", "training-stages.json", "games/arcade-engine.js", "games/arcade-worker.js", "games/arcade-browser.js", "games/arcade-shell.html",
           "vendor/arcade-pacman/source.json", "vendor/arcade-pacman/source.zip",
           "planner_data.py", "games/arcade-planner.js", "data/pacman-planner-v1.zip",
           "data/pacman-planner-v1-manifest.json", "data/pacman-planner-v1-quality.json"]


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


def notebook(local=False):
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

**You train Pac-Man. Four ghosts use the classic arcade engine.** The faithful browser recreation supplies the classic maze, original renderer/font/sounds, power pellets, fruit, tunnels, lives and chase/scatter phases. Kev chooses player directions; upstream code controls Blinky, Pinky, Inky and Clyde.

The lab has **one assessed checkpoint: CP1 — fine-tune and evaluate Kev on Pac-Man game states**. Interactive play with the trained adapter is a separate, ungraded activity. The 90-minute route is 0–20 overview and interactive play; 20–30 inspect planning labels; **30–60 mandatory fine-tuning**; 60–80 evaluate; 80–90 explain and submit the checkpoint evidence. Installation, downloads, label generation and the four general-decision training stages are prework.

The target runtime is **Colab with one NVIDIA RTX PRO 6000 Blackwell GPU**. The full Server Edition has 96 GB VRAM. Check the actual allocation in the prework cell. The notebook uses BF16 inference and training autocast, with FP32 stored backbone, adapter and head parameters in the initial recipe. Colab does not guarantee this GPU, including on paid plans. Arrange access before class and record actual runtime cost. All stages still need an instructor GPU preflight.

The notebook separates **Stage 1: initial decision training → Stage 2: dates/missing evidence → Stage 3: documents → Stage 4: skills/devtools → Stage 5: Pac-Man fine-tuning**. Each saves its own checkpoint and training logs. Calibration is a separate probability-fitting procedure and is not applied here; these fresh runs do not inherit the released model's benchmark scores or fitted temperature. [Recipe and compute](https://github.com/jaredpalmer/kev/blob/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4/docs/model-cards/kev-4b.md#training-procedure).""")

    md("""## Game provenance

Game source: [masonicGIT/pacman](https://github.com/masonicGIT/pacman), GPL-3.0, pinned `7407174c1d6a38be8cd230577489e39e0873145b`. Its source/assets are unchanged; the author documents small [accuracy differences](https://github.com/masonicGIT/pacman#accuracy). Browser and CPU evaluation execute this same arcade engine. Model source: [jaredpalmer/kev](https://github.com/jaredpalmer/kev). The model receives structured state, not screenshots. Inference stays inside the GPU runtime. The board identifies the active adapter, stage, steps, rank and checkpoint hashes.""")
    md("""## Prework: prepare the runtime

Save your own copy of this notebook in Colab. Open **Runtime > Change runtime type**, select the RTX PRO 6000 Blackwell option if your account offers it, then connect. Run the prework cells before class. Confirm the GPU name rather than relying on a menu label. A different allocation requires an instructor-approved, timed fallback.

Setup creates a separate Python 3.13 environment using Kev's locked dependencies, including PyTorch 2.8.0 with CUDA 12.8. It checks GPU identity, BF16 support and a CUDA forward/backward pass in that environment, then writes `runtime-preflight.json`. The notebook kernel only runs the teaching helpers and Colab callback. Complete installation, data/model downloads and all four general stages before class.

The target GPU is an assumption for this session, not a free-compute promise. [Colab availability](https://research.google.com/colaboratory/faq.html), [NVIDIA GPU specifications](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/), [PyTorch Blackwell support](https://pytorch.org/blog/pytorch-2-7/).""")
    lock = source_lock()
    code("""from pathlib import Path
from urllib.request import urlopen
import hashlib, json, os, sys
TRAINING_PROFILE = 'memory_safe'  # Later stages; Stage 1 explicitly selects 4 x 2 below
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
for name in ['cloud_runtime', 'training_monitor', 'lora_recovery', 'checkpoint_backup', 'optimized_training', 'training_stages', 'planner_data', 'pacman_lab', 'api_client']:
    sys.modules.pop(name, None)
from cloud_runtime import CloudRuntime
from lora_recovery import latest_snapshot
from training_stages import specifications, inspect_checkpoint, restore_checkpoint, backup_checkpoint
from pacman_lab import *
from planner_data import prepare_dataset, planner_rollout, PREFIX, RECIPE
from api_client import call, distribution
ensure_node(LAB_DIR)
runtime = CloudRuntime(LAB_DIR, training_profile=TRAINING_PROFILE)
CHECKPOINT_ROOT = LAB_DIR / 'checkpoints'
INITIAL = CHECKPOINT_ROOT / 'kev-4b-initial'
DATES = CHECKPOINT_ROOT / 'kev-4b-dates'
DOCUMENTS = CHECKPOINT_ROOT / 'kev-4b-documents'
SKILLS = CHECKPOINT_ROOT / 'kev-4b-skills'
STAGES = specifications()
manifest = prepare_dataset(LAB_DIR / 'data')
GAME = notebook_game()
STAGE_OWNERS = {}
print('Prepared classic arcade engine, four ghosts and episode-disjoint player snapshots.')
""")
    md("""Setup fetches readable helpers, a compressed planning dataset and a 9.7 MB archive of original game source/assets, verifying every SHA-256. It extracts the three declared dataset files after checking their individual hashes. The instructor generates planning labels before class; learners do not spend the training block generating demonstrations. Node.js executes that same engine for Python evaluation; Colab installs a checksum-pinned official Node binary only if needed, separate from Torch. Cached downloads are reused. Network access is needed on first use.

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
    code("runtime.setup()")
    md("""## Required optimized training stack

Setup extends Kev's frozen environment with hash-pinned wheels: **Flash Linear Attention / fla-core 0.5.2**, **causal-conv1d 1.7.0** (Python 3.13, Torch 2.8, CUDA 12, Linux x86_64, CXX11 ABI), einops 0.8.1 and Ninja 1.13.0. Torch 2.8.0 / CUDA 12.8, Triton 3.4.0, Transformers 5.17 and PEFT remain at Kev's locked versions. Dependencies are installed without replacing that stack. SDPA handles full attention; FLA's Triton kernels handle DeltaNet; the CUDA convolution handles its short convolution. Training uses fused AdamW, BF16 autocast, gradient checkpointing and gradient accumulation; the later-stage memory profile also bounds row passes. The LoRA/head recipe and curriculum remain in Kev's trainer.

Preparation audits the actual pinned base/LoRA/head on CPU and verifies the selected optimized function bindings and CUDA/BF16 availability. It does **not** run a separate model forward/backward by default. `optimized-training-preflight.json` records `bindings_verified`, zero optimizer steps and `cuda_loss_backward='not_run'`. Every training subprocess verifies bindings again and refuses reference-kernel fallback. The first real training step exercises loss/backward/fused AdamW and saves a recovery snapshot after successful completion.

Set `RUN_TRAINING_PREFLIGHT=True` below only when you want the additional two-record compatibility check. This loads the actual model on CUDA, checks finite adapter/head gradients and counts FLA, convolution and fused AdamW calls without running a CUDA profiler. It reports encoding, forward, backward, gradient checks and completed optimizer updates separately. Both this optional check and real training print Python stacks after 60 seconds without progress; logs retain the active call. Fifteen-second heartbeats show process liveness. The binding-only attempt has a two-minute timeout; the optional training check has a 20-minute timeout. These are limits, not expected durations. Initial kernel compilation/autotuning can still happen during the first real training step. Full-curriculum CUDA compatibility, VRAM and timing remain unvalidated by the course author.

To retry after a stalled old preflight, interrupt its cell and keep the Colab runtime connected. Copy the updated bootstrap cell into your notebook and run it to refresh the verified helpers; existing downloads, checkpoints and logs stay in place. With the existing environment already installed, run the preparation cell below with the default `False`. Installation and model verification now have separate cells.

Fused optimizer/kernel arithmetic can differ numerically from the reference implementation. The published-reference profile retains upstream training flags; both profiles now require optimized kernels. Optional inference fusion and serving CUDA graphs remain disabled for training because they do not implement this backward path. [FLA release](https://github.com/fla-org/flash-linear-attention/releases/tag/v0.5.2), [causal-conv1d release](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.7.0).""")
    code("""RUN_TRAINING_PREFLIGHT = False  # Optional extra model checks; optimized bindings remain required
audit = runtime.prepare_training(run_training_preflight=RUN_TRAINING_PREFLIGHT)
print('Training suite records:', audit['suite_records'])
print('Trainable parameters:', sum(audit['trainable_parameters'].values()))
print('Optimized training verification:', json.dumps(runtime.training_preflight, indent=2))
""")
    md("""## Memory profile and recovery storage

A learner's original **0.8B** batch-8 run failed at step 2,073/3,144 with 89.90 GiB actively allocated on a 94.97 GiB GPU. Startup warnings confirmed `causal_conv1d` and `flash-linear-attention` were missing, so Qwen used reference PyTorch kernels with substantial training intermediates. The peak reached 91.91 GiB by step 280 and then stayed flat through step 2,070; that cumulative maximum alone cannot diagnose a leak. [Qwen kernel documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5#usage-tips-and-notes). Record counts are not physical batch sizes: records can contain multiple question rows and none-pair siblings.

The **Stage 1 cell explicitly selects batch 4 × accumulation 2 and `row_budget=0`**, matching Kev's published initial execution. This assignment happens in the training cell before the command is printed, so the generic profile cannot silently replace it. The default **`memory_safe`** profile still supplies **batch 1 × accumulation 8**, gradient checkpointing and `row_budget=2048` to later stages. Both executions preserve the data, epochs, optimizer-step count, architecture, learning-rate schedule and augmentation settings. Microbatching, row grouping and dropout execution differ; a continued run that changes these settings is not an exact numerical reproduction. The published later-stage execution flags remain available as `TRAINING_PROFILE='published_reference'`. Full-run CUDA fit and timing remain unvalidated by the course author. A single question longer than a nonzero row budget still runs intact; records are not silently truncated or dropped.

Save at optimizer step 1, every 100 steps or five minutes (checked at optimizer boundaries), and the final step. Keep the latest two complete snapshots. Each contains the LoRA adapter/head, optimizer, scheduler, RNG and progress counters. Kev's native `--resume` supports full-weight runs only; the lab adds a separate LoRA recovery implementation. After a failure, Stage 1 uses its `RESUME_CHECKPOINT` selector; later stages use their `RESUME_*` flags. Keep the same inputs and output path, and rerun only the affected stage. Stage 1 permits its explicit batching change; other training arguments must match. Do not rerun completed earlier stages. A snapshot can resume training or supply an intermediate model; it is not a completed curriculum stage.

**The older notebook saved LoRA weights only at the end. Its failed step-2,073 run has logs/configuration but no automatic learned checkpoint.** New output names below preserve that failed directory. Local `/content` files disappear when the runtime is replaced.

Set **`SAVE_TO_DRIVE=True`** below before training. Colab mounts Drive once. Training and recovery saves stay on the local disk; after each complete recovery save, the notebook copies a verified archive to `MyDrive/QPlusLearning/lab-01-kev-pacman/backups`. It keeps the latest two backups per stage and also archives the completed final checkpoint. Adapter/head weights, optimizer, scheduler, RNG, progress and any custom training JSONL are included; foundation weights and packages are downloaded again by setup. Wait for **`Drive backup complete at optimizer step …`** before deleting a runtime. If backup fails, the local snapshot remains and training stops with the error.

On a new runtime, enable the same flag and run this storage cell. Available Drive backups automatically restore to their original local paths without overwriting existing local checkpoints or changed labels. Stage 1 then detects the restored snapshot and resumes by default. With no checkpoint it starts fresh. You can still select `latest` or a specific snapshot explicitly. Logs and TensorBoard events have their separate export control below.""")
    code("""SAVE_TO_DRIVE = False  # True enables automatic backup and restore; no manual file copying
RESTORE_RECOVERY_ARCHIVE = None  # ZIP from the inspection/export cell; restore into an absent root
CHECKPOINT_ROOT = LAB_DIR / 'checkpoints'
DRIVE_BACKUP_ROOT = None
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_BACKUP_ROOT = Path('/content/drive/MyDrive/QPlusLearning/lab-01-kev-pacman/backups')
if RESTORE_RECOVERY_ARCHIVE is not None:
    restore_checkpoint(RESTORE_RECOVERY_ARCHIVE, CHECKPOINT_ROOT)
else:
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
print('Automatic Drive backup:', json.dumps(runtime.configure_backup(DRIVE_BACKUP_ROOT, CHECKPOINT_ROOT), indent=2))
print('Later-stage training profile:', runtime.training_profile, 'overrides:', runtime.selected_recipe({}))
print('Stage 1 selects batch 4 x accumulation 2 / row_budget 0 in its own training cell.')
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

The initial cell performs this stage only. It explicitly sets `INITIAL_EXECUTION={'batch': 4, 'accum': 2, 'row_budget': 0}` before constructing the command; gradient checkpointing remains enabled. The generic memory profile applies to later stages. **By default it resumes a complete recovery snapshot when one exists locally or was restored from Drive; with no checkpoint it starts fresh.** You can select `latest` or a specific saved snapshot with `RESUME_CHECKPOINT`; an explicitly requested missing checkpoint stops the run. A completed initial checkpoint is reused. It streams progress, saves recovery snapshots and writes TensorBoard curves. The full run is prework with a configurable 180-minute attempt cap, not a promised duration. To restore a completed initial checkpoint manually, upload its ZIP, set `RESTORE_ARCHIVE` and declare learner/instructor ownership.""")
    md("""### Resume initial training

Keep the same `INITIAL` output. The cell below uses one setting: **`RESUME_CHECKPOINT`**. Its default detects whether a recovery snapshot exists. Set it to **`'latest'`** to require the latest checkpoint, or to a full snapshot directory path to choose a specific save. Set it to `None` for a fresh run in a new output. The launcher prints the saved optimizer step and selected path before restoring weights, optimizer, scheduler, RNG and progress. A missing or incomplete explicitly requested checkpoint stops the run; it does not start fresh.

The cell explicitly selects **batch 4 × accumulation 2 and row budget 0**, permitting only the batching changes needed to continue an earlier 1×8 run. Base, data, epochs, learning rates, precision, effective batch and total optimizer steps must match. Grouping and dropout draws can change, and the execution change is recorded in recovery receipts. Work after the chosen snapshot is repeated. For a student's first run with no checkpoint, explicitly set `RESUME_CHECKPOINT=None` to initialize a new output. Existing outputs are protected from accidental replacement.""")
    code("""INITIAL = CHECKPOINT_ROOT / 'kev-4b-initial'
INITIAL_EXECUTION = {'batch': 4, 'accum': 2, 'row_budget': 0}  # Explicit published initial execution
runtime.initial_execution = dict(INITIAL_EXECUTION)
print('Selected initial execution:', runtime.initial_execution, flush=True)
print('Selected command:', runtime.pretraining_command(INITIAL), flush=True)
from lora_recovery import latest_snapshot
RESUME_CHECKPOINT = 'latest' if latest_snapshot(Path(str(INITIAL) + '-recovery')) else None
# To choose a saved step, replace the line above with its full snapshot directory path.
RESTORE_ARCHIVE = None  # Example: '/content/pacman-initial-checkpoint.zip'
RESTORED_CHECKPOINT_OWNER = 'learner'  # 'instructor' for a supplied fallback
if RESTORE_ARCHIVE is None:
    if (INITIAL / 'run-evidence.json').is_file():
        print('Using completed initial checkpoint:', INITIAL)
    else:
        runtime.pretrain(INITIAL, steps=0, resume_from=RESUME_CHECKPOINT,
                         allow_execution_change=RESUME_CHECKPOINT is not None)
    STAGE_OWNERS['initial'] = 'learner'
else:
    restore_checkpoint(RESTORE_ARCHIVE, INITIAL)
    STAGE_OWNERS['initial'] = RESTORED_CHECKPOINT_OWNER
published = json.loads((runtime.repo / 'experiments/q35-4b-s23.json').read_text())[0]
initial_config, initial_metrics = inspect_checkpoint(INITIAL, stage='initial', owner=STAGE_OWNERS['initial'], recipe=runtime.selected_initial_recipe(published))
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

Warm-start from your **completed Stage 1** checkpoint. On a fresh runtime without a restored checkpoint, run Stage 1 to completion first; setup/model downloads do not create this trained checkpoint. Use 1,425 generated records (900 date-policy cases, 255 missing-fact cases and 270 intact controls) plus 2,000 replayed `decision-v7` training records. Published settings: one epoch, learning rate 2e-5, batch 4, accumulation 2, gradient checkpointing, BF16, seed 1 and 25% none minimal pairs. This is a separate run and a separate checkpoint. Its 180-minute attempt cap is a scheduling limit pending GPU measurements.""")
    code("""DATES = CHECKPOINT_ROOT / 'kev-4b-dates'
from lora_recovery import latest_snapshot
STAGES = specifications()
RESUME_DATES = latest_snapshot(Path(str(DATES) + '-recovery')) is not None
RESTORE_DATES_ARCHIVE = None
RESTORED_DATES_OWNER = 'learner'
if RESTORE_DATES_ARCHIVE is None:
    if (DATES / 'run-evidence.json').is_file():
        print('Using completed dates checkpoint:', DATES)
    else:
        print('Dates command:', runtime.intermediate_command('dates', DATES, INITIAL), flush=True)
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
    md('## Stage 3 — Documents (prework)\n\nWarm-start from **Stage 2**. If you restored a completed dates checkpoint, run runtime setup, optimized preparation, storage and intermediate-data preparation, then start here. You can skip the Stage 1/2 training cells; recovery files are not needed to initialize documents. Set `DATES` to the restored folder below. Earlier stage archives that are unavailable are reported as missing in the comparison/export.\n\nUse 5,219 consumer-finance complaint records plus 2,000 replayed decision-v7 training records. Published settings: one epoch, learning rate 2e-5, batch 2 × accumulation 4, BF16, FP32 stored weights, state budget 7,552, gradient checkpointing, seed 2 and 25% none minimal pairs. Expect 903 optimizer steps. The memory profile uses batch 1 × accumulation 8 with bounded row passes. Save this checkpoint before continuing; documents and skills are separate in Kev-4B.')
    code("""DOCUMENTS = CHECKPOINT_ROOT / 'kev-4b-documents'
from lora_recovery import latest_snapshot
STAGES = specifications()
DATES = CHECKPOINT_ROOT / 'kev-4b-dates'  # Set this to your completed dates folder
STAGE_OWNERS.setdefault('dates', 'learner')  # 'instructor' for a supplied checkpoint
RESUME_DOCUMENTS = latest_snapshot(Path(str(DOCUMENTS) + '-recovery')) is not None
RESTORE_DOCUMENTS_ARCHIVE = None
RESTORED_DOCUMENTS_OWNER = 'learner'
if RESTORE_DOCUMENTS_ARCHIVE is None:
    if (DOCUMENTS / 'run-evidence.json').is_file():
        print('Using completed documents checkpoint:', DOCUMENTS)
    else:
        print('Documents command:', runtime.intermediate_command('documents', DOCUMENTS, DATES), flush=True)
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
from lora_recovery import latest_snapshot
STAGES = specifications()
RESUME_SKILLS = latest_snapshot(Path(str(SKILLS) + '-recovery')) is not None
RESTORE_SKILLS_ARCHIVE = None
RESTORED_SKILLS_OWNER = 'learner'
if RESTORE_SKILLS_ARCHIVE is None:
    if (SKILLS / 'run-evidence.json').is_file():
        print('Using completed skills checkpoint:', SKILLS)
    else:
        print('Skills and developer tools command:', runtime.intermediate_command('skills', SKILLS, DOCUMENTS), flush=True)
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
""")
    md("""## Load completed Skills checkpoint — start the lab here

On a new runtime or after refreshing the bootstrap, run `runtime.setup()`, optimized preparation with `RUN_TRAINING_PREFLIGHT=False`, and the storage cell first. Bootstrap creates a new runtime helper; these cells establish its hardware, training and backup settings while reusing cached files. If using the automatic Drive backup, keep the complete `kev-4b-skills` backup folder (including `latest.json`, ZIPs and their `.zip.json` receipts) at `MyDrive/QPlusLearning/lab-01-kev-pacman/backups/kev-4b-skills`. With `SAVE_TO_DRIVE=True`, storage restores the completed native checkpoint to `/content/pacman-kev-lab/checkpoints/kev-4b-skills`.

**When Skills is already complete, skip the Stage 1–4 training and backup/export cells. Run the cell below, then the single CP1 exercise.** The interactive play cell is available before and after fine-tuning. This cell loads the completed Skills model, reads available saved stage metrics and starts inference. It does not resume Skills training or require the Documents checkpoint. Earlier stage weights and training curves are available only if you also restored them; their absent archives are reported in the submission.

Set Skills ownership to `instructor` below if the instructor supplied this checkpoint. Interactive play initially uses this general model; CP1 fine-tunes it on your reviewed Pac-Man labels.""")
    code("""# Load completed Skills checkpoint for the lab
from pathlib import Path
import json
CHECKPOINT_ROOT = LAB_DIR / 'checkpoints'
INITIAL = CHECKPOINT_ROOT / 'kev-4b-initial'
DATES = CHECKPOINT_ROOT / 'kev-4b-dates'
DOCUMENTS = CHECKPOINT_ROOT / 'kev-4b-documents'
SKILLS = CHECKPOINT_ROOT / 'kev-4b-skills'
GENERAL = SKILLS
required = ['head.pt', 'adapter_config.json', 'training_config.json', 'training_metrics.json', 'run-evidence.json']
missing = [name for name in required if not (GENERAL / name).is_file()]
if missing or not any(file.is_file() for file in GENERAL.glob('adapter_model.*')):
    raise RuntimeError(f'Restore the completed Skills checkpoint to {GENERAL}; missing files: {missing}, or adapter weights.')
STAGE_OWNERS = globals().get('STAGE_OWNERS', {})
STAGE_OWNERS.setdefault('skills', 'learner')  # 'instructor' for a supplied checkpoint
stage_checkpoints = {'initial': INITIAL, 'dates': DATES, 'documents': DOCUMENTS, 'skills': SKILLS}
stage_metrics, stage_configs = {}, {}
for stage, folder in stage_checkpoints.items():
    config_path, metrics_path = folder / 'training_config.json', folder / 'training_metrics.json'
    stage_configs[stage] = json.loads(config_path.read_text()) if config_path.is_file() else None
    stage_metrics[stage] = json.loads(metrics_path.read_text()) if metrics_path.is_file() else None
initial_config, dates_config, documents_config, skills_config = [stage_configs[name] for name in stage_checkpoints]
initial_metrics, dates_metrics, documents_metrics, skills_metrics = [stage_metrics[name] for name in stage_checkpoints]
skills_evidence = json.loads((GENERAL / 'run-evidence.json').read_text())
if (skills_evidence['stage'] != 'skills' or skills_metrics['optimizer_steps'] != 1915
        or skills_metrics['requested_records'] != 15320 or skills_config['args'].get('max_steps', 0)
        or skills_metrics.get('truncated_records', 0) or skills_metrics.get('rejected_records', 0)):
    raise ValueError('Complete the full Skills stage before starting the lab.')
missing_stage_archives = [stage for stage, folder in stage_checkpoints.items()
                         if not (folder / 'head.pt').is_file() or not list(folder.glob('adapter_model.*'))]
print('Earlier stage archives unavailable:', missing_stage_archives)
models = runtime.start(GENERAL)
print(json.dumps(models, indent=2))
""")
    md("""## Calibration is separate from training

Kev's release fitted one probability temperature after its four training stages. We do not copy that fitted value into freshly trained checkpoints. This notebook keeps their own raw probabilities. A workload calibration experiment needs suitable held-out labels and is outside the mandatory 30-minute Pac-Man fine-tuning block. It does not update LoRA/head weights or change the top-ranked action.""")
    md("""## Interactive play — trained Kev at the controls

Try **Human** mode with arrows/WASD; click the board for keyboard focus. Restart, select **Kev**, and watch its choices. Human mode runs at 60 simulation frames per second. Kev pauses the simulation while choosing a direction at each tile center, then player and all four ghosts advance using upstream speeds/timers. Wall-clock survival is not a fair skill metric. Pause before running training cells.

This activity is available before and after the checkpoint. The **Active LoRA + pointer head** badge initially shows `kev-4b-skills`: general Skills training, no Pac-Man fine-tuning. CP1's evaluation loads `kev-4b-pacman-planner-v1`; rerun this same play cell afterward to play with that adapter. Expand the details for paths and SHA-256 fingerprints. The badge and traces verify the serving model card; `kev-latest` alone is an API alias.

Colab supplies the notebook callback below. On Kaggle, use the Python evaluation and rollouts instead. API errors pause visibly; the game does not replace failed player decisions with a hidden rules controller. Pause play before training, and finish interactive play before exporting results.""")
    code("""from IPython.display import display, HTML, JSON
bridge = NotebookBridge(LAB_DIR / 'results/player-trace.jsonl', runtime.active_model_info)
try:
    from google.colab import output
except ImportError:
    print('Kaggle/local: continue with the decision and rollout cells below.')
else:
    output.register_callback('pacman.decide', lambda state: JSON(bridge.decide(state)))
    output.register_callback('pacman.model', lambda: JSON(bridge.model()))
    display(HTML(GAME))
""")
    md("""## CP1 — Fine-tune and evaluate Kev on Pac-Man game states

Use the committed planning-labelled game states to adapt your completed Skills LoRA/head, then compare the baseline and task adapter on the same held-out boards. This is the lab's only assessed checkpoint.

Complete the data review, full task training and paired evaluation below. Submit the trained adapter/head, reviewed training labels, curves and `comparison.json`, with a short explanation of one changed move or remaining mistake. Explain how the native ghost rules affect a planning label and why the search is approximate. Completion requires the full 762-update recipe, no dropped/truncated records, and recorded before/after measurements; improvement is an experimental result to measure.

### Inspect the planning data (20–30 minutes)

Data has **4,096 training, 256 development and 256 evaluation** snapshots from valid native-engine trajectories, starting at levels 1, 2, 3 and 5. At every labelled board, the teacher simulates up to **20 future player moves** for every legal first action, retaining **eight paths per first action** under **two independent frightened-mode randomness scenarios**. Chase/scatter movement is predicted from each ghost's deterministic native targeting rule, rather than a random walk. Native code handles speeds, timers, release counters, collisions, power pellets, fruit and score.

The objective ranks worst-scenario survival first, then mean `score gained + 5 × pellets collected − 4 × repeat visits − 6 × distance to the next pellet + 2 × min(dangerous-ghost distance, 8)`. The stored candidates show the best sequence found for each direction. **Beam pruning and the finite horizon mean this is an approximate planning teacher, not a globally optimal full-game solver.** The teacher uses current native timers/pixel offsets; the learner sees board, modes, history and elapsed frames, so this is privileged-state imitation. The teacher resamples future RNG rather than looking ahead at the actual episode RNG. Search sequences, labels, replay seeds and scores stay in `_meta`, outside model inputs.

Collection uses 70% planning, 20% the earlier heuristic and 10% random legal moves to include recovery states. Snapshots are spaced four moves apart. Whole episodes and exact observable snapshots are disjoint across splits; all use the classic maze. Inspect the manifest's episode counts, action balance, four-ghost, frightened, fruit and junction coverage, and the CPU teacher-comparison report.

Inspect three training boards and enter your own legal move labels in `EDITS` before checking the teacher. Keep all edits in the training partition. Fix the training file and one candidate before opening evaluation data. The versioned planner files and checkpoint preserve earlier heuristic experiments. Warm-start this candidate from Skills.""")
    code("""training_file = LAB_DIR / 'data/pacman-planner-v1-train-reviewed.jsonl'
label_source = training_file if training_file.is_file() else LAB_DIR / 'data/pacman-planner-v1-train.jsonl'
training = [json.loads(line) for line in label_source.read_text().splitlines()]
print('Training labels:', label_source)
for row in training[:3]:
    print(row['_meta']['id'], json.dumps(row['state'], indent=2))
    print('Legal:', list(row['questions']['move']['criteria']))
EDITS = {}  # Example after inspecting a board: {'train-board-0000': 'left'}
assert set(EDITS).issubset({r['_meta']['id'] for r in training}), 'Unknown training board ID'
for row in training:
    if row['_meta']['id'] in EDITS:
        label = EDITS[row['_meta']['id']]
        assert label in row['questions']['move']['criteria'], 'Choose a legal move'
        row['questions']['move'].update(label=label, src='learner_annotation')
        row['_meta']['label_source'] = 'learner annotation'
if EDITS or not training_file.is_file():
    training_file.write_text(''.join(json.dumps(row) + '\\n' for row in training))
print('Teacher labels for inspected boards:', [(r['_meta']['id'], r['questions']['move']['label']) for r in training[:3]])
print('Search alternatives:', training[0]['_meta'].get('teacher', {}).get('candidates', []))
print('Split sizes:', manifest['counts'])
print('Coverage:', manifest['coverage'])
print('CPU teacher comparison:', json.loads((LAB_DIR / 'data/pacman-planner-v1-quality.json').read_text())['summary'])
before_dev = evaluate(LAB_DIR / 'data/pacman-planner-v1-development.jsonl')
print('Development accuracy:', before_dev['accuracy'])
""")
    md("""### Stage 5 — Fine-tune Pac-Man decisions (30–60 minutes)

30-minute block: 5 minutes inspect a labelled request and the loss; target up to 20 minutes training; 5 minutes inspect and save the checkpoint. This is supervised imitation. Kev updates the same all-module rank-16 LoRA adapters and 256-dimensional pointer head as the preceding stages; original base matrices stay frozen. Planning happens only when producing labels, not inside the training loss or the model controller.

Warm-start from **your Stage 4 Skills checkpoint**. Following the intermediate-stage pattern, train **one complete epoch**, learning rate **2e-5**, with **2,000 decision-v7 training examples mixed as replay**. There are **6,096 requests / 762 optimizer updates**, versus the old 64-board exercise's 16 updates. Dates used 3,425 requests / 429 updates; Documents 7,219 / 903; Skills 15,320 / 1,915. More labels provide a substantive adaptation experiment; improved play must still be measured.

For the target 96 GB GPU this stage explicitly uses **batch 4 × accumulation 2, row budget 0**, BF16 autocast, FP32 stored weights, optimized FLA/conv kernels, fused AdamW and gradient checkpointing. The effective batch stays eight. The printed command is authoritative. None/distractor and none-pair augmentation are disabled for this mixed task stage, because the Pac-Man output must remain a legal move. Generic replay retains its recorded labels; option shuffling stays active. These task/execution choices depart from the generic augmentation settings.

The helper reads architecture from the checkpoint and stops inference to free GPU memory. The state budget is 4,096 tokens; truncation/rejected records fail the audit. Existing recovery and Drive backup apply. The full stage has no short-step cap and may exceed the classroom block: at 1.5 seconds/update, compute alone is 19 minutes; at 3 seconds/update, 38 minutes. Measure steady step time before class and complete a slower full run as prework. CPU checks do not establish GPU timing, memory fit or trained-model improvement.""")
    code("""print(json.dumps(training[0], indent=2))
CHECKPOINT = CHECKPOINT_ROOT / 'kev-4b-pacman-planner-v1'
from lora_recovery import latest_snapshot
RESUME_PACMAN = latest_snapshot(Path(str(CHECKPOINT) + '-recovery')) is not None
print('Pac-Man recipe:', RECIPE)
print('Selected Pac-Man command:', runtime.finetuning_command(training_file, CHECKPOINT, GENERAL))
if (CHECKPOINT / 'run-evidence.json').is_file():
    evidence = json.loads((CHECKPOINT / 'run-evidence.json').read_text())
    assert evidence['training_data_sha256'] == hashlib.sha256(training_file.read_bytes()).hexdigest(), 'Saved checkpoint used different labels; select a new checkpoint output to retrain.'
    checkpoint = CHECKPOINT
    print('Using completed Pac-Man checkpoint:', checkpoint)
else:
    checkpoint = runtime.finetune(training_file, CHECKPOINT, init_from=GENERAL, steps=0, resume=RESUME_PACMAN)
metrics = json.loads((checkpoint / 'training_metrics.json').read_text())
print(metrics)
assert len(training) == manifest['counts']['train'], 'Preserve the complete training partition'
saved_recipe = json.loads((checkpoint / 'training_config.json').read_text())['args']
for key, value in RECIPE.items():
    assert saved_recipe[key] == (float(value) if key == 'lr' else value), f'Checkpoint used another recipe: {key}'
assert metrics['optimizer_steps'] == manifest['expected_optimizer_steps'], 'Complete the full planner/replay stage'
assert not metrics.get('truncated_records', 0) and not metrics.get('rejected_records', 0), 'No state truncation or dropped records'
assert metrics['records_seen'] == metrics['requested_records'] == manifest['expected_training_requests'], 'Complete one epoch including replay'
""")
    md("""### Evaluate the baseline and task adapter (60–80 minutes)

The candidate is now fixed. Score both models on the same **256 evaluation snapshots** and save predictions by ID. Report strict and tie-aware teacher agreement, lower search-survival choices, search-value regret within the same survival class, and immediate captures. These are comparisons with an approximate teacher, not full-game win rates. Fine-tuning may leave answers unchanged or make them worse.""")
    code("""runtime.start(GENERAL)
before = evaluate(LAB_DIR / 'data/pacman-planner-v1-evaluation.jsonl')
before_identity = runtime.active_model_info()
before_run = rollout(max_turns=128)
runtime.start(checkpoint)
after = evaluate(LAB_DIR / 'data/pacman-planner-v1-evaluation.jsonl')
after_identity = runtime.active_model_info()
after_run = rollout(max_turns=128)
comparison = {'base_model': audit['base'], 'base_revision': audit['base_revision'], 'baseline_checkpoint': str(GENERAL), 'checkpoint_owners': STAGE_OWNERS, 'fine_tuned_checkpoint': str(checkpoint), 'active_adapters': {'before': before_identity, 'after': after_identity}, 'before': before, 'after': after, 'rollouts': {'general': before_run, 'fine_tuned': after_run}, 'general_training_stages': stage_metrics, 'missing_general_stage_archives': missing_stage_archives, 'pacman_training': metrics, 'runtime': runtime.gpu}
comparison['pacman_dataset'] = manifest
(LAB_DIR / 'comparison.json').write_text(json.dumps(comparison, indent=2))
for name, result in [('general', before), ('fine_tuned', after)]:
    print(name, {key:result[key] for key in ['accuracy', 'tie_aware_teacher_accuracy', 'lower_search_survival_choices', 'mean_same_survival_search_regret', 'caught_next_turn']})
for name, episode in [('general', before_run), ('fine_tuned', after_run)]:
    print(name, {key:episode[key] for key in ['turns', 'dots_collected', 'score', 'repeated_tiles', 'outcome']})
""")
    md("""The evaluation leaves your Pac-Man task adapter serving. You can now rerun **Interactive play** above, choose **Kev**, and start a new game. Verify the badge says `kev-4b-pacman-planner-v1`. This remains an ungraded activity. Both Python rollouts use the same starting board, seed, native engine and 128-decision cap, stopping at the first lost life or completed level; the browser retains three lives and level progression. The trajectories illustrate behavior rather than a win-rate estimate. The archived five-seed CPU report evaluates the planning teacher itself, which can also fail.

### Explain and export the checkpoint evidence (80–90 minutes)

Explain one changed move or remaining mistake. Relate a planning label to the board and ghost personalities. Identify the LoRA/head parameters that trained and the limitations of finite-horizon beam search.

Submit the executed notebook, reviewed training JSONL, `comparison.json`, all available small adapter/head checkpoints and stage configurations/metrics, training logs and TensorBoard events and `runtime-preflight.json`. A full fresh run produces five checkpoints. If you continued from a completed intermediate checkpoint without older archives, record those missing stages; the imported checkpoint retains its recorded parent provenance, but absent parent weights cannot be rechecked or exported. Record the Colab compute units consumed and elapsed GPU time from your session. Save outputs before the temporary runtime disconnects, then stop the server. On Colab, run the download cell.""")
    code("""runtime.stop()
archive = str(LAB_DIR.parent / 'pacman-lab-submission.zip')
print('Unavailable earlier stage archives:', missing_stage_archives)
# Export small adapters/heads and results, without foundation weights or packages.
import zipfile
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as out:
    for file in [LAB_DIR/'comparison.json', training_file, LAB_DIR/'data/pacman-planner-v1-manifest.json', LAB_DIR/'data/pacman-planner-v1-quality.json', LAB_DIR/'runtime-preflight.json', LAB_DIR/'optimized-training-preflight.json', LAB_DIR/'trainable-parameters.json']:
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
    if local:
        result = local_notebook(result)
    filename = 'pacman_kev_lab_local.ipynb' if local else 'pacman_kev_lab.ipynb'
    (ROOT / 'notebooks' / filename).write_text(json.dumps(result, indent=2) + '\n')


def local_notebook(result):
    """Keep the published Colab notebook intact and generate a lab Jupyter route."""
    cells = {cell['id']: cell for cell in result['cells']}

    def replace(cell_id, old, new):
        text = ''.join(cells[cell_id]['source'])
        if text.count(old) != 1:
            raise ValueError(f'Local notebook needs one {old!r} in {cell_id}')
        cells[cell_id]['source'] = text.replace(old, new, 1).splitlines(keepends=True)

    def markdown(cell_id, text):
        cells[cell_id]['source'] = text.strip().splitlines(keepends=True)

    runtime_sha = hashlib.sha256((ROOT / 'lab_runtime.py').read_bytes()).hexdigest()
    memory_sha = hashlib.sha256((ROOT / 'lab_memory_check.py').read_bytes()).hexdigest()
    native_sha = hashlib.sha256((ROOT / 'lab_native_kernels.py').read_bytes()).hexdigest()
    replace('cell-03', 'import hashlib, json, os, sys\n', '''import hashlib, json, os, sys
GPU = os.environ.get('QPLUS_GPU', '1')
if not GPU.isdecimal():
    raise ValueError('QPLUS_GPU must select one physical GPU index, such as 1')
os.environ['CUDA_VISIBLE_DEVICES'] = GPU
LAB_SOURCE = Path(os.environ.get('QPLUS_LAB_SOURCE', '/chronos_data/huixu/QPlusLearning/labs/lab-01-kev-pacman')).expanduser().resolve()
''')
    replace('cell-03', "LAB_DIR = Path.cwd() / 'pacman-kev-lab'", "LAB_DIR = Path(os.environ.get('QPLUS_LAB_DIR', '/chronos_data/huixu/qpluslearning-runtime-a6000')).expanduser().resolve()")
    replace('cell-03', 'LAB_DIR.mkdir(exist_ok=True)', '''if LAB_DIR.is_relative_to(LAB_SOURCE) or LAB_SOURCE.is_relative_to(LAB_DIR):
    raise ValueError('Use a persistent runtime directory outside the lab source directory')
LAB_DIR.mkdir(parents=True, exist_ok=True)
for name, folder in {'HF_HOME': 'huggingface', 'UV_CACHE_DIR': 'uv', 'UV_PYTHON_INSTALL_DIR': 'python',
                     'PIP_CACHE_DIR': 'pip', 'TORCH_HOME': 'torch', 'TRITON_CACHE_DIR': 'triton',
                     'CUDA_CACHE_PATH': 'cuda'}.items():
    os.environ.setdefault(name, str(LAB_DIR / 'cache' / folder))''')
    replace('cell-03', 'for name, expected_sha256 in FILES.items():',
            f"LOCAL_RUNTIME_SHA256 = {runtime_sha!r}\nLOCAL_MEMORY_CHECK_SHA256 = {memory_sha!r}\nLOCAL_NATIVE_KERNELS_SHA256 = {native_sha!r}\nLOCAL_FILES = {{**FILES, 'lab_runtime.py': LOCAL_RUNTIME_SHA256, 'lab_memory_check.py': LOCAL_MEMORY_CHECK_SHA256, 'lab_native_kernels.py': LOCAL_NATIVE_KERNELS_SHA256}}\nfor name, expected_sha256 in LOCAL_FILES.items():")
    replace('cell-03', '''    url = f'https://raw.githubusercontent.com/yxc20089/QPlusLearning/{COURSE_REVISION}/labs/lab-01-kev-pacman/{name}'
    with urlopen(url, timeout=60) as response:
        content = response.read()''', '''    source = LAB_SOURCE / name
    if not source.is_file():
        raise FileNotFoundError(f'Missing local lab helper: {source}; apply the lab changes to the checkout first')
    content = source.read_bytes()''')
    replace('cell-03', "print('Downloaded verified helper:', name, flush=True)",
            "print('Copied verified local helper:', name, flush=True)")
    replace('cell-03', "for name in ['cloud_runtime',", "for name in ['lab_memory_check', 'lab_native_kernels', 'lab_runtime', 'cloud_runtime',")
    replace('cell-03', 'runtime = CloudRuntime(LAB_DIR, training_profile=TRAINING_PROFILE)', '''from lab_runtime import LabRuntime
runtime = LabRuntime(LAB_DIR, training_profile=TRAINING_PROFILE)
print('Jupyter Python:', sys.executable)
print('Training Python:', runtime.python)
print('Physical GPU:', GPU)''')
    replace('cell-03', '# Later stages; Stage 1 explicitly selects 4 x 2 below',
            '# Lab execution: batch 1 x accumulation 8, row_budget 2048')
    replace('cell-05', 'SHOW_TENSORBOARD = True',
            'SHOW_TENSORBOARD = False  # Run TensorBoard in a lab terminal and forward port 6006')
    replace('cell-08', 'RUN_TRAINING_PREFLIGHT = False', 'RUN_TRAINING_PREFLIGHT = True')
    replace('cell-08', '# Optional extra model checks; optimized bindings remain required',
            '# Required kernel smoke check; each stage also tests long samples before full training.')
    replace('cell-10', 'DRIVE_BACKUP_ROOT = None',
            "DRIVE_BACKUP_ROOT = Path(os.environ.get('QPLUS_BACKUP_DIR', str(LAB_DIR / 'backups')))")
    replace('cell-10', "print('Automatic Drive backup:'", "print('Automatic lab archive backup:'")
    replace('cell-10', 'Stage 1 selects batch 4 x accumulation 2 / row_budget 0',
            'Stage 1 selects batch 1 x accumulation 8 / row_budget 2048')
    replace('cell-15', "INITIAL_EXECUTION = {'batch': 4, 'accum': 2, 'row_budget': 0}  # Explicit published initial execution",
            "INITIAL_EXECUTION = {'batch': 1, 'accum': 8, 'row_budget': 2048}  # A6000 execution; effective batch remains eight")
    replace('cell-36', "print('Pac-Man recipe:', RECIPE)",
            "PACMAN_RECIPE = runtime.selected_pacman_recipe(RECIPE)\nprint('Pac-Man recipe and lab execution:', PACMAN_RECIPE)")
    replace('cell-36', 'for key, value in RECIPE.items():', 'for key, value in PACMAN_RECIPE.items():')
    replace('cell-38', "comparison['pacman_dataset'] = manifest", '''comparison['pacman_dataset'] = manifest
comparison['lab_execution'] = {'initial': runtime.initial_execution, 'pacman': PACMAN_RECIPE,
                               'physical_gpu': GPU, 'training_python': str(runtime.python)}''')
    replace('cell-38', "comparison['lab_execution'] =", "comparison['lab_memory_preflights'] = runtime.memory_preflights\ncomparison['lab_execution'] =")
    replace('cell-38', "comparison['lab_execution'] =", "comparison['lab_native_kernels'] = runtime.native_training_kernels\ncomparison['lab_execution'] =")
    replace('cell-40', "LAB_DIR/'runtime-preflight.json',", "LAB_DIR/'runtime-preflight.json', LAB_DIR/'lab-native-kernels.json',")

    markdown('cell-02', '''## 实验室普通 Jupyter：先准备运行环境

从你的本地电脑 SSH 登录实验室，按照 [LAB_SETUP.md](../LAB_SETUP.md) 创建 Python 3.13 Conda 环境 `/chronos_data/conda_envs/qplus-jupyter-py313`，选择 **QPlusLearning (A6000, Python 3.13)** kernel。

本 notebook 默认使用物理 GPU 1、源码 `/chronos_data/huixu/QPlusLearning/labs/lab-01-kev-pacman` 和持久目录 `/chronos_data/huixu/qpluslearning-runtime-a6000`。运行前可设置 `QPLUS_GPU`、`QPLUS_LAB_SOURCE`、`QPLUS_LAB_DIR`。GPU 必须在第一次 CUDA 初始化之前选择；换卡时重启 kernel。

Jupyter 与训练器使用独立环境。`runtime.setup()` 会在 `/chronos_data/conda_envs/qplus-kev-training-py313` 建立 uv 环境（可用 `QPLUS_TRAINING_ENV` 指定），按 Kev 上游 lockfile 安装 Python 3.13 / Torch 2.8.0 / CUDA 12.8，再校验 GPU/BF16、CUDA backward 和优化内核。不要把训练环境指向 Jupyter Conda 环境，也不需要在 kernel 中手动安装 Torch。

本地编译优先复用上面已有的 Jupyter Conda 环境里的 CUDA 12.8 `nvcc` 和开发头文件；不新建第三个 CUDA toolkit 环境。`QPLUS_CUDA_HOME` 可指定已有的其他 CUDA 12.8 toolkit。`nvidia-smi` 显示 CUDA 12.8 不能证明 `nvcc` 已安装；开发工具只在原 wheel 与系统 GLIBC 不兼容、必须本地编译时使用。

按你提供的当前运行约 20 GB 显存占用，单张约 48 GB 的 A6000 足够，本 notebook 默认仅使用 GPU 1。这个占用由你提供，并非本次修改的 GPU 实测结果；不同阶段的占用以各自显存报告为准。所有阶段使用 batch 1 × accumulation 8、row_budget 2048，保持有效 batch 八，每阶段开始前保留长样本显存检查。历史 reference 路径的约 90 GiB OOM 记录不作为当前运行的显存需求；本版本要求 FLA/causal-conv1d 优化内核。''')
    markdown('cell-04', '''## 训练曲线

启动单元格从本地 checkout 复制并校验 21 个课程 helper/数据/游戏文件及三个本地运行 helper（`lab_runtime.py`、`lab_memory_check.py`、`lab_native_kernels.py`），保留已有数据、checkpoint 和日志。原课程 commit、模型 revision、数据和内核 SHA-256 保持固定。训练安装仍需要联网下载 Kev、锁定依赖、模型权重和训练数据。

在实验室终端运行 TensorBoard，并从本地 SSH 转发 6006：

```bash
conda activate /chronos_data/conda_envs/qplus-jupyter-py313
tensorboard --logdir /chronos_data/huixu/qpluslearning-runtime-a6000/logs --host 127.0.0.1 --port 6006
```

`logs/<stage>/<attempt>` 保存每步 CE、学习率、梯度、耗时、显存、TensorBoard events，以及原始日志和失败批次形状。''')
    markdown('cell-07', '''## 校验锁定的优化训练环境

训练使用 Python 3.13、Torch 2.8.0/CUDA 12.8、Triton 3.4.0、Transformers 5.17.0 和锁定 PEFT。FLA/fla-core 0.5.2、causal-conv1d 1.7.0、einops 0.8.1、Ninja 1.13.0 的 wheel 均校验 SHA-256，并用 `--no-deps` 安装。CUDA convolution wheel 要求 Linux x86_64、Python 3.13 和 CXX11 ABI。

若原 causal-conv1d wheel 导入失败，且诊断明确是系统缺少它要求的 GLIBC 版本，`lab_native_kernels.py` 才用 SHA-256 固定的官方 1.7.0 源码和已有 CUDA 12.8 `nvcc` 本地编译，默认两条编译任务。它在独立训练环境中安装并重新验证 convolution/FLA bindings；保留 Torch 锁定版本、系统 GLIBC、原 wheel checksum 和源码 license。其他导入错误会保留原诊断并停止，不会触发这种重编译。`lab-native-kernels.json` 记录使用原 wheel 还是已验证的本地构建；本地 wheel 按工具链指纹和 checksum 缓存复用。

下方必须保持 `RUN_TRAINING_PREFLIGHT=True`：下载并审计实际 4B 基座后，在 A6000 上用两条真实记录检查 loss、backward、有限 LoRA/head 梯度和 fused AdamW/FLA/convolution 调用。20 分钟是 timeout，不是预计耗时。

开始各阶段完整训练前，另用该阶段实际参数和训练数据扫描 tokenized forward-pass 形状，选择观测到最昂贵的样本，并考虑符合条件的 none-pair siblings。临时模型连续累积八个 microbatch 后更新，重复两次，记录 allocated/reserved 峰值和实际 pass 形状；这不是正式 checkpoint。`logs/<stage>/memory-check-*/memory-preflight.json` 保存检查结果。检查 timeout 为一小时，不是预计耗时。失败不开始完整阶段；通过也不能保证整个阶段所有形状和后续共享 GPU 占用都能适配。''')
    markdown('cell-09', '''## 持久存储与恢复

实验室 execution 使用 batch 1 × accumulation 8、gradient checkpointing 和 row_budget 2048；保留原模型、数据、epochs、有效 batch 八和 optimizer-step 数。分组、dropout 与优化内核会改变数值轨迹。长于 row budget 的单个问题仍完整运行，不会静默截断。

`SAVE_TO_DRIVE=False`；默认把验证过的恢复 ZIP 保存到 `LAB_DIR/backups`（可用 `QPLUS_BACKUP_DIR` 指定）。这不依赖 Google Drive，同盘副本不能代替异地备份。保留 step 1、每 100 步/五分钟、最终步的完整 LoRA/head、optimizer、scheduler、RNG 和输入数据。checkpoint 与 recovery 路径保持固定；已有文件不会作为新训练被删除。完整输出直接复用，部分输出从恢复快照继续。

Stage 1 支持显式改变 batch/accum/row_budget 的恢复；其他阶段恢复要求原参数一致。每次训练仍有 180 分钟 attempt cap，超时后保留恢复结果；这不是整个阶段的预计耗时。''')
    markdown('cell-14', '''## Stage 1：初始决策训练

使用固定 Qwen3.5-4B-Base 和完整 decision-v7 train，训练两个 epochs。实验室单元格明确选择 `INITIAL_EXECUTION={'batch':1,'accum':8,'row_budget':2048}`。默认有完整快照时恢复，没有快照时新建；完整 checkpoint 复用。`RESUME_CHECKPOINT='latest'` 或完整快照路径可以明确要求恢复，缺失时停止，避免无意从头训练。''')
    markdown('cell-28', '''## 加载已完成的 Skills checkpoint

在新 kernel 中先运行启动、`runtime.setup()`、优化环境准备和存储单元格。如果没有完整 Skills checkpoint，按顺序完成 Stage 1–4；下载基座不会产生已训练的 Skills adapter。

如果已有完整 Skills checkpoint，将 native 文件放在 `LAB_DIR/checkpoints/kev-4b-skills`，或恢复完整自动备份。下方单元格检查 stage、文件和完整步数，加载训练过的 LoRA/head，并明确报告缺失的早期阶段 archive。保留 learner/instructor ownership。''')
    markdown('cell-31', '''## 可选的交互游戏

原 Kev 浏览器控制使用 Colab kernel callback；普通 Jupyter 中下方单元格会提示跳过，不影响后续真实模型的结构化评估和 rollout。该单元格不会用规则控制器伪装成模型推理。''')
    markdown('cell-35', '''## CP1：训练 Pac-Man adapter

完整训练 4,096 planning-labelled boards 加 2,000 decision-v7 replay，one epoch、lr 2e-5、rank-16 LoRA、head-256、max_state 4096，共 **762 optimizer updates**。

实验室 execution 明确使用 batch 1 × accumulation 8、row_budget 2048。打印的命令、保存的 training_config、下面的 recipe 一致性检查和最终 evidence 使用同一选择。数据 manifest 保留课程发布 recipe，实际 lab execution 另外记录，不改源数据或删除断言。开始训练前释放推理模型显存；已有 recovery 必须匹配本次参数。''')
    markdown('cell-39', '''## 保存实验室运行结果

导出包含实际评估、已审阅标签、可用的 adapter/head、配置、运行 receipts、训练日志和 TensorBoard events。基础模型和包不进入 submission ZIP。缺少的早期阶段 archive 会明确列出。导出单元格打印本地文件路径；从你本地电脑通过 scp 下载，保留真实 GPU 型号、显存和耗时记录。''')
    replace('cell-00', 'The target runtime is **Colab with one NVIDIA RTX PRO 6000 Blackwell GPU**. The full Server Edition has 96 GB VRAM. Check the actual allocation in the prework cell. The notebook uses BF16 inference and training autocast, with FP32 stored backbone, adapter and head parameters in the initial recipe. Colab does not guarantee this GPU, including on paid plans. Arrange access before class and record actual runtime cost. All stages still need an instructor GPU preflight.',
            'This notebook runs in **ordinary Jupyter on one laboratory RTX A6000**. The user reports about 20 GB of VRAM use for the current run, which fits one 48 GB card. It uses a separate locked Python 3.13 training environment. BF16 and optimized CUDA kernels remain required. Stage memory checks record the actual usage; this change has been checked on CPU and has not been run on the lab GPU by the author.')
    result['metadata'].pop('colab', None)
    result['metadata']['kernelspec'] = {'display_name': 'QPlusLearning (A6000, Python 3.13)',
                                      'language': 'python', 'name': 'qplus-a6000-py313'}
    return result



def game():
    from pacman_lab import notebook_game
    html = notebook_game()
    (ROOT / 'games/pacman.html').write_text(html)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pin-source', action='store_true', help='pin committed helpers before rebuilding')
    args = parser.parse_args()
    if args.pin_source:
        source_lock(pin=True)
    notebook()
    notebook(local=True)
    game()
    print('Built Colab and laboratory notebooks and the Pac-Man browser game.')
