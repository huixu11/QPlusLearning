# Lab 1 — Train a Decision Model to Play Pac-Man

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yxc20089/QPlusLearning/blob/main/labs/lab-01-kev-pacman/notebooks/pacman_kev_lab.ipynb)

Train **Kev-4B** with its published LoRA plus pointer-head architecture, then teach it to control Pac-Man. Ghosts follow deterministic game code. The class lasts **90 minutes with 30 minutes for Pac-Man fine-tuning**; installation, downloads and four general training stages are prework.

Use [the notebook](notebooks/pacman_kev_lab.ipynb), [Colab setup](COLAB_SETUP.md) and the [native lab slides](https://docs.google.com/presentation/d/1_PfmbUEH38jMO_24S_AqtUUh-IIg2LyZYkEYKjZ5uXQ/edit). The slides retain the Kev/CLM comparison and now show the 4B curriculum; local exports are [PDF](slides/pacman-lab.pdf) and [PPTX](slides/pacman-lab.pptx).

The target is **one NVIDIA RTX PRO 6000 Blackwell GPU in Colab**. The full Server Edition has 96 GB VRAM. Arrange access before class, inspect the actual allocation, and record elapsed time and compute units. Colab does not guarantee this GPU or free access.

## The five training stages

Start from `Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b`. Only Stage 1 initializes fresh adapters/head; each later stage uses the preceding learner checkpoint. Original base matrices remain frozen while LoRA changes effective encoder features. This is decision-model training over an already pretrained LLM, rather than language-model pretraining from random weights. Old 0.8B adapters cannot initialize 4B.

| Stage | New source records | Replay from decision-v7 train | Published reference settings | Optimizer steps |
| --- | --- | --- | --- | --- |
| 1. Initial decisions | 12,576 decision-v7 | None | 2 epochs, lr 5e-5, batch 4 × accumulation 2, seed 2 | 3,144 |
| 2. Dates/missing evidence | 1,425 generated cases | 2,000 | 1 epoch, lr 2e-5, batch 4 × accumulation 2, seed 1 | 429 |
| 3. Documents | 5,219 finance complaints | 2,000 | 1 epoch, lr 2e-5, batch 2 × accumulation 4, seed 2 | 903 |
| 4. Skills/devtools | 6,000 hard-v1 + 5,320 devtools-v1 | 4,000 | 1 epoch, lr 2e-5, batch 2 × accumulation 4, seed 1 | 1,915 |
| 5. Pac-Man | 64 reviewed training boards | None | 2 epochs, lr 2e-5, batch 1 × accumulation 8, seed 7 | 16 |

All stages use BF16 autocast over FP32 stored parameters, gradient checkpointing, rank-16 LoRA (alpha 32) on all projections and a 256-dimensional pointer head. Upstream provides AdamW, OneCycleLR and cross-entropy over supplied options. General stages retain option shuffling, none/distractor insertion and 25% none minimal pairs. Documents and skills use `max_state=7552`; dates uses 384. Pac-Man uses 2,048 and disables none/distractor insertion because only legal directions are valid. Option shuffling remains active.

The initial recipe selects arm 0 of `experiments/q35-4b-s23.json`. [training-stages.json](training-stages.json) pins intermediate settings and data hashes. Documents remain separate from skills, as in the 4B release. Skills data concatenates verified hard-v1 train then devtools-v1 train; its combined checksum is course-specific. We do not assert byte identity to the unavailable historical round10 joint file. No development/test rows enter training. Calibration remains separate and is not fitted or inherited from the release; neither are its benchmark scores.

The notebook's **Stage 1 training cell explicitly uses batch 4 × accumulation 2, non-reentrant gradient checkpointing and `row_budget=0`**. It assigns `runtime.initial_execution` before constructing and printing the actual command, overriding the generic memory profile for this stage. The default `memory_safe` profile supplies **batch 1 × accumulation 8 and `row_budget=2048`** to later stages. Both retain source data, epochs, effective batch, optimizer-step count, architecture, learning-rate schedule and augmentation. Microbatch weighting, grouping, dropout execution and optimized arithmetic differ, so changing execution during a run is not an exact numerical reproduction. `TRAINING_PROFILE='published_reference'` retains original later-stage execution flags and still requires optimized kernels. A nonzero row budget splits questions into passes; a longer single question runs intact without silently truncating or dropping records.

The published initial execution is **batch 4 × accumulation 2, checkpointing enabled, BF16 autocast, FP32 stored weights and no row budget**. The [model card](https://github.com/jaredpalmer/kev/blob/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4/docs/model-cards/kev-4b.md#compute) reports about 56 minutes / 24.6 GB peak on one H100. These are not RTX PRO 6000 measurements.

For a running initial stage, wait for a recovery save and interrupt the cell. Keep the original checkpoint path. The Stage 1 cell has one resume setting: `RESUME_CHECKPOINT='latest'` uses the latest complete snapshot for that output, or provide a specific snapshot's full directory path. It prints the selected path and saved optimizer step, then resumes at 4×2 / row budget 0. A missing or incomplete requested checkpoint stops training without starting over. Explicitly set `RESUME_CHECKPOINT=None` only for a student's first run in a new output directory. The guarded continuation allows only batch/accumulation/row-budget changes at a plain single-GPU optimizer boundary, preserves effective batch and total steps, and maps progress to the same next source records. Weights, optimizer, scheduler and RNG are restored. Grouping changes the numerical trajectory; recovery receipts and final evidence retain the execution history. Other stages' ordinary resume still rejects changed arguments.

Stages 1–4 each have a separate run cell, checkpoint, backup ZIP, restore control and TensorBoard directory. Each has a configurable 180-minute attempt cap, a scheduling limit rather than a prediction. Keep all four checkpoints before class. Stage 4 is the class baseline. The notebook checks full optimizer-step counts, recipe/data pins, parent fingerprints and learner/instructor ownership. Kev's records-seen count includes augmented siblings and may exceed requested source records.

| Minutes | Activity | Evidence |
| --- | --- | --- |
| Prework | Install; verify kernels; initial → dates → documents → skills | Four checkpoints and separate training curves |
| 0–10 | Compare architectures and play | Player state and legal actions |
| 10–20 | Inspect general training stages and a decision | Configurations, curves and parameter audit |
| 20–30 | Review player labels | Edited train partition; evaluation closed |
| 30–60 | Inspect loss (5 min), fine-tune (20 min), inspect/save (5 min) | Pac-Man checkpoint and positive optimizer steps |
| 60–80 | Compare decisions, gameplay and cost | Paired predictions, captures, dots, latency and memory |
| 80–90 | Explain and submit | Notebook, labels, five adapters/heads and measurements |

## Required optimized training

[optimized_training.py](optimized_training.py) adds hash-verified wheels from [training-kernels.json](training-kernels.json) to Kev's frozen environment using `--no-deps`: flash-linear-attention/fla-core 0.5.2, causal-conv1d 1.7.0, einops 0.8.1 and Ninja 1.13.0. The CUDA convolution wheel requires Python 3.13, Linux x86_64, Torch 2.8 and CXX11 ABI. Kev's Torch 2.8.0/CUDA 12.8, Triton 3.4.0, Transformers 5.17.0 and PEFT stay locked.

FLA's Triton kernels handle Gated DeltaNet; the CUDA convolution handles its short convolution; PyTorch SDPA handles full attention. Training uses fused AdamW, BF16 autocast, checkpointing, accumulation and bounded row passes. Kev's inference fusion/CUDA graphs are disabled for this backward path. The trainer source remains pinned and checksum-verified; telemetry, fused optimizer and recovery hooks are inserted in memory.

`prepare_training()` downloads/audits the actual 4B model on CPU, then checks pinned optimized function bindings and CUDA/BF16 availability. The separate two-record training preflight is **skipped by default** (`RUN_TRAINING_PREFLIGHT=False`). Its JSON report records `bindings_verified`, zero optimizer steps and `cuda_loss_backward='not_run'`. Each real training subprocess rechecks bindings and refuses reference-kernel fallback. The first real optimizer step exercises training and saves a recovery checkpoint.

Use `prepare_training(run_training_preflight=True)` for the optional two-record loss/backward/fused AdamW check. It verifies finite adapter/head gradients and actual FLA/convolution/fused AdamW dispatch counters without a CUDA profiler. Phases identify encoding, forward, backward, gradient checks and completed updates. This optional check and real training print Python stacks after 60 seconds without progress. Heartbeats show process liveness. The binding-only attempt has a two-minute timeout and the optional training check has a 20-minute timeout; neither is a duration estimate. Skipping the extra check does not bypass first-use kernel compilation during training. **The optional CUDA check and full 4B curriculum have not been run by the course author.**

For an old preflight that is still running, interrupt its cell and keep the runtime connected. Copy and run the updated bootstrap cell to refresh helpers while keeping downloaded weights/kernels, checkpoints and logs. If the optimized environment is already installed, run the new separate preparation cell with `RUN_TRAINING_PREFLIGHT=False`, then proceed to Stage 1.

## Progress and TensorBoard

Colab supports the embedded TensorBoard cell; open it before training. Each optimizer step records CE, next-step learning rate, gradient norm, epoch, duration, records seen and peak/live allocated/reserved/free GPU memory. These are training curves, with no invented validation curve. Logs remain separate under `logs/<stage>/<attempt>`.

The monitor streams the first step, every ten steps and the final step, plus 15-second heartbeats during loading/compilation or long steps. Raw stdout/stderr, JSONL, CSV, TensorBoard events and failure/timeout status are preserved. `batches.jsonl` records physical rows, longest row, padded tokens and record IDs before each forward pass, including the failing pass. [training_monitor.py](training_monitor.py) verifies all insertion sites against the pinned upstream trainer.

The former `BUNDLED_FILES` embedded helpers and licensed game source. The notebook now downloads **12 small files** from a committed public source pin, checks every hash and reloads cached helper modules. [notebook-source.json](notebook-source.json) records that pin. Model weights, kernels and training data download separately. The notebook kernel hosts teaching helpers/TensorBoard; Kev runs in its own locked Python environment.

## Recovery and the earlier OOM

The learner's old 0.8B batch-8 run failed at step 2,073/3,144 on a 94.97 GiB GPU: 89.90 GiB was actively allocated, with only 73.88 MiB free. Logs confirmed both optimized DeltaNet/convolution packages were missing. The cumulative peak reached 91.91 GiB by step 280 and stayed there; that alone does not establish a leak. Optimized kernels and a smaller execution profile address the observed setup, while new telemetry measures batch shapes and live memory. Allocator `expandable_segments` cannot free active tensors.

**The old notebook saved LoRA weights only after completion. That failed run has logs/configuration but no automatic learned checkpoint.** We cannot reconstruct exited-process weights from those logs. The new `kev-4b-*` output paths preserve old files and start fresh, incompatible 4B adapters.

[lora_recovery.py](lora_recovery.py) saves after optimizer step 1, every 100 steps or five minutes (checked at optimizer boundaries), and the final step. It keeps the latest two complete snapshots under `<output>-recovery`. Each contains loadable adapter/head exports plus optimizer, scheduler, CPU/CUDA/Python RNG and progress counters. Atomic completion markers prevent selecting incomplete saves. Native Kev resume is full-weight-only; the lab implements LoRA recovery separately. CPU tests verify identical continuation including dropout, Adam moments, scheduler and actual pinned epoch-loop behavior. CUDA continuation remains unvalidated.

After interruption, ordinary resume requires identical profile, arguments, inputs, parent checkpoint and output paths. Stage 1 selects recovery with `RESUME_CHECKPOINT` and enables its explicit 4×2 execution-change path as described above. For later stages, set the affected `RESUME_DATES`, `RESUME_DOCUMENTS`, `RESUME_SKILLS` or `RESUME_PACMAN` to `True`. Completed stages stay completed. A partial snapshot is not a completed curriculum stage. Set `SAVE_TO_DRIVE=True` before training to persist weights across runtime loss, or export recovery/log ZIPs before `/content` disappears. Recovery archives must restore at their original absolute paths; completed checkpoint ZIPs have per-stage restore controls.

## Game, labels and evaluation

The community source is [codaaiteam/jev-pacman](https://github.com/codaaiteam/jev-pacman), MIT, pinned at `8446fe74690cd61909bda91acfadbccb0f02b422`. Its demo has a human player and Jev-controlled ghosts. This adaptation preserves its maze/renderer and changes controllers: Kev selects legal player directions and deterministic code advances two ghosts. There are no power pellets or ROM downloads. Full notices remain in [the game](games/pacman.html) and [third-party provenance](vendor/README.md).

[pacman_lab.py](pacman_lab.py) defines state, rules, labels and evaluation. [player-controller.js](games/player-controller.js) uses a Colab callback to the notebook-local server at `127.0.0.1:8009`; no public inference endpoint is needed. Kaggle can run structured decisions/Python rollouts but lacks this callback.

Starter data has 64 train, 16 development and 16 evaluation snapshots, with disjoint positions on one maze. The heuristic teacher prioritizes immediate safety, maze distance to dots, ghost separation and fixed ties. Learners can edit training labels. These are synthetic labels, not human recordings or optimal gameplay. Fix the candidate before viewing evaluation. Compare Stage 4 against Stage 5 on identical boards/rules; report teacher agreement, immediate captures, unchanged answers and regressions. Capped 24-turn rollouts illustrate behavior rather than a win rate. Compare turns because the game waits for inference.

## Validation and provenance

Local CPU tests cover rules/data splits, all stage commands, TensorBoard events, process failures, LoRA recovery, kernel installation guards and checkpoint provenance. Completion of the optimized 4B profile, CUDA recovery, timing, reloads and live Colab callback remains **unvalidated**. Run the complete notebook on the intended allocation before teaching and prepare compatible instructor checkpoints. Label supplied results as instructor results.

Kev code is pinned at `84847f0a883d900f7de5b7a57eaa341ca7f9a6b4`. [upstream.json](upstream.json) records base, suite-mirror and optional released-model pins. Downloaded source partitions are verified against their manifests/checksums. Vendored recipe files remain unchanged and licensed.

From the repository root:

```bash
python3 -m pip install -r labs/lab-01-kev-pacman/test-requirements.txt
python3 -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python3 -m unittest discover -s labs/lab-01-kev-pacman -p 'test_*.py'
python3 labs/lab-01-kev-pacman/build_artifacts.py
python3 scripts/check_notebook.py
```

After helper edits, commit them before rebuilding with `--pin-source`, then commit the notebook/lock. Ordinary rebuilds preserve the source pin. The builder does not write Google Slides.

Sources: [Kev-4B model card](https://huggingface.co/jaredpalmer/kev-4b), [initial recipe](https://github.com/jaredpalmer/kev/blob/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4/experiments/q35-4b-s23.json), [Qwen optimized kernels](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5), [FLA release](https://github.com/fla-org/flash-linear-attention/releases/tag/v0.5.2), [CUDA convolution release](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.7.0). Kev/CLM/llama.cpp/SGLang comparison sources remain in the slide notes.
