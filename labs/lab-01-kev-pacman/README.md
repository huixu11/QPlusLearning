# Lab 1 — Train a Decision Model to Play Pac-Man

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yxc20089/QPlusLearning/blob/main/labs/lab-01-kev-pacman/notebooks/pacman_kev_lab.ipynb)

The [native lab slides](https://docs.google.com/presentation/d/1_PfmbUEH38jMO_24S_AqtUUh-IIg2LyZYkEYKjZ5uXQ/edit) compare Kev, CLM and decision-serving implementations, then guide a **90-minute Pac-Man lab with 30 minutes of fine-tuning**. Pac-Man is the learned player. Ghosts follow deterministic game code. Local exports are [PDF](slides/pacman-lab.pdf) and [PPTX](slides/pacman-lab.pptx).

The lab follows **Kev's published LoRA plus pointer-head recipe**. Learners start with the pretrained Qwen3.5-0.8B base, initialize a fresh adapter/head, run the full decision-v7 base-training stage, continue through dates/missing-evidence and documents/skills stages, then fine-tune their own checkpoint on Pac-Man labels. Original base matrices remain frozen, while LoRA changes the encoder's effective features. CLM remains an architectural comparison.

Start with [the student notebook](notebooks/pacman_kev_lab.ipynb) and [Colab setup](COLAB_SETUP.md). The target is **one NVIDIA RTX PRO 6000 Blackwell GPU in Colab**. The full Server Edition has 96 GB VRAM. Confirm the actual allocation, BF16 support and CUDA kernels before training. Colab does not guarantee this GPU or free access. Record actual compute units and elapsed time.

## Recipe and timetable

The notebook executes checksum-verified upstream `kev.train` source with telemetry and LoRA recovery hooks in memory. The upstream checkout stays pinned. The published initial reference is the seed-2 arm of `experiments/q35-08b.json`: all 12,576 decision-v7 training records, two epochs, learning rate 1e-4, batch 8, accumulation 1, BF16 autocast over FP32 weights, LoRA rank 16 / alpha 32, and a 256-dimensional pointer head. Upstream supplies AdamW, OneCycleLR, cross-entropy, option shuffling, none/distractor augmentation and 25% none minimal pairs. The command has no `--init_from`.

The default `memory_safe` profile changes execution to **batch 1, accumulation 8, gradient checkpointing and `row_budget=2048`** for every stage. The data, epochs, effective batch, architecture, augmentation settings and learning-rate schedule remain as documented. Microbatch weighting, dropout and row grouping differ, so results are not an exact numerical reproduction. Set `training_profile='published_reference'` for the original settings. The change responds to a user-reported OOM at step 2,073/3,144 on a 94.97 GiB GPU with 89.90 GiB actively allocated, using Qwen's memory-heavy reference DeltaNet path. The new profile still requires a GPU rerun. The row budget splits questions into passes; a single longer question runs intact without silently dropping/truncating records.

**Complete all three general-decision training stages as prework**, along with installation and downloads. The published base run took about 20 minutes on an H100. We have not measured this Colab target, so the full initial run has no promised class-time duration. The helper gives prework a configurable 90-minute attempt cap. Keep each stage checkpoint; the documents/skills checkpoint is the class baseline.

Each general stage has its own run cell, output directory, backup ZIP and restore controls. After a disconnect, restore all three in order. The notebook checks the recipe, complete optimizer/record counts, data checksum and parent checkpoint fingerprint, and records learner/instructor ownership.

| Minutes | Activity | Evidence |
| --- | --- | --- |
| Prework | Install, audit parameters, run initial → dates/evidence → documents/skills | Three separate checkpoints and training runs |
| 0–10 | Compare architectures and play | Player state and legal actions |
| 10–20 | Inspect stage training curves and a player decision | Configuration, optimizer steps and parameter audit |
| 20–30 | Review Pac-Man labels | Reviewed training file, closed evaluation split |
| 30–60 | Inspect loss (5 min), fine-tune (20 min), inspect/save (5 min) | Positive optimizer steps and domain checkpoint |
| 60–80 | Compare decisions, gameplay and cost | Paired predictions, captures, dots, latency and memory |
| 80–90 | Explain and submit | Notebook, labels, four adapters/heads and measurements |

Pac-Man adaptation uses upstream's documented custom-data path, `--init_from` the learner's documents/skills checkpoint, two epochs, learning rate 2e-5, batch 1, accumulation 8, BF16 autocast and gradient checkpointing. With all 64 accepted rows this takes 16 optimizer steps. It disables none/distractor insertion because game outputs must remain legal directions. Option shuffling remains active. This deliberate task adaptation differs from the generic initial recipe. The 20-minute training cap needs a GPU preflight.

The two intermediate stages follow the published curriculum and save separate checkpoints:

| Stage | New training data | Replay from decision-v7 train | Settings |
| --- | --- | --- | --- |
| Dates and missing evidence | 1,425 generated records | 2,000 | 1 epoch, lr 4e-5, batch 8, accumulation 1, seed 1 |
| Documents and skills | 5,219 complaints + 6,000 skill records + 5,320 developer-tooling records | 6,000 | 1 epoch, lr 2e-5, batch 4, accumulation 2, state budget 7,552, checkpointing, seed 1 |

Both use BF16 and preserve 25% none minimal pairs. [training-stages.json](training-stages.json) records selected settings and exact data checksums. The joint file is reconstructed from verified training partitions and must match the published 16,539-record checksum. No development/test data enters training.

Calibration is separate and is not applied here. Fresh checkpoints do not inherit the release's measured accuracy or temperature. Initial decision training uses an already pretrained LLM; learners do not train a language foundation model from random weights.

## Progress and TensorBoard

Open the TensorBoard cell **before** training. Colab supports an embedded dashboard. Every optimizer step records cross-entropy, next-step learning rate, gradient norm before clipping, epoch, step duration, records seen and peak allocated GPU memory. Each stage has its own run directory under `logs/`. These are training curves, with no fabricated validation curve.

The helper captures stdout/stderr explicitly for Colab. Console progress appears at the first step, every ten steps and the last step, with a 15-second heartbeat during loading and long steps. It preserves raw logs, JSONL, CSV, TensorBoard event files and a completion/failure status, including when a run times out. GPU curves now include live allocated, reserved and free memory. `batches.jsonl` records row counts, longest rows, padded tokens and record IDs before each forward pass, so a failing batch remains inspectable. [training_monitor.py](training_monitor.py) checks the upstream checksum and every telemetry/recovery insertion site.

The former `BUNDLED_FILES` dictionary kept the notebook independent of a checkout by embedding helpers and licensed game source. Setup now downloads ten small files from a pinned public course commit and verifies their hashes; [notebook-source.json](notebook-source.json) records that pin. GitHub access is required on first use. TensorBoard is installed in the notebook kernel, while Kev's locked model environment remains separate.

## Recovery after interruption

[lora_recovery.py](lora_recovery.py) saves trainable parameters, AdamW state, OneCycleLR state, CPU/CUDA/Python RNG and training counters at optimizer boundaries: after step 1, every 100 steps or five minutes, and at the final step. Snapshots include loadable adapter/head exports, are committed through a completion marker and latest pointer, and retain the latest two complete saves under `<output>-recovery`. The native Kev `--resume` flag is full-weight-only; the lab implements LoRA recovery separately. CPU checks establish identical continuation with dropout, including optimizer moments and scheduler state; CUDA continuation is not yet validated.

After a failure, keep the same arguments, profile, data, parent checkpoint and output path, and set the relevant `RESUME_INITIAL`, `RESUME_DATES`, `RESUME_SKILLS` or `RESUME_PACMAN` to `True`. Changing the profile requires a new run. Optional `SAVE_TO_DRIVE=True` puts outputs and recovery on Drive before training. Otherwise download the recovery ZIP and logs before Colab discards its temporary filesystem. Recovery ZIPs must return to the original absolute checkpoint root; complete checkpoint ZIPs can be restored through each stage's existing controls.

**The previous notebook only saved LoRA weights after training finished. Its failed step-2,073 run has no automatic learned checkpoint.** Configuration and telemetry are not learned weights, and weights cannot be recovered after the child process exits. The new notebook uses `*-v2` output folders to preserve old files and includes an inspection/export cell. Partial snapshots cannot substitute for a completed curriculum stage.

## Game and data

The source is [codaaiteam/jev-pacman](https://github.com/codaaiteam/jev-pacman), MIT, pinned at `8446fe74690cd61909bda91acfadbccb0f02b422`. The original has a human player and Jev-controlled ghosts. Our adaptation preserves its maze and renderer and replaces both controllers. It has one maze, two ghosts and no power pellets. No ROM is needed.

[pacman_lab.py](pacman_lab.py) defines observations, legal moves, deterministic ghosts, transitions, labels and evaluation. [player-controller.js](games/player-controller.js) connects the browser to the notebook. [pacman.html](games/pacman.html) includes the complete MIT notice, with accompanying [LICENSE](games/LICENSE). Human mode runs locally. Model mode uses a Colab callback to notebook-local `127.0.0.1:8009`.

Starter data has 64 training, 16 development and 16 evaluation snapshots. Positions are disjoint across splits, all on one maze. The teacher prioritizes immediate safety, maze distance to dots, ghost separation and fixed direction ties. Learners can replace training labels with their own legal moves. These are synthetic states and heuristic labels, not human recordings or an optimal gameplay benchmark. Evaluation strips labels and metadata. Fix the candidate before opening evaluation data.

Compare the learner's documents/skills checkpoint against their Pac-Man checkpoint on identical boards and ghost rules. Accuracy measures teacher agreement. Immediate captures describe one move. Capped 24-turn rollouts illustrate gameplay and do not establish a win rate. Report unchanged answers and regressions. Compare game turns rather than wall-clock survival because the game waits for inference.

## Validation and provenance

The original initial profile ran on the user's GPU and **failed with OOM at step 2,073/3,144**. Completion of the revised profile, CUDA recovery, training timing, reloads and the live Colab callback remain **unvalidated**. Before teaching, run the complete notebook on the intended allocation and prepare compatible instructor checkpoints and recorded evaluations. Label supplied results as instructor results. The setup's parameter audit checks that only LoRA and the pointer head can train; it does not replace a real training run.

Kev source: `84847f0a883d900f7de5b7a57eaa341ca7f9a6b4`. Qwen base: `Qwen/Qwen3.5-0.8B-Base@dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. The suite loader verifies the training partition against its manifest checksum. [upstream.json](upstream.json) records these pins and the optional released-model reference. Vendored recipe and game files retain their upstream licenses; see [third-party provenance](vendor/README.md).

Clone this repository, then run the CPU checks and regenerate the notebook/game from the repository root. Python 3.10+ and Node.js are required; no GPU or downloaded model is needed for these checks:

```bash
python3 -m pip install -r labs/lab-01-kev-pacman/test-requirements.txt
python3 -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python3 -m unittest discover -s labs/lab-01-kev-pacman -p 'test_*.py'
python3 labs/lab-01-kev-pacman/build_artifacts.py
python3 scripts/check_notebook.py
```

The builder records helper hashes and a pinned course revision in the notebook and regenerates the adapted browser game. After changing helpers, commit those changes, then run the builder with `--pin-source` and commit the regenerated notebook/lock. Ordinary rebuilds preserve the existing source pin. The comparison slides retain their original base→Pac-Man walkthrough; use the notebook for the current four-stage procedure. The slides include PDF/PPTX exports and [reviewed content with speaker notes](slides/comparison-content.json); [deck metadata](slides/deck.json) records the native source and revision. The builder does not write Google Slides.

Sources: [Kev initial experiment](https://github.com/jaredpalmer/kev/blob/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4/experiments/q35-08b.json), [Kev model card](https://github.com/jaredpalmer/kev/blob/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4/docs/model-cards/kev-0.8b.md), [Kev custom-data guide](https://github.com/jaredpalmer/kev/blob/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4/README.md), [CLM trainer](https://github.com/Contrastive-LM/CLM/blob/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7/train/finetune.py), [llama.cpp release](https://huggingface.co/blog/ggml-org/decision-models-in-llamacpp), [SGLang documentation](https://lmsysorg.mintlify.app/docs/supported-models/decision_models).
