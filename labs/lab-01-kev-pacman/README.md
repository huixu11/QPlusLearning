# Lab 1 — Train a Decision Model to Play Pac-Man

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yxc20089/QPlusLearning/blob/main/labs/lab-01-kev-pacman/notebooks/pacman_kev_lab.ipynb)

The [native lab slides](https://docs.google.com/presentation/d/1_PfmbUEH38jMO_24S_AqtUUh-IIg2LyZYkEYKjZ5uXQ/edit) compare Kev, CLM and decision-serving implementations, then guide a **90-minute Pac-Man lab with 30 minutes of fine-tuning**. Pac-Man is the learned player. Ghosts follow deterministic game code. Local exports are [PDF](slides/pacman-lab.pdf) and [PPTX](slides/pacman-lab.pptx).

The lab follows **Kev's published LoRA plus pointer-head recipe**. Learners start with the pretrained Qwen3.5-0.8B base, initialize a fresh adapter/head, run the full decision-v7 base-training stage, then fine-tune their own checkpoint on Pac-Man labels. Original base matrices remain frozen, while LoRA changes the encoder's effective features. CLM remains an architectural comparison.

Start with [the student notebook](notebooks/pacman_kev_lab.ipynb) and [Colab setup](COLAB_SETUP.md). The target is **one NVIDIA RTX PRO 6000 Blackwell GPU in Colab**. The full Server Edition has 96 GB VRAM. Confirm the actual allocation, BF16 support and CUDA kernels before training. Colab does not guarantee this GPU or free access. Record actual compute units and elapsed time.

## Recipe and timetable

The notebook invokes the unchanged, pinned upstream `kev.train` module. Initial training reads the seed-2 arm from `experiments/q35-08b.json`: all 12,576 decision-v7 training records, two epochs, learning rate 1e-4, batch 8, accumulation 1, BF16 autocast over FP32 weights, LoRA rank 16 / alpha 32, and a 256-dimensional pointer head. Upstream supplies AdamW, OneCycleLR, cross-entropy, option shuffling, none/distractor augmentation and 25% none minimal pairs. The command has no `--init_from`.

**Complete initial training as prework**, along with installation and downloads. The published base run took about 20 minutes on an H100. We have not measured this Colab target, so the full initial run has no promised class-time duration. The helper gives prework a configurable 90-minute attempt cap. Keep its checkpoint for class.

Save the notebook's prework checkpoint ZIP to your computer. After a runtime disconnect, upload it and use `RESTORE_ARCHIVE` to restore it. The notebook checks recipe compatibility and records whether the checkpoint came from the learner or instructor.

| Minutes | Activity | Evidence |
| --- | --- | --- |
| Prework | Install, audit parameters, train the full initial decision model | Pinned base/data, fresh LoRA/head checkpoint |
| 0–10 | Compare architectures and play | Player state and legal actions |
| 10–20 | Inspect initial training and a player decision | Configuration, optimizer steps and parameter audit |
| 20–30 | Review Pac-Man labels | Reviewed training file, closed evaluation split |
| 30–60 | Inspect loss (5 min), fine-tune (20 min), inspect/save (5 min) | Positive optimizer steps and domain checkpoint |
| 60–80 | Compare decisions, gameplay and cost | Paired predictions, captures, dots, latency and memory |
| 80–90 | Explain and submit | Notebook, labels, both adapters/heads and measurements |

Pac-Man adaptation uses upstream's documented custom-data path, `--init_from` the learner's initial checkpoint, two epochs, learning rate 2e-5, batch 1, accumulation 8, BF16 autocast and gradient checkpointing. With all 64 accepted rows this takes 16 optimizer steps. It disables none/distractor insertion because game outputs must remain legal directions. Option shuffling remains active. This deliberate task adaptation differs from the generic initial recipe. The 20-minute training cap needs a GPU preflight.

The released Kev-0.8B also includes dates, documents/skills and calibration stages. This lab reproduces the base stage and adds Pac-Man adaptation. It does not reproduce the full released curriculum or inherit its benchmark scores or temperature calibration. Initial decision training uses an already pretrained LLM; learners do not train a language foundation model from random weights.

## Game and data

The source is [codaaiteam/jev-pacman](https://github.com/codaaiteam/jev-pacman), MIT, pinned at `8446fe74690cd61909bda91acfadbccb0f02b422`. The original has a human player and Jev-controlled ghosts. Our adaptation preserves its maze and renderer and replaces both controllers. It has one maze, two ghosts and no power pellets. No ROM is needed.

[pacman_lab.py](pacman_lab.py) defines observations, legal moves, deterministic ghosts, transitions, labels and evaluation. [player-controller.js](games/player-controller.js) connects the browser to the notebook. [pacman.html](games/pacman.html) includes the complete MIT notice, with accompanying [LICENSE](games/LICENSE). Human mode runs locally. Model mode uses a Colab callback to notebook-local `127.0.0.1:8009`.

Starter data has 64 training, 16 development and 16 evaluation snapshots. Positions are disjoint across splits, all on one maze. The teacher prioritizes immediate safety, maze distance to dots, ghost separation and fixed direction ties. Learners can replace training labels with their own legal moves. These are synthetic states and heuristic labels, not human recordings or an optimal gameplay benchmark. Evaluation strips labels and metadata. Fix the candidate before opening evaluation data.

Compare the learner's initial checkpoint against their Pac-Man checkpoint on identical boards and ghost rules. Accuracy measures teacher agreement. Immediate captures describe one move. Capped 24-turn rollouts illustrate gameplay and do not establish a win rate. Report unchanged answers and regressions. Compare game turns rather than wall-clock survival because the game waits for inference.

## Validation and provenance

GPU execution, training timing, reloads and the live Colab callback are **not yet tested**. Before teaching, run the complete notebook on the intended allocation and prepare compatible instructor checkpoints and recorded evaluations. Label supplied results as instructor results. The setup's parameter audit checks that only LoRA and the pointer head can train; it does not replace a real training run.

Kev source: `84847f0a883d900f7de5b7a57eaa341ca7f9a6b4`. Qwen base: `Qwen/Qwen3.5-0.8B-Base@dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. The suite loader verifies the training partition against its manifest checksum. [upstream.json](upstream.json) records these pins and the optional released-model reference. Vendored recipe and game files retain their upstream licenses; see [third-party provenance](vendor/README.md).

Clone this repository, then run the CPU checks and regenerate the notebook/game from the repository root. Python 3.10+ and Node.js are required; no GPU or downloaded model is needed for these checks:

```bash
python3 -m unittest discover -s labs/lab-01-kev-pacman -p 'test_*.py'
python3 labs/lab-01-kev-pacman/build_artifacts.py
python3 scripts/check_notebook.py
```

The builder bundles the teaching helpers and licensed community source into the notebook and regenerates the adapted browser game. The notebook runs independently of a GitHub checkout. The slides include PDF/PPTX exports and [reviewed content with speaker notes](slides/comparison-content.json); [deck metadata](slides/deck.json) records the native source and revision. The builder does not write Google Slides.

Sources: [Kev initial experiment](https://github.com/jaredpalmer/kev/blob/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4/experiments/q35-08b.json), [Kev model card](https://github.com/jaredpalmer/kev/blob/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4/docs/model-cards/kev-0.8b.md), [Kev custom-data guide](https://github.com/jaredpalmer/kev/blob/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4/README.md), [CLM trainer](https://github.com/Contrastive-LM/CLM/blob/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7/train/finetune.py), [llama.cpp release](https://huggingface.co/blog/ggml-org/decision-models-in-llamacpp), [SGLang documentation](https://lmsysorg.mintlify.app/docs/supported-models/decision_models).
