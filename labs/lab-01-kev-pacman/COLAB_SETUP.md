# Colab setup for Kev-4B Pac-Man

Target **one NVIDIA RTX PRO 6000 Blackwell GPU**, nominally 96 GB on the full Server Edition. Inspect the actual allocation. An RTX 6000 Ada or RTX A6000 is a different card. Arrange access before class; Colab does not guarantee this GPU, including on paid plans. Measure cost using actual compute units and elapsed time.

## Prework

1. Open [Lab 1 in Colab](https://colab.research.google.com/github/yxc20089/QPlusLearning/blob/main/labs/lab-01-kev-pacman/notebooks/pacman_kev_lab.ipynb) and save a fresh copy. Select the target GPU if offered.
2. Run setup. It downloads hash-verified helpers, classic-game source/assets and starter data, installs pinned Kev in a separate Python 3.13 environment and adds the hash-pinned optimized training wheels without replacing locked dependencies.
3. Keep `runtime-preflight.json` and run the separate preparation cell with `RUN_TRAINING_PREFLIGHT=False` (default). This fetches the pinned **Qwen3.5-4B-Base**, audits LoRA/head parameters on CPU, and checks pinned optimized bindings/CUDA/BF16 availability. Keep `optimized-training-preflight.json`; it records `bindings_verified`, zero optimizer steps and `cuda_loss_backward='not_run'`. The first real training step exercises loss/backward/fused AdamW and saves a recovery checkpoint. Reference-kernel fallback stops training.
4. Optional: set `RUN_TRAINING_PREFLIGHT=True` for two extra suite-record training checks. This runs without a CUDA profiler, verifies gradients, and counts actual FLA/convolution/fused AdamW calls. Both this check and real training dump Python stacks after 60 seconds without progress. First-use compilation/autotuning may still occur in training when the check is skipped. The binding-only attempt has a two-minute cap; the optional check has a 20-minute cap. These are timeout limits, not duration estimates.

If an older notebook is stuck in the two-record preflight, interrupt its cell and keep the runtime connected. Copy/run the updated bootstrap cell to refresh verified helpers without deleting downloads or checkpoints. If the optimized environment is already installed, run the new separate preparation cell with the default `False`, then continue to Stage 1.
4. Open TensorBoard before training. Select `SAVE_TO_DRIVE=True` before any run if weights should survive runtime loss. Local `/content` is temporary.
5. Complete **initial → dates/evidence → documents → skills/devtools** in the four separate prework sections. Each saves its own checkpoint and backup. Use fresh `kev-4b-*` directories: 0.8B adapters cannot initialize 4B. Keep all four ZIPs and stage curves.
6. The notebook starts the skills checkpoint for inference. Confirm `/v1/models` and a legal player decision. Record the starting compute-unit balance for the class.

The [recipe table](README.md#the-five-training-stages) lists exact source counts, replay and selected published settings. General stages use the complete published curriculum and configurable 180-minute attempt caps. These caps are scheduling limits, not measured durations. General training belongs outside the 90-minute class. The mandatory 30-minute class block fine-tunes Pac-Man for two complete epochs on 64 accepted rows: 16 optimizer steps, with a 20-minute training attempt cap.

## Optimized execution and precision

Kev's frozen stack uses Torch 2.8.0/CUDA 12.8, Triton 3.4.0, Transformers 5.17.0 and PEFT. The additive overlay pins FLA/fla-core 0.5.2, causal-conv1d 1.7.0, einops 0.8.1 and Ninja 1.13.0. The CUDA wheel matches Python 3.13, Linux x86_64, Torch 2.8 and CXX11 ABI. FLA handles DeltaNet, CUDA handles short convolution, and SDPA handles full attention. BF16 autocast, FP32 stored parameters and fused AdamW are required. Kev inference fusion/CUDA graphs are disabled for training.

Default `TRAINING_PROFILE='memory_safe'`: batch 1 × accumulation 8, non-reentrant checkpointing, `row_budget=2048` in every stage. Source data, effective batch and schedules remain unchanged; microbatch/dropout execution and kernel arithmetic differ. The published-reference profile retains original execution flags but still requires optimized kernels. A row budget does not truncate a single longer question. Optional allocator `expandable_segments` cannot release actively used tensors.

For initial-stage throughput, the optional benchmark compares memory-safe 1×8, bounded 2×4 and 4×2, and published 4×2 with no row budget. All retain gradient checkpointing. Stop any running training cell after a recovery save before running these separate 40-step trials. The first eight steps are excluded from timing; later compilation can still affect results. Selection requires a completed trial, at least 20% allocation headroom and a measured gain of at least 5% over the baseline. The report is not full-stage validation.

To retain initial training progress after changing the selected execution, use the same output with `RESUME_INITIAL=True` and `ALLOW_INITIAL_EXECUTION_CHANGE=True`. Only batch/accumulation/row-budget changes are admitted, effective batch and total steps stay fixed, and recovery maps to the same next records. It restores optimizer/scheduler/RNG and logs the execution history. Grouping/dropout may change numerics. Work after the latest completed snapshot is repeated. The default strict resume and all intermediate-stage settings remain intact.

This contract requires BF16 and rejects T4/P100 as drop-in training fallbacks. An explicitly allowed alternative GPU must satisfy the same software/kernel checks and receive its own full validation. Kaggle supports structured evaluation/rollouts but does not provide the Colab browser callback.

## Progress and interruption

TensorBoard runs in the notebook kernel while Kev trains in its locked environment. Separate `logs/<stage>/<attempt>` directories contain per-step events, CSV/JSONL, stdout/stderr and status. Curves include CE, learning rate, gradient norm, timing, records and peak/live allocated/reserved/free VRAM. `batches.jsonl` preserves token shapes and IDs before each forward pass. These are training curves, not fabricated validation results.

Snapshots save adapter/head, optimizer/scheduler/RNG and progress after step 1, every 100 steps or five minutes at optimizer boundaries, and the final step. The latest two complete saves remain under `<output>-recovery`. After interruption, keep identical arguments, data, parent and paths, set the affected `RESUME_* = True` and rerun that stage. Do not use Kev's native full-weight-only resume flag. Export recovery and logs after exceptions, or use Drive storage. Recovery ZIPs restore to the original absolute checkpoint root. Completed checkpoint ZIPs restore through `RESTORE_ARCHIVE`, `RESTORE_DATES_ARCHIVE`, `RESTORE_DOCUMENTS_ARCHIVE` and `RESTORE_SKILLS_ARCHIVE`, in order. Declare learner/instructor ownership.

The old 0.8B run failed at step 2,073 with missing FLA/convolution kernels and nearly all 95 GiB actively occupied. Its notebook saved weights only at completion, so that failed run has no automatic learned checkpoint. Inspect its old output directory before restarting; logs/configuration cannot reconstruct exited-process weights. Fresh 4B output names preserve those files.

## Before teaching

The local CPU tests do not establish CUDA compatibility or full-stage fit. **The optimized 4B CUDA preflight, full curriculum, CUDA continuation, reloads and live browser callback have not been run by the course author.** Complete them on the intended allocation, measure duration/compute-unit use and prepare compatible instructor checkpoints. Keep the 90-minute class and 30-minute fine-tuning block; report supplied results as instructor results.

Sources: [Colab FAQ](https://research.google.com/colaboratory/faq.html), [RTX PRO 6000 Server Edition](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/), [Kev-4B card](https://huggingface.co/jaredpalmer/kev-4b), [Qwen kernels](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5).

## Classic Pac-Man update

Use the new bootstrap and CP0–CP5 cells. On an existing runtime rerun bootstrap, **Load completed Skills checkpoint — start the lab here**, then CP0. On a fresh runtime run setup, optimized preparation and Drive storage first. Keep the completed Skills baseline. CP3 writes `kev-4b-pacman-arcade`, with new `pacman-arcade-*` data and a 4,096-token state budget. The badge identifies the actual live adapter and hashes.

Node.js executes the same pinned four-ghost engine for CPU evaluation. Setup uses an installed Node 18+ or a checksum-pinned official Node 22.17.0 binary. The native renderer, font, sounds and classic mechanisms are included. Human play uses 60 simulation frames per second; Kev pauses simulation time while deciding.
