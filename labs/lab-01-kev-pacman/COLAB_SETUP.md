# Colab setup for Kev-4B Pac-Man

Target **one NVIDIA RTX PRO 6000 Blackwell GPU**, nominally 96 GB on the full Server Edition. Inspect the actual allocation. An RTX 6000 Ada or RTX A6000 is a different card. Arrange access before class; Colab does not guarantee this GPU, including on paid plans. Measure cost using actual compute units and elapsed time.

## Prework

1. Open [Lab 1 in Colab](https://colab.research.google.com/github/yxc20089/QPlusLearning/blob/main/labs/lab-01-kev-pacman/notebooks/pacman_kev_lab.ipynb) and save a fresh copy. Select the target GPU if offered.
2. Run setup. It downloads 12 hash-verified helpers/game files, installs pinned Kev in a separate Python 3.13 environment and adds the hash-pinned optimized training wheels without replacing locked dependencies.
3. Keep `runtime-preflight.json` and run `prepare_training()`. This fetches the pinned **Qwen3.5-4B-Base**, audits LoRA/head parameters, then runs actual Kev loss/backward/fused AdamW on two suite records. Keep `optimized-training-preflight.json`. Reference-kernel fallback stops training. First Triton compilation may take time. Phase logs distinguish forward/backward, completed optimizer checks and profiler finalization; heartbeats repeat the last reported phase. The 20-minute timeout is not an expected duration.
4. Open TensorBoard before training. Select `SAVE_TO_DRIVE=True` before any run if weights should survive runtime loss. Local `/content` is temporary.
5. Complete **initial → dates/evidence → documents → skills/devtools** in the four separate prework sections. Each saves its own checkpoint and backup. Use fresh `kev-4b-*` directories: 0.8B adapters cannot initialize 4B. Keep all four ZIPs and stage curves.
6. The notebook starts the skills checkpoint for inference. Confirm `/v1/models` and a legal player decision. Record the starting compute-unit balance for the class.

The [recipe table](README.md#the-five-training-stages) lists exact source counts, replay and selected published settings. General stages use the complete published curriculum and configurable 180-minute attempt caps. These caps are scheduling limits, not measured durations. General training belongs outside the 90-minute class. The mandatory 30-minute class block fine-tunes Pac-Man for two complete epochs on 64 accepted rows: 16 optimizer steps, with a 20-minute training attempt cap.

## Optimized execution and precision

Kev's frozen stack uses Torch 2.8.0/CUDA 12.8, Triton 3.4.0, Transformers 5.17.0 and PEFT. The additive overlay pins FLA/fla-core 0.5.2, causal-conv1d 1.7.0, einops 0.8.1 and Ninja 1.13.0. The CUDA wheel matches Python 3.13, Linux x86_64, Torch 2.8 and CXX11 ABI. FLA handles DeltaNet, CUDA handles short convolution, and SDPA handles full attention. BF16 autocast, FP32 stored parameters and fused AdamW are required. Kev inference fusion/CUDA graphs are disabled for training.

Default `TRAINING_PROFILE='memory_safe'`: batch 1 × accumulation 8, non-reentrant checkpointing, `row_budget=2048` in every stage. Source data, effective batch and schedules remain unchanged; microbatch/dropout execution and kernel arithmetic differ. The published-reference profile retains original execution flags but still requires optimized kernels. A row budget does not truncate a single longer question. Optional allocator `expandable_segments` cannot release actively used tensors.

This contract requires BF16 and rejects T4/P100 as drop-in training fallbacks. An explicitly allowed alternative GPU must satisfy the same software/kernel checks and receive its own full validation. Kaggle supports structured evaluation/rollouts but does not provide the Colab browser callback.

## Progress and interruption

TensorBoard runs in the notebook kernel while Kev trains in its locked environment. Separate `logs/<stage>/<attempt>` directories contain per-step events, CSV/JSONL, stdout/stderr and status. Curves include CE, learning rate, gradient norm, timing, records and peak/live allocated/reserved/free VRAM. `batches.jsonl` preserves token shapes and IDs before each forward pass. These are training curves, not fabricated validation results.

Snapshots save adapter/head, optimizer/scheduler/RNG and progress after step 1, every 100 steps or five minutes at optimizer boundaries, and the final step. The latest two complete saves remain under `<output>-recovery`. After interruption, keep identical arguments, data, parent and paths, set the affected `RESUME_* = True` and rerun that stage. Do not use Kev's native full-weight-only resume flag. Export recovery and logs after exceptions, or use Drive storage. Recovery ZIPs restore to the original absolute checkpoint root. Completed checkpoint ZIPs restore through `RESTORE_ARCHIVE`, `RESTORE_DATES_ARCHIVE`, `RESTORE_DOCUMENTS_ARCHIVE` and `RESTORE_SKILLS_ARCHIVE`, in order. Declare learner/instructor ownership.

The old 0.8B run failed at step 2,073 with missing FLA/convolution kernels and nearly all 95 GiB actively occupied. Its notebook saved weights only at completion, so that failed run has no automatic learned checkpoint. Inspect its old output directory before restarting; logs/configuration cannot reconstruct exited-process weights. Fresh 4B output names preserve those files.

## Before teaching

The local CPU tests do not establish CUDA compatibility or full-stage fit. **The optimized 4B CUDA preflight, full curriculum, CUDA continuation, reloads and live browser callback have not been run by the course author.** Complete them on the intended allocation, measure duration/compute-unit use and prepare compatible instructor checkpoints. Keep the 90-minute class and 30-minute fine-tuning block; report supplied results as instructor results.

Sources: [Colab FAQ](https://research.google.com/colaboratory/faq.html), [RTX PRO 6000 Server Edition](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/), [Kev-4B card](https://huggingface.co/jaredpalmer/kev-4b), [Qwen kernels](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5).
