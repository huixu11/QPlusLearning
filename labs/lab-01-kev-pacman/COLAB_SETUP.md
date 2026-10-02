# Colab setup for the Pac-Man lab

This session assumes **one NVIDIA RTX PRO 6000 Blackwell GPU in Colab**. The full Server Edition has 96 GB of GPU memory. Confirm the actual device and memory available to your notebook. An RTX 6000 Ada or an older RTX A6000 is a different GPU.

Arrange access before class. Colab offers premium GPUs subject to availability and varies allocations by account and demand. This plan does not assume the target GPU is free or guaranteed by a subscription. If the target is unavailable, use an instructor-validated fallback or the instructor's prepared checkpoint and clearly label any supplied results.

## Before the 90-minute session

1. Open [Lab 1 in Colab](https://colab.research.google.com/github/yxc20089/QPlusLearning/blob/main/labs/lab-01-kev-pacman/notebooks/pacman_kev_lab.ipynb) and save your own copy.
2. Open **Runtime > Change runtime type** and choose the RTX PRO 6000 Blackwell option if offered. Connect and inspect the actual device shown by `nvidia-smi`.
3. Run the runtime prework cells. They fetch hash-verified helpers/game source from a pinned course commit, install pinned Kev in a separate Python 3.13 environment, check the GPU, fetch verified decision-v7 training data and the pinned Qwen base, and audit trainable parameters.
4. Keep `runtime-preflight.json`. It records GPU name, actual VRAM, compute capability, PyTorch and CUDA versions, BF16 support, and whether a small GPU forward/backward pass succeeded.
5. Open the TensorBoard cell, then run **Stage 1 (initial), Stage 2 (dates/evidence), and Stage 3 (documents/skills)** as separate prework cells. Each warm-starts from its preceding checkpoint except Stage 1, which initializes fresh LoRA/head weights. Save all three backup ZIPs. The helper starts Stage 3 for inference after training. Confirm `/v1/models` responds and that a player decision names a legal direction. All three general training stages, downloads and installation are outside the class timetable.
6. Record your starting Colab compute-unit balance so you can measure the session's cost. Save outputs to your computer before disconnecting.

The notebook creates `pacman-initial-checkpoint.zip`, `pacman-dates-checkpoint.zip` and `pacman-skills-checkpoint.zip` in separate backup cells. Save each one. To restore after a disconnect, upload all three and use `RESTORE_ARCHIVE`, `RESTORE_DATES_ARCHIVE` and `RESTORE_SKILLS_ARCHIVE` in order. The notebook verifies full stage completion, recipe/data compatibility and parent checkpoint fingerprints, while recording learner/instructor ownership. If switching from an older notebook, save any completed checkpoint before starting a fresh kernel with the updated helpers.

The model server runs inside the notebook at `127.0.0.1:8009`. The browser game uses a Colab callback to ask that server for player moves. Ghosts follow fixed game code. No public endpoint or paid inference API key is required.

## Precision and training

The RTX PRO 6000 target uses **BF16 inference and BF16 training autocast**. Kev's initial recipe stores backbone, adapter and head parameters in FP32. Domain training preserves that precision. Installation uses Kev's locked PyTorch 2.8.0 / CUDA 12.8 environment, which supports Blackwell. It does not use whichever PyTorch happens to be installed in the Colab notebook kernel.

Initial training uses all 12,576 decision-v7 training records, two epochs, batch 8, accumulation 1, learning rate 1e-4, rank-16 LoRA and a fresh 256-dimensional pointer head. It retains upstream augmentation and OneCycleLR. It passes no `--init_from`. The published 20-minute initial run used an H100 and is not a timing estimate for this lab. The complete run belongs in prework and has a configurable 90-minute attempt cap.

Those batch settings describe the **published reference**. The notebook defaults to `memory_safe`: batch 1 × accumulation 8, gradient checkpointing and a 2,048 padded-token row budget on every stage. This retains the curriculum and effective batch, but microbatch weighting/dropout/row execution differ. Select `TRAINING_PROFILE='published_reference'` in setup for the original settings. A user's original reference run exhausted 94.97 GiB VRAM at step 2,073; Qwen's fallback DeltaNet operators and expanded question rows make training much larger than the stored 0.8B parameters alone. The revised profile needs a GPU rerun. Allocator `expandable_segments` is enabled when no allocator configuration was supplied; it cannot free active training intermediates.

During the mandatory 30-minute class block, fine-tune **your own documents/skills adapter/head** with `--init_from`, batch 1, accumulation 8, two epochs, learning rate 2e-5, gradient checkpointing and a 2,048-token state budget. With all 64 accepted training boards this gives 16 steps. Stop inference while training, then reload the checkpoint. Legal directions are the complete action set, so this adaptation disables none/distractor insertion while retaining option shuffling. Record actual duration and peak memory. The training attempt has a 20-minute cap.

Stage 2 adds 1,425 dates/missing-evidence records plus 2,000 replayed decision-v7 records for one epoch at lr 4e-5. Stage 3 adds 16,539 documents/skills records plus 6,000 replayed records for one epoch at lr 2e-5, batch 4 × accumulation 2, a 7,552-token state budget and gradient checkpointing. Both preserve upstream none-pair augmentation. Their default attempt caps are 90 minutes each, configurable after GPU measurements. Calibration is discussed separately and is not fitted or copied from the release. All four training stages update LoRA and the pointer head while original base matrices remain fixed.

## Live training curves

GPU telemetry includes live allocated, reserved and free VRAM. `batches.jsonl` records physical question rows, longest row, padded tokens and record IDs before each forward pass, including an OOM-triggering pass. Check these alongside the TensorBoard curves before attributing an OOM to gradual growth or a single large batch.

The setup cell installs `tensorboard==2.20.0` in the notebook kernel. Open its dashboard before Stage 1. Training runs in Kev's separate locked environment; telemetry is relayed back to the notebook and written to `logs/<stage>/<attempt>/tensorboard`, alongside raw logs, CSV and JSONL. The dashboard updates during training. Console heartbeats distinguish loading/data preparation from completed optimizer steps. Download the submission archive to retain curves after disconnecting.

## Periodic LoRA recovery

The revised helper saves at optimizer step 1, then every 100 steps or five minutes, and the final step; the time interval is checked at optimizer boundaries. It retains the latest two complete snapshots under `<output>-recovery`, including adapter/head exports, optimizer, scheduler, RNG and progress. Resume with the affected stage's `RESUME_* = True` and the same profile, arguments, inputs and output path. Do not pass Kev's native `--resume`, which is restricted to full-weight runs. Resume refuses a completed run or a run without a recovery snapshot.

Set `SAVE_TO_DRIVE=True` **before** training to persist new outputs/recovery on Drive. Local `/content` storage is temporary. The notebook's inspection/export cell can download recovery and logs after an exception; set `INSPECT_OUTPUT` to the failed stage. Restore a recovery archive into an absent checkpoint root at its original absolute path. Completed checkpoint backups retain the existing per-stage restore controls. Keep the parent checkpoints and identical training data when resuming a later stage.

The previous notebook had no periodic LoRA save. Its step-2,073 failure normally leaves `training_config.json` and logs, with no adapter/head weights from that run. An older completed or manually saved checkpoint would be separate. The new `*-v2` directories preserve the failed run's files. Use the new notebook's inspection cell to check the old directory before starting again.

## Instructor preparation

Run the full notebook on the intended Colab allocation before teaching. Confirm dependency installation, context length, positive optimizer steps, checkpoint reloads, paired evaluation and the live browser callback. Prepare a compatible checkpoint and recorded evaluation for allocation failures. The original initial run OOMed; completion with the revised profile, CUDA recovery and the live callback still need validation.

Keep the 90-minute session and the mandatory 30-minute fine-tuning block. Measure setup separately. Measure cost using actual session duration and compute units consumed rather than inventing a fixed Colab dollar rate.

A T4 fallback requires `CloudRuntime(LAB_DIR, allow_other_gpu=True)` before setup, uses FP32 and cannot run the published BF16 initial recipe unchanged. Use compatible prepared checkpoints for all three general stages, label that substitution, and time domain training independently. The current CUDA 12.8 environment rejects P100/Pascal GPUs, so Kaggle's P100 is not a drop-in fallback. Kaggle supports Python evaluation and rollouts but does not supply the Colab browser callback.

Sources: [Colab FAQ](https://research.google.com/colaboratory/faq.html), [RTX PRO 6000 Blackwell Server Edition](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/), [PyTorch Blackwell support](https://pytorch.org/blog/pytorch-2-7/), [PyTorch CUDA architecture support](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128), [Kev-0.8B model card](https://huggingface.co/jaredpalmer/kev-0.8b).
