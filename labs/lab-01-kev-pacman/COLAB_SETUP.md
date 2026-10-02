# Colab setup for the Pac-Man lab

This session assumes **one NVIDIA RTX PRO 6000 Blackwell GPU in Colab**. The full Server Edition has 96 GB of GPU memory. Confirm the actual device and memory available to your notebook. An RTX 6000 Ada or an older RTX A6000 is a different GPU.

Arrange access before class. Colab offers premium GPUs subject to availability and varies allocations by account and demand. This plan does not assume the target GPU is free or guaranteed by a subscription. If the target is unavailable, use an instructor-validated fallback or the instructor's prepared checkpoint and clearly label any supplied results.

## Before the 90-minute session

1. Open [Lab 1 in Colab](https://colab.research.google.com/github/yxc20089/QPlusLearning/blob/main/labs/lab-01-kev-pacman/notebooks/pacman_kev_lab.ipynb) and save your own copy.
2. Open **Runtime > Change runtime type** and choose the RTX PRO 6000 Blackwell option if offered. Connect and inspect the actual device shown by `nvidia-smi`.
3. Run the runtime prework cells. They unpack the game and teaching helpers, install pinned Kev in a separate Python 3.13 environment, check the GPU, fetch verified decision-v7 training data and the pinned Qwen base, and audit trainable parameters.
4. Keep `runtime-preflight.json`. It records GPU name, actual VRAM, compute capability, PyTorch and CUDA versions, BF16 support, and whether a small GPU forward/backward pass succeeded.
5. Run the full initial-training cell before class. It trains a fresh adapter/head over the pretrained Qwen base using the published decision-v7 recipe. Keep this checkpoint. The helper starts it for inference after training. Confirm `/v1/models` responds and that a player decision names a legal direction. Initial training, downloads and installation are outside the class timetable.
6. Record your starting Colab compute-unit balance so you can measure the session's cost. Save outputs to your computer before disconnecting.

The notebook downloads `pacman-initial-checkpoint.zip` after initial training. Keep this backup. If a temporary runtime disconnects before class, upload that ZIP through the Colab Files sidebar and set `RESTORE_ARCHIVE` in the initial-training cell to its path. The notebook verifies that its configuration matches the published base-stage recipe and records whether you restored your own checkpoint or an instructor fallback.

The model server runs inside the notebook at `127.0.0.1:8009`. The browser game uses a Colab callback to ask that server for player moves. Ghosts follow fixed game code. No public endpoint or paid inference API key is required.

## Precision and training

The RTX PRO 6000 target uses **BF16 inference and BF16 training autocast**. Kev's initial recipe stores backbone, adapter and head parameters in FP32. Domain training preserves that precision. Installation uses Kev's locked PyTorch 2.8.0 / CUDA 12.8 environment, which supports Blackwell. It does not use whichever PyTorch happens to be installed in the Colab notebook kernel.

Initial training uses all 12,576 decision-v7 training records, two epochs, batch 8, accumulation 1, learning rate 1e-4, rank-16 LoRA and a fresh 256-dimensional pointer head. It retains upstream augmentation and OneCycleLR. It passes no `--init_from`. The published 20-minute initial run used an H100 and is not a timing estimate for this lab. The complete run belongs in prework and has a configurable 90-minute attempt cap.

During the mandatory 30-minute class block, fine-tune **your own initial adapter/head** with `--init_from`, batch 1, accumulation 8, two epochs, learning rate 2e-5, gradient checkpointing and a 2,048-token state budget. With all 64 accepted training boards this gives 16 steps. Stop inference while training, then reload the checkpoint. Legal directions are the complete action set, so this adaptation disables none/distractor insertion while retaining option shuffling. Record actual duration and peak memory. The training attempt has a 20-minute cap.

The lab reproduces Kev's published base stage and adds Pac-Man adaptation. The released model's later dates, documents/skills and calibration stages are outside this exercise. Both lab stages update LoRA and the pointer head, changing effective encoder features while original base matrices stay fixed.

## Instructor preparation

Run the full notebook on the intended Colab allocation before teaching. Confirm dependency installation, context length, positive optimizer steps, checkpoint reloads, paired evaluation and the live browser callback. Prepare a compatible checkpoint and recorded evaluation for allocation failures. GPU execution and the live callback have not yet been tested for this draft.

Keep the 90-minute session and the mandatory 30-minute fine-tuning block. Measure setup separately. Measure cost using actual session duration and compute units consumed rather than inventing a fixed Colab dollar rate.

A T4 fallback requires `CloudRuntime(LAB_DIR, allow_other_gpu=True)` before setup, uses FP32 and cannot run the published BF16 initial recipe unchanged. Use a compatible prepared initial checkpoint, label that substitution, and time domain training independently. The current CUDA 12.8 environment rejects P100/Pascal GPUs, so Kaggle's P100 is not a drop-in fallback. Kaggle supports Python evaluation and rollouts but does not supply the Colab browser callback.

Sources: [Colab FAQ](https://research.google.com/colaboratory/faq.html), [RTX PRO 6000 Blackwell Server Edition](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/), [PyTorch Blackwell support](https://pytorch.org/blog/pytorch-2-7/), [PyTorch CUDA architecture support](https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128), [Kev-0.8B model card](https://huggingface.co/jaredpalmer/kev-0.8b).
