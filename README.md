# Q+Learning

Hands-on labs for learning how language models make decisions, how to train them, and how to evaluate their behavior.

| Lab | What you build | Session |
| --- | --- | --- |
| [Lab 1 — Train a Decision Model to Play Pac-Man](labs/lab-01-kev-pacman/README.md) | A Kev-4B player using LoRA and a pointer head | 90 minutes, including 30 minutes of fine-tuning; four general training stages are prework |

[![Open Lab 1 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yxc20089/QPlusLearning/blob/main/labs/lab-01-kev-pacman/notebooks/pacman_kev_lab.ipynb)

Lab 1 includes a notebook with separate initial, intermediate and domain training stages, live TensorBoard curves, periodic LoRA recovery, setup instructions, a community browser-game adaptation, planning-labelled data, and [15 lab slides](labs/lab-01-kev-pacman/slides/pacman-lab.pdf). The notebook fetches verified helper files from a pinned course commit. The slides compare Kev, CLM, llama.cpp and SGLang, link the existing [CLM animation and maths lecture](https://docs.google.com/presentation/d/1sdnPkV6VyTW9tr6Xmtyyxns4UHlUGoTtTLj-vmxNhfo/edit), and guide the five training stages into one assessed CP1. Interactive play remains ungraded.

The lab targets Colab with an NVIDIA RTX PRO 6000 Blackwell GPU. Arrange access before class; that allocation is not guaranteed or assumed free. CPU and game checks are included. The original batch-8 initial run OOMed on a learner's 94.97 GiB GPU. The optimized 4B kernel preflight, full memory profile, CUDA recovery, timing and live Colab callback still require a validation run; see [setup](labs/lab-01-kev-pacman/COLAB_SETUP.md).

Course code and content use the [MIT license](LICENSE), with third-party materials covered by their [original licenses](labs/lab-01-kev-pacman/vendor/README.md).
