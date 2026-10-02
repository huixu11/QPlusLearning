# Q+Learning

Hands-on labs for learning how language models make decisions, how to train them, and how to evaluate their behavior.

| Lab | What you build | Session |
| --- | --- | --- |
| [Lab 1 — Train a Decision Model to Play Pac-Man](labs/lab-01-kev-pacman/README.md) | A Kev-0.8B player using LoRA and a pointer head | 90 minutes, including 30 minutes of fine-tuning; initial training is prework |

[![Open Lab 1 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yxc20089/QPlusLearning/blob/main/labs/lab-01-kev-pacman/notebooks/pacman_kev_lab.ipynb)

Lab 1 includes a self-contained notebook, setup instructions, a community browser-game adaptation, labelled starter data, checkpoints for learner progress, and [slides](labs/lab-01-kev-pacman/slides/pacman-lab.pdf). The slides compare Kev, CLM, llama.cpp and SGLang before introducing the exercise.

The lab targets Colab with an NVIDIA RTX PRO 6000 Blackwell GPU. Arrange access before class; that allocation is not guaranteed or assumed free. CPU and game checks are included. GPU training, timing and the live Colab callback still require an instructor validation run; see [setup](labs/lab-01-kev-pacman/COLAB_SETUP.md).

Course code and content use the [MIT license](LICENSE), with third-party materials covered by their [original licenses](labs/lab-01-kev-pacman/vendor/README.md).
