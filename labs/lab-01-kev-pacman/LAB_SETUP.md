# 在实验室 RTX A6000 上运行 notebook

使用 [pacman_kev_lab_local.ipynb](notebooks/pacman_kev_lab_local.ipynb)。连接由你的**本地电脑通过 SSH**建立，下面的安装和训练命令在登录后的实验室终端运行。

此版本采用单卡 GPU 1，Stages 1–5 默认均为 batch 1 × accumulation 8、梯度检查点和 `row_budget=2048`。训练数据、上游代码和模型版本、优化内核哈希、有效 batch 8、学习率和阶段步数保留；Pac-Man 仍为 762 个 optimizer updates。微批次和分组改变数值执行。`row_budget` 不会截断一条更长的样本，因此这些设置还需要实际显存验证。

## 1. 本地登录，检查实验室资源

在**本地电脑**运行：

```bash
ssh -p 130 huixu@130.245.132.100
```

在**实验室终端**运行：

```bash
cd /chronos_data/huixu/QPlusLearning
git status --short
uname -m
free -h
df -h /chronos_data
nvidia-smi
```

确认当前 checkout 已包含 `labs/lab-01-kev-pacman/lab_runtime.py`、本地 notebook 和 `scripts/setup_lab_jupyter.sh`。保留现有修改，通过你平时的同步方式取得这些文件。不要直接重置 checkout 或覆盖已有 notebook。

按你最新提供的约 20 GB 显存占用，本版本采用一张 48 GB A6000，默认 GPU 1；开始运行前重新检查该卡占用。此占用来自你的反馈，我们尚未在实验室 GPU 上实测，也不代表每个阶段都固定使用 20 GB。保留各阶段强制显存检查，用实际报告核对。当前 Kev 使用单 GPU，不需要为这个运行配置增加第二张卡。驱动 `570.133.20` 的 `nvidia-smi` 显示 CUDA 12.8，实际训练 Python 仍验证 Torch/CUDA/BF16 和优化内核。

旧 0.8B 的约 90 GiB OOM 属于历史 batch 8、缺少 FLA/convolution 优化内核而走 reference 路径的配置，与当前约 20 GB 占用区别记录。新版本要求优化内核并采用 1×8；较长样本仍以阶段检查结果为准。显存累计增长后达到平台本身不能证明泄漏，需结合实际活跃/保留显存、样本长度和各步日志判断。

## 2. 创建 Python 3.13 的 Jupyter Conda 环境

两种环境分开安装在 `/chronos_data/conda_envs`：

| 用途 | 路径 | 管理方式 |
| --- | --- | --- |
| Notebook kernel、JupyterLab、TensorBoard | `/chronos_data/conda_envs/qplus-jupyter-py313` | Conda，Python 3.13 |
| 实际训练和推理 | `/chronos_data/conda_envs/qplus-kev-training-py313` | Notebook 自动创建的 uv venv，Python 3.13 |

训练环境虽然位于 `conda_envs` 目录内，**不是 Conda 环境**。不要提前用 `conda create` 创建这个训练路径，也不要让它指向 Jupyter 环境。`uv sync` 会按上游锁文件同步依赖；共用路径会影响 Jupyter 包。

在**实验室终端**运行：

```bash
bash scripts/setup_lab_jupyter.sh
```

脚本检查 Linux x86_64、Git、Conda、GPU 和路径权限；所有路径变量必须以 `/` 开头，不能含字面量 `~`。不存在的 kernel prefix 才会创建。已有 prefix 必须是 Python 3.13 的 Conda 环境，否则停止并保留它。Jupyter 包安装为：

```bash
python -m pip install uv jupyterlab ipykernel tensorboard==2.20.0
python -m ipykernel install --sys-prefix \
  --name qplus-a6000-py313 \
  --display-name 'QPlusLearning (A6000, Python 3.13)'
```

这些命令由脚本通过 `conda run --prefix ...` 在正确环境执行，不依赖 `conda activate` 的 shell 初始化。脚本不会创建训练 prefix；本地 runtime 只接受新的空路径或有匹配所有权记录的已有 uv 环境，拒绝未知的已有环境。

无需手动安装 Torch 或系统 CUDA Toolkit。实际 trainer 用上游锁定的 Torch 2.8.0/CUDA 12.8、Triton 3.4.0、Transformers 5.17.0 和 PEFT；CUDA runtime 由 wheels 提供。优化 overlay 保留 FLA/fla-core 0.5.2、causal-conv1d 1.7.0 的 `cp313` wheel 及校验哈希。所需系统 NVIDIA 驱动仍必须可用。

## 3. 启动 Jupyter，并从本地转发端口

安装脚本最后会打印完整的环境变量和启动命令，复制到**实验室终端**执行。默认配置也可使用：

```bash
export QPLUS_GPU=1
export QPLUS_JUPYTER_ENV=/chronos_data/conda_envs/qplus-jupyter-py313
export QPLUS_TRAINING_ENV=/chronos_data/conda_envs/qplus-kev-training-py313
export QPLUS_LAB_SOURCE=/chronos_data/huixu/QPlusLearning/labs/lab-01-kev-pacman
export QPLUS_LAB_DIR=/chronos_data/huixu/qpluslearning-runtime-a6000
export QPLUS_BACKUP_DIR="$QPLUS_LAB_DIR/backups"
export CUDA_VISIBLE_DEVICES="$QPLUS_GPU"

conda run --no-capture-output --prefix "$QPLUS_JUPYTER_ENV" \
  jupyter lab --no-browser --ip=127.0.0.1 --port=8888 \
  --ServerApp.port_retries=0 \
  --ServerApp.root_dir=/chronos_data/huixu/QPlusLearning
```

在**本地电脑的另一个终端**运行并保持连接：

```bash
ssh -p 130 -N \
  -L 127.0.0.1:8888:127.0.0.1:8888 \
  -L 127.0.0.1:6006:127.0.0.1:6006 \
  huixu@130.245.132.100
```

用本地浏览器打开实验室 Jupyter 输出的 `http://127.0.0.1:8888/lab?token=...`。保留 Jupyter 自带 token 认证。`port_retries=0` 在 8888 被占用时直接停止，避免服务悄悄换端口而 SSH 仍转发到 8888。打开 `labs/lab-01-kev-pacman/notebooks/pacman_kev_lab_local.ipynb`，选择 **QPlusLearning (A6000, Python 3.13)** kernel。

`QPLUS_LAB_SOURCE` 指向 lab 目录，包含 helper 和数据；`QPLUS_LAB_DIR` 是独立的持久运行目录。Notebook 从本地 checkout 复制并校验 helpers，避免从 GitHub 下载旧版课程 bootstrap。上游 Kev、模型和 wheel 仍首次联网下载并验证。可通过上述变量改路径；换 GPU 后重启 kernel，避免 CUDA 已初始化时改变可见设备。

## 4. 按 notebook 验证，再开始训练

首次按顺序运行 bootstrap、runtime setup、optimized preparation 和 storage。`runtime.prepare_training(run_training_preflight=True)` 是必需步骤，不能改成 `False` 或通过只检查 bindings 跳过训练验证。它运行两条真实 4B 样本的 loss/backward/fused AdamW，作为优化内核 smoke test；该检查有 20 分钟超时和进度诊断。首次安装、模型下载、编译和 autotuning 需要时间。

| 检查 | 证明什么 |
| --- | --- |
| `runtime.setup()` | 锁定环境安装、GPU 小矩阵 forward/backward、CUDA/BF16、优化 wheels 安装 |
| `runtime.prepare_training(True)` | 基础模型/训练数据、LoRA/head 审计、优化绑定及两条真实样本的内核 smoke test |
| 每个阶段的强制显存检查 | 按该阶段实际参数和数据选出的高 token 负载，实际梯度累积和两次 fused AdamW updates |
| 完整阶段和恢复/reload | 该阶段的实际显存峰值、耗时、全部 optimizer steps 和恢复行为 |

每次正式阶段训练前，runtime 会先释放正在 serving 的模型，再按该阶段的实际训练数据、replay、参数和 LoRA `warm_start` 检查显存。独立 helper 扫描各 epoch 实际 tokenized forward-pass shapes，纳入 eligible none-pair 的保守变体，选择观察到的最高 padded pass-token record。随后连续运行 8 个 microbatches，再执行 fused AdamW，重复两次 optimizer updates；这些验证不会生成正式训练 checkpoint。

检查报告保存到 `logs/<stage>/memory-check-*/memory-preflight.json`，包含 allocated/reserved/free/total 显存、选中 shape 及优化内核调用记录。显存检查 OOM、失败或超时时，runtime 不会启动该阶段完整训练。其 3,600 秒超时是停止上限，并非预计耗时。即使通过，也只证明选中负载在检查当时可运行，不证明全部 shape、GPU 后续共享占用或完整 curriculum 都适配 48 GB。

查看运行目录中的 `runtime-preflight.json`、`optimized-training-preflight.json`、各阶段 `memory-preflight.json` 和训练日志。缺少优化内核、出现 reference fallback 或任一 preflight 失败时，必须停止训练并修复依赖。两条样本的 smoke test 不能替代阶段显存检查。先检查空闲显存和当前 GPU 用户，再运行完整阶段；如 OOM，保留报告、日志、checkpoint 和 recovery，不要通过随意升级依赖或删减校验来继续。

没有已完成 Skills checkpoint 时，按 **Initial → Dates → Documents → Skills → Pac-Man** 运行。已有兼容的完整 Skills checkpoint 可在 setup/preparation/storage 后使用 notebook 的加载或恢复单元格，直接开始 CP1。模型下载本身不产生 Skills adapter。

Notebook 默认 `SHOW_TENSORBOARD=False`。在另一个**实验室终端**启动：

```bash
conda run --no-capture-output \
  --prefix /chronos_data/conda_envs/qplus-jupyter-py313 \
  tensorboard \
  --logdir /chronos_data/huixu/qpluslearning-runtime-a6000/logs \
  --host 127.0.0.1 --port 6006
```

本地浏览器打开 `http://127.0.0.1:6006`。命令默认使用上面的路径；若修改了 `QPLUS_LAB_DIR` 或 Jupyter prefix，同步修改 TensorBoard 参数。

## 5. 保存与恢复

Checkpoint、数据、训练日志（包括各阶段显存检查报告）和缓存保存在持久 `QPLUS_LAB_DIR`，实际 trainer venv 保存在 `QPLUS_TRAINING_ENV`。Notebook 在 `QPLUS_BACKUP_DIR` 自动生成本地备份，保留 optimizer、scheduler、RNG 和进度恢复信息。训练在 step 1、每 100 步或五分钟的 optimizer 边界、最终步保存 recovery，保留最近两个完整快照。

中断后保持路径和数据一致，重新运行相应阶段；Initial 使用 `RESUME_CHECKPOINT`，后续阶段自动查找参数匹配的 recovery。更换 batch/accum/row-budget 后，Initial 的受控恢复可记录执行变化，后续阶段要求恢复参数匹配。已完成输出会复用。不要删除非空输出目录来绕过恢复检查。

同一磁盘上的备份是本地副本；需要抵御磁盘故障时，再通过你已有的备份方式复制到其他存储。Notebook 可导出结构化 evaluation、rollouts 和提交材料；依赖 Colab 浏览器回调的交互模型游戏在普通 Jupyter 中跳过。

首次安装需能访问 GitHub（包括 releases）、PyPI、PyTorch wheel 源、Hugging Face，以及必要时 Node.js 下载站点。缺少网络或模型权限时，根据具体失败配置实验室网络/已有认证；不要把 token 写进 notebook 或在聊天中发送。

此修改仅做 CPU、语法和命令构造验证。实验室 GPU 安装、完整训练、显存峰值和耗时需要在上述机器执行后确认。
