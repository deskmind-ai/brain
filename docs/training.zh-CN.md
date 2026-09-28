# 训练 · [English](training.md)

## 流程

1. **问题和老师。** 带类型的问题（选择题、是非题、打分题）来自网页和表单任务、桌面任务、证据题，每道题由一个老师模型给出概率分布。我们的网页、表单和证据题数据是用云端前沿模型生成和标注的，这部分生成器不在本仓库里。如果要加这类数据，请用条款允许的老师模型自行生成，写成 `items_labeled.jsonl`（格式见 `deskmind_brain.train.data`）。`deskmind_brain.train.label_teacher` 可以用本地模型标注，并会拒绝使用托管的 System One 接口当老师。
2. **桌面 DAgger。** 让当前模型在 [deskmind-ai/hands](https://github.com/deskmind-ai/hands) 的沙箱任务上实际操作，由自动标注程序（oracle）给每个经过的状态标注正确答案。`scripts/import_hands_dagger.py` 把这些轨迹转成训练样本，并保留每个请求里没标注的问题，让训练时和实际运行时的提示格式一致。
3. **反例。** 有些界面元素容易让模型学到「标签捷径」，比如撤销按钮、已保存的文档、另开的第二个窗口。遇到这类元素，就把同一个状态配上能打破捷径的标签再加一份（见 `scripts/shortcut_counterexamples.py`，以及 `import_hands_dagger.py` 里的撤销相关选项）。
4. **蒸馏。** 在 Qwen3.5-0.8B 和 4B 上训练 LoRA，损失 = 对老师分布的 KL + 对标签的交叉熵。加上 `--shuffle-options` 时，每条样本的选项顺序随机打乱，让字母位置不携带答案信息。`round_size` 是每个提示里最多的选项数（52 = A..Z、a..z）：部署时，更长的选项列表分成几轮淘汰赛打分；训练时，更长的列表会截成包含正确答案的 `round_size` 个选项。
5. **部署。** 用 `scripts/merge_lora.py` 合并权重，再用 MLX 启动服务（`deskmind-brain-serve --two-stage`）。

第三方评测数据（公开的 Hugging Face 数据集、TypeSafe 公开评测集）在运行时从原始来源下载，本仓库不重新分发。

## 训练命令

在有 CUDA 的机器上运行（训练使用 `device_map="cuda"`；`--model` 默认是 Qwen/Qwen3.5-2B，请显式指定要用的基座）：

```bash
uv sync --extra local --extra train          # Linux 上可再加 --extra cuda，启用更快的线性注意力内核
uv run python -m deskmind_brain.train.train --model Qwen/Qwen3.5-4B \
  --data data/train/desktop:3000 --data data/train/counterexamples:800 --data data/train/control \
  --prompt-format 2 --shuffle-options --round-size 52 --compact-targets \
  --max-tokens 8192 --token-budget 8192 --batch-size 4 --lr 5e-5 --out runs/brain-4b
uv run python scripts/merge_lora.py runs/brain-4b/Qwen3.5-4B-lora --base Qwen/Qwen3.5-4B
uv run deskmind-brain-serve --predictor mlx:runs/brain-4b/Qwen3.5-4B-merged --port 8793 --two-stage   # 在 Mac 上运行
```

每个 `--data` 目录里放一个 `items_labeled.jsonl`；`dir:N` 表示从中抽 N 条。桌面数据和反例数据由上面的 DAgger 脚本和反例脚本生成。`data/train/control` 代表你自己准备的、由老师模型标注的数据（见第 1 步）。

## 每条都让我们多花了一轮训练的教训

- **提示的格式本身会泄露答案。** 如果某个标签总是以另一种提示格式出现，模型就会去认格式，而不是读页面状态。
- **自动标注只能标它「看得到」的东西。** 有一个输入框填好之后，界面上仍显示为空，结果每个模型都学会了把值再填一遍。
- **用真实失败时的请求来检测。** 重放真实失败时的原始请求，能把合成检测集分不出高下的版本区分开。
- **从同一批轨迹里留出的离线测试集发现不了捷径。** 只有在真实桌面上闭环运行才能发现。
