<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/banner-dark.svg">
    <img src="assets/brand/banner-light.svg" alt="得心 DeskMind — 得心，应手。" width="720">
  </picture>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-262B28"></a>
  <a href="https://huggingface.co/deskmind"><img alt="Models on Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20models-deskmind-C95536"></a>
  <img alt="Python 3.12" src="https://img.shields.io/badge/python-3.12-262B28">
  <img alt="MLX on Apple Silicon" src="https://img.shields.io/badge/MLX-Apple%20Silicon-262B28">
  <a href="docs/results.zh-CN.md"><img alt="bench v25: 39/39" src="https://img.shields.io/badge/bench%20v25-39%2F39-C95536"></a>
</p>

<p align="center">
  <b>DeskMind Brain · 得心</b>：电脑操作 agent 的每一步决策，带可信的把握程度，在你自己的 Mac 上运行。<br>
  <a href="README.md">English</a> · <a href="docs/results.zh-CN.md">成绩</a> · <a href="docs/training.zh-CN.md">训练</a>
</p>

---

**得心，应手。** 「得心」（DeskMind）取自「得心应手」：心里想到，手上就做到。它是一组开源项目，让 agent 在你的电脑上**看**懂屏幕、**想**好下一步、**做**到真实的桌面上，全程在本机完成。

本仓库是其中的 **Brain（决策）**：两个小模型，回答带类型的问题，给出的是概率而不是一段文字：
- 下一步做什么操作？
- 点哪个元素？
- 目标是不是已经达成？
- 要不要先问用户？

| | 仓库 | 分工 |
|---|---|---|
| 👁 | [deskmind-ai/eyes](https://github.com/deskmind-ai/eyes) | 在截图里找到要操作的目标 |
| 🧠 | **deskmind-ai/brain** | 决定下一步，并给出把握有多大 |
| ✋ | [deskmind-ai/hands](https://github.com/deskmind-ai/hands) | 在真实的 macOS 桌面上执行 |
| 📐 | [deskmind-ai/bench](https://github.com/deskmind-ai/bench) | 沙箱桌面任务和评分器，用来复现我们的成绩 |

## 特点

- **输入带类型的问题，输出校准过的概率。** 选择题、是非题、打分题都从答案字母的 logits 读出：只读 logits，不生成文本，也不解析文本，概率天然加起来等于 1。
- **本地优先。** 在 Apple Silicon 上用 MLX 运行，默认情况下屏幕内容不离开你的电脑。路由的升级层也可以指向远程服务（比如云端 API），这样做时，该层会看到发给它的请求。
- **接口可直接替换。** `POST /v1/systemone` 接收 `{state, questions}`、返回 `{answers}`，和 System One 类决策接口格式一致，已有客户端只需改服务地址。
- **两级路由。** 默认由 0.8B 回答。遇到没把握的步骤，或容易出大错的步骤（宣布完成、撤销、少见的快捷键），`scripts/router_serve.py` 会交给 4B 复核。

## 成绩（2026 年 10 月，发布版 G18b）

真实 macOS 桌面，13 个沙箱任务，每个跑 3 轮，按严格标准判定通过。测试集为 [deskmind-ai/bench](https://github.com/deskmind-ai/bench) v25，在 M4 Pro 上通过 DeskMind app 运行：

| | 通过 | 没做完就说完成 | 每次决策耗时（中位数） |
|---|---|---|---|
| **DeskMind 路由 G18b**（0.8B → 4B，8 位，门槛 0.96） | **39/39** | **0** | 0.8B 直接回答 0.48 秒，交给 4B 复核 3.6 秒（约 70% 的步骤） |
| DeskMind 路由 G14（上一版） | 36/39 | 0 | 0.57 秒 |

在较早的 v23 测试集上，G14 路由通过 35/38（92%），Jev（TypeSafe，云端）33/38（87%）；v25 上没有 Jev 的成绩。JevBench v1.4.2 的 231 道公开题（即榜单的 `public_accuracy` 一列）：G18b 4B **0.835**，G14 4B 0.866。榜单主分（JevBench Score，0–100）还计入密封题、校准、速度和成本，我们还没有密封题成绩。详情见 [docs/results.zh-CN.md](docs/results.zh-CN.md)。

**已知不足：**
- **速度：** G18b 的 0.8B 把握集中在一个窄段，大部分步骤交给了 4B，一次决策通常约 3 秒；下一轮的目标是把快速路径的比例拿回来。
- **通用判断：** G18b 的 4B 在 JevBench hard 档退步（111 道中 85 → 76），换来了真实桌面上的提升。
- **宣布完成：** 有一个任务（中文精确文本）文件已经写对，模型却一直没有宣布完成，直到用完步数。

方法和完整数据见 [docs/results.zh-CN.md](docs/results.zh-CN.md)。

## 快速上手

```bash
uv sync --extra mlx
uv run hf download deskmind/brain-4b --revision g18b-q8 --local-dir models/brain-4b
uv run deskmind-brain-serve --predictor mlx:models/brain-4b --port 8793 --two-stage
curl -s localhost:8793/v1/systemone -H 'Content-Type: application/json' -d @examples/request.json
```

`examples/request.json` 是一个沙箱 Finder 任务里的真实一步：页面状态，加上操作及其目标的各个问题。

4B 下载约 4.2 GB，0.8B 约 0.8 GB。如果 `hf download` 报 `CAS Client Error`（Xet 传输通道出错），在命令前加
`HF_HUB_DISABLE_XET=1` 重试。

两级路由，一个进程搞定（0.8B 先答，有风险或没把握的步骤交给 4B）：

```bash
uv run hf download deskmind/brain-0.8b --revision g18b-q8 --local-dir models/brain-0.8b
uv run deskmind-brain-serve --predictor mlx:models/brain-0.8b --escalate-to mlx:models/brain-4b \
  --two-stage --port 8796
```

路由门槛随权重一起发布（0.8B 的 `deskmind.json` 里的 `router_threshold`，G18b 为 0.96）；`--threshold` 可以覆盖。

每个回复都带一条 `routing` 记录，说明是谁答的、为什么。两级也可以拆成两个服务分开跑，0.8B 在 8794 端口，4B 在 8793 端口：`uv run python scripts/router_serve.py --fast http://127.0.0.1:8794 --strong http://127.0.0.1:8793 --keep-done-over-undo --port 8796`。

## 模型

| 模型 | 基座 | 许可 |
|---|---|---|
| `deskmind/brain-0.8b` | Qwen3.5-0.8B + LoRA（已合并） | Apache-2.0 |
| `deskmind/brain-4b` | Qwen3.5-4B + LoRA（已合并） | Apache-2.0 |

每个模型目录里都有一个 `deskmind.json`，记录训练时用的提示格式。

## 训练

见 [docs/training.zh-CN.md](docs/training.zh-CN.md)，内容包括：
- 带自动标注的桌面 DAgger 数据采集；
- 针对「标签捷径」的反例；
- 对齐老师概率分布的 LoRA 蒸馏；
- 提示格式和两段式推理。

## 品牌

方框是你的桌面，橙色的点是一个稳稳落下的决定。吉祥物叫**小方**，是那个方框活了过来。素材在 [`assets/brand`](assets/brand)。

<p>
  <img src="assets/brand/xiaofang-idle.svg" height="72" alt="小方·静候">
  <img src="assets/brand/xiaofang-think.svg" height="72" alt="小方·思考">
  <img src="assets/brand/xiaofang-working.svg" height="72" alt="小方·执行">
  <img src="assets/brand/xiaofang-done.svg" height="72" alt="小方·完成">
</p>

## 许可

代码和模型权重采用 Apache-2.0。DeskMind、得心这两个名称，以及 logo 和小方，不在代码许可范围内：可以用来指代本项目，但不能修改后使用，也不能暗示我们为别的产品背书。
