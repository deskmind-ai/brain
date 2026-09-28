# 为得心 Brain 做贡献 · [English](CONTRIBUTING.md)

感谢参与！得心 Brain 是个小项目，所有改动都用同一个标准衡量：在真实桌面上，每一步的决策有没有更准、把握程度更可信，或者更快？组织层面的通用规范（[deskmind-ai/.github](https://github.com/deskmind-ai/.github)）在这里同样适用。

## 环境

```bash
git clone https://github.com/deskmind-ai/brain && cd brain
uv sync --extra mlx          # Apple Silicon：本地运行和评测
uv run --group dev pytest -q # 提 PR 前必须通过
```

- 训练需要 CUDA 机器：`uv sync --extra local --extra train`，见 [docs/training.zh-CN.md](docs/training.zh-CN.md)。
- 大多数 issue 不需要 GPU，也不需要下载模型，测试用的是模拟服务。

## 从哪里开始

- 带 [`good first issue`](https://github.com/deskmind-ai/brain/labels/good%20first%20issue) 标签的 issue 都只涉及一个文件或一个命令，并写明了怎样算完成。
- 不确定改动该放在哪个仓库，就先在 Discussions 里问。Eyes（视觉定位）、Hands（桌面驱动）、Bench（任务和评分）各有自己的仓库。

## 提交 PR

- **一个 PR 只做一件事**，简单说明改了什么、怎么验证的。
- **改变模型行为要附证据。** 跑一下相关的检测集或评测，贴出改动前后的数字，可以用 `scripts/probe_ops.py`、`deskmind-brain-eval score` 或 bench。一个改动让某个测试集变好、另一个变差，也有价值，只要在 PR 里写清楚。
- **保持接口稳定。** `POST /v1/systemone` 接收 `{state, questions}`、返回 `{answers}`，已有客户端依赖这个格式；扩展时要保持兼容。
- **遵守提示格式。** 改动提示的渲染方式，必须新开一个 `prompt_format` 编号并写进 `deskmind.json`，因为已有模型只认它训练时用的格式。
- **代码风格：** 跟周围代码保持一致，函数小，注释写「为什么」，不引入新框架。推送前跑一遍 `pytest`。

## 数据与隐私

- 不要提交真实用户数据：你自己的应用和账号里的截图、页面文字、文档都不行。这类数据放在本地的 `private/` 目录里，这个目录不会提交到 git。
- 训练和评测数据必须有允许再分发的来源和许可。新增数据时写明出处。
- 用商业模型的输出当训练标签之前，先确认对方的使用条款。

## 反馈问题

- **Bug：** 附上命令、模型，以及一个能复现问题的最小请求（去掉个人信息）。
- **安全问题：** 不要公开提 issue，见 [SECURITY.md](https://github.com/deskmind-ai/.github/blob/main/SECURITY.md)。

提交贡献即表示你同意贡献内容按本仓库的 Apache-2.0 许可发布。
