<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/banner-dark.svg">
    <img src="assets/brand/banner-light.svg" alt="DeskMind 得心 — 得心，应手。" width="720">
  </picture>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-262B28"></a>
  <a href="https://huggingface.co/deskmind"><img alt="Models on Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20models-deskmind-C95536"></a>
  <img alt="Python 3.12" src="https://img.shields.io/badge/python-3.12-262B28">
  <img alt="MLX on Apple Silicon" src="https://img.shields.io/badge/MLX-Apple%20Silicon-262B28">
  <a href="docs/results.md"><img alt="bench v23: 92%" src="https://img.shields.io/badge/bench%20v23-92%25-C95536"></a>
</p>

<p align="center">
  <b>DeskMind Brain</b> · Typed, calibrated step decisions for computer-use agents, running on your own Mac.<br>
  <a href="README.zh-CN.md">中文</a> · <a href="docs/results.md">Results</a> · <a href="docs/training.md">Training</a>
</p>

---

**DeskMind · 得心** — *得心，应手。* (from 得心应手: what the mind decides, the hand carries out) is a family of open-source
projects that let an agent **see** your screen, **decide** the next step, and **act** on the real desktop, all locally.

This repository is the **Brain**. It is a pair of small models that answer typed questions and return probabilities
instead of free text:
- *Which operation next?*
- *Which element?*
- *Is the goal already met?*
- *Should I ask the user first?*

| | Repository | Role |
|---|---|---|
| 👁 | [deskmind-ai/eyes](https://github.com/deskmind-ai/eyes) | find the target on a screenshot |
| 🧠 | **deskmind-ai/brain** | decide the next step, with calibrated confidence |
| ✋ | [deskmind-ai/hands](https://github.com/deskmind-ai/hands) | drive the real macOS desktop |
| 📐 | [deskmind-ai/bench](https://github.com/deskmind-ai/bench) | sandbox desktop tasks and graders to reproduce our numbers |

## What it does

- **Typed questions in, calibrated probabilities out.** Choice, yes/no and score questions are read from answer-letter
  logits only, no decoding: nothing is generated or parsed, and the probabilities sum to 1 by construction.
- **Local first.** Serving uses MLX on Apple Silicon, and by default your screen never leaves your machine. The
  router's escalation tier can be pointed at a remote server (for example a cloud API); if you do that, that tier sees
  the requests it is sent.
- **Drop-in API.** `POST /v1/systemone` takes `{state, questions}` and returns `{answers}`, the same shape as System
  One–style decision APIs, so an existing client only changes its base URL.
- **Two tiers.** The 0.8B model answers by default. When it is unsure, or when a step is costly to get wrong (finishing
  a task, undoing work, unusual shortcuts), `scripts/router_serve.py` hands the step to the 4B model.

## Results (September 2026)

Real macOS desktop, 13 sandbox tasks × 3 runs, strict pass. The suite is
[deskmind-ai/bench](https://github.com/deskmind-ai/bench) v23.

| | pass | false "done" | time per step (p50) |
|---|---|---|---|
| Jev (TypeSafe, cloud) | 87% | 2 | 0.36 s |
| **DeskMind router** (0.8B → 4B, 8-bit, M4 Pro) | **92%** | **0** | **0.59 s** |

JevBench v1.4.2: on the 231 public items (the board's `public_accuracy` column), DeskMind Brain 4B scores **0.866**,
the same as Jev 1.13. The board's headline JevBench Score (0–100) also weighs sealed items, calibration, speed and cost;
we have not been scored on the sealed set yet.

**Known gaps:**
- **One unsolved task:** extracting rows from a web page into a new document (Jev does not solve it either); the
  model types the rows into the save dialog's file-name field.
- **Slow steps:** p95 is about 4.6 s, on steps that confirm a task is done.

Method and full tables: [docs/results.md](docs/results.md).

## Quick start

```bash
uv sync --extra mlx
uv run hf download deskmind/brain-4b --local-dir models/brain-4b
uv run deskmind-brain-serve --predictor mlx:models/brain-4b --port 8793 --two-stage
curl -s localhost:8793/v1/systemone -H 'Content-Type: application/json' -d @examples/request.json
```

`examples/request.json` is a real step from a sandbox Finder task: the page state plus the questions for the operation
and its targets.

Two-tier serving in one process (the 0.8B answers, and risky or unsure steps go to the 4B):

```bash
uv run hf download deskmind/brain-0.8b --local-dir models/brain-0.8b
uv run deskmind-brain-serve --predictor mlx:models/brain-0.8b --escalate-to mlx:models/brain-4b \
  --two-stage --port 8796
```

Each reply carries a `routing` record: who answered, and why. The tiers can also run as separate servers, with the 0.8B
on 8794 and the 4B on 8793: `uv run python scripts/router_serve.py --fast http://127.0.0.1:8794 --strong http://127.0.0.1:8793
--keep-done-over-undo --port 8796`.

## Models

| Model | Base | Licence |
|---|---|---|
| `deskmind/brain-0.8b` | Qwen3.5-0.8B + LoRA (merged) | Apache-2.0 |
| `deskmind/brain-4b` | Qwen3.5-4B + LoRA (merged) | Apache-2.0 |

Each model directory carries a `deskmind.json` with the prompt format it was trained with.

## Training

See [docs/training.md](docs/training.md). It covers:
- desktop DAgger with an oracle;
- counterexamples for label shortcuts;
- LoRA distillation with KL to teacher distributions;
- the prompt format and two-stage serving.

## Brand

The frame is your desk. The orange point is a decision, placed carefully. The mascot is **Xiaofang (小方)**, the frame
come to life. Assets are in [`assets/brand`](assets/brand).

<p>
  <img src="assets/brand/xiaofang-idle.svg" height="72" alt="Xiaofang idle">
  <img src="assets/brand/xiaofang-think.svg" height="72" alt="Xiaofang thinking">
  <img src="assets/brand/xiaofang-working.svg" height="72" alt="Xiaofang working">
  <img src="assets/brand/xiaofang-done.svg" height="72" alt="Xiaofang done">
</p>

## License

Code and weights are licensed Apache-2.0. The DeskMind name, 得心, the logo and Xiaofang are not covered by the code
licence. You may use them to refer to the project, but not in modified form or to imply endorsement.
