# Contributing to DeskMind Brain · [中文](CONTRIBUTING.zh-CN.md)

Thanks for helping. DeskMind Brain is small, and every change is judged the same way: does it make step decisions more
accurate, better calibrated, or faster on a real desktop? The org-wide guidelines
([deskmind-ai/.github](https://github.com/deskmind-ai/.github)) apply here too.

## Set up

```bash
git clone https://github.com/deskmind-ai/brain && cd brain
uv sync --extra mlx          # Apple Silicon; serving and evals
uv run --group dev pytest -q # must pass before you open a PR
```

- Training needs a CUDA machine: `uv sync --extra local --extra train`. See [docs/training.md](docs/training.md).
- You don't need a GPU or the models to work on most issues. The tests use a stub server.

## Where to start

- Issues labelled [`good first issue`](https://github.com/deskmind-ai/brain/labels/good%20first%20issue) are scoped to one
  file or one command, and each says what "done" looks like.
- Unsure where a change belongs? Open a Discussion first. Eyes (grounding), Hands (desktop driver) and Bench (tasks and
  graders) live in their own repositories.

## Pull requests

- **One change per PR,** with a short description of what changed and how you checked it.
- **Behaviour changes need evidence.** Run the probe or benchmark that covers the change and paste the before/after
  numbers: `scripts/probe_ops.py`, `deskmind-brain-eval score`, or a bench run. A change that helps one suite and hurts
  another is still useful, as long as the PR says so.
- **Keep the API stable.** `POST /v1/systemone` takes `{state, questions}` and returns `{answers}`, and clients depend on
  that shape. Extend it compatibly.
- **Follow the prompt format.** A change to how prompts are rendered needs a new `prompt_format` number, recorded in
  `deskmind.json`, because existing models are tied to the format they were trained with.
- **Style.** Match the surrounding code: small functions, comments that explain why, no new frameworks. Run `pytest`
  before pushing.

## Data and privacy

- Never commit real user data: screenshots, page text or documents from your own apps and accounts. Keep that kind of
  data under a local `private/` directory, which is git-ignored.
- Training and evaluation data must have a source and licence that allow redistribution. Say where new data came from.
- Don't add outputs of commercial models as training labels without checking their terms.

## Reporting problems

- **Bugs:** give the command, the model, and a minimal request body that reproduces the problem, with anything
  personal removed.
- **Security issues:** don't open a public issue. See [SECURITY.md](https://github.com/deskmind-ai/.github/blob/main/SECURITY.md).

By contributing you agree that your contribution is licensed under Apache-2.0, the licence of this repository.
