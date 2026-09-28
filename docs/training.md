# Training · [中文](training.zh-CN.md)

## Pipeline

1. **Questions and teachers.** Typed questions (choice, yes/no, score) come from web and form tasks, desktop tasks and
   evidence questions, and a teacher gives each one a probability distribution. Our web, form and evidence data was
   generated and labelled with hosted frontier models; those generators are not part of this repository. To add such
   data, write it as `items_labeled.jsonl` (the format `deskmind_brain.train.data` reads) with any teacher whose terms
   allow it. `deskmind_brain.train.label_teacher` labels items with a local model, and refuses the hosted System One
   API.
2. **Desktop DAgger.** The current model drives the sandbox tasks in
   [deskmind-ai/hands](https://github.com/deskmind-ai/hands), and an oracle labels every visited state.
   `scripts/import_hands_dagger.py` turns the traces into items. It keeps each request's unlabelled questions so the
   prompt has the same shape at training and serving time.
3. **Counterexamples.** Some UI elements make a label shortcut easy to learn: an undo button, a saved document, a
   second window. For these, the same state is added with a label that breaks the shortcut
   (`scripts/shortcut_counterexamples.py`, and the undo options in `import_hands_dagger.py`).
4. **Distillation.** LoRA on Qwen3.5-0.8B and 4B. The loss mixes KL to the teacher distribution with cross-entropy to
   the label. With `--shuffle-options`, answer options are shuffled per example so letter position carries no signal.
   `round_size` is the maximum number of options per prompt (52 = A..Z, a..z): at serving time a longer option list is
   scored in tournament rounds, and in training a longer list is cut to `round_size` options that include the gold.
5. **Serving.** Merge the adapter (`scripts/merge_lora.py`), then serve with MLX (`deskmind-brain-serve --two-stage`).

Third-party evaluation data (public Hugging Face datasets, the TypeSafe public set) is downloaded from its source at
runtime and is not redistributed in this repository.

## Training command

On a CUDA machine (training uses `device_map="cuda"`; the default `--model` is Qwen/Qwen3.5-2B, so pass the base you
want):

```bash
uv sync --extra local --extra train          # add --extra cuda on Linux for the fast linear-attention kernels
uv run python -m deskmind_brain.train.train --model Qwen/Qwen3.5-4B \
  --data data/train/desktop:3000 --data data/train/counterexamples:800 --data data/train/control \
  --prompt-format 2 --shuffle-options --round-size 52 --compact-targets \
  --max-tokens 8192 --token-budget 8192 --batch-size 4 --lr 5e-5 --out runs/brain-4b
uv run python scripts/merge_lora.py runs/brain-4b/Qwen3.5-4B-lora --base Qwen/Qwen3.5-4B
uv run deskmind-brain-serve --predictor mlx:runs/brain-4b/Qwen3.5-4B-merged --port 8793 --two-stage   # on the Mac
```

Each `--data` directory holds an `items_labeled.jsonl`; `dir:N` samples N items from it. The desktop and
counterexample data come from the DAgger and counterexample scripts above; `data/train/control` stands for any
teacher-labelled data you bring (step 1).

## Lessons that cost us a training round each

- **Prompt shape can leak the label.** If one label always arrives in a different prompt layout, the model reads the
  layout instead of the state.
- **The oracle can only label what the harness shows.** A field that looks empty after a successful fill taught every
  model to type the value again.
- **Probe with real failing requests.** Replaying the exact requests from real failures separates checkpoints that
  synthetic probes rate as equal.
- **Offline holdouts from the same trajectories cannot catch shortcuts.** Only closed-loop runs on the real desktop
  can.
