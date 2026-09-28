# Results · [中文](results.zh-CN.md)

All numbers are our own runs. The bench suite, graders and harness commit for each number are listed in
[deskmind-ai/bench](https://github.com/deskmind-ai/bench).

## Real desktop (bench suite v23, 13 tasks × 3 runs)

| config | strict pass | tasks not solved | false "done" | step p50 / p95 |
|---|---|---|---|---|
| Jev (TypeSafe API, hosted) | 33/38 = 87% | 1, plus 2 of 3 runs of a second | 2 | 0.36 / 0.44 s |
| DeskMind router, earlier checkpoint | 33/39 = 85% | 2 | 3 | 0.57 / 4.0 s |
| **DeskMind router (0.8B → 4B, 8-bit), current** | **35/38 = 92%** | **1** | **0** | **0.59 / 4.6 s** |

- **Same setup for both:** the same suite and harness commit, 3 runs per task. Jev's step time is the hosted API's
  latency; ours is local inference on an M4 Pro.
- **One run per system was not scored:** the accessibility tree came back incomplete at the first step of the same
  task (an environment failure). A rerun of that task went 3/3 for both.
- **Where Jev loses runs:** it declares a text edit done before it is saved (2 of 3 runs), and it loops on the
  web-extraction task, which neither system solves.
- **Ambiguous goals:** when the goal matches more than one record, the model asks first, then writes the full row,
  saves and finishes.
- **Remaining failure:** the web-extraction task; the model types rows into the save dialog's file-name field.
- **Runs cluster by task:** almost every task passes 3/3 or 0/3, so read the table per task rather than per pooled run.

## JevBench v1.4.2, public set

These are the 231 items in the public JevBench repository, the same items as the board's `public_accuracy`
column. The board's headline JevBench Score (0–100; Jev 1.13: 63.3) blends in 308 sealed items plus calibration,
speed and cost, and its tier columns cover more items than the public repository holds, so our tier counts are not
comparable with the board's tier columns.

Run locally with the official `jevbench.cli` and the typesafe adapter against a local server:

| | easy (48) | original (72) | hard (111) | public (231) |
|---|---|---|---|---|
| DeskMind Brain 4B (earlier checkpoint) | 47 | 71 | 80 | 0.857 |
| **DeskMind Brain 4B (current)** | **48** | **67** | **85** | **0.866** |
| DeskMind Brain 0.8B | 48 | 54 | 61 | 0.706 |
| Jev 1.13 (board `public_accuracy`) | | | | 0.866 |

- These are public items only; the sealed set is scored by the benchmark maintainers.
- Our training data has no exact overlap with the 231 public items.

## Speed on an M4 Pro (48 GB)

- **Where the time goes:** prefill is compute-bound at about 850 tokens/s for the 4B. A desktop step is about 1.9k
  prompt tokens, so time scales with prompt length, not weight bytes.
- **Quantization does not help:**
  - 8-bit gives the same speed with 119/120 identical answers;
  - 4-bit is not faster and changes 30% of answers.
- **What made it fast (0.95 s → 0.59 s per step, p50):**
  - the shared goal and rules come before the page state, so the prefix is reused across steps;
  - decorative elements are dropped from the prompt;
  - when both models agree a task is done, the 4B scores only the "done" answer instead of every field.
