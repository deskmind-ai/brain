# Results · [中文](results.zh-CN.md)

All numbers are our own runs. The bench suite, graders and harness commit for each number are listed in
[deskmind-ai/bench](https://github.com/deskmind-ai/bench).

## Real desktop (bench suite v25, 13 tasks × 3 runs): current release

| config | strict pass | false "done" | decision p50 / p95 |
|---|---|---|---|
| DeskMind router G14 (0.8B → 4B, 8-bit, threshold 0.94), earlier release | 36/39 = 92% | 0 | 0.57 / 5.25 s |
| **DeskMind router G18b (0.8B → 4B, 8-bit, threshold 0.96), current** | **39/39 = 100%** | **0** | **2.85 / 9.82 s** |

- **G18b setup:** run through the DeskMind app (hands 694eb97, optional checks and notes off), 3 runs per task, no
  environment errors, no no-progress loops. Decision times are over 208 decisions.
- **Where the time goes:** steps the 0.8B answers itself take p50 0.48 s / p95 0.66 s; steps escalated to the 4B take
  p50 3.6 s / p95 9.8 s, and about 70% of steps escalate. The G18b 0.8B's confidences sit in a narrow band (about
  0.94–0.97), so a threshold that keeps it correct sends most steps to the 4B. Restoring a wider fast path is work for
  the next round.
- **Not perfect:** in all 3 runs of the Chinese exact-text task (G04) the file was right but the model never said
  "done"; each run used the full 20-step budget. The grader checks the final state, so these count as passes.
- **No Jev number on v25:** the Jev comparison below is on v23 only.

## Real desktop (bench suite v23, 13 tasks × 3 runs)

| config | strict pass | tasks not solved | false "done" | step p50 / p95 |
|---|---|---|---|---|
| Jev (TypeSafe API, hosted) | 33/38 = 87% | 1, plus 2 of 3 runs of a second | 2 | 0.36 / 0.44 s |
| DeskMind router, earlier checkpoint | 33/39 = 85% | 2 | 3 | 0.57 / 4.0 s |
| DeskMind router G14 (0.8B → 4B, 8-bit) | 35/38 = 92% | 1 | 0 | 0.59 / 4.6 s |

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
| DeskMind Brain 4B, G10b | 47 | 71 | 80 | 0.857 |
| DeskMind Brain 4B, G14 | 48 | 67 | 85 | 0.866 |
| **DeskMind Brain 4B, G18b (current)** | **48** | **69** | **76** | **0.835** |
| DeskMind Brain 0.8B, G14 | 48 | 54 | 61 | 0.706 |
| DeskMind Brain 0.8B, G18b (current) | 48 | 56 | 63 | 0.723 |
| DeskMind router G14 (0.8B → 4B) | | | | 0.790 |
| DeskMind router G18b (0.8B → 4B, threshold 0.96) | 48 | 65 | 71 | 0.797 |
| Jev 1.13 (board `public_accuracy`) | | | | 0.866 |

- These are public items only; the sealed set is scored by the benchmark maintainers.
- **G18b vs G14 on the 4B:** the hard tier drops from 85 to 76, mostly temporal/numeric, hard-judgement and multi-hop
  items. G18b's training mix leans further toward desktop states, and general judgement suffered; the release keeps
  G18b for its real-desktop reliability (above).
- **Calibration (G18b 4B):** Brier 0.269, ECE 0.089 (G14: 0.230, 0.079).
- **No overlap:** the G18b training mix (100,703 items) shares no word 13-gram and no option set with the 231 public
  items.

## Speed on an M4 Pro (48 GB)

- **Where the time goes:** prefill is compute-bound at about 850 tokens/s for the 4B. A desktop step is about 1.9k
  prompt tokens, so time scales with prompt length, not weight bytes.
- **Quantization does not help:**
  - 8-bit gives the same speed with 119/120 identical answers;
  - 4-bit is not faster and changes 30% of answers.
- **What made the G14 router fast (0.95 s → 0.59 s per step, p50, v23):**
  - the shared goal and rules come before the page state, so the prefix is reused across steps;
  - decorative elements are dropped from the prompt;
  - when both models agree a task is done, the 4B scores only the "done" answer instead of every field.
