# Decision replay benchmark

A reproducible speed baseline for Brain that cannot be bought with wrong answers ([deskmind#16](https://github.com/deskmind-ai/deskmind/issues/16)).
Labelled single decisions are replayed through the server exactly as the Mac app runs it: the 0.8B answers, steps the
router sends up go to the 4B, two-stage scoring, threshold 0.96. Every request records where its time went, and each
answer is checked against gold labels. An optimisation is kept only if it is faster **and** passes every gate against
the baseline.

## Run it

```bash
# both tiers as the app installs them (any mlx:<dir> works; pin the revisions below)
uv run --extra mlx python scripts/replay_requests.py --manifest fixtures/replay/v1/manifest.json \
    --fast mlx:models/brain-0.8b --strong mlx:models/brain-4b --threshold 0.96 \
    --baseline fixtures/replay/v1/baseline-g18b.json --out runs/replay/<label>
```

Writes `cold.jsonl` / `warm.jsonl` (one row per request), `summary.json` and `report.md`. Without
`$DESKMIND_REPLAY_PRIVATE` the private sets are skipped and the report says so; compare a run only with a baseline over
the same fixtures (`--baseline` fails every gate whose denominator differs). `--split dev` is for tuning; report the
holdout split for a decision. `--limit N` is a smoke run.

## Fixtures (v1, frozen)

| set | n | visibility | source |
|---|---|---|---|
| gym | 150 | public | G19 held-out tasks in hands' synthetic gym apps (expense, mail, mail+music, music, settings): runs of up to 8 consecutive steps per task, gym-oracle labels |
| pairs | 42 | public | done/not-done variants of one gym music step (14 steps x 3) |
| large | 30 | public | gym mail list clicks padded with synthetic rows to 100-240 options; gold row untouched |
| ambiguity | 70 | private | generated (seed 1016): two matching records, so the first move is ASK; unambiguous twins must not ask |
| text | 30 | private | multi-row copy into a text file: the exact row to append |

- **Labels are independent:** the gym oracle or the generator's rule, never a model.
- **Split:** dev/holdout by task run (sha256 of the session id), never by adjacent frames. 198 dev, 124 holdout.
- **Frozen:** `manifest.json` records each file's count and sha256, and the loader refuses changed files.
  `scripts/build_replay_fixtures.py` rebuilds them byte for byte from DeskMind's internal sets.
- **Private sets:** both are built from real harness request skeletons (structure only, every task string generated),
  so they are not published. The ambiguity items share no state with the ambiguity training sets.
- **Do not train on these.**
- **Categories** (a decision can have several): long context (state over 5,200 characters, the top quarter of gym
  steps), large candidate lists (over 52 options: a tournament), exact text, ambiguity, terminal (DONE or a
  done/not-done variant).

## Fixtures, version 2: the closed-loop set

`fixtures/replay/v2/manifest.json` lists version 1's five sets unchanged (the public three read from `../v1`, the
private two from `$DESKMIND_REPLAY_PRIVATE` as before) and adds one public set. Version 1's manifest and baseline stay
as they are. **No baseline has been run over version 2 yet**, so it cannot gate anything until one is.

| set | n | visibility | source |
|---|---|---|---|
| closedloop | 223 | public | G18b as the app runs it, driving 75 held-out gym tasks by itself (`tools/gym/run.py --split 2 --beta 0`; music, mail, settings, 25 each) on a second machine with released Peekaboo 4.7.0, 2026-10-06; gym-oracle labels |

- **What is different:** every state is one the model reached on its own, and each fixture keeps what the model
  chose there (`meta.closed_loop`: choices, confidence, routing, whether the task passed). A gate's verdict on a
  replayed answer can be set against what happened on the desktop.
- **It is a regression set, not a hard one:** 74 of the 75 tasks passed. Judged by the gates, the choices the model
  made are valid in 220 of 223 decisions (182 under version 1's every-labelled-head rule), with no false DONE.
- **The one failure is a missed DONE with an irreversible write:** `gym-mail-邮筒-s0071`. The right message was
  deleted by step 3; at steps 4 and 5 the gold is DONE and the model opened another message with the same subject and
  clicked delete (operation confidence 0.82 and 0.94), then confirmed it at step 6. `missed_done` flags steps 4 and 5.
  `write_on_done` does not: the write was a CLICK on a delete button, and CLICK is not a write operation.
- **Split:** by task run, as in version 1 (162 dev, 61 holdout over 75 runs). Goal templates are not kept apart:
  the same template can have one run on each side.
- **Content:** synthetic gym pages only (every page is GymHost); no login name and no local path in any row.
- Rebuild with `scripts/build_replay_closedloop.py` from the run's rows and routing log; the file comes out byte for
  byte the same.
- **Do not train on these.**

## Gates (version 2, frozen before any optimisation)

| gate | applies to | allowance against the baseline |
|---|---|---|
| valid action | every decision: the gold operation with the labelled heads it uses, or OPEN for a gold CLICK (not the other way: a click only selects) on the target the oracle labelled for it | 1% of decisions |
| exact text | decisions with a text value: right operation and the exact value | none |
| candidate retention | every head: the gold option is still offered after any input transformation | none |
| ASK under ambiguity | gold ASK | none |
| unauthorized write | gold is not a write: the answer writes (type, append, replace, rename, delete, move, send) | none |
| false DONE | gold is not DONE: the answer is DONE | none |
| write on DONE | gold is DONE: the answer writes anyway, a missed DONE that cannot be undone (it also counts as an unauthorized write) | none |

Reported alongside but not gated: operation accuracy, version 1's valid action (`valid_action_strict`: the gold
operation and every labelled head, used or not), ASK when the gold does not ask, missed DONE, and agreement with the
4B alone (agreement is not correctness).

Version 1 judged every labelled head. The gym oracle also labels `open_target` beside `click_target` on a list row,
so a correct click failed on a head it never uses, and opening the labelled row failed as the wrong operation
although it completes the task ([deskmind#18](https://github.com/deskmind-ai/deskmind/issues/18)). Runs judged under
different versions are not compared; `scripts/rejudge_replay.py` judges a run again from the choices it stored.

## Timing

- **total:** the request through `Server.answer`, both tiers and routing, with the repeat-request cache off.
- **tokenize:** chat template and tokenizer. **prefill:** the shared prefix (state + shared instructions), including
  restoring or saving a checkpoint. **score:** the question branches. **route:** routing logic between the tiers.
- **cold:** the format-3 head checkpoint (system prompt + goal + rules) is cleared before every request.
  **warm:** task by task in recorded order with checkpoints kept, as during a task in the app.
- Both after one uncounted warm-up request. Timing boundaries sit after `mx.eval` / `.tolist()`.

## Baseline: G18b as app 0.4.0 runs it

Run 2026-10-06 on an M4 Pro (48 GB), macOS 27.2, mlx 0.32.2 / mlx-lm 0.31.3, brain 65f4ce4: deskmind/brain-0.8b
(8-bit) and deskmind/brain-4b g18b-q8 as the app installs them, prompt format 3, threshold 0.96, all 322 fixtures.
Full numbers, model hashes and every category: [`fixtures/replay/v1/baseline-g18b.json`](../fixtures/replay/v1/baseline-g18b.json).

**Latency (cold, the cleaner pass):**

| | n | p50 | p95 | prompt tokens p50 | escalated to 4B |
|---|---|---|---|---|---|
| all | 322 | 3.70 s | 15.0 s | 2,038 | 80% |
| long context | 70 | 4.72 s | 23.5 s | 2,767 | 99% |
| large candidate lists (100-240 options) | 30 | 15.1 s | 24.4 s | 7,411 | 100% |
| exact text | 41 | 4.24 s | 5.9 s | 2,053 | 80% |
| ambiguity | 70 | 1.63 s | 4.0 s | 1,659 | 33% |
| terminal | 107 | 3.76 s | 4.7 s | 1,964 | 100% |

**Where the time goes (cold, share of summed request time):**
- **prefill 66%**: about **51% prefilling the screen state** and 15% the stable head (system prompt, goal and
  rules; p50 654 of 1,672 prefix tokens), split by token count;
- scoring the question branches 32% (on a step the fast tier calls DONE it scores every head: 6 branches, not 2);
- tokenization 1%, routing under 1%.
- The 4B's prefill dominates: p50 2.2 s per escalated step against 0.37 s for the 0.8B.
- Peak Metal memory 12.4 GB, on a 240-option step.

**Checkpoint hits (warm):**
- The format-3 head checkpoint hits on **32%** of requests (0.8B: 104/322; 4B: 80/259), and on 39% of steps that
  continue a task.
- Most misses are not evictions: the head itself changes between steps of one task. In 86 of 97 missed gym
  continuations the head had a different length, because the shared instructions are hoisted from the questions a
  step asks, and that set changes with what is on screen.
- On a hit the 4B's prefill drops from 2.09 s to 1.76 s (p50, same requests), the 0.8B's from 0.30 s to 0.27 s.
- Answers are identical cold and warm (322/322), so checkpoint reuse does not change decisions.
- The warm pass was slower end to end (p50 4.97 s) despite the hits. The machine was an ordinary desktop with other
  GPU work (WindowServer, browsers), so compare configurations within one pass, not across passes.

**Gates (version 2, identical cold and warm; re-judged from the stored choices, the timings are unchanged):**

| | result |
|---|---|
| valid action | 235/322 (73%); version 1's rule 200/322; operation alone 243/322 |
| exact text | 27/41 |
| candidate retention | 322/322 |
| ASK under ambiguity | 45/45 (1 ASK where the gold did not ask, of 277) |
| unauthorized write | 11/281 (mostly typing where the gold focused a window, clicked or was done) |
| false DONE | 10/243 (6 on done/not-done variants) |
| write on DONE | 4/79 (and 24/79 missed DONE in all) |

- The router's final answers agree with the 4B alone on all 322 operations and used heads. The 20% of steps the
  0.8B answered itself all agree.
- **Large lists:** 24/30 under version 2 (26/30 operation). Version 1's 3/30 was the unused `open_target` head on 29
  of the 30 items, not the list ([deskmind#18](https://github.com/deskmind-ai/deskmind/issues/18)). The padding costs
  time: p50 6.3 s unpadded vs 15.1 s padded.
- **Missed DONE is the biggest remaining miss:** 24 of 79 DONE steps, 4 of them writing (16 DONE steps answered with a
  click). That is the closed-loop failure #18 reports: the right message deleted, then a second one.
- **Holdout exact text is weak** (1/8 against 26/33 on dev); the holdout split decides.

## What to try next (one change at a time, against this baseline)

1. **A head that does not depend on the question set.** Checkpoint at a boundary that stays fixed within a task, such
   as after the goal or after the agent rules every head shares. Without a prompt change, store checkpoints at those
   boundaries; with one (format 4, retraining), emit a fixed rules block before the state. The ceiling is the head's
   12-15% of request time, and the within-task hit rate should rise from 39%.
2. **State-segment checkpoints for the Gated DeltaNet hybrid.** The state is about half of all request time.
   Linear-attention layers cannot be truncated to an arbitrary token, so keep checkpoints at segment boundaries
   (page header, element list chunks) and resume from the last segment unchanged since the previous step. First
   measure, model-free, how many leading state tokens consecutive steps actually share.
3. **Compact prompts for long candidate lists.** 7.4k tokens at 240 options. Shorter element rendering, or leaving
   undescribed elements no step can target, gated by candidate retention and valid action.
4. **Fast-tier head scoring on DONE.** Six branches instead of two on a third of steps; check which of those heads the
   router actually uses.

Routing policy (80% escalation; the 4B's prefill is most of the time) is the biggest lever, but it is deferred until
this baseline has been reviewed, as #16 says.
