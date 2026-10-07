# What changes on `next`, for the 0.6 cut

brain's counterpart of hands' `docs/next-model-visible.md` (deskmind#65). Every change on `next` that alters what a
client gets back is listed here as it merges: what the model is shown (prompts, options, their order), or which answer
the router returns. The 0.6 cut reads this list: the replay baseline is re-run over it, and a retrain decides which of
these its data must cover. A change to the protocol's schema bumps the protocol version (`2.0-next.N`) instead.

| PR | What changes | Model sees it? | Replay |
|---|---|---|---|
| brain#17 | The router sends a CLICK on a commit control (save, submit, send, delete, pay, publish, share) to the strong tier at any confidence (`risky_commit`); a fast DONE (p ≥ 0.8) stands when the strong tier's alternative scores below 0.96 (`done_kept_low_override`, risky_DONE only) | No: which tier's answer is returned | pending (v1 ~1 h) |
