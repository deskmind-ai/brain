"""Judge a replay run again under the current gates (replay.GATES_VERSION), from the choices its rows stored: no model.

For a gates change: the run's latency is kept as it was, every gate is recomputed, and the summary records which
version judged it. Private fixture sets need $DESKMIND_REPLAY_PRIVATE, as for the run itself.

    python scripts/rejudge_replay.py fixtures/replay/v1/manifest.json runs/replay/baseline-v1 \
        --summary fixtures/replay/v1/baseline-g18b.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deskmind_brain.eval.replay import GATES_VERSION, judge_choices, load_fixtures, load_manifest, markdown, summarize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("run", help="a run folder with <mode>.jsonl rows (replay_requests.py --manifest --out)")
    ap.add_argument("--summary", help="the run's summary.json to update in place (default: <run>/summary.json)")
    args = ap.parse_args()
    items, notes = load_fixtures(load_manifest(args.manifest))
    by_id = {it.id: it for it in items}
    run = Path(args.run)
    summary_path = Path(args.summary or run / "summary.json")
    summary = json.loads(summary_path.read_text())
    runs = {}
    for mode in summary["modes"]:
        rows = [json.loads(line) for line in (run / f"{mode}.jsonl").open() if line.strip()]
        missing = [r["id"] for r in rows if r["id"] not in by_id]
        if missing:
            raise SystemExit(f"{mode}: {len(missing)} rows have no fixture here (private sets need the env var): {missing[:3]}")
        runs[mode] = [{**r, **judge_choices(by_id[r["id"]], r["choices"])} for r in rows]
    summary["modes"] = summarize(runs)
    summary["notes"] = notes
    summary["rejudged"] = {"gates_version": GATES_VERSION, "from_rows": str(run)}
    summary_path.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print(markdown(summary["modes"], summary["env"], notes))


if __name__ == "__main__":
    main()
