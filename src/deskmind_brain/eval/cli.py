"""deskmind-brain-eval: build eval suites, run predictors, score runs.

  deskmind-brain-eval fetch typesafe-public
  deskmind-brain-eval predict --predictor uniform --out runs/uniform.jsonl
  deskmind-brain-eval score --published --run uniform=runs/uniform.jsonl --common
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from deskmind_brain.eval.data import EvalItem, Prediction, load_items, load_predictions, write_jsonl
from deskmind_brain.eval.metrics import report, score_run
from deskmind_brain.eval.predictors import make_predictor
from deskmind_brain.eval.runner import run_predictions

DEFAULT_DATA = Path("data")
DEFAULT_SUITE = DEFAULT_DATA / "eval" / "typesafe_public"


def _fmt(x: object, digits: int = 3) -> str:
    if isinstance(x, float):
        return "–" if math.isnan(x) else f"{x:.{digits}f}"
    return str(x)


def _per_case(items: list[EvalItem], preds: dict[str, Prediction], field: str) -> float:
    """Mean over cases of the summed per-item latency or cost."""
    totals: dict[str, float] = defaultdict(float)
    seen = False
    for it in items:
        p = preds.get(it.id)
        value = getattr(p, field) if p else None
        if value is not None:
            totals[it.meta.get("case_id", it.id)] += value
            seen = True
    return sum(totals.values()) / len(totals) if seen else float("nan")


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def cmd_fetch(args: argparse.Namespace) -> None:
    if args.source == "typesafe-public":
        from deskmind_brain.eval.sources import typesafe_public as source

        summary = source.build_suite(
            out_dir=args.data_dir / "eval" / source.SUITE, raw_dir=args.data_dir / "raw" / source.SUITE, refresh=args.refresh
        )
    else:
        from deskmind_brain.eval.sources import hf_tasks as source

        summary = source.build_suite(
            out_dir=args.data_dir / "eval" / source.SUITE,
            raw_dir=args.data_dir / "raw" / source.SUITE,
            n=args.n,
            seed=args.seed,
            only=args.only.split(",") if args.only else None,
            refresh=args.refresh,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def cmd_predict(args: argparse.Namespace) -> None:
    items = load_items(args.suite)
    predictor = make_predictor(args.predictor)
    out = args.out or Path("runs") / args.suite.name / f"{predictor.name.replace(':', '_').replace('/', '_')}.jsonl"
    preds = run_predictions(predictor, items, out, resume=not args.no_resume, limit=args.limit)
    errors = sum(1 for p in preds.values() if p.error)
    print(f"wrote {out} ({len(preds)} predictions, {errors} errors)")


def cmd_score(args: argparse.Namespace) -> None:
    items = load_items(args.suite)
    runs: dict[str, dict[str, Prediction]] = {}
    if args.published:
        for path in sorted((args.suite / "published").glob("*.jsonl")):
            runs[f"published:{path.stem}"] = load_predictions(path)
    for spec in args.run or []:
        label, _, path = spec.partition("=")
        if not path:
            label, path = Path(spec).stem, spec
        runs[label] = load_predictions(Path(path))
    if not runs:
        raise SystemExit("nothing to score: pass --published and/or --run LABEL=PATH")

    scored = {label: score_run(items, preds) for label, preds in runs.items()}
    if args.common:
        # Keep only (item, question) pairs every run answered, so runs are compared on identical work.
        keys = set.intersection(*({(r.item_id, r.qid) for r in rs if r.answered} for rs in scored.values()))
        scored = {label: [r for r in rs if (r.item_id, r.qid) in keys] for label, rs in scored.items()}

    reports = {label: report(rs) for label, rs in scored.items()}
    subset = "common subset" if args.common else "all reference pairs"
    print(f"## {args.suite.name} — {subset}\n")
    headers = ["run", "pairs", "cov", "acc", "bal acc", "macro acc", "nll", "brier", "ece", "aurc", "sel@80", "cov@r5", "cov@r10", "kl", "tv", "score mae", "s/case", "$/case"]
    rows = []
    for label, rep in reports.items():
        o, m = rep["overall"], rep["macro_by_group"]
        rows.append(
            [label, _fmt(o["pairs"]), _fmt(o["coverage"], 2), _fmt(o["accuracy"]), _fmt(o["balanced_accuracy"]), _fmt(m["accuracy"]), _fmt(o["nll"]),
             _fmt(o["brier"]), _fmt(o["ece"]), _fmt(o["aurc"]), _fmt(o["sel_acc@80"]), _fmt(o["cov@risk5"]), _fmt(o["cov@risk10"]), _fmt(o["kl"]), _fmt(o["tv"]),
             _fmt(o["score_mae"]), _fmt(_per_case(items, runs[label], "latency_s"), 2),
             _fmt(_per_case(items, runs[label], "cost_usd"), 5)]
        )
    print(_table(headers, rows))

    for section, title in (("by_group", "accuracy / kl by group"), ("by_type", "accuracy / kl by question type"), ("by_k", "accuracy / ece by option count")):
        names = sorted({n for rep in reports.values() for n in rep[section]})
        second = "ece" if section == "by_k" else "kl"
        print(f"\n### {title}\n")
        header = ["run"] + [f"{n} (n)" for n in names]
        body = []
        for label, rep in reports.items():
            cells = []
            for n in names:
                s = rep[section].get(n)
                cells.append("–" if s is None else f"{_fmt(s['accuracy'])} / {_fmt(s[second])} ({s['pairs']})")
            body.append([label] + cells)
        print(_table(header, body))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n")


def cmd_calibrate(args: argparse.Namespace) -> None:
    from deskmind_brain.eval.calibrate import apply, crossfit, fit

    items = load_items(args.suite)
    preds = load_predictions(args.run)
    if args.crossfit:
        calibrated, fits = crossfit(items, preds)
        print(json.dumps({"fold_temperatures": fits}))
    else:
        temps = fit(items, preds)
        print(json.dumps(temps))
        if args.temps_out:
            args.temps_out.parent.mkdir(parents=True, exist_ok=True)
            args.temps_out.write_text(json.dumps(temps, indent=2) + "\n")
        calibrated = {it.id: apply(it, preds[it.id], temps) for it in items if it.id in preds}
    if args.out:
        write_jsonl(args.out, (p.to_json() for p in calibrated.values()))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="deskmind-brain-eval")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download and build an eval suite")
    p.add_argument("source", choices=["typesafe-public", "hf-tasks"])
    p.add_argument("--n", type=int, default=100, help="hf-tasks: items per task")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", help="hf-tasks: comma-separated task names")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--refresh", action="store_true", help="re-download even if cached")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("predict", help="run a predictor over a suite")
    p.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    p.add_argument("--predictor", required=True, help="uniform | reference | systemone[:model[@base_url]] | local[-noshare]:<model>[+<lora>] | mlx:<model>[@4bit|@8bit]; append ~<temps.json> to apply fitted temperatures")
    p.add_argument("--out", type=Path)
    p.add_argument("--limit", type=int)
    p.add_argument("--no-resume", action="store_true")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("score", help="score prediction files against a suite")
    p.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    p.add_argument("--published", action="store_true", help="include answers shipped with the suite (Jev, Opus, Sol)")
    p.add_argument("--run", action="append", help="LABEL=PATH to a predictions JSONL (repeatable)")
    p.add_argument("--common", action="store_true", help="restrict to pairs answered by every run")
    p.add_argument("--json", type=Path, help="also write the full report as JSON")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("calibrate", help="fit per-kind temperatures on a run (never on the suite you report)")
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True, help="predictions JSONL to fit on")
    p.add_argument("--crossfit", action="store_true", help="2-fold fit/apply on this suite, to estimate the effect")
    p.add_argument("--temps-out", type=Path, help="write the fitted {kind: T} JSON")
    p.add_argument("--out", type=Path, help="write the rescaled predictions")
    p.set_defaults(func=cmd_calibrate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
