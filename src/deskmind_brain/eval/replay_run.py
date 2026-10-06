"""Run the replay fixtures through the server as the Mac app runs it (see replay.py for the fixtures and gates).

The fast and strong tiers are loaded in this process with deskmind_brain.serve.Server and --two-stage, exactly the
path `deskmind-brain-serve --escalate-to` takes, minus HTTP. The repeat-request cache is off (it would answer a
replayed step from memory). Every request is timed end to end, and each tier's predictor reports where its time went
(LogitsPredictorBase.stats).

Modes:
  cold  the format-3 head checkpoints are cleared before every request: a step whose goal and rules were never seen.
  warm  fixtures replay task by task in recorded order with the checkpoints kept: the app during a task.
Both run after one warm-up request (kernels compiled, weights paged in), which is reported but not counted.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from deskmind_brain.eval.data import EvalItem
from deskmind_brain.eval.replay import (GATES_VERSION, check_release, compare, head_agreement, judge, load_fixtures,
                                        load_manifest, markdown, max_options, order_for_replay, pred_choice, sha256_file,
                                        summarize)
from deskmind_brain.router import heads_for


def _inner(pred):
    return getattr(pred, "inner", pred)


def _body(item: EvalItem) -> dict[str, Any]:
    return {"state": item.state, "model": "replay",
            "questions": {qid: q.model_dump(exclude_none=True) for qid, q in item.questions.items()}}


def _clear_checkpoints(server) -> None:
    for pred in (server.predictor, server.strong):
        if pred is not None and hasattr(_inner(pred), "_checkpoints"):
            _inner(pred)._checkpoints.clear()


def _model_info(spec: str) -> dict[str, Any]:
    path = Path(spec.split(":", 1)[-1])
    info: dict[str, Any] = {"spec": spec.replace(str(Path.home()), "~")}  # reports get published: no user names
    fmt = path / "deskmind.json"
    if fmt.exists():
        info["deskmind.json"] = json.loads(fmt.read_text())
    weights = sorted(path.glob("*.safetensors"))
    if weights:
        h = hashlib.sha256()
        for w in weights:
            with w.open("rb") as f:
                for block in iter(lambda: f.read(1 << 24), b""):
                    h.update(block)
        info["weights_sha256"] = h.hexdigest()
    cfg = path / "config.json"
    if cfg.exists():
        q = json.loads(cfg.read_text()).get("quantization")
        info["quantization"] = q
    return info


def environment(manifest_path: Path, fast: str, strong: str | None, threshold: float) -> dict[str, Any]:
    import mlx.core as mx
    import mlx_lm

    def sh(*cmd: str) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except OSError:
            return ""

    repo = Path(__file__).resolve().parents[3]
    return {
        "brain_commit": sh("git", "-C", str(repo), "rev-parse", "--short", "HEAD") + (
            "+dirty" if sh("git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no") else ""),
        "manifest_sha256": sha256_file(manifest_path),
        "hardware": f"{sh('sysctl', '-n', 'machdep.cpu.brand_string')}, "
                    f"{int(sh('sysctl', '-n', 'hw.memsize') or 0) / 2**30:.0f} GB",
        "os": f"macOS {platform.mac_ver()[0]}" if sys.platform == "darwin" else platform.platform(),
        "python": platform.python_version(),
        "mlx": mx.__version__,
        "mlx_lm": mlx_lm.__version__,
        "fast": _model_info(fast),
        "strong": _model_info(strong) if strong else None,
        "threshold": threshold,
        "two_stage": True,
        "repeat_cache": "off",
    }


def replay(server, items: list[EvalItem], mode: str, log=print) -> list[dict[str, Any]]:
    import mlx.core as mx

    fast, strong = _inner(server.predictor), _inner(server.strong) if server.strong else None
    rows = []
    if mode == "warm":
        _clear_checkpoints(server)
    for i, item in enumerate(items, 1):
        if mode == "cold":
            _clear_checkpoints(server)
        fast.stats = None
        if strong is not None:
            strong.stats = None
        mx.reset_peak_memory()
        start = time.perf_counter()
        reply = server.answer(_body(item))
        total = time.perf_counter() - start
        fs, ss = fast.stats, (strong.stats if strong is not None else None)
        answers = reply["answers"]
        op = pred_choice(item, "operation", answers)
        rows.append({
            "id": item.id, "set": item.meta["replay"]["set"], "session": item.meta["replay"]["session"],
            "split": item.meta["replay"]["split"], "categories": item.meta["replay"]["categories"],
            "max_options": max_options(item), "total_s": total, "fast": fs, "strong": ss,
            "route_s": max(0.0, total - fs["total_s"] - (ss["total_s"] if ss else 0.0)),
            "routing": reply.get("routing"), "peak_memory_gb": mx.get_peak_memory() / 2**30,
            "choices": {qid: pred_choice(item, qid, answers) for qid in item.questions},
            "used_heads": heads_for(op, item.questions) if op else [],
            **judge(item, answers),
        })
        if i % 25 == 0 or i == len(items):
            log(f"[{mode}] {i}/{len(items)}  last {total:.2f}s")
    return rows


def reference_choices(pred, items: list[EvalItem], log=print) -> dict[str, dict[str, str | None]]:
    """The strong tier alone on every fixture (two-stage, every head on terminal steps), for head agreement."""
    inner = _inner(pred)
    saved = inner.terminal_heads
    inner.terminal_heads = True
    out = {}
    try:
        for i, item in enumerate(items, 1):
            answers = pred.predict(item).answers
            out[item.id] = {qid: pred_choice(item, qid, answers) for qid in item.questions}
            if i % 50 == 0 or i == len(items):
                log(f"[reference] {i}/{len(items)}")
    finally:
        inner.terminal_heads = saved
    return out


def run(manifest: str, fast: str, strong: str | None, threshold: float, modes: list[str], out: str,
        reference: bool = True, split: str | None = None, limit: int | None = None, label: str = "baseline",
        baseline: str | None = None, release: str | None = None) -> dict:
    """baseline: a summary.json of an earlier run on the same fixtures; its gates decide keep or reject.
    release: the release the weights must be (replay.RELEASES). A release named in `label` or the --out folder's name
    is checked the same way; the run stops before loading a model when the weights' sha256 is another release's."""
    from deskmind_brain.serve import Server, use_two_stage

    manifest_path = Path(manifest)
    items, notes = load_fixtures(load_manifest(manifest_path), split=split)
    items = order_for_replay(items)[: limit or None]
    out_dir = Path(out)
    env = {"label": label, **environment(manifest_path, fast, strong, threshold), "fixtures": len(items),
           "split": split or "all"}
    env["release"] = check_release(env["fast"].get("weights_sha256"),
                                   env["strong"].get("weights_sha256") if env["strong"] else None,
                                   label, out_dir.name, expect=release)
    out_dir.mkdir(parents=True, exist_ok=True)

    t = time.perf_counter()
    server = Server(fast, "replay", cache_size=0, escalate_to=strong, threshold=threshold)
    use_two_stage(server)
    env["load_s"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    server.answer(_body(items[0]))  # warm-up, not counted
    env["first_request_s"] = round(time.perf_counter() - t, 2)

    runs = {}
    for mode in modes:
        runs[mode] = replay(server, items, mode)
        with (out_dir / f"{mode}.jsonl").open("w") as f:
            for r in runs[mode]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"env": env, "notes": notes, "modes": summarize(runs)}
    if reference and server.strong is not None:
        ref = reference_choices(server.strong, items)
        summary["head_agreement_with_strong"] = {m: head_agreement(rows, ref) for m, rows in runs.items()}
    if baseline:
        base = json.loads(Path(baseline).read_text())["modes"]
        stale = {m: v.get("gates_version", "1") for m, v in base.items() if v.get("gates_version", "1") != GATES_VERSION}
        if stale:
            raise SystemExit(f"{baseline} was judged with gates version {stale}, this run with {GATES_VERSION}: "
                             "re-judge it first (scripts/rejudge_replay.py)")
        summary["gates_vs_baseline"] = {m: compare(base[m]["gates"]["all"], s["gates"]["all"])
                                        for m, s in summary["modes"].items() if m in base}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    md = markdown(summary["modes"], env, notes)
    for m, verdict in summary.get("gates_vs_baseline", {}).items():
        md += f"\n## Gates against {baseline} ({m}): {'PASS' if all(v['pass'] for v in verdict.values()) else 'FAIL'}\n\n"
        md += "\n".join(f"- {k}: {'pass' if v['pass'] else 'FAIL'} ({v['why']})" for k, v in verdict.items()) + "\n"
    if "head_agreement_with_strong" in summary:
        md += "\n## Head agreement with the strong tier alone (not correctness)\n\n" + "\n".join(
            f"- {m}: operation {a['operation']:.1%}, heads the step uses {a['used_heads']:.1%} (n={a['n']})"
            for m, a in summary["head_agreement_with_strong"].items()) + "\n"
    (out_dir / "report.md").write_text(md)
    return summary
