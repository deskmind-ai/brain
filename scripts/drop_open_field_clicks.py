"""Drop browser training items whose gold is an "Open <field>" click on a plain text field.

The browser agent's action schema offers an "Open <label>" click on every editable field. On a plain textbox there is
nothing to open, so the click is a no-op, yet a large share of "Open" labels in browser step data sit on
textboxes/searchboxes: Mind2Web's recorded click-to-focus before typing, and travel-site form fields. Models learned to
prefer it over typing (the desktop harness's file page went BLOCKED until those clicks were filtered). Comboboxes, date
pickers and buttons keep it.

    python scripts/drop_open_field_clicks.py path/to/browser_steps [more dirs...]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PLAIN = {"textbox", "searchbox"}


def is_plain_open(item: dict) -> bool:
    ref = item["references"].get("click_target")
    crit = (item["questions"].get("click_target") or {}).get("criteria") or {}
    if not ref or not isinstance(crit, dict):
        return False
    gold = max(ref["probs"], key=ref["probs"].get)
    text = json.dumps(crit.get(gold, ""), ensure_ascii=False)
    role = re.search(r'"role": "([^"]+)"', text)
    return bool(re.search(r"\bOpen ", text)) and (role.group(1) if role else "") in PLAIN


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    kept, dropped = [], 0
    for line in (src / "items_labeled.jsonl").read_text().splitlines():
        if line.strip():
            item = json.loads(line)
            if is_plain_open(item):
                dropped += 1
            else:
                kept.append(line)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "items_labeled.jsonl").write_text("\n".join(kept) + "\n")
    print(json.dumps({"src": str(src), "kept": len(kept), "dropped": dropped}))


if __name__ == "__main__":
    main()
