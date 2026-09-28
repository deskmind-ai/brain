"""Turn "what should I type here" into a typed Choice question, so the policy needs no text model.

An agent that answers TYPE_TEXT by calling a separate LLM for the string leaves one part of the loop outside the
model's control — and that part fails in practice (a local generator returned {"text": null} for a combobox
labelled "Where else?"). But nearly all values typed in browser runs are substrings of the goal, so the value can be
*chosen* instead of generated: extract candidate spans from the goal
(and the page's own values), and let the same logit-scoring machinery pick one.

`candidates` is deliberately over-generous — recall matters more than precision, since a wrong extra option only
costs a little probability mass, while a missing one makes the question unanswerable.
"""

from __future__ import annotations

import re

# Dates as goals state them ("September 20, 2026", "2026-09-20", "Sep 20"), quoted strings, capitalised phrases,
# bare numbers, and email/URL-ish tokens: the shapes that actually appear in a browsing goal.
MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
PATTERNS = [
    rf"{MONTH}\s+\d{{1,2}}(?:,\s*\d{{4}})?",
    r"\d{4}-\d{2}-\d{2}",
    r"\d{1,2}/\d{1,2}(?:/\d{2,4})?",
    r"\"([^\"]{1,60})\"",
    r"“([^”]{1,60})”",
    r"「([^」]{1,60})」",
    r"[\w.+-]+@[\w-]+\.[\w.]+",
    r"\b[A-Z][\w'’-]*(?:\s+(?:of|de|del|van|von|the)\s+[A-Z][\w'’-]*|\s+[A-Z][\w'’-]*)*\b",
    r"[一-鿿]{2,12}",
    r"\b\d+\b",
]
# Words that start a sentence and would otherwise be captured as a capitalised phrase.
STOP = {"find", "search", "open", "go", "show", "set", "select", "click", "stop", "do", "the", "a", "an", "and",
        "for", "from", "to", "on", "in", "with", "then", "when", "if", "not", "book", "please"}


def spans(text: str, limit: int = 40) -> list[str]:
    """Candidate values a goal (or page text) offers, longest first, deduplicated case-insensitively."""
    out: list[str] = []
    for pattern in PATTERNS:
        for match in re.finditer(pattern, text or ""):
            value = (match.group(1) if match.groups() else match.group(0)).strip(" ,.;:'\"")
            if not value or value.lower() in STOP or len(value) > 80:
                continue
            out.append(value)
    seen, unique = set(), []
    for value in sorted(out, key=len, reverse=True):
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return unique[:limit]


def candidates(goal: str, field_label: str = "", page_values: list[str] | None = None, limit: int = 26) -> list[str]:
    """Options for the value question: goal spans first, then values already present on the page.

    The field label is used only to order candidates — a field named "Departure" prefers a date-shaped span —
    never to filter, because a wrong ordering costs nothing while a missing option costs the whole question.
    """
    found = spans(goal)
    label = (field_label or "").lower()
    wants_date = any(word in label for word in ("date", "depart", "return", "check", "日期", "出发", "返回"))
    wants_number = any(word in label for word in ("passenger", "adult", "guest", "room", "qty", "quantity", "人数"))

    def rank(value: str) -> tuple[int, int]:
        dated = bool(re.match(rf"{MONTH}|\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}/\d{{1,2}}", value))
        numeric = value.isdigit()
        if wants_date:
            return (0 if dated else 1, -len(value))
        if wants_number:
            return (0 if numeric else 1, -len(value))
        return (1 if dated or numeric else 0, -len(value))

    ordered = sorted(found, key=rank)
    for value in page_values or []:
        value = (value or "").strip()
        if value and len(value) <= 80 and value.lower() not in {v.lower() for v in ordered}:
            ordered.append(value)
    return ordered[:limit]
