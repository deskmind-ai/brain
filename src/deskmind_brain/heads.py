"""The questions (heads) an agent step uses for each operation, in one place for the scorer and the router.

Agent-step naming, as hands sends it: `<op>_target`, plus `type_text_value` for the operations that type a value and
`replace_from` for REPLACE_TEXT. deskmind protocol/agent/operations.yaml is the registry; tests/test_heads.py checks this
against it when the deskmind checkout is next to this one.
"""

from __future__ import annotations

# TYPE_FOCUSED types a value where the keyboard focus is: its one head is type_text_value. It was missing here, so a
# two-stage server never scored it and hands typed candidate 1 whatever the model would have chosen (protocol G1).
TEXT_OPS = {"TYPE_TEXT", "REPLACE_TEXT", "APPEND_TEXT", "RENAME", "TYPE_FOCUSED"}
TERMINAL_OPS = {"DONE", "BLOCKED", "ASK"}


def heads_for(op: str, questions: dict) -> list[str]:
    """The questions an agent step with operation `op` actually uses."""
    prefix = op.lower() + "_"
    heads = [q for q in questions if q.startswith(prefix)]
    if op in TEXT_OPS and "type_text_value" in questions and "type_text_value" not in heads:
        heads.append("type_text_value")   # TYPE_TEXT's prefix already finds it: once is enough
    if op == "REPLACE_TEXT" and "replace_from" in questions:
        heads.append("replace_from")
    return heads
