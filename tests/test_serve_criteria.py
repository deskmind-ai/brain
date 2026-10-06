from deskmind_brain.eval.logits_base import question_block
from deskmind_brain.serve import Server
from deskmind_brain.types import SystemOneRequest

STATE = {"goal": "pick a team"}
MAP = {"billing": "b", "technical": "t", "sales": "s"}


def body(criteria):
    return {"state": STATE, "questions": {"team": {"type": "choice", "instructions": "team?", "criteria": criteria}}}


def test_list_and_map_forms_render_the_same_prompt():
    as_list = [{"key": k, "description": v} for k, v in MAP.items()]
    a = SystemOneRequest(**body(MAP)).questions["team"]
    b = SystemOneRequest(**body(as_list)).questions["team"]
    assert question_block(a, a.options()) == question_block(b, b.options())
    assert question_block(a, a.options(), compact=True) == question_block(b, b.options(), compact=True)


def test_cache_key_follows_option_order():
    server = Server("uniform", "test", cache_size=8)
    reordered = dict(reversed(list(MAP.items())))
    assert "cached" not in server.answer(body(MAP))
    assert "cached" not in server.answer(body(reordered))
    assert server.answer(body(MAP))["cached"] is True
    assert server.answer(body([{"key": k, "description": v} for k, v in MAP.items()]))["cached"] is True
    assert (server.hits, server.misses) == (2, 2)
