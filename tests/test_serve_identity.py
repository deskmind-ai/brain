"""Request identity (deskmind#36 item 4): a reply echoes request_id, session_id and step, the routing log records
them, and they are not part of the answer cache key."""
import json

import pytest
from pydantic import ValidationError

from deskmind_brain.serve import Server
from deskmind_brain.types import SystemOneRequest

QUESTIONS = {"operation": {"type": "choice", "instructions": "go?", "criteria": {"CLICK": "c", "DONE": "d"}}}


def body(**ident):
    return {"state": {"page": {"title": "w"}}, "questions": QUESTIONS, **ident}


def test_reply_echoes_the_identity_and_the_cache_ignores_it(tmp_path):
    server = Server("uniform", "t", cache_size=8)
    first = server.answer(body(request_id="r-1", session_id="run-1", step=3, observation_id="o-7"))
    assert (first["request_id"], first["session_id"], first["step"]) == ("r-1", "run-1", 3)
    assert "observation_id" not in first
    again = server.answer(body(request_id="r-2", session_id="run-1", step=4))
    assert again["cached"] is True and server.hits == 1                      # same state and questions: a hit
    assert (again["request_id"], again["step"]) == ("r-2", 4)                 # ...echoing its own request
    plain = server.answer(body())
    assert plain["cached"] is True and not {"request_id", "session_id", "step"} & plain.keys()


def test_routing_log_records_the_identity(tmp_path):
    log = tmp_path / "routing.jsonl"
    server = Server("uniform", "t", cache_size=0, escalate_to="uniform", threshold=0.96, routing_log=str(log))
    server.answer(body(request_id="r-9", session_id="run-2", step=1))
    rec = json.loads(log.read_text().splitlines()[-1])
    assert (rec["request_id"], rec["session_id"], rec["step"]) == ("r-9", "run-2", 1)
    assert "state" not in rec and "questions" not in rec                      # still metadata only


@pytest.mark.parametrize("bad", [{"step": 0}, {"step": "3"}, {"step": True}, {"step": 3.0}, {"request_id": ""},
                                 {"state_digest": "md5:abc"}])
def test_identity_fields_are_validated(bad):
    with pytest.raises(ValidationError):
        SystemOneRequest(**body(**bad))
