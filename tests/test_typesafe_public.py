import json

import pytest

from deskmind_brain.eval.data import load_items, load_predictions
from deskmind_brain.eval.sources.typesafe_public import build_workflow, consensus, parse_viewer
from deskmind_brain.types import Question

NOUL = Question(type="noul", instructions="unauthorized?")
SCORE = Question(type="score", instructions="evidence?", criteria=["weak", "some", "strong"])


def viewer_fixture() -> dict:
    node_questions = {"is_tp": 0, "strength": 1}
    answers = {
        "is_tp": {"type": "noul", "noul": 0.3},
        "strength": {"type": "score", "score": 1.2, "confidence": 0.5, "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}},
    }
    return {
        "id": "security_incidents",
        "questions": [NOUL.model_dump(exclude_none=True), SCORE.model_dump()],
        "documents": [{"alert": "doc zero"}, {"alert": "doc one"}],
        "examples": [{"case_id": "c1", "policy": "playbook", "name": "case one"}],
        "cases": {
            "c1": {
                "models": {
                    "typesafe": {
                        "model": "typesafe:v13",
                        "seconds": 0.4,
                        "cost": {"usd": 0.0002},
                        "nodes": [
                            {"node": "triage", "ran": True, "doc": 1, "questions": node_questions, "answers": answers},
                            {"node": "containment", "ran": False, "doc": None, "questions": {}, "answers": {}},
                        ],
                    },
                    "opus": {
                        "model": "anthropic:claude-opus-5",
                        "seconds": 10.0,
                        "cost": {"usd": 0.04},
                        "nodes": [{"node": "triage", "ran": True, "doc": 1, "questions": node_questions, "answers": {}}],
                    },
                },
                "references": [{"title": "astra"}, {"title": "fable"}],
                "reference_answers": {
                    "triage": {
                        "is_tp": {"type": "noul", "sets": [
                            {"value": False, "probabilities": {"true": 0.2, "false": 0.8}},
                            {"value": True},
                        ]},
                        "strength": {"type": "score", "sets": [
                            {"value": "1", "probabilities": {"0": 0.0, "1": 1.0, "2": 0.0}},
                            {"value": "2", "probabilities": {"0": 0.0, "1": 0.0, "2": 1.0}},
                        ]},
                    },
                    "containment": {"is_tp": {"type": "noul", "sets": [{"value": True}]}},
                },
            }
        },
    }


def test_parse_viewer_payload():
    raw = "__VIEWER_DATA__(" + json.dumps({"eval": {"id": "x"}}) + ");\n"
    assert parse_viewer(raw) == {"id": "x"}
    with pytest.raises(ValueError):
        parse_viewer("nothing here")


def test_consensus_mixes_soft_and_hard_sets():
    ref = consensus({"sets": [{"probabilities": {"true": 0.2, "false": 0.8}}, {"value": True}]}, NOUL)
    assert ref.probs == pytest.approx({"false": 0.4, "true": 0.6})
    assert ref.n_sets == 2 and ref.soft is False
    assert consensus({"sets": []}, NOUL) is None


def test_build_workflow():
    items, published = build_workflow(viewer_fixture())
    # containment never ran for any model, so it has no document and is skipped.
    assert [it.id for it in items] == ["security_incidents/c1/triage"]
    item = items[0]
    assert item.state == {"alert": "doc one"}
    assert set(item.questions) == {"is_tp", "strength"}
    assert item.references["strength"].probs == pytest.approx({"0": 0.0, "1": 0.5, "2": 0.5})

    (jev,) = published["typesafe"]
    assert jev.item_id == item.id and set(jev.answers) == {"is_tp", "strength"}
    assert jev.latency_s == pytest.approx(0.4) and jev.cost_usd == pytest.approx(0.0002)
    assert "opus" not in published  # no answers on the node


def test_items_round_trip(tmp_path):
    from deskmind_brain.eval.data import write_jsonl

    items, published = build_workflow(viewer_fixture())
    write_jsonl(tmp_path / "items.jsonl", (it.to_json() for it in items))
    write_jsonl(tmp_path / "p.jsonl", (p.to_json() for p in published["typesafe"]))
    (loaded,) = load_items(tmp_path)
    assert loaded.to_json() == items[0].to_json()
    assert load_predictions(tmp_path / "p.jsonl")[loaded.id].answers == published["typesafe"][0].answers
