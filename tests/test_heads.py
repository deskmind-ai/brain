"""One heads_for for scorer and router, matching the protocol registry; G1: TYPE_FOCUSED's value head is scored."""
import os
from pathlib import Path

import pytest

from deskmind_brain.eval import logits_base
from deskmind_brain import router
from deskmind_brain.heads import heads_for

# The protocol registry, from a deskmind checkout: $DESKMIND_REPO (CI checks it out), else one next to this repo.
_REPO = Path(os.environ.get("DESKMIND_REPO") or Path(__file__).resolve().parents[2] / "deskmind")
REGISTRY = _REPO / "protocol" / "agent" / "operations.yaml"


def test_scorer_and_router_share_one_definition():
    assert logits_base.heads_for is heads_for and router.heads_for is heads_for


def test_type_focused_scores_its_value():
    qs = {"operation": {}, "type_text_value": {}, "click_target": {}}
    assert heads_for("TYPE_FOCUSED", qs) == ["type_text_value"]


@pytest.mark.skipif(not REGISTRY.exists(), reason="deskmind checkout not next to brain")
def test_matches_the_protocol_registry():
    yaml = pytest.importorskip("yaml")
    reg = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    all_heads = {h: {} for spec in reg["operations"].values() for h in spec.get("heads", [])}
    for op, spec in reg["operations"].items():
        assert sorted(heads_for(op, {"operation": {}, **all_heads})) == sorted(spec.get("heads", [])), op
