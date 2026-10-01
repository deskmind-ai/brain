import json

from deskmind_brain.serve import default_threshold


def test_threshold_from_manifest(tmp_path):
    (tmp_path / "deskmind.json").write_text(json.dumps({"prompt_format": 3, "router_threshold": 0.96}))
    assert default_threshold(f"mlx:{tmp_path}") == 0.96


def test_threshold_fallback(tmp_path):
    (tmp_path / "deskmind.json").write_text(json.dumps({"prompt_format": 3}))
    assert default_threshold(f"mlx:{tmp_path}") == 0.94
    assert default_threshold(f"mlx:{tmp_path}/missing") == 0.94
