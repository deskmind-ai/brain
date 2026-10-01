import json

import pytest

from deskmind_brain.train.label_teacher import check_predictions_file, check_teacher, load_teacher_predictions


@pytest.mark.parametrize("spec", ["local:Qwen/Qwen3.5-4B", "mlx:models/brain-4b", "systemone:brain-4b@http://127.0.0.1:8793"])
def test_local_teachers_allowed(spec):
    check_teacher(spec)


@pytest.mark.parametrize("spec", ["systemone", "systemone:jev-latest", "systemone:x@https://api.typesafe.ai", "systemone~1.2"])
def test_hosted_system_one_refused(spec):
    with pytest.raises(SystemExit):
        check_teacher(spec)


def _write_preds(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


_ROW = {"item_id": "a", "answers": {"q": {"choice": "x"}}}


def test_own_predictions_file_allowed(tmp_path):
    path = _write_preds(tmp_path / "runs" / "teacher_brain-4b.jsonl", [_ROW])
    check_predictions_file(path)
    assert set(load_teacher_predictions(path)) == {"a"}


@pytest.mark.parametrize("rel", ["eval/typesafe_public/x.jsonl", "runs/TypeSafe.jsonl", "runs/teacher_jev-latest.jsonl",
                                 "eval/some_suite/published/model.jsonl"])
def test_hosted_predictions_path_refused(tmp_path, rel):
    path = _write_preds(tmp_path / rel, [_ROW])
    with pytest.raises(SystemExit, match="refusing predictions"):
        load_teacher_predictions(path)


@pytest.mark.parametrize("field,value", [("model", "jev-latest"), ("model", "typesafe:v13"), ("teacher", "systemone:Jev")])
def test_hosted_predictions_records_refused(tmp_path, field, value):
    path = _write_preds(tmp_path / "runs" / "preds.jsonl", [_ROW, {**_ROW, "item_id": "b", field: value}])
    with pytest.raises(SystemExit, match="record 2"):
        check_predictions_file(path)
