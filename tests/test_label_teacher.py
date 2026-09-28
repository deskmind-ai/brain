import pytest

from deskmind_brain.train.label_teacher import check_teacher


@pytest.mark.parametrize("spec", ["local:Qwen/Qwen3.5-4B", "mlx:models/brain-4b", "systemone:brain-4b@http://127.0.0.1:8793"])
def test_local_teachers_allowed(spec):
    check_teacher(spec)


@pytest.mark.parametrize("spec", ["systemone", "systemone:jev-latest", "systemone:x@https://api.typesafe.ai", "systemone~1.2"])
def test_hosted_system_one_refused(spec):
    with pytest.raises(SystemExit):
        check_teacher(spec)
