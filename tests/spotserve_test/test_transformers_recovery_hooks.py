import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("datasets")

from sllm.backends.backend_utils import BackendStatus
from sllm.backends.transformers_backend import DeletingException, InferenceStatus


class StreamValue:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


def test_transformers_streamer_forced_preemption_reports_tokens():
    status = InferenceStatus(BackendStatus.RUNNING)
    status.configure_forced_failure(
        "preempted", fail_after_tokens=2, prompt_tokens=3
    )

    status.put(StreamValue([[10, 11, 12, 13]]))

    with pytest.raises(DeletingException):
        status.put(StreamValue([[14]]))

    assert status.forced_failure is None
    assert len(status.get()[0]) == 5


def test_transformers_streamer_forced_failure_can_clear_tokens():
    status = InferenceStatus(BackendStatus.RUNNING)
    status.configure_forced_failure(
        "error",
        fail_after_tokens=1,
        prompt_tokens=3,
        no_current_tokens=True,
    )

    with pytest.raises(RuntimeError, match="Forced transformers backend failure"):
        status.put(StreamValue([[10, 11, 12, 13]]))

    assert status.forced_failure is None
    assert status.get() == []
