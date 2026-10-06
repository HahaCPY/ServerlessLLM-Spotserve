import argparse
import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_pipeline():
    path = REPO_ROOT / "scripts/run_k8s_two_workspace_f1_f2.py"
    spec = importlib.util.spec_from_file_location("k8s_two_workspace_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(path: Path, model_path: str = "/models/checkpoint") -> Path:
    path.write_text(
        json.dumps({"model": {"path": model_path}}), encoding="utf-8"
    )
    return path


def test_output_must_not_overlap_checkpoint(tmp_path):
    pipeline = load_pipeline()
    checkpoint = tmp_path / "models/checkpoint"
    config_path = config(tmp_path / "config.json", str(checkpoint))

    with pytest.raises(pipeline.PipelineError, match="must not overlap"):
        pipeline.ensure_safe_output(config_path, checkpoint / "results")

    result_path = tmp_path / "results"
    assert pipeline.ensure_safe_output(config_path, result_path)["model"][
        "path"
    ] == str(checkpoint)


def test_driver_command_uses_one_output_and_endpoint(tmp_path):
    pipeline = load_pipeline()
    args = argparse.Namespace(
        config=tmp_path / "config.json",
        output_dir=tmp_path / "results",
        endpoint="http://head:8343/v1/chat/completions",
        ray_address="auto",
        ray_namespace="sllm",
        repeats=2,
    )

    command = pipeline.driver_command(args, "formal")

    assert command[0] == pipeline.sys.executable
    assert command[command.index("--phase") + 1] == "formal"
    assert command[command.index("--repeats") + 1] == "2"
    assert command[command.index("--output-dir") + 1] == str(args.output_dir)
    assert command[command.index("--endpoint") + 1] == args.endpoint


def test_summary_states_checkpoint_was_not_deleted(tmp_path):
    pipeline = load_pipeline()
    payload = {
        "status": "passed",
        "formal_repeats": 2,
        "stages": [{"name": "formal-f1-f2", "status": "passed"}],
    }

    pipeline.render_summary(tmp_path, payload)

    report = (tmp_path / "pipeline-summary.md").read_text(encoding="utf-8")
    assert "不刪除 checkpoint" in report
    assert "F1" in report and "F2" in report
    assert "每個 treatment 2 次" in report
