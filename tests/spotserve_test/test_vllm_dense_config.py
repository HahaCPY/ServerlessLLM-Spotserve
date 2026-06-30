import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_json(relative_path: str):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_vllm_dense_configs_define_expected_policies():
    cases = {
        "config-vllm-dense-baseline.json": (
            "vllm-dense-baseline",
            "none",
            0,
        ),
        "config-vllm-dense-none.json": ("vllm-dense-none", "none", 0),
        "config-vllm-dense-naive-retry.json": (
            "vllm-dense-naive-retry",
            "naive_retry",
            2,
        ),
        "config-vllm-dense-token-replay.json": (
            "vllm-dense-token-replay",
            "generated_token_replay",
            2,
        ),
    }

    for filename, (model, policy, max_retries) in cases.items():
        config = load_json(f"examples/spotserve/{filename}")

        assert config["model"] == model
        assert config["backend"] == "vllm"
        assert config["router_num_cpus"] == 0
        assert config["num_gpus"] == 1
        assert config["backend_config"]["pretrained_model_name_or_path"] == (
            "Qwen/Qwen3-0.6B"
        )
        assert config["backend_config"]["trace_debug"] is True
        assert config["router_config"]["recovery_policy"] == policy
        assert config["router_config"]["max_retries"] == max_retries
        assert config["router_config"]["metrics_path"].endswith(
            f"{model}-router.jsonl"
        )


def test_vllm_dense_matrix_covers_baseline_and_preemption_runs():
    matrix = load_json("benchmarks/spotserve/benchmark_matrix_vllm_dense.yaml")
    runs = {run["name"]: run for run in matrix["runs"]}

    assert set(runs) == {
        "vllm-dense-no-preemption",
        "vllm-dense-preemption-none",
        "vllm-dense-naive-retry",
        "vllm-dense-token-replay",
    }
    assert runs["vllm-dense-no-preemption"]["trace"] is None
    assert runs["vllm-dense-no-preemption"]["model"] == (
        "vllm-dense-baseline"
    )

    expected_traces = {
        "vllm-dense-preemption-none": (
            "examples/spotserve/spot_trace_vllm_dense_none.jsonl"
        ),
        "vllm-dense-naive-retry": (
            "examples/spotserve/spot_trace_vllm_dense_naive_retry.jsonl"
        ),
        "vllm-dense-token-replay": (
            "examples/spotserve/spot_trace_vllm_dense_token_replay.jsonl"
        ),
    }
    for name, trace in expected_traces.items():
        run = runs[name]
        assert run["trace"] == trace
        assert run["workload"] == (
            "benchmarks/spotserve/workloads/vllm_dense_trace.jsonl"
        )
        assert run["router_metrics_path"].endswith(
            f"{run['model']}-router.jsonl"
        )
