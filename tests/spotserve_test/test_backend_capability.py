import json

import pytest

from sllm.backends.capability import get_backend_capability
from sllm.backends.vllm_capability import get_vllm_capability


def test_granite_moe_is_detected_from_config_not_directory_name(tmp_path):
    model = tmp_path / "granite-3.1-3b-a800m-instruct"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "architectures": ["GraniteMoeForCausalLM"], "model_type": "granitemoe",
        "num_local_experts": 40, "num_experts_per_tok": 8,
    }))
    cap = get_vllm_capability({"model": model.name, "num_gpus": 4,
        "backend_config": {"pretrained_model_name_or_path": str(model),
                           "tensor_parallel_size": 1}})
    assert {plan.tensor_parallel_size for plan in cap.supported_configs} == {1, 2, 4}
    assert any(plan.num_gpus == 1 for plan in cap.supported_configs)


def test_local_dense_config_overrides_a_misleading_moe_name(tmp_path):
    model = tmp_path / "not-actually-moe"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2",
    }))
    cap = get_vllm_capability({"model": model.name, "num_gpus": 1,
        "backend_config": {"pretrained_model_name_or_path": str(model),
                           "tensor_parallel_size": 1}})
    assert len(cap.supported_configs) == 1
    assert cap.supports_ep is False


@pytest.mark.parametrize("content", ["{broken-json", "[]", None])
def test_invalid_local_config_is_not_silently_classified_by_name(tmp_path, content):
    model = tmp_path / "moe-checkpoint"
    model.mkdir()
    if content is not None:
        (model / "config.json").write_text(content)
    with pytest.raises(ValueError, match="config"):
        get_vllm_capability({"model": model.name, "num_gpus": 1,
            "backend_config": {"pretrained_model_name_or_path": str(model)}})


def test_vllm_capability_advertises_current_tp_shape_only():
    capability = get_vllm_capability(
        {
            "model": "qwen3-dense",
            "backend": "vllm",
            "num_gpus": 4,
            "backend_config": {
                "tensor_parallel_size": 2,
            },
        }
    )

    assert capability.backend == "vllm"
    assert capability.model_name == "qwen3-dense"
    assert capability.supports_tp is True
    assert capability.supports_dp is True
    assert capability.supports_ep is False
    assert capability.supports_state_export is False
    assert capability.supports_state_restore is False
    assert capability.max_num_gpus == 4

    assert len(capability.supported_configs) == 1
    plan = capability.supported_configs[0]
    assert plan.model_name == "qwen3-dense"
    assert plan.backend == "vllm"
    assert plan.tensor_parallel_size == 2
    assert plan.data_parallel_size == 1
    assert plan.pipeline_parallel_size == 1
    assert plan.replica_count == 1
    assert plan.enable_expert_parallel is False
    assert plan.effective_expert_parallel_size == 1
    assert plan.expert_parallel_size == 1
    assert plan.num_replicas == 1
    assert plan.num_gpus == 4
    assert plan.reason == "current_vllm_config"


def test_vllm_moe_capability_advertises_verified_shapes():
    capability = get_vllm_capability(
        {
            "model": "vllm-moe",
            "backend": "vllm",
            "num_gpus": 1,
            "backend_config": {
                "pretrained_model_name_or_path": "Qwen/Qwen1.5-MoE-A2.7B",
                "tensor_parallel_size": 1,
                "trust_remote_code": True,
            },
        }
    )

    configs = {
        (
            plan.tensor_parallel_size,
            plan.data_parallel_size,
            plan.pipeline_parallel_size,
            plan.replica_count,
            plan.enable_expert_parallel,
            plan.effective_expert_parallel_size,
            plan.num_gpus,
            plan.num_replicas,
            plan.reason,
        )
        for plan in capability.supported_configs
    }

    assert capability.supports_ep is True
    assert capability.max_num_gpus == 4
    assert configs == {
        (4, 1, 1, 1, False, 1, 4, 1, "static_vllm_moe_catalogue"),
        (4, 1, 1, 1, True, 4, 4, 1, "static_vllm_moe_catalogue"),
        (2, 1, 1, 1, False, 1, 2, 1, "static_vllm_moe_catalogue"),
        (2, 1, 1, 2, False, 1, 4, 2, "static_vllm_moe_catalogue"),
        (2, 1, 1, 1, True, 2, 2, 1, "static_vllm_moe_catalogue"),
        (1, 2, 1, 1, True, 2, 2, 1, "static_vllm_moe_catalogue"),
        (1, 4, 1, 1, True, 4, 4, 1, "static_vllm_moe_catalogue"),
        (2, 2, 1, 1, True, 4, 4, 1, "static_vllm_moe_catalogue"),
        (2, 1, 1, 2, True, 2, 4, 2, "static_vllm_moe_catalogue"),
        (1, 1, 1, 1, False, 1, 1, 1, "static_vllm_moe_catalogue"),
        (1, 1, 1, 2, False, 1, 2, 2, "static_vllm_moe_catalogue"),
        (1, 1, 1, 3, False, 1, 3, 3, "static_vllm_moe_catalogue"),
        (1, 1, 1, 4, False, 1, 4, 4, "static_vllm_moe_catalogue"),
        (2, 1, 2, 1, False, 1, 4, 1, "static_vllm_moe_catalogue"),
        (2, 1, 2, 1, True, 2, 4, 1, "static_vllm_moe_catalogue"),
    }


def test_vllm_capability_defaults_to_configured_num_gpus_as_tp():
    capability = get_vllm_capability(
        {
            "model": "qwen3",
            "backend": "vllm",
            "num_gpus": 2,
            "backend_config": {},
        }
    )

    plan = capability.supported_configs[0]
    assert capability.max_num_gpus == 2
    assert plan.tensor_parallel_size == 2
    assert plan.num_gpus == 2


def test_backend_capability_dispatches_vllm_and_ignores_unknown_backends():
    capability = get_backend_capability(
        {
            "model": "qwen3",
            "backend": "vllm",
            "num_gpus": 1,
            "backend_config": {},
        }
    )

    assert capability is not None
    assert capability.backend == "vllm"
    assert get_backend_capability({"backend": "transformers"}) is None


def test_backend_capability_serializes_supported_configs():
    capability = get_vllm_capability(
        {
            "model": "qwen3",
            "backend": "vllm",
            "num_gpus": 1,
            "backend_config": {},
        }
    )

    payload = capability.to_dict()

    assert payload["backend"] == "vllm"
    assert payload["model_name"] == "qwen3"
    assert payload["supports_tp"] is True
    assert payload["supports_dp"] is True
    assert payload["supports_ep"] is False
    assert payload["supports_state_export"] is False
    assert payload["supports_state_restore"] is False
    assert payload["max_num_gpus"] == 1
    assert len(payload["supported_configs"]) == 1

    plan = payload["supported_configs"][0]
    assert plan["model_name"] == "qwen3"
    assert plan["backend"] == "vllm"
    assert plan["tensor_parallel_size"] == 1
    assert plan["data_parallel_size"] == 1
    assert plan["pipeline_parallel_size"] == 1
    assert plan["replica_count"] == 1
    assert plan["enable_expert_parallel"] is False
    assert plan["effective_expert_parallel_size"] == 1
    assert plan["expert_parallel_size"] == 1
    assert plan["num_replicas"] == 1
    assert plan["num_gpus"] == 1
    assert plan["target_nodes"] == []
    assert plan["expert_placement_plan"] is None
    assert plan["reason"] == "current_vllm_config"
