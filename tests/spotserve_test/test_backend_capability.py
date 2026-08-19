from sllm.backends.capability import get_backend_capability
from sllm.backends.vllm_capability import get_vllm_capability


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
        (4, 1, 1, 1, False, 1, 4, 1, "verified_vllm_moe_config"),
        (4, 1, 1, 1, True, 4, 4, 1, "verified_vllm_moe_config"),
        (2, 1, 1, 1, False, 1, 2, 1, "verified_vllm_moe_config"),
        (2, 1, 1, 2, False, 1, 4, 2, "verified_vllm_moe_config"),
        (2, 1, 1, 1, True, 2, 2, 1, "verified_vllm_moe_config"),
        (2, 1, 1, 2, True, 2, 4, 2, "verified_vllm_moe_config"),
        (1, 1, 1, 1, False, 1, 1, 1, "verified_vllm_moe_config"),
        (1, 1, 1, 2, False, 1, 2, 2, "verified_vllm_moe_config"),
        (1, 1, 1, 3, False, 1, 3, 3, "verified_vllm_moe_config"),
        (1, 1, 1, 4, False, 1, 4, 4, "verified_vllm_moe_config"),
        (2, 1, 2, 1, False, 1, 4, 1, "verified_vllm_moe_config"),
        (2, 1, 2, 1, True, 2, 4, 1, "verified_vllm_moe_config"),
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

    assert capability.to_dict() == {
        "backend": "vllm",
        "model_name": "qwen3",
        "supports_tp": True,
        "supports_dp": True,
        "supports_ep": False,
        "supports_state_export": False,
        "supports_state_restore": False,
        "max_num_gpus": 1,
        "supported_configs": [
            {
                "model_name": "qwen3",
                "backend": "vllm",
                "tensor_parallel_size": 1,
                "data_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "replica_count": 1,
                "enable_expert_parallel": False,
                "effective_expert_parallel_size": 1,
                "expert_parallel_size": 1,
                "num_replicas": 1,
                "num_gpus": 1,
                "target_nodes": [],
                "reason": "current_vllm_config",
            }
        ],
    }
