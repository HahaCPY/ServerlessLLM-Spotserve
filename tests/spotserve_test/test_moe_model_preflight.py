from copy import deepcopy
import json

import pytest

from scripts.preflight_moe_models import (
    main, moe_config_evidence, registered_architectures,
    screen_snapshot, tp_check, unquantized_weight_bytes,
)


def granite_config():
    return {
        "model_type": "granitemoe", "architectures": ["GraniteMoeForCausalLM"],
        "num_attention_heads": 24, "num_key_value_heads": 8,
        "num_local_experts": 40, "num_experts_per_tok": 8,
        "intermediate_size": 512, "max_position_embeddings": 131072,
    }


def snapshot(config=None):
    return {
        "model_id": "unit-fixture-not-a-published-model", "revision": "fixture",
        "config": config or granite_config(),
        "safetensors": {"parameters": {"BF16": 3300000000}},
    }


def screen(value):
    return screen_snapshot(value, {"GraniteMoeForCausalLM"}, (1, 2, 3),
                           4096, 512, 16303)


def test_granite_tp3_rejects_both_kv_and_expert_dimensions():
    check = tp_check(granite_config(), 3)
    assert check["status"] == "incompatible"
    assert len(check["reasons"]) == 2
    assert any("kv_heads_8" in reason for reason in check["reasons"])
    assert any("expert_intermediate_size_512" in reason for reason in check["reasons"])


@pytest.mark.parametrize("tp", [1, 2])
def test_granite_tp1_and_tp2_pass_necessary_checks_only(tp):
    assert tp_check(granite_config(), tp)["status"] == "compatible_necessary_checks_only"


def test_kv_heads_can_replicate_when_fewer_than_tp():
    config = {**granite_config(), "num_key_value_heads": 1, "intermediate_size": 768}
    assert tp_check(config, 3)["status"] == "compatible_necessary_checks_only"


def test_query_heads_must_partition_too():
    config = {**granite_config(), "num_attention_heads": 16,
              "num_key_value_heads": 1, "intermediate_size": 768}
    assert tp_check(config, 3)["status"] == "incompatible"


@pytest.mark.parametrize("tp", [0, -1, True])
def test_invalid_tp_is_rejected(tp):
    with pytest.raises(ValueError):
        tp_check(granite_config(), tp)


def test_dense_model_with_moe_in_name_is_not_moe_evidence():
    value = snapshot({"model_type": "llama", "architectures": ["LlamaForCausalLM"]})
    value["model_id"] = "misleading-MoE-name"
    assert screen(value)["moe_config_evidence"]["available"] is False
    assert screen(value)["eligible_for_gpu_canary_after_static_screen"] is False


def test_architecture_mismatch_is_rejected():
    config = {**granite_config(), "architectures": ["GraniteForCausalLM"]}
    assert moe_config_evidence(config)["available"] is False


def test_registered_dense_fallback_cannot_authorize_moe_canary():
    config = {**granite_config(), "num_key_value_heads": 6, "intermediate_size": 768,
              "architectures": ["GraniteMoeForCausalLM", "GraniteForCausalLM"]}
    result = screen_snapshot(snapshot(config), {"GraniteForCausalLM"}, (1, 2, 3),
                             4096, 512, 16303)
    assert result["architecture_registered_in_local_tree"] is False
    assert result["eligible_for_gpu_canary_after_static_screen"] is False


def test_all_experts_selected_is_not_sparse_routing_evidence():
    config = {**granite_config(), "num_experts_per_tok": 40}
    assert moe_config_evidence(config)["available"] is False


def test_missing_dimensions_are_unknown_not_compatible():
    config = granite_config()
    del config["num_key_value_heads"]
    assert tp_check(config, 1)["status"] == "unknown"


def test_unknown_runtime_layout_is_fail_closed():
    config = {**granite_config(), "model_type": "unaudited"}
    assert tp_check(config, 1)["status"] == "unknown"
    assert screen(snapshot(config))["eligible_for_gpu_canary_after_static_screen"] is False


def test_config_compatible_fixture_does_not_authorize_formal_experiment():
    config = {**granite_config(), "num_key_value_heads": 6, "intermediate_size": 768}
    result = screen(snapshot(config))
    assert result["eligible_for_gpu_canary_after_static_screen"] is True
    assert result["formal_experiment_eligible"] is False
    assert result["runtime_moe_verified"] is None
    assert result["runtime_routing_verified"] is None
    assert result["runtime_tp_verified"] is None


def test_output_tokens_count_against_context_limit():
    config = {**granite_config(), "max_position_embeddings": 4096}
    result = screen(snapshot(config))
    assert result["requested_context"] == 4608
    assert result["context_with_output_supported"] is False


def test_quantized_parameter_metadata_is_not_estimated_as_bf16_weights():
    value = snapshot()
    value["config"]["quantization_config"] = {"quant_method": "mxfp4"}
    assert unquantized_weight_bytes(value) is None
    assert unquantized_weight_bytes({"safetensors": {"parameters": {"U8": 5}}}) is None


def test_qwen_config_with_all_dense_layers_is_rejected():
    config = {
        "model_type": "qwen2_moe", "architectures": ["Qwen2MoeForCausalLM"],
        "num_experts": 60, "num_experts_per_tok": 4, "num_hidden_layers": 2,
        "decoder_sparse_step": 1, "mlp_only_layers": [0, 1],
    }
    assert moe_config_evidence(config)["available"] is False


def test_disabled_moe_flag_is_not_overridden_by_sparse_layer_metadata():
    config = {
        "model_type": "qwen2_moe", "architectures": ["Qwen2MoeForCausalLM"],
        "num_experts": 60, "num_experts_per_tok": 4, "num_hidden_layers": 2,
        "decoder_sparse_step": 1, "use_moe": False,
    }
    assert moe_config_evidence(config)["available"] is False


def test_registry_read_does_not_execute_module(tmp_path):
    path = tmp_path / "registry.py"
    path.write_text("raise RuntimeError('must not execute')\n"
                    "_TEXT_GENERATION_MODELS = {'GraniteMoeForCausalLM': ('x', 'y')}\n")
    assert registered_architectures(path) == {"GraniteMoeForCausalLM"}


def test_blocked_cli_returns_failure_and_no_formal_claim(tmp_path, capsys):
    path = tmp_path / "snapshots.json"
    path.write_text(json.dumps({"candidates": [deepcopy(snapshot())]}))
    registry = tmp_path / "registry.py"
    registry.write_text("_TEXT_GENERATION_MODELS = {'GraniteMoeForCausalLM': ('x', 'y')}\n")
    assert main(["--snapshots", str(path), "--registry", str(registry),
                 "--required-tps", "1", "2", "3"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "blocked_no_static_candidate"
    assert report["selected_model"] is None
    assert report["formal_experiment_started"] is False


def test_default_cli_requires_tp1_and_tp2_not_tp3(tmp_path, capsys):
    path = tmp_path / "snapshots.json"
    path.write_text(json.dumps({"candidates": [snapshot()]}))
    registry = tmp_path / "registry.py"
    registry.write_text("_TEXT_GENERATION_MODELS = {'GraniteMoeForCausalLM': ('x', 'y')}\n")
    assert main(["--snapshots", str(path), "--registry", str(registry)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["required_tps"] == [1, 2]
    assert report["static_canary_candidate_count"] == 1
    assert report["selected_model"] is None
    assert report["candidates"][0]["formal_experiment_eligible"] is False
