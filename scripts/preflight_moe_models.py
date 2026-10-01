"""Offline, fail-closed screening of published MoE configs for GPU canaries.

Config evidence and necessary TP checks are not proof of runtime MoE execution.
This command never imports vLLM, downloads weights, or launches GPU workers.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


# Audited conventional attention/FusedMoE implementations in the local tree.
# Tuple: expected architecture, expert-count field, top-k field, expert FFN field.
STANDARD_PROFILES = {
    "granitemoe": (
        "GraniteMoeForCausalLM", "num_local_experts",
        "num_experts_per_tok", "intermediate_size",
    ),
    "olmoe": (
        "OlmoeForCausalLM", "num_experts",
        "num_experts_per_tok", "intermediate_size",
    ),
    "gpt_oss": (
        "GptOssForCausalLM", "num_local_experts",
        "num_experts_per_tok", "intermediate_size",
    ),
    "afmoe": (
        "AfmoeForCausalLM", "num_experts",
        "num_experts_per_tok", "moe_intermediate_size",
    ),
    "ernie4_5_moe": (
        "Ernie4_5_MoeForCausalLM", "moe_num_experts",
        "moe_k", "moe_intermediate_size",
    ),
    "bailing_moe": (
        "BailingMoeForCausalLM", "num_experts",
        "num_experts_per_tok", "moe_intermediate_size",
    ),
    "qwen2_moe": (
        "Qwen2MoeForCausalLM", "num_experts",
        "num_experts_per_tok", "moe_intermediate_size",
    ),
}


def positive_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def registered_architectures(path: Path) -> set[str]:
    """Read registry dict keys without executing the module or remote code."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    architectures = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id.endswith("_MODELS")
            for target in node.targets
        ):
            continue
        architectures.update(
            key.value for key in node.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        )
    return architectures


def tp_check(config: Mapping[str, Any], tp: int) -> dict:
    """Necessary checks only; kernels/quantization still need a live canary."""
    if positive_int(tp) is None:
        raise ValueError("TP must be a positive integer")
    profile = STANDARD_PROFILES.get(config.get("model_type"))
    if profile is None:
        return {"status": "unknown", "reasons": ["unaudited_parallel_layout"]}
    dimensions = {
        "query_heads": positive_int(config.get("num_attention_heads")),
        "kv_heads": positive_int(config.get("num_key_value_heads")),
        "expert_intermediate_size": positive_int(config.get(profile[3])),
    }
    reasons = []
    for name, value in dimensions.items():
        if value is None:
            reasons.append(f"missing_{name}")
        elif name == "kv_heads":
            if (value >= tp and value % tp) or (value < tp and tp % value):
                reasons.append(f"kv_heads_{value}_cannot_partition_or_replicate_at_tp_{tp}")
        elif value % tp:
            reasons.append(f"{name}_{value}_not_divisible_by_tp_{tp}")
    status = "compatible_necessary_checks_only" if not reasons else "incompatible"
    if any(reason.startswith("missing_") for reason in reasons):
        status = "unknown" if len(reasons) == sum(
            reason.startswith("missing_") for reason in reasons
        ) else "incompatible"
    return {"status": status, "dimensions": dimensions, "reasons": reasons}


def moe_config_evidence(config: Mapping[str, Any]) -> dict:
    model_type = config.get("model_type")
    profile = STANDARD_PROFILES.get(model_type)
    if profile:
        architecture, experts_key, top_k_key, _ = profile
        architecture_matches = architecture in config.get("architectures", [])
    elif model_type == "jetmoe":
        experts_key, top_k_key = "moe_num_experts", "moe_top_k"
        architecture_matches = "JetMoEForCausalLM" in config.get("architectures", [])
    elif model_type == "molmo" and config.get("block_type") == "moe":
        experts_key, top_k_key = "moe_num_experts", "moe_top_k"
        architecture_matches = bool(config.get("auto_map", {}).get("AutoModelForCausalLM"))
    else:
        return {"available": False, "reason": "no_audited_moe_config_profile"}
    experts = positive_int(config.get(experts_key))
    top_k = positive_int(config.get(top_k_key))
    sparse = bool(experts and top_k and top_k < experts)
    enabled = config.get("use_moe", True) is not False
    if model_type == "qwen2_moe":
        layers = positive_int(config.get("num_hidden_layers"))
        step = positive_int(config.get("decoder_sparse_step", 1))
        only_dense = config.get("mlp_only_layers", [])
        enabled = enabled and bool(layers and step and any(
            (index + 1) % step == 0 and index not in only_dense
            for index in range(layers)
        ))
    return {
        "available": bool(architecture_matches and sparse and enabled),
        "num_experts": experts, "top_k": top_k,
        "architecture_matches_profile": architecture_matches,
        "scope": "published_config_not_loaded_weights_or_runtime",
    }


def unquantized_weight_bytes(snapshot: Mapping[str, Any]) -> int | None:
    if snapshot.get("config", {}).get("quantization_config"):
        return None
    parameters = (snapshot.get("safetensors") or {}).get("parameters")
    if not parameters:
        return None
    widths = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8}
    if any(dtype not in widths or positive_int(count) is None
           for dtype, count in parameters.items()):
        return None
    return sum(count * widths[dtype] for dtype, count in parameters.items())


def screen_snapshot(
    snapshot: Mapping[str, Any], registered: set[str], required_tps: tuple[int, ...],
    prompt_tokens: int, output_tokens: int, capacity_mib: int,
) -> dict:
    config = snapshot.get("config") or {}
    identity = moe_config_evidence(config)
    profile = STANDARD_PROFILES.get(config.get("model_type"))
    architecture_registered = bool(profile and profile[0] in registered)
    checks = {str(tp): tp_check(config, tp) for tp in required_tps}
    weight_bytes = unquantized_weight_bytes(snapshot)
    weight_check = None if weight_bytes is None else weight_bytes < capacity_mib * 2**20
    context_limit = positive_int(config.get("max_sequence_length")) or positive_int(
        config.get("max_position_embeddings")
    ) or positive_int(config.get("n_positions"))
    requested_context = prompt_tokens + output_tokens
    context_check = None if context_limit is None else requested_context <= context_limit
    static_ready = bool(
        not snapshot.get("fetch_error") and snapshot.get("revision")
        and identity["available"] and architecture_registered
        and all(check["status"] == "compatible_necessary_checks_only"
                for check in checks.values())
        and context_check is True and weight_check is not False
    )
    return {
        "model_id": snapshot["model_id"], "revision": snapshot.get("revision"),
        "config_url": snapshot.get("config_url"),
        "config_sha256": hashlib.sha256(json.dumps(
            config, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest(),
        "moe_config_evidence": identity,
        "architecture_registered_in_local_tree": architecture_registered,
        "tp_preflight": checks,
        "unquantized_weight_bytes_from_metadata": weight_bytes,
        "weights_alone_fit_device": weight_check,
        "weight_estimate_excludes_kv_cache_activations_and_runtime": True,
        "context_limit": context_limit, "requested_context": requested_context,
        "context_with_output_supported": context_check,
        "eligible_for_gpu_canary_after_static_screen": static_ready,
        "runtime_moe_verified": None, "runtime_routing_verified": None,
        "runtime_tp_verified": None, "memory_canary_passed": None,
        "kv_recovery_verified": None, "formal_experiment_eligible": False,
        "fetch_error": snapshot.get("fetch_error"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--required-tps", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--capacity-mib", type=int, default=16303)
    args = parser.parse_args(argv)
    if any(positive_int(value) is None for value in (
        *args.required_tps, args.prompt_tokens, args.output_tokens, args.capacity_mib,
    )):
        parser.error("TP, token counts, and capacity must be positive integers")
    snapshots = json.loads(args.snapshots.read_text(encoding="utf-8"))
    registered = registered_architectures(args.registry)
    rows = [screen_snapshot(
        snapshot, registered, tuple(args.required_tps),
        args.prompt_tokens, args.output_tokens, args.capacity_mib,
    ) for snapshot in snapshots["candidates"]]
    ready = [row["model_id"] for row in rows
             if row["eligible_for_gpu_canary_after_static_screen"]]
    print(json.dumps({
        "scope": "offline_config_screen_not_gpu_experiment",
        "required_tps": args.required_tps, "candidate_count": len(rows),
        "static_canary_candidate_count": len(ready), "static_canary_candidates": ready,
        "status": "canary_required" if ready else "blocked_no_static_candidate",
        "selected_model": None, "formal_experiment_started": False,
        "registry_sha256": hashlib.sha256(args.registry.read_bytes()).hexdigest(),
        "snapshots_sha256": hashlib.sha256(args.snapshots.read_bytes()).hexdigest(),
        "candidates": rows,
    }, indent=2))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
