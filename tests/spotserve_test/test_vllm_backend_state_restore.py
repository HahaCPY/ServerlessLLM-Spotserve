import asyncio
import os

import pytest


pytest.importorskip("vllm")

from sllm.backends.backend_utils import BackendStatus
from sllm.backends.vllm_backend import LLMEngineStatusDict, VllmBackend


class FakeStatefulEngine:
    def supports_state_restore(self):
        return True

    def export_inference_state(self, request_id, **kwargs):
        return {
            "request_id": request_id,
            "instance_id": "old-vllm-0",
            "node_id": "node-0",
            "state_kind": "vllm_kv_snapshot",
            "supports_restore": True,
            "runtime_state": {"snapshot_handle": "snapshot-1"},
            "metadata": {
                "kv_block_count": 2,
                "block_ids": [10, 11],
                "block_table": {request_id: [10, 11]},
                "cache_block_size": 16,
                "cache_dtype": "auto",
                "can_restore_same_node": True,
                "can_restore_cross_node": False,
            },
        }

    def restore_inference_state(self, state, request_id, **kwargs):
        return {
            "restored": True,
            "restored_blocks": len(state["metadata"]["block_ids"]),
        }


class FakeMetadataEngine:
    def get_all_request_kv_metadata(self):
        return [self.get_request_kv_metadata("req-live")]

    def get_request_kv_metadata(self, request_id):
        return {
            "request_id": request_id,
            "found": True,
            "tokens": [1, 2, 3],
            "prompt_tokens": [1, 2],
            "output_tokens": [3],
            "completed_tokens": 3,
            "kv_block_count": 2,
            "allocated_kv_block_count": 3,
            "block_ids": [10, 11, 20],
            "kv_block_ids_by_group": [[10, 11], [20]],
            "raw_block_ids_by_group": [[10, 11], [0, 20]],
            "null_block_mask_by_group": [
                [False, False],
                [True, False],
            ],
            "block_table": {
                "group_0": [10, 11],
                "group_1": [0, 20],
            },
            "cache_block_size": 16,
            "cache_dtype": "torch.float16",
            "cache_layout": "NHD",
            "reusable_tokens_by_target": {},
            "reusable_blocks_by_target": {},
        }


class FakeMoeMetadataEngine(FakeMetadataEngine):
    def get_request_moe_metadata(self, request_id):
        return {
            "per_request_expert_route_histogram": {
                request_id: {
                    "layer:0/expert:1": 5,
                    "layer:0/expert:2": 1,
                }
            },
            "moe_route_histogram_source": "runtime_hook",
            "moe_route_histogram_kind": "runtime_observed_topk",
        }

    def get_moe_runtime_metadata(self, instance_id="", node_id=""):
        return {
            "expert_placement_available": True,
            "expert_placement_snapshot": {
                "layer:0/expert:1": {
                    "rank_id": "rank-1",
                    "node_id": node_id,
                }
            },
            "placement_source": "runtime_hook",
        }


class FakeExpertPlacementRuntime:
    def __init__(self):
        self.applied_plan = None
        self.verified_plan = None

    async def apply_expert_placement_plan(self, expert_placement_plan):
        self.applied_plan = dict(expert_placement_plan)
        return {"applied": True, "reason": "runtime_apply_succeeded"}

    def verify_expert_placement_plan(self, expert_placement_plan):
        self.verified_plan = dict(expert_placement_plan)
        return {"verified": True, "reason": "runtime_verify_succeeded"}


class FakeEmptyRestoreEngine(FakeStatefulEngine):
    def restore_inference_state(self, state, request_id, **kwargs):
        return {"restored": True, "restored_blocks": 0}


class FakeWrongStateKindEngine(FakeStatefulEngine):
    def export_inference_state(self, request_id, **kwargs):
        state = super().export_inference_state(request_id, **kwargs)
        state["state_kind"] = "token_snapshot"
        return state


class FakeStagingEngine(FakeStatefulEngine):
    def restore_inference_state(self, state, request_id, **kwargs):
        return {"restored": False, "staged": True, "expected_blocks": 2}


def make_backend(engine=None, **backend_config):
    backend = VllmBackend.__new__(VllmBackend)
    backend.model_name = "model"
    backend.backend_config = backend_config
    backend.engine = engine
    backend.request_trace = LLMEngineStatusDict()
    backend.pending_kv_restores = {}
    backend.expert_placement_runtime_status = {}
    backend.request_expert_route_histograms = {}
    backend.request_expert_route_histogram_sources = {}
    backend.request_expert_route_histogram_kinds = {}
    backend.global_expert_hotness = {}
    backend.recent_window_expert_hotness = {}
    backend._model_config_cache = None
    backend._forced_failures_seen = set()
    backend.status = BackendStatus.RUNNING
    backend.status_lock = asyncio.Lock()
    return backend


def test_spotserve_request_controls_are_removed_before_sampling_params():
    backend = make_backend(object())
    request_data = {
        "temperature": 0.0,
        "force_failure": "preempted",
        "force_fail_after_tokens": 2,
        "force_fail_once": True,
        "_completed_tokens": 2,
    }

    forced_failure = backend._pop_spotserve_request_controls(
        request_data, "req-1"
    )

    assert forced_failure["failure_mode"] == "preempted"
    assert forced_failure["fail_after_tokens"] == 2
    assert request_data == {"temperature": 0.0}


def test_spotserve_request_controls_skip_replay_requests():
    backend = make_backend(object())
    request_data = {
        "temperature": 0.0,
        "force_failure": "preempted",
        "force_fail_after_tokens": 2,
        "input_tokens": [1, 2],
    }

    forced_failure = backend._pop_spotserve_request_controls(
        request_data, "req-1", skip_forced_failure=True
    )

    assert forced_failure is None
    assert request_data == {"temperature": 0.0, "input_tokens": [1, 2]}


def test_spotserve_moe_route_histogram_is_private_request_instrumentation():
    backend = make_backend(object())
    request_data = {
        "temperature": 0.0,
        "_spotserve_per_request_expert_route_histogram": {
            "layer:0/expert:1": 4,
            "layer:1/expert:2": 3,
        },
        "_spotserve_moe_route_histogram_source": "request_fixture",
        "_spotserve_moe_route_histogram_kind": "request_fixture",
    }

    backend._pop_request_expert_route_histogram(request_data, "req-live")

    assert request_data == {"temperature": 0.0}
    assert backend.request_expert_route_histograms == {
        "req-live": {
            "layer:0/expert:1": 4,
            "layer:1/expert:2": 3,
        }
    }
    assert backend._request_expert_route_metadata("req-live") == {
        "per_request_expert_route_histogram": {
            "req-live": {
                "layer:0/expert:1": 4,
                "layer:1/expert:2": 3,
            }
        },
        "moe_route_histogram_available": True,
        "moe_route_histogram_source": "request_fixture",
        "moe_route_histogram_kind": "request_fixture",
    }


def test_spotserve_nixl_side_channel_port_is_actor_specific():
    backend = make_backend(
        object(),
        kv_transfer_config={"kv_connector": "NixlConnector"},
        nixl_side_channel_base_port=5600,
        nixl_side_channel_port_span=20000,
    )

    port_a = backend._derive_spotserve_nixl_side_channel_port("actor-a")
    port_b = backend._derive_spotserve_nixl_side_channel_port("actor-b")

    assert 5600 <= port_a < 25600
    assert 5600 <= port_b < 25600
    assert port_a != port_b


def test_spotserve_nixl_side_channel_port_allows_exact_override():
    backend = make_backend(
        object(),
        kv_transfer_config={"kv_connector": "NixlConnector"},
        nixl_side_channel_port=6123,
    )

    assert backend._derive_spotserve_nixl_side_channel_port("actor-a") == 6123


def test_spotserve_moe_route_tracing_env_is_configured(monkeypatch):
    backend = make_backend(
        object(),
        enable_moe_route_instrumentation="true",
    )
    monkeypatch.delenv("VLLM_SPOTSERVE_MOE_TRACE", raising=False)

    backend._configure_spotserve_moe_route_tracing()

    assert os.environ["VLLM_SPOTSERVE_MOE_TRACE"] == "1"

    backend.backend_config["enable_moe_route_instrumentation"] = False
    backend._configure_spotserve_moe_route_tracing()

    assert os.environ["VLLM_SPOTSERVE_MOE_TRACE"] == "0"


@pytest.mark.asyncio
async def test_export_is_restorable_only_when_runtime_hooks_exist():
    unsupported = make_backend(object())
    state = await unsupported.export_inference_state(
        request_data={"request_id": "req-1"},
        current_output=[[1, 2, 3]],
        completed_tokens=3,
    )
    assert state["supports_restore"] is False
    assert state["state_kind"] == "token_snapshot"

    supported = make_backend(FakeStatefulEngine())
    state = await supported.export_inference_state(
        request_data={"request_id": "req-1"},
        current_output=[[1, 2, 3]],
        completed_tokens=3,
    )
    assert state["supports_restore"] is True
    assert state["state_kind"] == "vllm_kv_snapshot"
    assert state["metadata"]["kv_block_count"] == 2

    wrong_kind = make_backend(FakeWrongStateKindEngine())
    state = await wrong_kind.export_inference_state(
        request_data={"request_id": "req-1"},
        current_output=[[1, 2, 3]],
        completed_tokens=3,
    )
    assert state["supports_restore"] is False
    assert state["state_kind"] == "token_snapshot"


@pytest.mark.asyncio
async def test_restore_attaches_state_and_rejects_incompatible_cache():
    backend = make_backend(
        FakeStatefulEngine(), block_size=16, kv_cache_dtype="auto"
    )
    state = await backend.export_inference_state(
        request_data={"request_id": "req-1"},
        current_output=[[1, 2, 3]],
        completed_tokens=3,
    )
    restored = await backend.restore_inference_state(
        state, {"request_id": "req-1", "node_id": "node-0"}
    )
    assert restored == {
        "restored": True,
        "restored_blocks": 2,
        "state_kind": "vllm_kv_snapshot",
        "recovered_tokens": 3,
        "restore_scope": "same_node",
    }

    incompatible = make_backend(FakeStatefulEngine(), block_size=32)
    result = await incompatible.restore_inference_state(
        state, {"request_id": "req-1", "node_id": "node-0"}
    )
    assert result["restored"] is False
    assert result["reason"] == "incompatible_cache_config"

    state["metadata"]["expert_parallel_enabled"] = True
    ep_mismatch = make_backend(
        FakeStatefulEngine(), enable_expert_parallel=False
    )
    result = await ep_mismatch.restore_inference_state(
        state, {"request_id": "req-1", "node_id": "node-0"}
    )
    assert result["restored"] is True
    assert result["restored_blocks"] == 2

    state["metadata"]["state_restore_requires_ep_layout"] = True
    ep_required = make_backend(
        FakeStatefulEngine(), enable_expert_parallel=False
    )
    result = await ep_required.restore_inference_state(
        state, {"request_id": "req-1", "node_id": "node-0"}
    )
    assert result["restored"] is False
    assert result["reason"] == "incompatible_ep_layout"


@pytest.mark.asyncio
async def test_restore_rejects_cross_node_without_transfer_support():
    backend = make_backend(FakeStatefulEngine())
    state = await backend.export_inference_state(
        request_data={"request_id": "req-1"}, current_output=[[1]]
    )
    result = await backend.restore_inference_state(
        state, {"request_id": "req-1", "node_id": "node-1"}
    )
    assert result["restored"] is False
    assert result["reason"] == "cross_node_restore_unsupported"


@pytest.mark.asyncio
async def test_restore_rejects_success_without_restored_kv_blocks():
    backend = make_backend(FakeEmptyRestoreEngine())
    state = await backend.export_inference_state(
        request_data={"request_id": "req-1"}, current_output=[[1]]
    )
    result = await backend.restore_inference_state(
        state, {"request_id": "req-1", "node_id": "node-0"}
    )
    assert result == {
        "restored": False,
        "reason": "vllm_kv_restore_empty",
        "state_kind": "vllm_kv_snapshot",
    }


@pytest.mark.asyncio
async def test_restore_stages_nixl_payload_without_reporting_early_success():
    backend = make_backend(FakeStagingEngine())
    state = await backend.export_inference_state(
        request_data={"request_id": "req-1"}, current_output=[[1]]
    )

    result = await backend.restore_inference_state(
        state, {"request_id": "req-1", "node_id": "node-0"}
    )

    assert result == {
        "restored": False,
        "staged": True,
        "expected_blocks": 2,
        "state_kind": "vllm_kv_snapshot",
    }
    assert backend.pending_kv_restores == {"req-1": 2}


@pytest.mark.asyncio
async def test_context_metadata_reads_active_requests_from_runtime():
    backend = make_backend(FakeMetadataEngine())

    contexts = await backend.get_context_metadata("instance-0", "node-0")

    assert len(contexts) == 1
    assert contexts[0]["request_id"] == "req-live"
    assert contexts[0]["num_tokens"] == 3
    assert contexts[0]["context_blocks"] == 2
    assert contexts[0]["metadata"]["block_table"] == {
        "group_0": [10, 11],
        "group_1": [0, 20],
    }
    assert contexts[0]["metadata"]["cache_dtype"] == "torch.float16"
    assert contexts[0]["supports_state_restore"] is False


@pytest.mark.asyncio
async def test_context_metadata_merges_moe_route_histogram_instrumentation():
    backend = make_backend(FakeMetadataEngine())
    backend._record_request_expert_route_histogram(
        "req-live",
        {"layer:0/expert:1": 6},
        "request_fixture",
        "request_fixture",
    )

    contexts = await backend.get_context_metadata("instance-0", "node-0")

    assert contexts[0]["metadata"]["moe_route_histogram_available"] is True
    assert contexts[0]["metadata"]["moe_route_histogram_source"] == (
        "request_fixture"
    )
    assert contexts[0]["metadata"]["moe_route_histogram_kind"] == (
        "request_fixture"
    )
    assert contexts[0]["metadata"]["per_request_expert_route_histogram"] == {
        "req-live": {"layer:0/expert:1": 6}
    }


@pytest.mark.asyncio
async def test_context_metadata_reads_moe_route_histogram_from_runtime_hook():
    backend = make_backend(FakeMoeMetadataEngine())

    contexts = await backend.get_context_metadata("instance-0", "node-0")

    assert contexts[0]["metadata"]["moe_route_histogram_available"] is True
    assert contexts[0]["metadata"]["moe_route_histogram_source"] == (
        "runtime_hook"
    )
    assert contexts[0]["metadata"]["moe_route_histogram_kind"] == (
        "runtime_observed_topk"
    )
    assert contexts[0]["metadata"]["per_request_expert_route_histogram"] == {
        "req-live": {
            "layer:0/expert:1": 5,
            "layer:0/expert:2": 1,
        }
    }


def test_engine_parallel_metadata_derives_moe_placement_from_model_config(
    tmp_path,
):
    model_path = tmp_path / "moe-model"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        '{"num_hidden_layers": 2, "num_experts": 3}',
        encoding="utf-8",
    )
    backend = make_backend(
        object(),
        pretrained_model_name_or_path=str(model_path),
        enable_expert_parallel=True,
        tensor_parallel_size=2,
    )

    metadata = backend._engine_parallel_metadata(
        instance_id="instance-0",
        node_id="node-0",
    )

    assert metadata["expert_placement_available"] is True
    assert len(metadata["expert_placement_snapshot"]) == 6
    assert metadata["placement_source"] == "derived_from_model_config"
    assert metadata["expert_placement_snapshot"]["layer:0/expert:2"][
        "rank_id"
    ] == "ep-rank-0"


def test_engine_parallel_metadata_separates_plan_contract_from_snapshot():
    snapshot = {
        "layer:0/expert:1": {
            "layer_id": 0,
            "expert_id": 1,
            "rank_id": "replica:0/ep-rank:1",
            "node_id": "node-0",
        }
    }
    backend = make_backend(
        object(),
        enable_expert_parallel=True,
        tensor_parallel_size=2,
        placement_epoch=8,
        expert_placement_plan={
            "expert_placement_available": True,
            "placement_epoch": 8,
            "placement_source": "logical_reparallelization_planner",
            "placement_fingerprint": "plan-fp",
            "expert_placement_snapshot": snapshot,
        },
        expert_placement_snapshot=snapshot,
    )

    metadata = backend._engine_parallel_metadata(
        instance_id="instance-0",
        node_id="node-0",
    )

    assert metadata["expert_placement_available"] is True
    assert metadata["expert_placement_contract_available"] is True
    assert metadata["expert_placement_contract_bound"] is True
    assert metadata["expert_placement_fingerprint"] == "plan-fp"
    assert metadata["expert_placement_plan_fingerprint"] == "plan-fp"
    assert metadata["expert_placement_snapshot_fingerprint"]
    assert metadata["expert_placement_contract_snapshot_fingerprint"]
    assert metadata["expert_placement_contract_snapshot_match"] is True
    assert metadata["expert_placement_plan_applied"] is False
    assert metadata["expert_placement_plan_verified"] is False
    assert metadata["expert_placement_contract_reason"] == "runtime_not_applied"


def test_engine_parallel_metadata_can_report_runtime_verified_plan():
    snapshot = {
        "layer:0/expert:1": {
            "layer_id": 0,
            "expert_id": 1,
            "rank_id": "replica:0/ep-rank:1",
            "node_id": "node-0",
        }
    }
    backend = make_backend(
        object(),
        enable_expert_parallel=True,
        tensor_parallel_size=2,
        placement_epoch=8,
        expert_placement_plan={
            "expert_placement_available": True,
            "placement_epoch": 8,
            "placement_source": "logical_reparallelization_planner",
            "placement_fingerprint": "plan-fp",
            "expert_placement_snapshot": snapshot,
        },
        expert_placement_snapshot=snapshot,
        expert_placement_plan_applied=True,
        expert_placement_plan_verified=True,
    )

    metadata = backend._engine_parallel_metadata(
        instance_id="instance-0",
        node_id="node-0",
    )

    assert metadata["expert_placement_plan_applied"] is True
    assert metadata["expert_placement_plan_verified"] is True
    assert metadata["expert_placement_contract_reason"] == "verified_runtime_plan"


@pytest.mark.asyncio
async def test_backend_calls_runtime_expert_placement_apply_and_verify_hooks():
    snapshot = {
        "layer:0/expert:1": {
            "layer_id": 0,
            "expert_id": 1,
            "rank_id": "replica:0/ep-rank:1",
            "node_id": "node-0",
        }
    }
    runtime = FakeExpertPlacementRuntime()
    backend = make_backend(
        runtime,
        enable_expert_parallel=True,
        tensor_parallel_size=2,
        placement_epoch=8,
        expert_placement_plan={
            "expert_placement_available": True,
            "placement_epoch": 8,
            "placement_source": "logical_reparallelization_planner",
            "placement_fingerprint": "plan-fp",
            "expert_placement_snapshot": snapshot,
        },
        expert_placement_snapshot=snapshot,
    )

    await backend._apply_configured_expert_placement_plan()
    metadata = backend._engine_parallel_metadata(
        instance_id="instance-0",
        node_id="node-0",
    )

    assert runtime.applied_plan["placement_fingerprint"] == "plan-fp"
    assert runtime.verified_plan["placement_fingerprint"] == "plan-fp"
    assert metadata["expert_placement_apply_hook_available"] is True
    assert metadata["expert_placement_apply_attempted"] is True
    assert metadata["expert_placement_apply_success"] is True
    assert metadata["expert_placement_apply_reason"] == (
        "runtime_apply_succeeded"
    )
    assert metadata["expert_placement_verify_hook_available"] is True
    assert metadata["expert_placement_verify_attempted"] is True
    assert metadata["expert_placement_verify_success"] is True
    assert metadata["expert_placement_verify_reason"] == (
        "runtime_verify_succeeded"
    )
    assert metadata["expert_placement_plan_applied"] is True
    assert metadata["expert_placement_plan_verified"] is True
    assert metadata["expert_placement_contract_reason"] == "verified_runtime_plan"


@pytest.mark.asyncio
async def test_backend_reports_missing_runtime_expert_placement_apply_hook():
    snapshot = {
        "layer:0/expert:1": {
            "layer_id": 0,
            "expert_id": 1,
            "rank_id": "replica:0/ep-rank:1",
            "node_id": "node-0",
        }
    }
    backend = make_backend(
        object(),
        enable_expert_parallel=True,
        tensor_parallel_size=2,
        placement_epoch=8,
        expert_placement_plan={
            "expert_placement_available": True,
            "placement_epoch": 8,
            "placement_source": "logical_reparallelization_planner",
            "placement_fingerprint": "plan-fp",
            "expert_placement_snapshot": snapshot,
        },
        expert_placement_snapshot=snapshot,
    )

    await backend._apply_configured_expert_placement_plan()
    metadata = backend._engine_parallel_metadata(
        instance_id="instance-0",
        node_id="node-0",
    )

    assert metadata["expert_placement_apply_hook_available"] is False
    assert metadata["expert_placement_apply_attempted"] is False
    assert metadata["expert_placement_apply_success"] is False
    assert metadata["expert_placement_apply_reason"] == (
        "runtime_apply_hook_unavailable"
    )
    assert metadata["expert_placement_verify_hook_available"] is False
    assert metadata["expert_placement_verify_attempted"] is False
    assert metadata["expert_placement_verify_success"] is False
    assert metadata["expert_placement_verify_reason"] == (
        "runtime_verify_hook_unavailable"
    )
    assert metadata["expert_placement_plan_applied"] is False
    assert metadata["expert_placement_plan_verified"] is False
    assert metadata["expert_placement_contract_reason"] == (
        "runtime_apply_hook_unavailable"
    )


@pytest.mark.asyncio
async def test_runtime_metadata_reads_moe_placement_from_runtime_hook():
    backend = make_backend(FakeMoeMetadataEngine())

    metadata = await backend.get_runtime_metadata("instance-0", "node-0")

    profile = metadata["model_resource_profile"]
    assert profile["expert_placement_available"] is True
    assert profile["placement_source"] == "runtime_hook"
    assert profile["expert_placement_snapshot"] == {
        "layer:0/expert:1": {
            "rank_id": "rank-1",
            "node_id": "node-0",
        }
    }


@pytest.mark.asyncio
async def test_export_queries_runtime_when_request_trace_has_no_output():
    backend = make_backend(FakeMetadataEngine())

    state = await backend.export_inference_state(
        request_data={"request_id": "req-live"}
    )

    assert state["tokens"] == [1, 2, 3]
    assert state["completed_tokens"] == 3
    assert state["metadata"]["kv_block_count"] == 2
    assert state["metadata"]["cache_layout"] == "NHD"
    assert state["supports_restore"] is False
