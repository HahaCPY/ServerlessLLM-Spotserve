import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sllm.vllm_expert_remap import (
    _Layer,
    _cross_node_movement,
    _layers_for_plan,
    _target_ep_rank,
    aggregate_dp_engine_hook_results,
    aggregate_dp_engine_moe_metadata,
    get_all_to_all_counters,
    prepare_expert_remap,
    record_all_to_all_collective,
    record_sparse_all_to_all_transfer,
    reset_all_to_all_counters,
)
from vllm.v1.engine.core import EngineCore


class FakeQuant:
    pass


class FakeMoE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.global_num_experts = 4
        self.local_num_experts = 2
        self.expert_map = torch.tensor([0, 1, -1, -1])
        self.quant_method = FakeQuant()
        self.use_ep = True
        self.enable_eplb = False
        self.num_fused_shared_experts = 0
        self.rocm_aiter_fmoe_enabled = False
        self.ep_rank = 0
        self.ep_size = 2
        self.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                tensor_parallel_size=2, data_parallel_size=1
            )
        )
        self.moe_parallel_config = SimpleNamespace(
            all2all_backend="allgather_reducescatter"
        )
        self.w13_weight = torch.nn.Parameter(torch.arange(8.0).view(2, 4))
        self.w2_weight = torch.nn.Parameter(torch.arange(4.0).view(2, 2))

    def get_expert_weights(self):
        return [self.w13_weight.view(2, -1), self.w2_weight.view(2, -1)]


class FakeGroup:
    def rank(self):
        return 0

    def size(self):
        return 2


def make_model():
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.layers[0].experts = FakeMoE()
    return model


def make_plan():
    return {
        "live_expert_remap": True,
        "placement_fingerprint": "test-placement",
        "target_rank_count": 2,
        "target_parallel_plan": {
            "enable_expert_parallel": True,
            "tensor_parallel_size": 2,
            "data_parallel_size": 1,
        },
        "expert_to_target_rank": {
            f"layer:0/expert:{expert_id}": (
                "replica:0/ep-rank:0" if expert_id % 2 == 0
                else "replica:0/ep-rank:1"
            )
            for expert_id in range(4)
        },
    }


class ExpertRemapPreflightTests(unittest.TestCase):
    def setUp(self):
        self.model = make_model()
        self.plan = make_plan()
        self.patches = [
            patch(
                "vllm.model_executor.layers.fused_moe.layer.FusedMoE",
                FakeMoE,
            ),
            patch(
                "vllm.model_executor.layers.fused_moe."
                "unquantized_fused_moe_method.UnquantizedFusedMoEMethod",
                FakeQuant,
            ),
            patch(
                "vllm.distributed.parallel_state.get_ep_group",
                return_value=SimpleNamespace(device_group=FakeGroup()),
            ),
            patch.dict(os.environ, {"VLLM_SPOTSERVE_EXPERT_REMAP": "1"}),
        ]
        for active in self.patches:
            active.start()
            self.addCleanup(active.stop)

    def test_supported_fixed_ep_plan(self):
        result = prepare_expert_remap(self.model, self.plan)
        self.assertTrue(result["ready"], result)
        layers, _ = _layers_for_plan(self.model, self.plan)
        self.assertEqual(layers[0].old_local_ids, [0, 1])
        self.assertEqual(layers[0].new_local_ids, [0, 2])

    def test_rejects_fixed_ep_remap_across_data_parallel_engines(self):
        experts = self.model.layers[0].experts
        experts.vllm_config.parallel_config.tensor_parallel_size = 1
        experts.vllm_config.parallel_config.data_parallel_size = 2
        self.plan["target_parallel_plan"]["tensor_parallel_size"] = 1
        self.plan["target_parallel_plan"]["data_parallel_size"] = 2

        result = prepare_expert_remap(self.model, self.plan)

        self.assertFalse(result["ready"])
        self.assertEqual(
            result["reason"], "data_parallel_expert_remap_not_coordinated"
        )

        self.plan["dp_engine_coordinated"] = True
        self.plan["dp_engine_count"] = 2
        result = prepare_expert_remap(self.model, self.plan)
        self.assertTrue(result["ready"], result)

    def test_aggregates_every_dp_engine_result(self):
        result = aggregate_dp_engine_hook_results(
            [
                {
                    "applied": True,
                    "worker_results": [
                        {
                            "applied": True,
                            "worker_rank": 0,
                            "physical_weight_migration": True,
                            "moved_local_expert_shards": 2,
                            "moved_local_weight_bytes": 128,
                        },
                    ],
                    "step_boundary_barrier": True,
                },
                {
                    "applied": True,
                    "worker_results": [
                        {
                            "applied": True,
                            "worker_rank": 1,
                            "physical_weight_migration": True,
                            "moved_local_expert_shards": 2,
                            "moved_local_weight_bytes": 128,
                        },
                    ],
                    "step_boundary_barrier": True,
                },
            ],
            "applied",
        )
        self.assertTrue(result["applied"])
        self.assertEqual(result["dp_engine_count"], 2)
        self.assertEqual(result["dp_engine_success_count"], 2)
        self.assertEqual(result["worker_count"], 2)
        self.assertTrue(result["physical_weight_migration"])
        self.assertEqual(result["moved_local_expert_shards"], 4)
        self.assertEqual(result["moved_local_weight_bytes"], 256)

    def test_dp_preflight_failure_is_global(self):
        result = aggregate_dp_engine_hook_results(
            [
                {
                    "ready": False,
                    "reason": "active_requests_must_be_drained_before_expert_remap",
                    "active_requests_at_barrier": True,
                },
                {"ready": True, "worker_results": [{"ready": True}]},
            ],
            "ready",
        )
        self.assertFalse(result["ready"])
        self.assertFalse(result["success"])
        self.assertTrue(result["active_requests_at_barrier"])
        self.assertEqual(result["dp_engine_success_count"], 1)

    def test_aggregates_dp_engine_runtime_metadata(self):
        result = aggregate_dp_engine_moe_metadata([
            {
                "global_expert_hotness": {"0": 3},
                "runtime_expert_placement_shards": {
                    "layer:0/expert:0": [{"ep_rank": 0}]
                },
                "runtime_expert_placement_worker_snapshots": {"0": {}},
                "all_to_all_collective_calls": 4,
                "all_to_all_measurement_kind": (
                    "runtime_sparse_transfer_payload"
                ),
            },
            {
                "global_expert_hotness": {"1": 5},
                "runtime_expert_placement_shards": {
                    "layer:0/expert:1": [{"ep_rank": 1}]
                },
                "runtime_expert_placement_worker_snapshots": {"0": {}},
                "all_to_all_collective_calls": 6,
                "all_to_all_measurement_kind": (
                    "runtime_sparse_transfer_payload"
                ),
            },
        ])
        self.assertEqual(result["runtime_expert_placement_worker_count"], 2)
        self.assertEqual(result["runtime_expert_placement_shard_count"], 2)
        self.assertEqual(result["all_to_all_collective_calls"], 10)
        self.assertEqual(
            result["all_to_all_measurement_kind"],
            "runtime_sparse_transfer_payload",
        )

    def test_rejects_parallel_size_change(self):
        self.plan["target_parallel_plan"]["tensor_parallel_size"] = 4
        result = prepare_expert_remap(self.model, self.plan)
        self.assertFalse(result["ready"])
        self.assertEqual(result["reason"], "changing_tensor_parallel_size_not_supported")

    def test_rejects_missing_expert(self):
        del self.plan["expert_to_target_rank"]["layer:0/expert:3"]
        result = prepare_expert_remap(self.model, self.plan)
        self.assertFalse(result["ready"])
        self.assertEqual(result["reason"], "missing_target_expert:layer:0/expert:3")

    def test_rejects_duplicate_placement_on_rank(self):
        self.plan["expert_to_target_rank"]["layer:0/expert:3"] = "ep-rank:0"
        result = prepare_expert_remap(self.model, self.plan)
        self.assertFalse(result["ready"])
        self.assertTrue(result["reason"].startswith("local_expert_count_change"))

    def test_rejects_invalid_rank_encoding(self):
        self.assertEqual(_target_ep_rank("replica:0/ep-rank:1", 2), 1)
        with self.assertRaises(ValueError):
            _target_ep_rank("replica:1/ep-rank:1", 2)


class ActiveRequestBarrierTests(unittest.TestCase):
    def setUp(self):
        active_opt_in = patch.dict(
            os.environ, {"VLLM_SPOTSERVE_ACTIVE_REQUEST_REMAP": "1"}
        )
        active_opt_in.start()
        self.addCleanup(active_opt_in.stop)
        self.core = SimpleNamespace(
            scheduler=SimpleNamespace(has_unfinished_requests=lambda: True),
            batch_queue=None,
            async_scheduling=False,
            model_executor=SimpleNamespace(
                collective_rpc=lambda method, args: (
                    [{"ready": True}] if method == "prepare_expert_placement_plan"
                    else [{"applied": True, "success": True}]
                )
            ),
            _aggregate_expert_placement_hook_results=(
                EngineCore._aggregate_expert_placement_hook_results
            ),
        )
        self.core.prepare_expert_placement_plan = lambda plan: (
            EngineCore.prepare_expert_placement_plan(self.core, plan)
        )
        self.plan = make_plan()

    def test_default_rejects_active_requests(self):
        result = EngineCore.apply_expert_placement_plan(self.core, self.plan)
        self.assertFalse(result["applied"])
        self.assertEqual(
            result["reason"], "active_requests_must_be_drained_before_expert_remap"
        )

    def test_preflight_reports_active_request_without_entering_apply(self):
        calls = []

        def rpc(method, args):
            calls.append(method)
            return [{"ready": True}]

        self.core.model_executor.collective_rpc = rpc
        result = EngineCore.prepare_expert_placement_plan(self.core, self.plan)

        self.assertFalse(result["ready"])
        self.assertTrue(result["active_requests_at_barrier"])
        self.assertEqual(calls, [])

    def test_coordinated_preflight_enters_apply_without_local_recheck(self):
        calls = []

        def rpc(method, args):
            calls.append(method)
            return [{"applied": True, "success": True}]

        self.core.model_executor.collective_rpc = rpc
        self.plan.update({
            "dp_engine_coordinated": True,
            "dp_engine_count": 2,
            "dp_preflight_coordinated": True,
            "active_requests_at_barrier": False,
        })
        result = EngineCore.apply_expert_placement_plan(self.core, self.plan)

        self.assertTrue(result["applied"])
        self.assertEqual(calls, ["apply_expert_placement_plan"])

    def test_rejects_async_and_batch_queue(self):
        self.plan["allow_active_requests"] = True
        for field, value in (("async_scheduling", True), ("batch_queue", [])):
            setattr(self.core, field, value)
            result = EngineCore.apply_expert_placement_plan(self.core, self.plan)
            self.assertFalse(result["applied"])
            self.assertEqual(
                result["reason"], "active_request_remap_requires_sync_step_boundary"
            )
            setattr(self.core, field, False if field == "async_scheduling" else None)

    def test_rejects_without_runtime_opt_in(self):
        self.plan["allow_active_requests"] = True
        with patch.dict(os.environ, {"VLLM_SPOTSERVE_ACTIVE_REQUEST_REMAP": "0"}):
            result = EngineCore.apply_expert_placement_plan(self.core, self.plan)
        self.assertFalse(result["applied"])
        self.assertEqual(result["reason"], "active_request_remap_not_enabled")

    def test_sync_step_boundary_preserves_active_request(self):
        self.plan["allow_active_requests"] = True
        result = EngineCore.apply_expert_placement_plan(self.core, self.plan)
        self.assertTrue(result["applied"])
        self.assertTrue(result["active_requests_at_barrier"])
        self.assertTrue(result["step_boundary_barrier"])

    def test_collective_failure_poison_engine(self):
        self.plan["allow_active_requests"] = True
        def rpc(method, args):
            if method == "apply_expert_placement_plan":
                raise RuntimeError("transfer failed")
            return [{"ready": True}]

        self.core.model_executor.collective_rpc = rpc
        with self.assertRaisesRegex(RuntimeError, "transfer failed"):
            EngineCore.apply_expert_placement_plan(self.core, self.plan)
        self.assertTrue(self.core._spotserve_expert_remap_failed)
        with self.assertRaisesRegex(RuntimeError, "weights_inconsistent"):
            EngineCore.step(self.core)


class CrossNodeMovementTests(unittest.TestCase):
    def test_counts_only_experts_received_from_another_physical_host(self):
        module = FakeMoE()
        layer = _Layer(
            "model.layers.0.experts", 0, module, [0, 1], [0, 2],
            module.get_expert_weights(),
        )
        old_indices = [[0, 1, 2, 3]]
        self.assertEqual(
            _cross_node_movement([layer], old_indices, ["host-a", "host-b"], 0),
            (1, 24),
        )
        self.assertEqual(
            _cross_node_movement([layer], old_indices, ["host-a", "host-a"], 0),
            (0, 0),
        )
        self.assertEqual(
            _cross_node_movement([layer], old_indices, ["", ""], 0),
            (0, 0),
        )


class AllToAllCounterTests(unittest.TestCase):
    def setUp(self):
        reset_all_to_all_counters()

    def tearDown(self):
        reset_all_to_all_counters()

    def test_disabled_counter_does_not_claim_measurement(self):
        manager = SimpleNamespace(internode=False)
        with patch.dict(os.environ, {"VLLM_SPOTSERVE_A2A_TRACE": "0"}):
            record_all_to_all_collective(
                "dispatch", torch.zeros(2, 4), torch.zeros(4, 4), manager
            )
            counters = get_all_to_all_counters()
        self.assertFalse(counters["enabled"])
        self.assertFalse(counters["available"])
        self.assertEqual(counters["collective_calls"], 0)

    def test_records_real_collective_boundary_payload(self):
        manager = SimpleNamespace(internode=True)
        with patch.dict(os.environ, {"VLLM_SPOTSERVE_A2A_TRACE": "1"}):
            record_all_to_all_collective(
                "dispatch",
                (torch.zeros(2, 4), torch.zeros(2, 8)),
                (torch.zeros(4, 4), torch.zeros(4, 8)),
                manager,
            )
            record_all_to_all_collective(
                "combine", torch.zeros(4, 4), torch.zeros(2, 4), manager
            )
            counters = get_all_to_all_counters()
        self.assertTrue(counters["enabled"])
        self.assertTrue(counters["available"])
        self.assertEqual(counters["dispatch_calls"], 1)
        self.assertEqual(counters["combine_calls"], 1)
        self.assertEqual(counters["collective_calls"], 2)
        self.assertEqual(counters["internode_calls"], 2)
        self.assertEqual(counters["observed_input_bytes"], 160)
        self.assertEqual(counters["observed_output_bytes"], 224)

    def test_records_sparse_transfer_payload(self):
        with patch.dict(os.environ, {"VLLM_SPOTSERVE_A2A_TRACE": "1"}):
            record_sparse_all_to_all_transfer(
                "dispatch",
                sent_bytes=128,
                received_bytes=96,
                remote_rows=3,
                local_rows=5,
            )
            record_sparse_all_to_all_transfer(
                "combine",
                sent_bytes=64,
                received_bytes=80,
                remote_rows=3,
                local_rows=5,
            )
            counters = get_all_to_all_counters()
        self.assertTrue(counters["available"])
        self.assertEqual(counters["collective_calls"], 2)
        self.assertEqual(counters["observed_input_bytes"], 192)
        self.assertEqual(counters["observed_output_bytes"], 176)
        self.assertEqual(counters["sparse_remote_rows"], 6)
        self.assertEqual(counters["sparse_local_rows"], 10)
        self.assertEqual(
            counters["measurement_kind"],
            "runtime_sparse_transfer_payload",
        )


if __name__ == "__main__":
    unittest.main()
