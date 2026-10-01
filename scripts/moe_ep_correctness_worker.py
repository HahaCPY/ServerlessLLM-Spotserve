"""Read-only runtime correctness and kernel profiling; native operations unchanged."""

from collections import defaultdict
from contextlib import nullcontext
import functools
import inspect

from scripts import moe_diagnostic_trace as trace
from scripts.moe_architecture_diagnostic_worker import ArchitectureDiagnosticWorker
import torch


class EPCorrectnessDiagnosticWorker(ArchitectureDiagnosticWorker):
    def _install_timers(self):
        super()._install_timers()
        from vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter
        from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
        self.correctness_enabled = False
        self.oracle_seen = set()
        self.oracle_records = []
        self.router_totals = defaultdict(float)
        self.router_objects = set()
        self.router_histograms = defaultdict(lambda: [0] * 40)
        self.kernel_profiler = None
        self.collective_diagnostics_enabled = False
        self.collective_records = {}

        original_route = FusedMoERouter.select_experts
        route_signature = inspect.signature(original_route)

        @functools.wraps(original_route)
        def route(owner, *args, **kwargs):
            with self.annotation("router"):
                result = original_route(owner, *args, **kwargs)
            if self.correctness_enabled:
                b = route_signature.bind(owner, *args, **kwargs)
                self.check_router(owner, b.arguments["router_logits"], *result)
            return result

        FusedMoERouter.select_experts = route
        trace.timed(FusedMoERouter, "select_experts", "router", gpu=True)
        original_experts = TritonExperts.apply
        expert_signature = inspect.signature(original_experts)

        @functools.wraps(original_experts)
        def experts(owner, *args, **kwargs):
            with self.annotation("expert_apply"):
                result = original_experts(owner, *args, **kwargs)
            if self.correctness_enabled and id(owner) not in self.oracle_seen:
                b = expert_signature.bind(owner, *args, **kwargs).arguments
                if b["hidden_states"].shape[0]:
                    self.oracle_seen.add(id(owner))
                    self.check_expert(b)
            return result

        TritonExperts.apply = experts
        trace.timed(TritonExperts, "apply", "expert_apply", gpu=True)

        # Optional small-input gate.  Native collectives run first; these
        # wrappers only copy their inputs/outputs for host-side conservation
        # checks and are disabled during performance measurements.
        from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
        original_dispatch = AgRsAll2AllManager.dispatch
        original_combine = AgRsAll2AllManager.combine
        original_reduce = MoERunner._maybe_reduce_final_output

        def payload(tensor):
            value = tensor.detach()
            return {"shape": list(value.shape), "dtype": str(value.dtype),
                    "values": value.float().cpu().tolist()}

        @functools.wraps(original_dispatch)
        def dispatch(owner, hidden_states, topk_weights, topk_ids, *args, **kwargs):
            capture = self.collective_diagnostics_enabled and "agrs_dispatch" not in self.collective_records
            before = ({"hidden": payload(hidden_states), "weights": payload(topk_weights),
                       "ids": payload(topk_ids)} if capture else None)
            result = original_dispatch(owner, hidden_states, topk_weights, topk_ids, *args, **kwargs)
            if capture:
                self.collective_records["agrs_dispatch"] = {"input": before,
                    "output": {"hidden": payload(result[0]), "weights": payload(result[1]),
                               "ids": payload(result[2])}}
            return result

        @functools.wraps(original_combine)
        def combine(owner, hidden_states, *args, **kwargs):
            capture = self.collective_diagnostics_enabled and "agrs_combine" not in self.collective_records
            before = payload(hidden_states) if capture else None
            result = original_combine(owner, hidden_states, *args, **kwargs)
            if capture:
                self.collective_records["agrs_combine"] = {
                    "input": before, "output": payload(result)}
            return result

        @functools.wraps(original_reduce)
        def reduce(owner, states, trunc_size):
            capture = self.collective_diagnostics_enabled and "moe_final_reduce" not in self.collective_records
            before = payload(states) if capture else None
            result = original_reduce(owner, states, trunc_size)
            if capture:
                from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
                config = owner.moe_config
                self.collective_records["moe_final_reduce"] = {
                    "tp_size": config.tp_size, "ep_size": config.ep_size,
                    "actual_tp_group_size": get_tensor_model_parallel_world_size(),
                    "is_sequence_parallel": config.is_sequence_parallel,
                    "skip_final_all_reduce": config.skip_final_all_reduce,
                    "fused_output_is_reduced": owner._fused_output_is_reduced,
                    "input": before, "output": payload(result)}
            return result

        AgRsAll2AllManager.dispatch = dispatch
        AgRsAll2AllManager.combine = combine
        MoERunner._maybe_reduce_final_output = reduce

    def annotation(self, label):
        return (torch.profiler.record_function("MoEDiag:" + label)
                if self.kernel_profiler is not None else nullcontext())

    def check_router(self, owner, logits, weights, ids):
        self.router_objects.add(id(owner))
        if not ids.numel():
            return
        t = self.router_totals
        reference = torch.softmax(logits.float(), dim=-1)
        expected_values, expected_ids = reference.topk(8, dim=-1)
        normalized = reference.gather(-1, ids.long().clamp(0, 39))
        normalized = normalized / normalized.sum(-1, keepdim=True)
        t["rows"] += ids.shape[0]
        t["assignments"] += ids.numel()
        t["invalid_ids"] += ((ids < 0) | (ids >= 40)).sum().item()
        t["duplicate_rows"] += (ids.sort(-1).values.diff(dim=-1) == 0).any(-1).sum().item()
        different = (ids.sort(-1).values != expected_ids.sort(-1).values).any(-1)
        boundary = reference.topk(9, dim=-1).values
        t["topk_set_different_rows"] += different.sum().item()
        t["topk_non_tie_different_rows"] += (different & ((boundary[:, 7] - boundary[:, 8]) > 1e-5)).sum().item()
        t["max_weight_reference_error"] = max(t["max_weight_reference_error"], (weights-normalized).abs().max().item())
        t["max_weight_sum_error"] = max(t["max_weight_sum_error"], (weights.sum(-1)-1).abs().max().item())
        t["nonfinite_weights"] += (~torch.isfinite(weights)).sum().item()
        histogram = torch.bincount(ids.flatten().long().clamp(0, 39), minlength=40).cpu().tolist()
        key = str(id(owner))
        self.router_histograms[key] = [a+b for a,b in zip(self.router_histograms[key], histogram)]

    def check_expert(self, b):
        # Oracle samples include the tail under large dispatch loads. It uses
        # the actual rank-local weights in FP32, not a different checkpoint.
        x, ids, weights = b["hidden_states"], b["topk_ids"], b["topk_weights"]
        rows = (torch.arange(x.shape[0], device=x.device) if x.shape[0] <= 128 else
                torch.linspace(0, x.shape[0]-1, 16, device=x.device).long().unique())
        xs, selected, ws = x[rows].float(), ids[rows].long(), weights[rows].float()
        mapping = b["expert_map"]
        local_ids = selected if mapping is None else mapping[selected].long()
        expected = torch.zeros_like(xs)
        w13, w2 = b["w1"], b["w2"]
        with torch.inference_mode():
            for local in local_ids.unique().tolist():
                if local < 0:
                    continue
                r, k = (local_ids == local).nonzero(as_tuple=True)
                projected = torch.nn.functional.linear(xs[r], w13[local].float())
                first, second = projected.chunk(2, dim=-1)
                activated = torch.nn.functional.silu(first) * second
                values = torch.nn.functional.linear(activated, w2[local].float()) * ws[r, k, None]
                expected.index_add_(0, r, values)
        observed = b["output"][rows].float()
        error = observed - expected
        relative = error.norm().item() / max(expected.norm().item(), 1e-8)
        maximum, scale = error.abs().max().item(), expected.abs().max().item()
        self.oracle_records.append({"sample_rows": rows.cpu().tolist(), "dispatch_rows": x.shape[0],
            "relative_l2_error": relative, "max_abs_error": maximum, "reference_max_abs": scale,
            "nonfinite_output_values": (~torch.isfinite(observed)).sum().item(),
            "passes_predeclared_tolerance": relative <= 0.03 and maximum <= 0.02 + 0.1*scale,
            "w13_shape": list(w13.shape), "w2_shape": list(w2.shape),
            "activation": str(b["activation"]), "apply_router_weight_on_input": b["apply_router_weight_on_input"]})

    def set_collective_diagnostics(self, enabled):
        self.collective_diagnostics_enabled = bool(enabled)
        self.collective_records = {}

    def collective_snapshot(self):
        model = self.model_runner.model
        first = next(module for module in model.modules()
                     if hasattr(module, "expert_map") and hasattr(module, "moe_config"))
        mapping = first.expert_map
        return {"rank": self.rank, "local_rank": self.local_rank,
                "actual_gpu_uuid": str(torch.cuda.get_device_properties(self.device).uuid),
                "global_expert_ids": (list(range(first.moe_config.num_experts)) if mapping is None else
                    [i for i, local in enumerate(mapping.cpu().tolist()) if local >= 0]),
                "records": self.collective_records}

    def reset_capacity_peak(self):
        torch.cuda.reset_peak_memory_stats(self.device)

    def capacity_snapshot(self):
        def storage_bytes(item, seen):
            if isinstance(item, torch.Tensor):
                ptr = item.untyped_storage().data_ptr()
                if ptr in seen:
                    return 0
                seen.add(ptr)
                return item.untyped_storage().nbytes()
            if isinstance(item, dict):
                return sum(storage_bytes(x, seen) for x in item.values())
            if isinstance(item, (tuple, list)):
                return sum(storage_bytes(x, seen) for x in item)
            return 0

        runner = self.model_runner
        cache_config = runner.vllm_config.cache_config
        free, total = torch.cuda.mem_get_info(self.device)
        return {"rank": self.rank, "local_rank": self.local_rank,
            "actual_gpu_uuid": str(torch.cuda.get_device_properties(self.device).uuid),
            "total_bytes": total, "free_bytes": free,
            "allocated_bytes": torch.cuda.memory_allocated(self.device),
            "reserved_bytes": torch.cuda.memory_reserved(self.device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            "parameter_bytes": sum(x.numel()*x.element_size() for x in runner.model.parameters()),
            "kv_storage_bytes": storage_bytes(getattr(runner, "kv_caches", None), set()),
            "gpu_blocks": cache_config.num_gpu_blocks, "block_size": cache_config.block_size}

    def set_correctness_diagnostics(self, enabled):
        self.correctness_enabled = enabled
        self.oracle_seen.clear()
        self.oracle_records.clear()
        self.router_totals.clear()
        self.router_objects.clear()
        self.router_histograms.clear()

    def correctness_snapshot(self):
        return {"rank": self.rank, "local_rank": self.local_rank,
                "router_totals": dict(self.router_totals), "router_objects": len(self.router_objects),
                "router_histograms": dict(self.router_histograms), "expert_oracle": self.oracle_records,
                "oracle_scope": "all rows for <=128 rows per expert-kernel object; otherwise 16 including first/tail"}

    def start_kernel_profiler(self):
        if self.kernel_profiler is not None:
            raise ValueError("profiler already active")
        self.kernel_profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
        self.kernel_profiler.start()

    def stop_kernel_profiler(self):
        profiler = self.kernel_profiler
        profiler.stop()
        self.kernel_profiler = None
        events = [{"name": e.key, "device_type": str(e.device_type), "calls": e.count,
                   "self_cpu_us": e.self_cpu_time_total,
                   "self_device_us": e.self_device_time_total,
                   "inclusive_device_us": e.device_time_total}
                  for e in profiler.key_averages()]
        return {"rank": self.rank, "local_rank": self.local_rank,
                "cuda_kernels": sorted([e for e in events if "CUDA" in e["device_type"]],
                                       key=lambda e:e["self_device_us"], reverse=True)[:80],
                "annotations": [e for e in events if e["name"].startswith("MoEDiag:")],
                "cpu_operations": sorted([e for e in events if "CPU" in e["device_type"]],
                                         key=lambda e:e["self_cpu_us"], reverse=True)[:25]}
