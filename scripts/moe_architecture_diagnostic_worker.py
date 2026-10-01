"""Diagnostic-only worker subclass. No vLLM source or loading policy edits."""

import dataclasses
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import time

from scripts import moe_diagnostic_trace as trace

trace.emit("diagnostic_worker_module_entry")
with trace.span("diagnostic_worker_imports"):
    import torch
    import vllm
    import vllm.v1.worker.gpu_worker as worker_module
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker


class ArchitectureDiagnosticWorker(Worker):
    def __init__(self, *args, **kwargs):
        with trace.span("worker_constructor"):
            super().__init__(*args, **kwargs)
        self.startup_trace = None
        self._install_timers()

    def _install_timers(self):
        import vllm.model_executor.model_loader.base_loader as base
        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
        import vllm.model_executor.layers.fused_moe.layer as layer
        from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
        import vllm.model_executor.layers.fused_moe.runner.moe_runner as moe_module

        trace.marked(worker_module, "init_worker_distributed_environment", "distributed_initialization")
        original_cuda_init = torch.cuda._lazy_init

        def cuda_init():
            if torch.cuda.is_initialized():
                return original_cuda_init()
            with trace.span("cuda_lazy_initialization"):
                return original_cuda_init()

        torch.cuda._lazy_init = cuda_init
        trace.marked(base, "initialize_model", "model_structure_and_weight_allocation")
        trace.marked(base, "process_weights_after_loading", "weight_postprocessing")
        trace.marked(DefaultModelLoader, "load_weights", "checkpoint_and_weight_loading")
        trace.marked(DefaultModelLoader, "_prepare_weights", "checkpoint_file_discovery")
        original_iterator = DefaultModelLoader.get_all_weights

        def iterator(loader, *args, **kwargs):
            source = iter(original_iterator(loader, *args, **kwargs))
            while True:
                before = time.monotonic()
                try:
                    item = next(source)
                except StopIteration:
                    return
                entry = trace.TOTALS["startup:checkpoint_iterator_advance"]
                entry["calls"] += 1
                entry["wall_s"] += time.monotonic() - before
                yield item

        DefaultModelLoader.get_all_weights = iterator
        expert_class = layer.FusedMoE if inspect.isclass(layer.FusedMoE) else None
        if expert_class is None:
            from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
            expert_class = RoutedExperts
        trace.timed(expert_class, "weight_loader", "expert_weight_loader_callbacks", startup=True)
        for method in ("profile_run", "_dummy_run", "_dummy_sampler_run"):
            trace.marked(GPUModelRunner, method, "runner_" + method)
        for method in ("dispatch", "dispatch_router_logits", "combine"):
            trace.timed(AgRsAll2AllManager, method, "agrs_" + method, gpu=True)
        trace.timed(MoERunner, "_maybe_reduce_final_output", "moe_final_reduction", gpu=True)
        trace.timed(moe_module, "tensor_model_parallel_all_reduce", "actual_moe_all_reduce", gpu=True)

    def init_device(self):
        trace.emit("cuda_state_before_init_device", initialized=torch.cuda.is_initialized())
        with trace.span("worker_init_device"):
            return super().init_device()

    def load_model(self, **kwargs):
        before_io = trace.io_snapshot()
        with trace.span("worker_load_model"):
            result = super().load_model(**kwargs)
            with trace.span("weight_loading_final_cuda_sync"):
                torch.cuda.synchronize(self.device)
        trace.emit("loading_io_observation", before=before_io, after=trace.io_snapshot())
        return result

    def determine_available_memory(self):
        with trace.span("engine_memory_profiling"):
            if os.environ.get("MOE_DIAG_CPROFILE") != "1":
                return super().determine_available_memory()
            import cProfile
            import pstats
            profiler = cProfile.Profile()
            with profiler:
                result = super().determine_available_memory()
            stats = pstats.Stats(profiler)
            trace.emit("memory_cpu_profile", functions=[
                {"file": key[0], "line": key[1], "function": key[2],
                 "calls": value[1], "self_s": value[2], "cumulative_s": value[3]}
                for key, value in sorted(stats.stats.items(),
                                         key=lambda row: row[1][3], reverse=True)[:35]])
            return result

    def initialize_from_config(self, *args, **kwargs):
        with trace.span("kv_cache_initialization"):
            return super().initialize_from_config(*args, **kwargs)

    def compile_or_warm_up_model(self):
        with trace.span("engine_kernel_warmup"):
            result = super().compile_or_warm_up_model()
        self.startup_trace = trace.snapshot()
        return result

    def get_diagnostic_snapshot(self):
        return {"rank": self.rank, "local_rank": self.local_rank,
                "startup": self.startup_trace, "current": trace.snapshot(),
                "module_files": {"worker": worker_module.__file__,
                                 "runner": inspect.getfile(GPUModelRunner)}}

    def set_diagnostic_measurement(self, label=None):
        return trace.set_measurement(label)

    def get_expert_placement_evidence(self):
        model = self.model_runner.model
        checkpoint = Path(self.model_config.model)
        index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
        from safetensors import safe_open
        modules = []
        for name, module in model.named_modules():
            if not hasattr(module, "w13_weight") or not hasattr(module, "w2_weight"):
                continue
            config = module.moe_config
            parallel = config.moe_parallel_config
            mapping = module.expert_map
            global_ids = (list(range(config.num_experts)) if mapping is None else
                          [i for i, local in enumerate(mapping.cpu().tolist()) if local >= 0])
            layer = int(re.search(r"layers\.(\d+)", name).group(1))
            prefix = f"model.layers.{layer}.block_sparse_moe."
            keys = [prefix + "input_linear.weight", prefix + "output_linear.weight"]
            tensors = []
            for key in keys:
                with safe_open(str(checkpoint / index[key]), framework="pt", device="cpu") as handle:
                    tensors.append(handle.get_tensor(key))
            full_input, full_output = tensors
            partition = config.intermediate_size_per_partition
            offset = parallel.tp_rank * partition
            observed, expected = [], []
            for global_id in global_ids:
                local_id = global_id if mapping is None else int(mapping[global_id].item())
                observed += [module.w13_weight[local_id, 0, :8].cpu().tolist(),
                             module.w13_weight[local_id, partition, :8].cpu().tolist(),
                             module.w2_weight[local_id, :8, 0].cpu().tolist()]
                expected += [full_input[global_id, offset, :8].tolist(),
                             full_input[global_id, full_input.shape[1] // 2 + offset, :8].tolist(),
                             full_output[global_id, :8, offset].tolist()]
            kernel = getattr(module.quant_method, "moe_kernel", None)
            implementation = getattr(kernel, "impl", kernel)
            prepare = getattr(implementation, "prepare_finalize", None)
            modules.append({"name": name, "layer": layer,
                "parallel": dataclasses.asdict(parallel),
                "use_all2all_kernels": parallel.use_all2all_kernels,
                "global_expert_ids": global_ids,
                "expert_map": None if mapping is None else mapping.cpu().tolist(),
                "w13_shape": list(module.w13_weight.shape), "w2_shape": list(module.w2_weight.shape),
                "weight_device": str(module.w13_weight.device),
                "intermediate_partition": partition,
                "checkpoint_samples_match": observed == expected,
                "checked_scalar_values": sum(len(row) for row in observed),
                "observed_samples_sha256": hashlib.sha256(json.dumps(observed).encode()).hexdigest(),
                "expected_samples_sha256": hashlib.sha256(json.dumps(expected).encode()).hexdigest(),
                "quant_method": type(module.quant_method).__name__,
                "prepare_finalize_class": type(prepare).__name__})
        return {"rank": self.rank, "local_rank": self.local_rank,
                "actual_gpu_uuid": str(torch.cuda.get_device_properties(self.device).uuid),
                "model_class": type(model).__name__, "expert_modules": modules,
                "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
                "vllm_version": vllm.__version__, "vllm_module_file": vllm.__file__,
                "worker_module_file": worker_module.__file__}
