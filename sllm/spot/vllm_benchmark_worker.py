"""Opt-in named-RPC identity evidence; no unsafe function serialization."""


class BenchmarkWorkerExtension:
    def get_benchmark_rank_identity(self):
        import torch
        return {"rank": self.rank, "local_rank": self.local_rank,
                "actual_gpu_uuid": str(torch.cuda.get_device_properties(self.device).uuid),
                "model_class": type(self.model_runner.model).__name__}
