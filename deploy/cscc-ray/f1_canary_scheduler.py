"""vLLM 0.11.2-only C=1 barrier and audit for the CSCC recovery canary.

This opt-in scheduler changes no production planner or cross-host capability.
It stops EngineCore scheduling after an exact synchronous output boundary;
utility calls remain available while the request and its KV stay allocated.
"""

import json
import os
import time
from pathlib import Path

from vllm.v1.core.sched.scheduler import Scheduler

FREEZE_KEY = "cscc_f1_canary_freeze_at"


class CanaryScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self.vllm_config
        if (
            config.scheduler_config.async_scheduling is not False
            or config.scheduler_config.max_num_seqs != 1
            or config.speculative_config is not None
        ):
            raise ValueError("CSCC canary requires synchronous C=1 without speculation")
        self._frozen_request_id = None
        self._audit_path = Path(os.environ["CSCC_F1_AUDIT_PATH"])
        self._resume_path = Path(os.environ["CSCC_F1_RESUME_PATH"])

    def _record(self, event, **values):
        with self._audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": event, "monotonic_s": time.monotonic(),
                                     **values}, sort_keys=True) + "\n")

    def has_requests(self):
        if self._frozen_request_id is not None:
            request = self.requests.get(self._frozen_request_id)
            # Exported/aborted requests must continue connector polling so
            # the producer can receive the consumer completion notification.
            if request is None or request.is_finished() or self._resume_path.exists():
                self._frozen_request_id = None
            else:
                return False
        return super().has_requests()

    def schedule(self):
        output = super().schedule()
        for request_id, count in output.num_scheduled_tokens.items():
            if count:
                request = self.requests[request_id]
                self._record(
                    "scheduled", request_id=request_id, scheduled_tokens=int(count),
                    computed_tokens=int(request.num_computed_tokens),
                    prompt_tokens=len(request.prompt_token_ids or []),
                )
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        connector_output = model_runner_output.kv_connector_output
        if connector_output is not None:
            self._record(
                "connector_completion",
                finished_recving=sorted(connector_output.finished_recving or []),
                finished_sending=sorted(connector_output.finished_sending or []),
                invalid_block_ids=sorted(connector_output.invalid_block_ids or []),
            )
        output = super().update_from_output(scheduler_output, model_runner_output)
        for request in self.running:
            extra = request.sampling_params.extra_args or {}
            threshold = extra.get(FREEZE_KEY)
            if threshold is None:
                continue
            if not 0 < threshold < request.max_tokens:
                raise ValueError("freeze threshold must precede request completion")
            count = len(request.output_token_ids)
            if count > threshold or request.num_output_placeholders:
                raise RuntimeError("invalid synchronous canary boundary")
            if count == threshold:
                self._frozen_request_id = request.request_id
                request.sampling_params.extra_args = {
                    key: value for key, value in extra.items() if key != FREEZE_KEY
                }
                self._record(
                    "frozen", request_id=request.request_id,
                    generated_tokens=count, computed_tokens=request.num_computed_tokens,
                )
        return output
