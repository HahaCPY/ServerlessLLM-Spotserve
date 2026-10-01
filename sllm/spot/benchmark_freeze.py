"""Controlled benchmark boundary logic, without CUDA/vLLM imports."""

FREEZE_KEY = "spotserve_benchmark_freeze_generated_tokens"


def freeze_sampling_extra_args(threshold, max_tokens):
    if type(threshold) is not int or not 0 < threshold < max_tokens:
        raise ValueError("native freeze threshold must be inside an unfinished request")
    return {FREEZE_KEY: threshold}


class NativeFreezeMixin:
    """Pause inside synchronous EngineCore, before another decode is scheduled."""

    def update_from_output(self, *args, **kwargs):
        outputs = super().update_from_output(*args, **kwargs)
        for request in self.running:
            extra = getattr(request.sampling_params, "extra_args", None) or {}
            threshold = extra.get(FREEZE_KEY)
            if threshold is None:
                continue
            freeze_sampling_extra_args(threshold, request.max_tokens)
            count = len(request.output_token_ids)
            if count > threshold:
                raise RuntimeError("native_freeze_overshoot_not_a_valid_boundary")
            if count == threshold:
                if getattr(request, "num_output_placeholders", 0):
                    raise RuntimeError("native_freeze_has_async_tokens_in_flight")
                self.set_pause_state(self.freeze_pause_state)
                # A benchmark barrier is one-shot; public resume may continue
                # this same request after the controller verifies its state.
                request.sampling_params.extra_args = {
                    key: value for key, value in extra.items() if key != FREEZE_KEY}
        return outputs
