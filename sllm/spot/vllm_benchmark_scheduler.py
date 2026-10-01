"""Opt-in controlled native freeze for one or two requests."""

from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.scheduler import Scheduler

from .benchmark_freeze import NativeFreezeMixin


class NativeFreezeScheduler(NativeFreezeMixin, Scheduler):
    freeze_pause_state = PauseState.PAUSED_ALL

    def __init__(self, *args, **kwargs):
        config = kwargs["vllm_config"]
        if (config.scheduler_config.async_scheduling is not False
                or config.speculative_config is not None
                or config.scheduler_config.max_num_seqs not in (1, 2)):
            raise ValueError("benchmark freeze needs synchronous non-speculative C=1/C=2")
        super().__init__(*args, **kwargs)
