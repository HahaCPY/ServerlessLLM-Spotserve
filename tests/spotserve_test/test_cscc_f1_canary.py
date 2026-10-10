"""CPU checks for exact freeze, connector evidence, and event ordering."""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
CSCC = ROOT / "deploy" / "cscc-ray"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


canary = load("cscc_f1_canary_test", CSCC / "run_f1_canary.py")


class FakeScheduler:
    def __init__(self, config):
        self.vllm_config = config
        self.requests = {}
        self.running = []

    def has_requests(self):
        return bool(self.requests)

    def update_from_output(self, *args):
        return {"output": True}


fake_scheduler_module = types.ModuleType("vllm.v1.core.sched.scheduler")
fake_scheduler_module.Scheduler = FakeScheduler
with patch.dict(sys.modules, {"vllm.v1.core.sched.scheduler": fake_scheduler_module}):
    barrier = load("cscc_f1_barrier_test", CSCC / "f1_canary_scheduler.py")


def request(count, threshold=256):
    return types.SimpleNamespace(
        request_id="r", max_tokens=384, output_token_ids=[1] * count,
        num_computed_tokens=512 + count - 1, num_output_placeholders=0,
        sampling_params=types.SimpleNamespace(extra_args={barrier.FREEZE_KEY: threshold}),
        is_finished=lambda: False,
    )


class BarrierTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.audit = Path(self.directory.name) / "audit.jsonl"
        self.resume = Path(self.directory.name) / "resume"
        self.env = patch.dict("os.environ", {
            "CSCC_F1_AUDIT_PATH": str(self.audit),
            "CSCC_F1_RESUME_PATH": str(self.resume),
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.config = types.SimpleNamespace(
            scheduler_config=types.SimpleNamespace(async_scheduling=False, max_num_seqs=1),
            speculative_config=None,
        )
        self.scheduler = barrier.CanaryScheduler(self.config)
        self.output = types.SimpleNamespace(kv_connector_output=None)

    def install(self, req):
        self.scheduler.requests = {req.request_id: req}
        self.scheduler.running = [req]

    def test_freeze_exactly_256_without_freeing_kv(self):
        req = request(255)
        self.install(req)
        self.scheduler.update_from_output(None, self.output)
        self.assertTrue(self.scheduler.has_requests())
        req.output_token_ids.append(1)
        self.scheduler.update_from_output(None, self.output)
        self.assertFalse(self.scheduler.has_requests())
        self.assertIs(self.scheduler.requests["r"], req)
        self.assertNotIn(barrier.FREEZE_KEY, req.sampling_params.extra_args)
        self.assertEqual(json.loads(self.audit.read_text())["generated_tokens"], 256)

    def test_resume_releases_barrier(self):
        self.install(request(256))
        self.scheduler.update_from_output(None, self.output)
        self.resume.touch()
        self.assertTrue(self.scheduler.has_requests())

    def test_aborted_request_allows_connector_polling(self):
        req = request(256)
        self.install(req)
        self.scheduler.update_from_output(None, self.output)
        req.is_finished = lambda: True
        self.assertTrue(self.scheduler.has_requests())

    def test_overshoot_is_rejected(self):
        self.install(request(257))
        with self.assertRaisesRegex(RuntimeError, "boundary"):
            self.scheduler.update_from_output(None, self.output)

    def test_async_scheduling_is_rejected(self):
        self.config.scheduler_config.async_scheduling = True
        with self.assertRaisesRegex(ValueError, "synchronous"):
            barrier.CanaryScheduler(self.config)


class EvidenceTests(unittest.TestCase):
    def evidence(self, first, received=True, invalid=False):
        return canary.restore_evidence([
            {"event": "scheduled", "scheduled_tokens": first},
            {"event": "connector_completion", "finished_recving": ["r"] if received else [],
             "invalid_block_ids": [9] if invalid else []},
        ], 768)

    def test_real_receive_and_one_boundary_token(self):
        result = self.evidence(1)
        self.assertTrue(result["restore_success"])
        self.assertEqual(result["prefix_reused_tokens"], 767)

    def test_full_recompute_is_not_restore(self):
        self.assertFalse(self.evidence(768)["restore_success"])

    def test_missing_completion_is_not_restore(self):
        self.assertFalse(self.evidence(1, received=False)["restore_success"])

    def test_invalid_blocks_are_not_restore(self):
        self.assertFalse(self.evidence(1, invalid=True)["restore_success"])

    def test_early_done_is_preserved_while_waiting_for_resume_ack(self):
        process = canary.EngineProcess()
        process._pending = []
        process._config = {"timeout_s": 1}
        messages = [{"event": "done", "tokens": [1, 2]}, {"event": "resumed"}]
        process._conn = types.SimpleNamespace(poll=lambda timeout: True,
                                               recv=lambda: messages.pop(0))
        process._process = types.SimpleNamespace(poll=lambda: None)
        self.assertEqual(process.wait("resumed")["event"], "resumed")
        self.assertEqual(process.wait("done")["tokens"], [1, 2])


if __name__ == "__main__":
    unittest.main()
