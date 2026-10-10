import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_driver():
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "scripts" / "run_k8s_moe_f1_f2.py"
    spec = importlib.util.spec_from_file_location("k8s_moe_f1_f2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_default_spec():
    repo_root = Path(__file__).resolve().parents[2]
    return json.loads((
        repo_root
        / "benchmarks/spotserve/formal/k8s_qwen15_moe_a27b_8gpu.json"
    ).read_text(encoding="utf-8"))


def fake_hardware():
    return {
        "nodes": [
            {
                "ray_node_id": f"ray-{index}",
                "address": f"10.0.0.{index + 1}",
                "gpu_count": 1,
                "worker_ids": [str(index)],
                "physical_host_markers": [f"spotserve_physical_host_{index // 4}"],
            } for index in range(8)
        ],
        "gpus": [],
        "hardware_fingerprint": "hardware-fp",
    }


def fake_capability():
    return {
        "source": "measured_k8s_offline_profiles",
        "supported_configs": [
            {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "data_parallel_size": dp,
                "replica_count": 1,
                "enable_expert_parallel": True,
                "num_gpus": dp,
                "latency_estimate_ms": 1000 / dp,
                "throughput_estimate_req_s": float(dp),
                "load_time_estimate_ms": float(dp * 1000),
                "migration_cost_estimate_ms": 10.0,
                "expert_weight_movement_cost_estimate_ms": 5.0,
            }
            for dp in (2, 3, 4)
        ],
    }


def test_core_probe_dry_run_never_touches_gpu_or_formal_ledger(tmp_path, monkeypatch):
    driver = load_driver()
    monkeypatch.setattr(driver, "probe_cluster", lambda *args: pytest.fail("GPU probe"))
    monkeypatch.setattr(driver, "execute_benchmark", lambda *args: pytest.fail("benchmark"))
    args = SimpleNamespace(dry_run=True, force_core_probe=False)
    assert driver.run_core_probe(args, load_default_spec(), tmp_path, "endpoint") == 0
    assert driver.read_json(tmp_path / "core-probe-results.json")["status"] == "dry_run_passed"
    assert not (tmp_path / "run-ledger.json").exists()
    assert not (tmp_path / "state.json").exists()


def test_core_probe_missing_cache_stops_before_http_gpu_and_benchmark(tmp_path, monkeypatch):
    driver = load_driver()
    monkeypatch.setattr(driver, "check_existing_endpoint", lambda *args: pytest.fail("HTTP"))
    monkeypatch.setattr(driver, "probe_cluster", lambda *args: pytest.fail("GPU probe"))
    args = SimpleNamespace(dry_run=False, force_core_probe=False)
    assert driver.run_core_probe(args, load_default_spec(), tmp_path, "endpoint") == 2
    result = driver.read_json(tmp_path / "core-probe-results.json")
    assert "--phase profile once" in result["error"]
    prior = (tmp_path / "core-probe-results.json").read_bytes()
    assert driver.run_core_probe(args, load_default_spec(), tmp_path, "endpoint") == 2
    assert (tmp_path / "core-probe-results.json").read_bytes() == prior
    assert len(list((tmp_path / "core-probe-attempts").iterdir())) == 1


def test_cache_only_profiles_never_launch_gpu_benchmark(tmp_path, monkeypatch):
    driver = load_driver()
    monkeypatch.setattr(driver, "execute_benchmark", lambda *args: pytest.fail("profiling"))
    with pytest.raises(driver.ExperimentError, match="matching measured profile"):
        driver.ensure_candidate_profiles(
            load_default_spec(), fake_hardware(), {}, [f"prompt-{i}" for i in range(12)], tmp_path,
            "endpoint", "auto", "sllm", False, cache_only=True)


@pytest.mark.parametrize("cache_only", [True, False])
def test_profiles_preserve_measured_failed_shape_without_implicit_retry(tmp_path, monkeypatch, cache_only):
    driver = load_driver()
    spec = load_default_spec()
    monkeypatch.setattr(driver, "digest_value", lambda *args: "input-hash")
    monkeypatch.setattr(driver, "execute_benchmark", lambda *args: pytest.fail("profiling"))
    for shape in spec["candidate_shapes"]:
        driver.write_json(tmp_path / "profiles" / f"{driver.shape_label(shape)}.json", {
            "status": "failed" if shape["data_parallel_size"] == 3 else "passed",
            "profile_input_hash": "input-hash", "shape": shape,
        })
    profiles = driver.ensure_candidate_profiles(
        spec, fake_hardware(), {}, [f"prompt-{i}" for i in range(12)], tmp_path,
        "endpoint", "auto", "sllm", False, cache_only=cache_only)
    assert [row["shape"]["data_parallel_size"] for row in profiles] == [2, 4]


def test_core_probe_runs_only_paired_baseline_and_dynamic_native_recovery(tmp_path, monkeypatch):
    driver = load_driver()
    spec = load_default_spec()
    driver.write_json(tmp_path / "preflight" / "hardware.json", fake_hardware())
    driver.write_json(tmp_path / "preflight" / "network.json", {})
    monkeypatch.setattr(driver, "check_existing_endpoint", lambda *args: {"health": "ok"})
    monkeypatch.setattr(driver, "probe_cluster", lambda *args: fake_hardware())
    monkeypatch.setattr(driver, "exact_chat_prompts", lambda model, tokens, count: [f"p-{i}" for i in range(count)])
    def cached(*args, **kwargs):
        assert kwargs["cache_only"] is True
        assert args[-1] is False
        return []
    monkeypatch.setattr(driver, "ensure_candidate_profiles", cached)
    monkeypatch.setattr(driver, "build_capability", lambda *args: fake_capability())
    monkeypatch.setattr(driver, "derive_preemption_schedule", lambda *args: {"effective_preemption_time_s": 45})
    output_tokens = int(spec["workload"]["output_tokens"])
    calls = []
    workload_hashes = []
    def execute(matrix_path, *args):
        matrix = driver.read_json(matrix_path)
        run = matrix["runs"][0]
        deploy = driver.read_json(Path(run["deploy_config"]))
        recovery = "trace" in run
        calls.append(recovery)
        assert run["delete_models_before_run"] == []
        assert run["capture_instance_states_after_workload"] is True
        assert deploy["router_config"]["require_native_kv_restore"] is recovery
        assert deploy["router_config"]["enable_reparallelization"] is recovery
        workload_hashes.append(run["workload_sha256"])
        rows = [{"request_id": f"measured-{i:03d}", "benchmark_phase": "measured",
                 "success": True, "response": {"usage": {"completion_tokens": output_tokens},
                 "_spotserve_token_ids": [3] * output_tokens, "_spotserve_prompt_token_ids": [1, 2]}}
                for i in range(24)]
        run_dir = Path(matrix["output_dir"])
        if recovery:
            rows[0]["response"]["_spotserve_kv_restore"] = {"restored": True, "cached_tokens": 3}
            driver.write_json(run_dir / "instance_states.json", {"old-a": {}})
            driver.write_json(run_dir / "final_instance_states.json", {
                "new-a": {"pool": "ready", "state": "ready", "member_node_ids": ["4", "5"]}})
            metrics = [
                {"type": "reparallelization", "event": "preempt", "execution_status": "applied",
                 "target_nodes": ["4", "5"], "event_state_marked_at_s": 100, "planner_started_at_s": 101},
                {"type": "native_kv_receipt", "request_id": "measured-000", "cached_tokens": 3,
                 "restored_blocks": 1, "source_instance_id": "old-a", "target_instance_id": "new-a", "timestamp": 110},
                {"type": "native_kv_source_release_authorized", "request_ids": ["measured-000"],
                 "source_instance_ids": ["old-a"], "timestamp": 111, "deadline_time_s": 130},
            ] + [{"type": "request", "request_id": row["request_id"], "success": True,
                  "recovery_fallback": False, "state_restore_fallback": False} for row in rows]
            driver.write_jsonl(Path(run["router_metrics_path"]), metrics)
        driver.write_jsonl(run_dir / "raw_requests.jsonl", rows)
        return [{"run_dir": str(run_dir), "runtime_ep_audit_verified": True}]
    monkeypatch.setattr(driver, "execute_benchmark", execute)
    args = SimpleNamespace(dry_run=False, force_core_probe=False, ray_address="auto", ray_namespace="sllm")
    assert driver.run_core_probe(args, spec, tmp_path, "endpoint") == 0
    assert calls == [False, True]
    assert workload_hashes[0] == workload_hashes[1]
    result = driver.read_json(tmp_path / "core-probe-results.json")
    assert result["status"] == "live_kv_core_verified"
    assert result["full_spotserve_verified"] is False
    assert not (tmp_path / "run-ledger.json").exists()


def test_stateful_formal_treatments_require_native_kv_not_replay():
    driver = load_driver()
    for experiment, treatments in (("f1", driver.F1_TREATMENTS), ("f2", driver.F2_TREATMENTS)):
        for treatment in treatments:
            config = driver.formal_deploy_config(
                load_default_spec(), fake_hardware(), fake_capability(), experiment,
                treatment, "model", Path("/tmp/test-metrics.jsonl"))
            required = config["router_config"]["require_native_kv_restore"]
            assert required == (driver.treatment_flags(experiment, treatment)["recovery_policy"] == "stateful_recovery")


def test_repeat_workloads_are_paired_without_overwriting_previous_input(tmp_path):
    driver = load_driver()
    spec = load_default_spec()
    count = int(spec["workload"]["warmup_requests"]) + int(
        spec["workload"]["request_count"])
    prompts = [f"unique prompt {index}" for index in range(count)]

    def artifacts(treatment, repeat):
        matrix_path, _ = driver.write_one_run_artifacts(
            spec, fake_hardware(), fake_capability(), prompts, tmp_path,
            "f1", treatment, repeat, "http://localhost:8343")
        return driver.read_json(matrix_path)["runs"][0]

    original = artifacts("original_spotserve", 1)
    initial_bytes = Path(original["workload"]).read_bytes()
    full = artifacts("moe_spotserve", 1)
    second_repeat = artifacts("original_spotserve", 2)

    assert original["workload"] == full["workload"]
    assert original["workload_sha256"] == full["workload_sha256"]
    assert second_repeat["workload"] != original["workload"]
    assert second_repeat["workload_sha256"] != original["workload_sha256"]
    assert Path(original["workload"]).read_bytes() == initial_bytes
    trace = [
        json.loads(line)
        for line in Path(original["trace"]).read_text().splitlines()
        if line.strip()
    ]
    assert [row["event"] for row in trace] == [
        "preempt", "preempt", "preempt", "add", "add", "add",
    ]
    assert [row["node_id"] for row in trace] == [
        "3", "1", "5", "3", "1", "5",
    ]
    assert [row["time"] for row in trace] == [
        570.0, 630.0, 630.0, 690.0, 750.0, 810.0,
    ]
    assert trace[0]["grace_period_s"] == 30.0
    assert trace[3]["node_info"] == {
        "free_gpu": 1, "state": "ready", "total_gpu": 1,
    }
    assert all(row["gpu_count"] == 1 for row in trace)
    assert all(row["capacity_unit"] == "gpu" for row in trace)
    assert all("instance_selector" not in row for row in trace)
    assert original["initial_unavailable_worker_nodes"] == []
    assert original["restore_worker_nodes_after_run"] == ["1", "3", "5"]


def test_trace_schedule_is_calibrated_and_never_exceeds_eight_gpus():
    driver = load_driver()
    spec = load_default_spec()
    initial_shape = dict(spec["initial_parallel"])
    profile = {
        "shape": initial_shape,
        "calibration_output_tokens": 256,
        "decode_ms_per_token": 100.0,
        "estimated_time_to_preemption_target_ms": 26000.0,
    }

    schedule = driver.derive_preemption_schedule(spec, [profile])

    assert schedule["effective_preemption_time_s"] == 570.0
    assert schedule["effective_add_time_s"] == 690.0
    assert schedule["preemption_notice_times_s"] == [570.0, 630.0]
    assert schedule["calibrated_anchor_arrival_times_s"] == [544.0, 604.0]
    assert schedule["formal_burst_start_times_s"] == [
        20.0, 240.0, 420.0, 544.0, 604.0, 840.0,
    ]
    assert schedule["configured_output_tokens"] == 2048
    assert [row["available_gpus"] for row in schedule["capacity_timeline"]] == [
        8, 7, 5, 6, 7, 8,
    ]
    assert max(
        row["available_gpus"] for row in schedule["capacity_timeline"]
    ) == 8


def test_trace_schedule_rejects_anchor_that_would_start_before_workload():
    driver = load_driver()
    spec = load_default_spec()
    profile = {
        "shape": dict(spec["initial_parallel"]),
        "calibration_output_tokens": 256,
        "decode_ms_per_token": 1000.0,
        "estimated_time_to_preemption_target_ms": 600000.0,
    }

    with pytest.raises(driver.ExperimentError, match="model is too slow"):
        driver.derive_preemption_schedule(spec, [profile])


def smoke_summary(driver, directory):
    directory.mkdir(parents=True, exist_ok=True)
    states = {
        name: {"pool": "ready", "state": "ready", "node_id": members[0],
               "member_node_ids": members}
        for name, members in (("a", ["0", "1"]), ("b", ["2", "3"]))
    }
    driver.write_json(directory / "instance_states.json", states)
    driver.write_json(directory / "run_metadata.json", {
        "runtime_ep_audit": {
            "verified": True, "ready_instance_count": 2, "audited_instance_count": 2,
            "instances": [{"instance_id": name, "verified": True,
                           "observed_ep_sizes": [2], "observed_ep_ranks": [0, 1]}
                          for name in states],
        },
    })
    driver.write_jsonl(directory / "raw_requests.jsonl", [
        {"benchmark_phase": "smoke", "success": True,
         "response": {"usage": {"completion_tokens": 64}}} for _ in range(4)
    ])
    return {"run_dir": str(directory), "runtime_ep_audit_verified": True,
            "success_rate": 1.0}


def test_smoke_validates_two_disjoint_ep2_and_exact_output_tokens(tmp_path):
    driver = load_driver()
    summary = smoke_summary(driver, tmp_path / "run")
    checks = driver.validate_smoke_summary(
        driver.smoke_spec(load_default_spec()), fake_hardware(), summary)
    assert checks["reserved_worker_count"] == 4
    assert checks["unreserved_worker_count"] == 4
    assert checks["ep_rank_size_readback_verified"] is True


@pytest.mark.parametrize("problem", ["overlap", "missing_member", "missing_rank", "short_output"])
def test_smoke_rejects_invalid_bookkeeping_or_incomplete_decode(tmp_path, problem):
    driver = load_driver()
    directory = tmp_path / "run"
    summary = smoke_summary(driver, directory)
    states = driver.read_json(directory / "instance_states.json")
    metadata = driver.read_json(directory / "run_metadata.json")
    if problem == "overlap":
        states["b"].update(node_id="1", member_node_ids=["1", "2"])
    elif problem == "missing_member":
        states["b"]["member_node_ids"] = ["2"]
    elif problem == "missing_rank":
        metadata["runtime_ep_audit"]["instances"][1]["observed_ep_ranks"] = [0]
    elif problem == "short_output":
        driver.write_jsonl(directory / "raw_requests.jsonl", [
            {"benchmark_phase": "smoke", "success": True,
             "response": {"usage": {"completion_tokens": 12}}} for _ in range(4)])
    driver.write_json(directory / "instance_states.json", states)
    driver.write_json(directory / "run_metadata.json", metadata)
    with pytest.raises(driver.ExperimentError):
        driver.validate_smoke_summary(
            driver.smoke_spec(load_default_spec()), fake_hardware(), summary)


def test_smoke_dry_run_never_calls_endpoint_ray_or_benchmarks(tmp_path, monkeypatch):
    driver = load_driver()
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not access the cluster")
    for name in ("check_existing_endpoint", "probe_cluster", "execute_benchmark"):
        monkeypatch.setattr(driver, name, forbidden)
    monkeypatch.setattr("sys.argv", ["driver", "--phase", "smoke", "--dry-run",
                                    "--repeats", "1",
                                    "--model-path", "/mounted/checkpoint",
                                    "--output-dir", str(tmp_path)])
    assert driver.main() == 0
    result = driver.read_json(tmp_path / "smoke-results.json")
    assert result["status"] == "dry_run_passed"
    effective = driver.read_json(Path(result["attempt_dir"]) / "smoke-config.json")
    assert effective["model"]["path"] == "/mounted/checkpoint"
    assert not (tmp_path / "profiles").exists()
    assert not (tmp_path / "run-ledger.json").exists()


def test_smoke_endpoint_check_is_read_only_and_bounded(monkeypatch):
    import io

    driver = load_driver()
    urls = []
    def response(url, timeout):
        urls.append((url, timeout))
        return io.StringIO(json.dumps({"status": "ok"} if url.endswith("/health") else {"models": []}))
    monkeypatch.setattr(driver.request, "urlopen", response)
    assert driver.check_existing_endpoint("http://head:8343/v1/chat/completions")["models_endpoint_reachable"] is True
    assert urls == [("http://head:8343/health", 10.0), ("http://head:8343/v1/models", 10.0)]


def test_model_path_override_cannot_relabel_prior_formal_evidence(monkeypatch):
    driver = load_driver()
    monkeypatch.setattr("sys.argv", ["driver", "--phase", "formal", "--model-path", "/new-model",
                                    "--repeats", "1",
                                    "--output-dir", "/unused"])
    with pytest.raises(SystemExit) as exc:
        driver.parse_args()
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["driver", "--output-dir", "/unused"],
        ["driver", "--output-dir", "/unused", "--repeats", "0"],
    ],
)
def test_formal_repeat_count_is_required_and_positive(monkeypatch, argv):
    driver = load_driver()
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit) as exc:
        driver.parse_args()
    assert exc.value.code == 2


def test_repeat_count_mismatch_does_not_modify_existing_results(
    tmp_path, monkeypatch
):
    driver = load_driver()
    spec_path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/spotserve/formal/k8s_qwen15_moe_a27b_8gpu.json"
    )
    output = tmp_path / "results"
    base_argv = [
        "driver", "--config", str(spec_path), "--output-dir", str(output),
        "--dry-run", "--repeats",
    ]
    monkeypatch.setattr("sys.argv", base_argv + ["2"])
    assert driver.main() == 0
    state_before = (output / "state.json").read_bytes()
    report_before = (output / "results.md").read_bytes()

    monkeypatch.setattr("sys.argv", base_argv + ["3"])
    assert driver.main() == 2
    assert (output / "state.json").read_bytes() == state_before
    assert (output / "results.md").read_bytes() == report_before


def test_smoke_endpoint_failure_is_reported_before_gpu_probe(tmp_path, monkeypatch):
    driver = load_driver()
    def fail(*args, **kwargs):
        raise driver.ExperimentError("HTTP service missing")
    monkeypatch.setattr(driver, "check_existing_endpoint", fail)
    monkeypatch.setattr(driver, "probe_cluster", lambda *args: pytest.fail("GPU probe called"))
    args = SimpleNamespace(force_smoke=False, dry_run=False)
    assert driver.run_smoke(args, load_default_spec(), tmp_path, "endpoint") == 2
    result = driver.read_json(tmp_path / "smoke-results.json")
    assert result["status"] == "blocked"
    assert result["error"] == "HTTP service missing"
    assert (tmp_path / "smoke-results.md").is_file()


def test_smoke_is_one_run_without_profiles_and_does_not_retry_implicitly(tmp_path, monkeypatch):
    driver = load_driver()
    calls = []
    monkeypatch.setattr(driver, "check_existing_endpoint", lambda *args: {"health": {"status": "ok"}})
    monkeypatch.setattr(driver, "probe_cluster", lambda *args: fake_hardware())
    monkeypatch.setattr(driver, "exact_chat_prompts", lambda *args: [f"p{i}" for i in range(5)])
    monkeypatch.setattr(driver, "ensure_candidate_profiles", lambda *args: pytest.fail("profile called"))
    def execute(matrix, *args):
        calls.append(matrix)
        run = driver.read_json(matrix)["runs"][0]
        deploy = driver.read_json(Path(run["deploy_config"]))
        assert run["capture_instance_states"] is True
        assert run["delete_after_run"] is True
        assert "delete_models_before_run" not in run
        assert "trace" not in run
        assert deploy["auto_scaling_config"]["min_instances"] == 2
        assert deploy["router_config"]["enable_reparallelization"] is False
        assert "kv_transfer_config" not in deploy["backend_config"]
        return [smoke_summary(driver, matrix.parent / "run")]
    monkeypatch.setattr(driver, "execute_benchmark", execute)
    args = SimpleNamespace(force_smoke=False, dry_run=False,
                           ray_address="auto", ray_namespace="sllm")
    assert driver.run_smoke(args, load_default_spec(), tmp_path, "endpoint") == 0
    result = driver.read_json(tmp_path / "smoke-results.json")
    assert result["status"] == "passed"
    assert "cross_pod_kv_restore" in result["not_verified"]
    prior_attempt = Path(result["attempt_dir"])
    assert driver.run_smoke(args, load_default_spec(), tmp_path, "endpoint") == 2
    assert len(calls) == 1
    args.force_smoke = True
    assert driver.run_smoke(args, load_default_spec(), tmp_path, "endpoint") == 0
    assert len(calls) == 2
    assert prior_attempt != Path(driver.read_json(tmp_path / "smoke-results.json")["attempt_dir"])
    assert (prior_attempt / "smoke-results.json").is_file()


def test_default_spec_has_three_real_ep_shapes():
    driver = load_driver()
    spec = load_default_spec()

    driver.validate_spec(spec)

    signatures = {
        driver.shape_signature(shape) for shape in spec["candidate_shapes"]
    }
    assert signatures == {
        (1, 1, 2, 1, True),
        (1, 1, 3, 1, True),
        (1, 1, 4, 1, True),
    }


def test_gpu_id_only_variants_do_not_count_as_distinct_candidates():
    driver = load_driver()
    spec = load_default_spec()
    first = copy.deepcopy(spec["candidate_shapes"][0])
    second = copy.deepcopy(first)
    first["target_worker_nodes"] = ["gpu-0", "gpu-1"]
    second["target_worker_nodes"] = ["gpu-2", "gpu-3"]
    spec["candidate_shapes"] = [first, second]

    with pytest.raises(driver.ExperimentError, match="distinct parallel shapes"):
        driver.validate_spec(spec)


@pytest.mark.parametrize(
    "experiment,treatment",
    [
        *(('f1', treatment) for treatment in (
            "rerouting", "reparallelization", "original_spotserve",
            "moe_spotserve",
        )),
        *(('f2', treatment) for treatment in (
            "original_spotserve", "reparallelization_only",
            "migration_only", "full_moe_spotserve",
            "original_spotserve_replay", "reparallelization_only_replay",
            "migration_only_replay", "full_moe_spotserve_replay",
        )),
    ],
)
def test_every_treatment_uses_ep_and_measured_candidates(
    tmp_path, experiment, treatment
):
    driver = load_driver()
    spec = load_default_spec()
    config = driver.formal_deploy_config(
        spec,
        fake_hardware(),
        fake_capability(),
        experiment,
        treatment,
        "model-under-test",
        tmp_path / "router.jsonl",
    )

    backend = config["backend_config"]
    candidates = backend["spotserve_backend_capability"]["supported_configs"]
    assert backend["enable_expert_parallel"] is True
    assert backend["planned_effective_expert_parallel_size"] == 2
    assert len({driver.shape_signature(row) for row in candidates}) == 3
    assert all(row["enable_expert_parallel"] for row in candidates)
    assert all(row["data_parallel_size"] > 1 for row in candidates)


def test_f2_changes_only_declared_moe_ablation_flags(tmp_path):
    driver = load_driver()
    spec = load_default_spec()
    configs = {
        treatment: driver.formal_deploy_config(
            spec,
            fake_hardware(),
            fake_capability(),
            "f2",
            treatment,
            f"model-{treatment}",
            tmp_path / f"{treatment}.jsonl",
        )
        for treatment in (*driver.F2_TREATMENTS, *driver.F2_REPLAY_CONTROLS)
    }

    audits = {
        treatment: config["router_config"]["treatment_audit"]
        for treatment, config in configs.items()
    }
    assert audits["original_spotserve"]["moe_reparallelization"] is False
    assert audits["original_spotserve"]["moe_migration"] is False
    assert audits["reparallelization_only"]["moe_reparallelization"] is True
    assert audits["reparallelization_only"]["moe_migration"] is False
    assert audits["migration_only"]["moe_reparallelization"] is False
    assert audits["migration_only"]["moe_migration"] is True
    assert audits["full_moe_spotserve"]["moe_reparallelization"] is True
    assert audits["full_moe_spotserve"]["moe_migration"] is True
    assert {
        config["router_config"]["recovery_policy"] for config in configs.values()
    } == {"stateful_recovery", "generated_token_replay"}
    for treatment in (
        "original_spotserve", "reparallelization_only",
        "migration_only", "full_moe_spotserve",
    ):
        assert audits[treatment]["kv_state_restore"] is True
        assert audits[f"{treatment}_replay"]["kv_state_restore"] is False
        assert audits[treatment]["transition_mode"] == "make_before_break"
        assert configs[treatment]["router_config"][
            "reparallelization_config"
        ]["migrate_before_create"] is True
        assert configs[treatment]["router_config"][
            "reparallelization_config"
        ]["allow_stop_before_recreate"] is False
        assert audits[treatment]["persistent_context_daemon"] is False
        assert audits[f"{treatment}_replay"]["transition_mode"] == (
            "break_before_make"
        )


def test_cluster_profile_rejects_mixed_gpu_models():
    driver = load_driver()
    spec = load_default_spec()
    config = {
        "architectures": ["Qwen2MoeForCausalLM"],
        "num_experts": 60,
    }
    manifest = {
        "manifest_sha256": "same-model",
        "config": config,
    }
    profile = {
        "nodes": fake_hardware()["nodes"],
        "gpus": [
            {
                "gpu": {
                    "uuid": f"uuid-{index}",
                    "name": "RTX 5090" if index < 7 else "RTX PRO 6000",
                    "memory_used_mib": 0,
                },
                "model_manifest": manifest,
            }
            for index in range(8)
        ],
    }

    with pytest.raises(driver.ExperimentError, match="mixed GPU models"):
        driver.validate_cluster_profile(spec, profile)


def test_cluster_profile_rejects_cross_node_image_mismatch():
    driver = load_driver()
    spec = load_default_spec()
    manifest = {
        "manifest_sha256": "same-model",
        "config": {
            "architectures": ["Qwen2MoeForCausalLM"],
            "num_experts": 60,
        },
    }
    profile = {
        "image_digest": "sha256:head",
        "nodes": fake_hardware()["nodes"],
        "gpus": [
            {
                "gpu": {
                    "uuid": f"uuid-{index}",
                    "name": "RTX 5090",
                    "memory_total_mib": 32768,
                    "memory_used_mib": 0,
                    "driver_version": "580.0",
                },
                "torch_version": "2.8.0",
                "torch_cuda_version": "12.8",
                "vllm_version": "0.10.0",
                "image_digest": "sha256:worker",
                "model_manifest": manifest,
            }
            for index in range(8)
        ],
    }

    with pytest.raises(driver.ExperimentError, match="immutable image digest"):
        driver.validate_cluster_profile(spec, profile)


def test_expert_placement_only_divergence_is_valid_contribution_evidence():
    driver = load_driver()
    plan = json.dumps({
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 4,
        "replica_count": 1,
        "enable_expert_parallel": True,
    })

    def summary(moe, latency=100.0, throughput=2.0):
        return {
            "latency_p95_ms": latency,
            "throughput_req_s": throughput,
            "replanning_latest_plan": plan,
            "context_migration_latest_normalized_plan": json.dumps([
                {"source_node": "node-a", "target_node": "node-b"}
            ], sort_keys=True),
            "replanning_latest_normalized_expert_placement": (
                json.dumps({
                    "target_rank_count": 4,
                    "expert_to_target_rank": (
                        {"layer:0/expert:0": "ep-rank:1"} if moe else {}
                    ),
                }, sort_keys=True)
            ),
        }

    pilot = driver.pilot_contribution_gate([
        {"treatment": "original_spotserve", "summary": summary(False)},
        {"treatment": "moe_spotserve", "summary": summary(True)},
    ])
    assert pilot["passed"] is True
    assert pilot["shape_diverged"] is False
    assert pilot["migration_diverged"] is False
    assert pilot["expert_placement_diverged"] is True

    ledger = []
    for repeat in (1, 2, 3):
        ledger.extend([
            {
                "stage": "formal",
                "experiment": "f1",
                "treatment": "original_spotserve",
                "repeat": repeat,
                "summary": summary(False),
            },
            {
                "stage": "formal",
                "experiment": "f1",
                "treatment": "moe_spotserve",
                "repeat": repeat,
                "summary": summary(True, latency=90.0, throughput=2.2),
            },
        ])
    contribution = driver.contribution_result(ledger)
    assert contribution["status"] == "supported"
    assert contribution["expert_placement_divergence_observed"] is True


def test_single_pass_reuses_identical_f1_rows_and_reports_f2_latency():
    driver = load_driver()

    def formal_row(experiment, treatment, p95):
        return {
            "status": "valid",
            "stage": "formal",
            "experiment": experiment,
            "treatment": treatment,
            "repeat": 1,
            "summary": {"latency_p95_ms": p95},
        }

    ledger = [
        formal_row("f1", "original_spotserve", 100.0),
        formal_row("f1", "moe_spotserve", 70.0),
    ]
    added = driver.reuse_f1_rows_for_f2(ledger, 1)

    assert len(added) == 2
    assert all(row["reused_without_gpu_rerun"] is True for row in added)
    assert {
        (row["treatment"], row["reused_from"]["treatment"])
        for row in added
    } == {
        ("original_spotserve", "original_spotserve"),
        ("full_moe_spotserve", "moe_spotserve"),
    }

    ledger.extend([
        formal_row("f2", "reparallelization_only", 90.0),
        formal_row("f2", "migration_only", 80.0),
    ])
    result = driver.f2_latency_ablation_result(ledger, expected_repeats=1)
    assert result["status"] == "complete"
    assert result["reparallelization_only_vs_original_percent"] == pytest.approx(10.0)
    assert result["migration_only_vs_original_percent"] == pytest.approx(20.0)
    assert result["full_vs_original_percent"] == pytest.approx(30.0)
    assert result[
        "reparallelization_marginal_with_migration_percent"
    ] == pytest.approx(12.5)


def test_generated_token_gate_rejects_incomplete_request_set(tmp_path):
    driver = load_driver()
    (tmp_path / "raw_requests.jsonl").write_text(
        json.dumps({
            "benchmark_phase": "measured",
            "success": True,
            "response": {"usage": {"completion_tokens": 1024}},
        }) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(driver.ExperimentError, match="exactly 2 requests"):
        driver.validate_generated_tokens(
            {"run_dir": str(tmp_path)},
            1024,
            expected_requests=2,
        )


def test_full_pipeline_uses_cli_repeat_count_and_resumes(
    tmp_path, monkeypatch
):
    driver = load_driver()
    spec_path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/spotserve/formal/k8s_qwen15_moe_a27b_8gpu.json"
    )
    spec = load_default_spec()
    profiles = [
        {
            "status": "passed",
            "shape": shape,
            "latency_avg_ms": 100.0,
            "latency_p95_ms": 120.0,
            "throughput_req_s": 2.0,
            "deployment_ready_latency_ms": 1000.0,
            "artifact": f"/simulated/{driver.shape_label(shape)}.json",
            "artifact_sha256": f"sha-{driver.shape_label(shape)}",
            "calibration_output_tokens": 256,
            "decode_ms_per_token": 100.0,
            "estimated_time_to_preemption_target_ms": 512.0,
        }
        for shape in spec["candidate_shapes"]
    ]
    calls = []

    monkeypatch.setattr(
        driver, "probe_cluster", lambda *args, **kwargs: fake_hardware()
    )
    monkeypatch.setattr(
        driver,
        "probe_network",
        lambda *args, **kwargs: {"minimum_throughput_gbps": 20.0},
    )
    monkeypatch.setattr(
        driver,
        "exact_chat_prompts",
        lambda *args, **kwargs: [
            f"prompt-{index}" for index in range(int(args[2]))
        ],
    )
    monkeypatch.setattr(
        driver,
        "ensure_candidate_profiles",
        lambda *args, **kwargs: profiles,
    )
    monkeypatch.setattr(
        driver, "build_capability", lambda *args, **kwargs: fake_capability()
    )

    def fake_run(
        spec, hardware, capability, prompts, output, endpoint,
        ray_address, ray_namespace, experiment, treatment, repeat,
    ):
        calls.append((experiment, treatment, repeat))
        is_moe = treatment in ("moe_spotserve", "full_moe_spotserve")
        latency = 80.0 if is_moe else 100.0
        return {
            "status": "valid",
            "experiment": experiment,
            "treatment": treatment,
            "repeat": repeat,
            "requested_formal_repeats": int(
                spec["experiment"]["formal_repeats"]
            ),
            "completed_at": driver.taipei_now(),
            "matrix": "/simulated/matrix.json",
            "deploy_config": "/simulated/deploy.json",
            "summary": {
                "success_rate": 1.0,
                "latency_avg_ms": latency,
                "latency_p95_ms": latency * 1.2,
                "throughput_req_s": 2.5 if is_moe else 2.0,
                "true_kv_restored_blocks_total": 64 if is_moe else 0,
                "replanning_avg_execution_duration_ms": 12.0,
                "replanning_latest_plan": json.dumps({
                    "tensor_parallel_size": 1,
                    "pipeline_parallel_size": 1,
                    "data_parallel_size": 4,
                    "replica_count": 1,
                    "enable_expert_parallel": True,
                }),
                "context_migration_latest_normalized_plan": json.dumps(
                    [{
                        "source_node": "node-a",
                        "target_node": "node-moe" if is_moe else "node-original",
                    }],
                    sort_keys=True,
                ),
                "replanning_latest_normalized_expert_placement": json.dumps({
                    "target_rank_count": 4,
                    "expert_to_target_rank": (
                        {"layer:0/expert:0": "ep-rank:1"} if is_moe else {}
                    ),
                }, sort_keys=True),
            },
        }

    monkeypatch.setattr(driver, "run_one_formal", fake_run)
    output = tmp_path / "results"
    original_pilot = fake_run(
        spec, fake_hardware(), fake_capability(), "prompt", output,
        "endpoint", "auto", "sllm", "f1", "original_spotserve", 0,
    )
    original_pilot["stage"] = "pilot"
    driver.write_json(output / "run-ledger.json", [original_pilot])
    calls.clear()
    argv = [
        str(spec_path),
        "--config", str(spec_path),
        "--output-dir", str(output),
        "--repeats", "2",
    ]
    monkeypatch.setattr("sys.argv", argv)

    assert driver.main() == 0
    ledger = json.loads((output / "run-ledger.json").read_text())
    state = json.loads((output / "state.json").read_text())
    report = (output / "results.md").read_text()
    assert len(ledger) == 18
    assert sum(row["stage"] == "pilot" for row in ledger) == 2
    assert sum(row["stage"] == "formal" for row in ledger) == 16
    assert all(
        row.get("requested_formal_repeats") == 2
        for row in ledger if row["stage"] == "formal"
    )
    assert state["requested_formal_repeats"] == 2
    assert state["status"] == "passed"
    assert state["contribution"]["status"] == "supported"
    assert "no_recovery" not in report
    assert (
        "| full_moe_spotserve | `stateful_recovery` | 2 |" in report
    )
    assert ("f1", "original_spotserve", 0) not in calls
    assert ("f1", "moe_spotserve", 0) in calls

    completed_call_count = len(calls)
    assert driver.main() == 0
    assert len(calls) == completed_call_count
