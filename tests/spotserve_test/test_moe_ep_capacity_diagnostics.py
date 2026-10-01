"""No GPU required: diagnostics must not mistake AGRS for a second allreduce."""

from scripts.analyze_moe_phase2_diagnostics import duplicate_normal_request_gate, verify_collective_gate
from scripts.run_moe_ep_capacity_sweep import config_summary


def test_collective_checks_dispatch_combine_and_sequence_parallel_noop():
    def tensor(values):
        return {"shape":[len(values),len(values[0])],"values":values,"dtype":"torch.float32"}

    records=[]
    for rank in range(2):
        own=[[rank+1.0,rank+2.0]]
        other=[[2.0 if rank==0 else 1.0,3.0 if rank==0 else 2.0]]
        ordered=[own[0],other[0]] if rank==0 else [other[0],own[0]]
        records.append({"rank":rank,"global_expert_ids":list(range(rank*20,(rank+1)*20)),
            "records":{"agrs_dispatch":{"input":{"hidden":tensor(own),
                "weights":tensor([[rank+0.5]]),"ids":tensor([[rank]])},
                "output":{"hidden":tensor(ordered),"weights":tensor([[0.5],[1.5]]),
                          "ids":tensor([[0],[1]])}},
                "agrs_combine":{"input":tensor([[1,3],[4,5]] if rank==0 else [[2,4],[5,6]]),
                                "output":tensor([[3,7]] if rank==0 else [[9,11]])},
                "moe_final_reduce":{"tp_size":1,"ep_size":2,"is_sequence_parallel":True,
                                    "input":tensor(own),"output":tensor(own)}}})
    report={"collective_gate":{"collectives":records,"runtime_correctness":[
        {"router_totals":{"assignments":8},"expert_oracle":[
            {"passes_predeclared_tolerance":True,"nonfinite_output_values":0}]}
        for _ in range(2)]}}
    gate=verify_collective_gate(report)
    assert gate["passes"], gate["checks"]
    records[1]["records"]["agrs_dispatch"]["output"]["ids"]["values"][1][0]=2
    assert not verify_collective_gate(report)["passes"]


def test_full_replica_throughput_uses_two_requests_and_slower_wall():
    def report(delay):
        request={"latency_s":delay,"ttft_s":0.1,"tpot_s":0.01,"passed_stream":True}
        return {"cells":[{"prompt_tokens":512,"output_tokens":128,"concurrency":1,
            "status":"complete","samples":[{"wall_s":delay,"requests":[request]}]}]}
    result=config_summary([report(1.0),report(2.0)],2)[0]
    assert result["total_concurrency"]==2
    assert result["generated_tokens_per_s"]==128.0
    assert result["completed_requests"]==2


def test_tp2_ep2_uses_outer_tp_group_even_if_moe_tp_size_is_one():
    def tensor(values):
        return {"shape":[len(values),len(values[0])],"values":values,"dtype":"torch.bfloat16"}
    records=[{"rank":rank,"global_expert_ids":list(range(rank*20,(rank+1)*20)),
        "records":{"moe_final_reduce":{"tp_size":1,"ep_size":2,
            "actual_tp_group_size":2,"input":tensor([[rank+1.0]]),
            "output":tensor([[3.0]])}}} for rank in range(2)]
    report={"tp":2,"collective_gate":{"collectives":records,"runtime_correctness":[
        {"router_totals":{"assignments":8},"expert_oracle":[
            {"passes_predeclared_tolerance":True,"nonfinite_output_values":0}]}
        for _ in range(2)]}}
    assert verify_collective_gate(report)["passes"]


def test_duplicate_normal_request_only_compares_scores_at_common_prefix():
    same={"prompt_sha256":"same"}
    requests=[{**same,"output_token_ids":[1,2,3],"logprobs":[{},
        {"2":{"logprob":-0.2},"5":{"logprob":-0.21}},{}]} for _ in range(8)]
    requests[4]={**same,"output_token_ids":[1,5,6],"logprobs":[{},
        {"2":{"logprob":-0.22},"5":{"logprob":-0.19}},{}]}
    value=duplicate_normal_request_gate({"batch_shape_gate":{
        "requests":requests,"normal_train_code_duplicated_at_indices":[0,4]}})
    assert value["first_different_token_1based"]==2
    assert value["common_previous_tokens"]==1
    assert abs(value["max_shared_logprob_abs_error"]-0.02)<1e-9
