from scripts.run_granite_moe_tp_canary import audit_routes


def test_live_route_audit_requires_all_layers_and_valid_sparse_experts():
    report = audit_routes([[[0, 1], [2, 3]], [[1, 3], [0, 2]]], 2, 4, 2)
    assert report["verified"] is True
    assert report["observed_layer_count"] == 2
    assert report["assignments"] == 8


def test_missing_runtime_routes_do_not_verify_moe():
    assert audit_routes(None, 2, 4, 2)["verified"] is False
    assert audit_routes([], 2, 4, 2)["verified"] is False


def test_missing_layer_does_not_verify_moe():
    assert audit_routes([[[0, 1]]], 2, 4, 2)["verified"] is False


def test_invalid_expert_ids_do_not_verify_moe():
    assert audit_routes([[[0, -1], [2, 4]]], 2, 4, 2)["verified"] is False


def test_duplicate_topk_ids_do_not_verify_sparse_expert_selection():
    assert audit_routes([[[0, 0], [2, 3]]], 2, 4, 2)["verified"] is False
