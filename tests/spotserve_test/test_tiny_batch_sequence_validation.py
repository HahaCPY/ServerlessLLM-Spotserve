from run_tiny_batch_recovery import validate_recovery_sequence


def test_accepts_audited_continuation_before_late_greedy_divergence():
    reference = list(range(80))
    actual = reference[:72] + [999] + reference[73:]

    result = validate_recovery_sequence(
        reference,
        actual,
        source_completed_tokens=40,
        min_continuation_match_tokens=32,
    )

    assert result["source_prefix_equal_reference"] is True
    assert result["matched_continuation_tokens"] == 32
    assert result["equal"] is False
    assert result["validation_passed"] is True


def test_rejects_a_mismatch_inside_the_source_prefix():
    reference = list(range(80))
    actual = reference.copy()
    actual[39] = 999

    result = validate_recovery_sequence(reference, actual, 40, 32)

    assert result["source_prefix_equal_reference"] is False
    assert result["validation_passed"] is False


def test_rejects_too_few_matching_continuation_tokens():
    reference = list(range(80))
    actual = reference.copy()
    actual[71] = 999

    result = validate_recovery_sequence(reference, actual, 40, 32)

    assert result["matched_continuation_tokens"] == 31
    assert result["validation_passed"] is False


def test_clamps_required_match_to_short_remaining_suffix():
    reference = list(range(45))

    result = validate_recovery_sequence(reference, reference.copy(), 40, 32)

    assert result["required_continuation_match_tokens"] == 5
    assert result["validation_passed"] is True
