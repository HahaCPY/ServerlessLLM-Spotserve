"""Reassemble a non-streaming response after a committed decode boundary."""


def completed_response_tokens(prefix, suffix, prompt_tokens, restored_input):
    if any(type(token) is not int or token < 0 for token in [*prefix, *suffix]):
        raise ValueError("invalid_committed_output_tokens")
    if type(prompt_tokens) is not int or prompt_tokens < 0:
        raise ValueError("invalid_original_prompt_length")
    if list(restored_input) != list(restored_input[:prompt_tokens]) + list(prefix):
        raise ValueError("committed_prefix_does_not_match_restore_input")
    return list(prefix) + list(suffix)
