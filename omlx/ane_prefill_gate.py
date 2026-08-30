"""Prompt-length gate for the opt-in Qwen ANE prefill modes.

The new ANE prefill modes (SwiGLU-in-ANE / MoE shared expert / o_proj split)
only pay off once the prefill is long enough to amortize their fixed per-call
overhead, so a short prompt is better served by plain GPU prefill. This module
centralizes the *decision* (it has no MLX dependency, so it is unit-testable
anywhere) that the engine uses to decide whether to engage the ANE backends
for a given request.
"""

from __future__ import annotations


def ane_prefill_should_engage(
    num_prompt_tokens: int,
    min_pp_tokens: int,
) -> bool:
    """Return True when the ANE prefill modes should run for this request.

    Args:
        num_prompt_tokens: token count of the incoming prompt.
        min_pp_tokens: the configured ``qwen35_ane_prefill_min_pp_tokens``
            threshold. ``0`` (or None) disables the gate -> always engage.

    Returns:
        True when ANE prefill should be used (long enough prompt, or the gate
        is disabled); False when the prompt is shorter than the threshold and
        the request should fall back to plain GPU prefill.
    """
    if min_pp_tokens is None or min_pp_tokens <= 0:
        return True
    return int(num_prompt_tokens) >= int(min_pp_tokens)
