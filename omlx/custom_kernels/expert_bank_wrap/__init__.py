# SPDX-License-Identifier: Apache-2.0
"""Full-bank external wrap for MoE expert streaming (Fase M).

Adapted from jundot/omlx PR #3437 (qwen4_moe_stream) by @alytaphoenix,
Apache-2.0. See docs/expert-streaming.md, "Fase M".
"""

from . import fast

__all__ = ["fast"]
