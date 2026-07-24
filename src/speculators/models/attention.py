"""Shared attention utilities for speculator models.

This module contains attention functions and utilities shared across different
speculator architectures (EAGLE3, DFlash, etc.) to avoid code duplication.
"""

from collections.abc import Callable

import torch
from transformers.modeling_utils import AttentionInterface

# ---------------------------------------------------------------------------
# Conditional flex_attention imports (Triton unavailable on Ascend NPU)
# ---------------------------------------------------------------------------
try:
    from torch.nn.attention.flex_attention import (  # noqa: WPS433
        BlockMask,
        create_mask as _create_mask,
        flex_attention,
    )

    _FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    _FLEX_ATTENTION_AVAILABLE = False
    BlockMask = None  # type: ignore[assignment,misc]
    _create_mask = None  # type: ignore[assignment]
    flex_attention = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Conditional torch_npu import (Ascend NPU)
# ---------------------------------------------------------------------------
try:
    import torch_npu

    _NPU_AVAILABLE = True
    _NPU_FUSION_ATTENTION_AVAILABLE = hasattr(torch_npu, "npu_fusion_attention")
except ImportError:
    torch_npu = None
    _NPU_AVAILABLE = False
    _NPU_FUSION_ATTENTION_AVAILABLE = False


# ---------------------------------------------------------------------------
# Pure-Python mask combinators (mirrors flex_attention.or_masks / and_masks
# so that eagle3/dflash attention modules don't need a flex_attention import
# just for these trivial closures).
# ---------------------------------------------------------------------------

def or_masks(*mask_mods: Callable) -> Callable:
    """Logical OR over mask_mod callables (equivalent to flex_attention.or_masks)."""

    def combined(_b, _h, q_idx, kv_idx):
        result = mask_mods[0](_b, _h, q_idx, kv_idx)
        for mod in mask_mods[1:]:
            result = result | mod(_b, _h, q_idx, kv_idx)
        return result

    return combined


def and_masks(*mask_mods: Callable) -> Callable:
    """Logical AND over mask_mod callables (equivalent to flex_attention.and_masks)."""

    def combined(_b, _h, q_idx, kv_idx):
        result = mask_mods[0](_b, _h, q_idx, kv_idx)
        for mod in mask_mods[1:]:
            result = result & mod(_b, _h, q_idx, kv_idx)
        return result

    return combined


def flex_attention_forward(
    module: torch.nn.Module,  # noqa: ARG001
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    scaling: float | None = None,
    **_kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Shared flex attention forward implementation.

    This function is used by both EAGLE3 and DFlash attention mechanisms to avoid
    code duplication and ensure consistent behavior.

    Args:
        module: The attention module (unused but required for interface compatibility).
        query: Query tensor of shape (batch, num_heads, seq_len, head_dim).
        key: Key tensor of shape (batch, num_heads, seq_len, head_dim).
        value: Value tensor of shape (batch, num_heads, seq_len, head_dim).
        attention_mask: BlockMask for flex attention.
        scaling: Optional scaling factor for attention scores.
        **_kwargs: Additional unused kwargs for interface compatibility.

    Returns:
        Tuple of (attention_output, None) where attention_output has shape
        (batch, seq_len, num_heads, head_dim) and None represents no attention weights.
    """
    if not _FLEX_ATTENTION_AVAILABLE:
        raise ImportError(
            "flex_attention is not available on this platform. "
            "Use 'npu_fusion_attention' on Ascend NPU or 'sdpa'/'eager' as fallback."
        )

    num_query_heads = query.shape[1]
    num_key_value_heads = key.shape[1]
    enable_gqa = num_query_heads != num_key_value_heads

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    flex_attention_output = flex_attention(
        query,
        key,
        value,
        score_mod=None,
        block_mask=attention_mask,
        enable_gqa=enable_gqa,
        scale=scaling,
    )
    attention_output: torch.Tensor = flex_attention_output
    attention_output = attention_output.transpose(1, 2).contiguous()
    return attention_output, None


def _prepare_npu_mask(
    attention_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    """Convert mask to NPU convention (True=masked).

    Handles:
      - None -> None (no masking)
      - bool tensor in flex convention (True=attend) -> invert
      - float tensor in SDPA convention (0 attend, -inf masked) -> invert
    """
    if attention_mask is None:
        return None
    if attention_mask.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        return ~(attention_mask != float("-inf"))
    if attention_mask.dtype == torch.bool:
        return ~attention_mask
    raise TypeError(
        f"Unsupported attention_mask dtype: {attention_mask.dtype}"
    )


@torch.compiler.disable
def npu_fusion_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask,
    scaling: float | None = None,
    dropout: float = 0.0,
    **_kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Forward using torch_npu.npu_fusion_attention (with v2 fallback).

    Converts mask from flex convention (True=attend) to NPU convention
    (True=masked) internally. Prefers v3 for graph-mode compatibility.
    """
    if torch_npu is None:
        raise ImportError("torch_npu is not available")

    num_heads = query.shape[1]
    npu_mask = _prepare_npu_mask(attention_mask)
    scale = scaling if scaling is not None else (query.shape[-1] ** -0.5)
    keep_prob = 1.0 - dropout if module.training else 1.0

    if _NPU_FUSION_ATTENTION_AVAILABLE:
        output = torch_npu.npu_fusion_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            num_heads,
            "BNSD",
            atten_mask=npu_mask,
            scale=scale,
            keep_prob=keep_prob,
            sparse_mode=1,
        )[0]
    else:
        raise ImportError(
            "npu_fusion_attention is not available on this platform. "
            "Use 'flex_attention' on CUDA or 'sdpa'/'eager' as fallback."
        )

    attention_output = output.transpose(1, 2).contiguous()
    return attention_output, None


def create_float_mask(
    mask_mod: Callable,
    B: int | None = None,  # noqa: N803
    H: int | None = None,  # noqa: N803
    Q_LEN: int = 0,  # noqa: N803
    KV_LEN: int = 0,  # noqa: N803
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Wrap ``create_mask`` and convert the boolean result to a float mask.

    Non-flex attention backends (eager, SDPA) add the mask numerically
    (``scores + mask``) and need 0 for attended and ``-inf`` for masked.
    """
    bool_mask = _create_mask(
        mask_mod, B=B, H=H, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device
    )
    float_mask = torch.zeros(bool_mask.shape, dtype=dtype, device=device)
    float_mask.masked_fill_(~bool_mask, float("-inf"))
    return float_mask


def block_mask_to_dense_attention_mask(
    block_mask: BlockMask, device: torch.device, dtype: torch.dtype
):
    attention_mask = torch.ones(block_mask.shape, device=device, dtype=dtype)

    for q_idx in range(attention_mask.shape[2]):
        attention_mask[0, 0, q_idx, :] = block_mask.mask_mod(
            torch.zeros(1, device=device, dtype=torch.long),
            torch.zeros(1, device=device, dtype=torch.long),
            torch.ones(1, device=device, dtype=torch.long) * q_idx,
            torch.arange(attention_mask.shape[3], device=device, dtype=torch.long),
        )
    return attention_mask


# Singleton registry for attention functions (shared across all models)
ALL_ATTENTION_FUNCTIONS = AttentionInterface()
ALL_ATTENTION_FUNCTIONS.register("simple_flex_attention", flex_attention_forward)
if _NPU_AVAILABLE:
    ALL_ATTENTION_FUNCTIONS.register(
        "npu_fusion_attention", npu_fusion_attention_forward
    )
