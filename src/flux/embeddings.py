from typing import Tuple, Union

import torch


# modified from diffusers.models.embeddings, support batch dim
def apply_rotary_emb(
        x: torch.Tensor,
        freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
        use_real: bool = True,
        use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings with extended support for 4D frequency tensors.

    Args:
        x: Input tensor of shape [B, H, S, D]
        freqs_cis: Tuple of (cos, sin) tensors. Supports shapes:
                   - 2D: ([S, D], [S, D])        (no batch)
                   - 3D: ([B, S, D], [B, S, D]) (shared across heads)
                   - 4D: ([B, H, S, D], [B, H, S, D]) or ([B, 1, S, D], [B, 1, S, D])
        use_real: Use real number computation method
        use_real_unbind_dim: Dimension for splitting real/imaginary parts

    Returns:
        Tensor with rotary embeddings applied
    """
    if use_real:
        cos, sin = freqs_cis
        B, H, S, D = x.shape  # Original input shape

        # Dimension handling for different cases
        if cos.dim() == 2:  # [S, D] → [1, 1, S, D]
            cos = cos[None, None].expand(B, H, -1, -1)
            sin = sin[None, None].expand(B, H, -1, -1)
        elif cos.dim() == 3:  # [B, S, D] → [B, 1, S, D]
            cos = cos[:, None].expand(-1, H, -1, -1)
            sin = sin[:, None].expand(-1, H, -1, -1)
        elif cos.dim() == 4:
            # Case 1: [B, 1, S, D] → [B, H, S, D] via broadcasting
            if cos.shape[1] == 1:
                cos = cos.expand(-1, H, -1, -1)
                sin = sin.expand(-1, H, -1, -1)
            # Case 2: [B, H, S, D] directly use
            elif cos.shape[1] != H:
                raise ValueError(f"4D freq_cis has {cos.shape[1]} heads but input has {H} heads")
        else:
            raise ValueError(f"Unsupported frequency tensor dim: {cos.dim()}")

        cos, sin = cos.to(x.device), sin.to(x.device)

        # Rotation computation (保持原有逻辑不变)
        if use_real_unbind_dim == -1:
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"Invalid unbind dim: {use_real_unbind_dim}")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
        return out
    else:
        # 非 real 模式的处理（保持原样）
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
        return x_out.type_as(x)