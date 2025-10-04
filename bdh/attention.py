"""
Attention mechanisms for BDH.
This module implements the RoPE-based linear attention used in BDH.
Currently implements causal masking; will be modified in Phase 1 for true linear attention.
"""
import math
import torch
import torch.nn as nn
from typing import Tuple

def get_freqs(n: int, theta: float = 2**16, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Generate frequency values for RoPE (Rotary Position Embedding).
    Args:
        n: Dimension size
        theta: Base for frequency computation
        dtype: Data type for output tensor
        
    Returns:
        Tensor of shape (n,) containing frequency values
    """
    def quantize(t: torch.Tensor, q: int = 2) -> torch.Tensor:
        """Quantize tensor values to multiples of q."""
        return (t / q).floor() * q

    positions = quantize(torch.arange(0, n, 1, dtype=dtype))
    freqs = 1.0 / (theta ** (positions / n)) / (2 * math.pi)
    return freqs

class Attention(nn.Module):
    """RoPE-based attention mechanism for BDH.
    This implements a linear attention variant with Rotary Position Embeddings.
    Currently uses causal masking (.tril) which makes it O(T²).
    In Phase 1, this will be replaced with true O(N²) linear attention with state.

    Args:
        config: Model configuration
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.n_internal
        
        # RoPE frequencies
        self.register_buffer(
            'freqs',
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )

    @staticmethod
    def phases_cos_sin(phases: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert phases to cos and sin components.
        
        Args:
            phases: Phase values in range [0, 1)
            
        Returns:
            Tuple of (cos_values, sin_values)
        """
        phases = (phases % 1) * (2 * math.pi)
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)
        return phases_cos, phases_sin

    @staticmethod
    def rope(phases: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply Rotary Position Embedding.
        
        Args:
            phases: Position-dependent phase values
            v: Input tensor to rotate
            
        Returns:
            Rotated tensor with same shape as v
        """
        # Create rotated version by swapping and negating alternate dimensions
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        
        phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        
        # Apply rotation: v' = v * cos(θ) + v_rot * sin(θ)
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Compute attention output.
        
        Args:
            Q: Queries of shape (B, nh, T, N)
            K: Keys of shape (B, nh, T, N)
            V: Values of shape (B, nh, T, D)
            
        Returns:
            Attention output of shape (B, nh, T, D)
            
        Note:
            Currently assumes Q is K (same tensor).
            Uses causal masking which makes this O(T²).
            Will be replaced with stateful linear attention in Phase 1.
        """
        assert self.freqs.dtype == torch.float32
        assert K is Q, "Current implementation assumes K and Q are the same"
        
        _, _, T, _ = Q.size()
        
        # Compute position-dependent phases for RoPE
        r_phases = (
            torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype)
            .view(1, 1, -1, 1)
        ) * self.freqs
        
        # Apply RoPE to queries and keys
        QR = self.rope(r_phases, Q)
        KR = QR  # Since K is Q
        
        # Compute attention scores with causal masking
        # NOTE: This is O(T²) and will be replaced in Phase 1
        scores = (QR @ KR.mT).tril(diagonal=-1)
        
        # Apply attention to values
        return scores @ V