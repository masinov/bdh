"""
Stateful linear attention for BDH models.

This module implements true O(N²) linear attention with persistent state,
replacing the O(T²) causal-masked attention from Phase 0.

Key differences from original attention.py:
- Autoregressive state accumulation instead of causal masking
- Hebbian updates: S ← γS + v ⊗ k^T
- O(1) complexity per token during inference
- State persists across batches (for long documents)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from ..attention import get_freqs, Attention  # Reuse RoPE utilities


class LinearAttentionWithState(nn.Module):
    """Linear attention mechanism with persistent synaptic state.
    
    Implements the attention mechanism from BDH paper Equation 8:
        S^(t,ℓ) = γS^(t-1,ℓ) + v^(t,ℓ) ⊗ k^(t,ℓ)ᵀ
        yKV^(t,ℓ) = S^(t,ℓ) × k^(t,ℓ)
    
    This replaces the O(T²) causal masking with O(N²) state updates.
    Causality is guaranteed by sequential accumulation.
    
    Args:
        config: Model configuration
        state_manager: StateManager instance for decay operations
    """
    
    def __init__(self, config, state_manager):
        super().__init__()
        self.config = config
        self.state_manager = state_manager
        
        nh = config.n_head
        N = config.n_internal
        
        # RoPE frequencies (reuse from original attention)
        self.register_buffer(
            'freqs',
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )
    
    @staticmethod
    def phases_cos_sin(phases: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert phases to cos and sin components.
        
        Reuses the static method from original Attention class.
        """
        return Attention.phases_cos_sin(phases)
    
    @staticmethod
    def rope(phases: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply Rotary Position Embedding.
        
        Reuses the static method from original Attention class.
        """
        return Attention.rope(phases, v)
    
    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        S_prev: torch.Tensor,
        layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute stateful linear attention.
        
        Args:
            Q: Queries of shape (B, nh, T, N)
            K: Keys of shape (B, nh, T, N)
            V: Values of shape (B, nh, T, D)
            S_prev: Previous state of shape (B, nh, N, D)
            layer_idx: Layer index for decay rate lookup
            
        Returns:
            Tuple of:
                - Attention output of shape (B, nh, T, D)
                - Updated state of shape (B, nh, N, D)
        
        Note:
            This is O(T × N × D) = O(T × N²) during training,
            but O(N²) per token during inference (T=1).
        """
        assert self.freqs.dtype == torch.float32
        assert K is Q, "Current implementation assumes K and Q are the same"
        
        B, nh, T, N = Q.size()
        D = V.size(-1)
        
        # Compute position-dependent phases for RoPE
        r_phases = (
            torch.arange(0, T, device=self.freqs.device, dtype=self.freqs.dtype)
            .view(1, 1, -1, 1)
        ) * self.freqs
        
        # Apply RoPE to queries and keys
        QR = self.rope(r_phases, Q)
        KR = QR  # Since K is Q
        
        # CRITICAL: Autoregressive loop - no .tril() masking!
        # Causality comes from sequential accumulation
        S = S_prev.clone()
        outputs = []
        
        for t in range(T):
            # 1. Query current state (inference from accumulated past)
            # This is the attention operation: query × state
            q_t = QR[:, :, t, :]  # (B, nh, N)
            output_t = torch.einsum('bhn,bhnd->bhd', q_t, S)
            outputs.append(output_t)
            
            # 2. Hebbian update: strengthen synapses for co-active neurons
            # S ← S + k ⊗ v (outer product)
            k_t = KR[:, :, t, :]  # (B, nh, N)
            v_t = V[:, :, t, :]   # (B, nh, D)
            
            # Outer product: (B, nh, N, 1) × (B, nh, 1, D) → (B, nh, N, D)
            S = S + torch.einsum('bhn,bhd->bhnd', k_t, v_t)
            
            # 3. Apply decay after update
            # This prevents unbounded growth and implements forgetting
            S = self.state_manager.apply_decay(S, layer_idx, step=1)
        
        # Stack outputs: list of (B, nh, D) → (B, nh, T, D)
        output = torch.stack(outputs, dim=2)
        
        return output, S
    
    @torch.no_grad()
    def forward_inference(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        S_prev: torch.Tensor,
        layer_idx: int,
        position: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimized forward for single-token inference (generation).
        
        This is the O(1) per token path used during generation.
        
        Args:
            Q: Query of shape (B, nh, 1, N) - single token
            K: Key of shape (B, nh, 1, N) - single token
            V: Value of shape (B, nh, 1, D) - single token
            S_prev: Previous state of shape (B, nh, N, D)
            layer_idx: Layer index for decay rate lookup
            position: Current position (for RoPE)
            
        Returns:
            Tuple of:
                - Attention output of shape (B, nh, 1, D)
                - Updated state of shape (B, nh, N, D)
        """
        assert Q.size(2) == 1, "Inference mode expects single token"
        
        # Compute RoPE phase for current position
        r_phase = (
            torch.tensor([position], device=self.freqs.device, dtype=self.freqs.dtype)
            .view(1, 1, 1, 1)
        ) * self.freqs
        
        # Apply RoPE
        QR = self.rope(r_phase, Q)
        KR = QR
        
        # Query state (O(N × D))
        q = QR.squeeze(2)  # (B, nh, N)
        output = torch.einsum('bhn,bhnd->bhd', q, S_prev)
        output = output.unsqueeze(2)  # (B, nh, 1, D)
        
        # Hebbian update (O(N × D))
        k = KR.squeeze(2)  # (B, nh, N)
        v = V.squeeze(2)   # (B, nh, D)
        S_new = S_prev + torch.einsum('bhn,bhd->bhnd', k, v)
        
        # Apply decay
        S_new = self.state_manager.apply_decay(S_new, layer_idx, step=1)
        
        return output, S_new


class LinearAttentionNoState(nn.Module):
    """Fallback: Linear attention without state (for backward compatibility).
    
    This maintains the same interface but doesn't use persistent state.
    Used when use_persistent_state=False in config.
    
    Essentially wraps the original attention with the new interface.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention = Attention(config)
    
    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        S_prev: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass without state (falls back to original attention).
        
        Args:
            Q, K, V: Same as stateful version
            S_prev: Ignored (for interface compatibility)
            layer_idx: Ignored (for interface compatibility)
            
        Returns:
            Tuple of (output, None) - no state to return
        """
        output = self.attention(Q, K, V)
        return output, None


def create_attention_layer(
    config,
    use_state: bool = True,
    state_manager = None
):
    """Factory function to create appropriate attention layer.
    
    Args:
        config: Model configuration
        use_state: Whether to use stateful attention
        state_manager: StateManager instance (required if use_state=True)
        
    Returns:
        LinearAttentionWithState or LinearAttentionNoState
    """
    if use_state:
        assert state_manager is not None, "state_manager required for stateful attention"
        return LinearAttentionWithState(config, state_manager)
    else:
        return LinearAttentionNoState(config)