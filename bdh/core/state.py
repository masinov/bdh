"""
State management for BDH models.

This module implements persistent synaptic state matrices that accumulate information
across time steps. The state represents Hebbian-learned associations between neurons,
implementing "neurons that fire together wire together."

Key concepts:
- State lives on edges (synapses) between neurons, shape (B, nh, N, D)
- N is the synaptic dimension (~8K-32K), D is the neuronal dimension (~256-512)
- State updates via outer product: S ← γS + v ⊗ k^T
- Decay prevents unbounded growth and implements forgetting
"""

import torch
import torch.nn as nn
from typing import Optional, List


class StateManager(nn.Module):
    """Manages persistent synaptic state matrices across time and layers.
    
    The state represents learned associations between neurons, updated via
    Hebbian-like rules during inference. Each layer maintains independent state.
    
    State shape: (B, nh, N, D) where:
        B = batch size
        nh = number of heads
        N = synaptic dimension (mlp_internal_dim_multiplier * n_embd // n_head)
        D = neuronal dimension (n_embd)
    
    Args:
        config: BDHConfig instance containing all state configuration
    """
    
    def __init__(self, config):
        super().__init__()
        self.n_layers = config.n_layer
        self.n_head = config.n_head
        self.N = config.n_internal
        self.D = config.n_embd
        
        # Store state configuration from BDHConfig
        self.decay_rate = config.state_decay_rate
        self.decay_mode = config.state_decay_mode
        self.init_mode = config.state_init_mode
        self.enable_regularization = config.enable_state_regularization
        self.spectral_penalty_weight = config.state_spectral_penalty_weight
        self.sparsity_target = config.state_sparsity_target
        
        # Initialize state storage (will be populated on first forward)
        # Use register_buffer so it's moved with model but not trained
        self.register_buffer('_state_initialized', torch.tensor(False))
        
        # Learned decay rates (if using learned mode)
        if self.decay_mode == 'learned':
            self.decay_rates = nn.Parameter(
                torch.ones(self.n_layers, self.n_head, self.N) * self.decay_rate
            )
        
        # For RoPE decay mode, we need frequencies
        if self.decay_mode == 'rope':
            # Use same frequency generation as attention
            from ..attention import get_freqs
            freqs = get_freqs(self.N, theta=2**16, dtype=torch.float32)
            self.register_buffer('rope_freqs', freqs.view(1, 1, -1))
        
        # Storage for previous state (for warm initialization)
        self._prev_final_state: Optional[List[torch.Tensor]] = None
    
    def initialize_state(
        self, 
        batch_size: int, 
        device: torch.device,
        dtype: torch.dtype = torch.float32
    ) -> List[torch.Tensor]:
        """Initialize state matrices for all layers.
        
        Args:
            batch_size: Batch size
            device: Device to create tensors on
            dtype: Data type for state matrices
            
        Returns:
            List of state tensors, one per layer, each of shape (B, nh, N, D)
        """
        if self.init_mode == 'zeros' or self._prev_final_state is None:
            # Fresh initialization
            state = [
                torch.zeros(
                    batch_size, self.n_head, self.N, self.D,
                    device=device, dtype=dtype
                )
                for _ in range(self.n_layers)
            ]
        else:
            # Warm start: use previous state scaled down
            warm_factor = 0.1
            state = [
                s.detach().clone() * warm_factor 
                if s.size(0) == batch_size 
                else torch.zeros(batch_size, self.n_head, self.N, self.D, 
                                device=device, dtype=dtype)
                for s in self._prev_final_state
            ]
        
        self._state_initialized = torch.tensor(True)
        return state
    
    def reset_state(self, batch_size: Optional[int] = None) -> None:
        """Reset state to uninitialized or fresh zeros.
        
        Args:
            batch_size: If provided, initializes state; otherwise marks as uninitialized
        """
        self._state_initialized = torch.tensor(False)
        if batch_size is None:
            self._prev_final_state = None
    
    def checkpoint_state(self, state: List[torch.Tensor]) -> None:
        """Save state for potential warm restart.
        
        Args:
            state: Current state to checkpoint
        """
        self._prev_final_state = [s.detach().clone() for s in state]
    
    def apply_decay(
        self, 
        S: torch.Tensor, 
        layer_idx: int,
        step: int = 1
    ) -> torch.Tensor:
        """Apply temporal decay to state matrix.
        
        Implements forgetting and prevents unbounded growth.
        
        Args:
            S: State matrix of shape (B, nh, N, D)
            layer_idx: Which layer this state belongs to
            step: Time steps elapsed (for decay calculation)
            
        Returns:
            Decayed state matrix
        """
        if self.decay_mode == 'exponential':
            # Simple exponential decay: S ← γS
            decay = self.decay_rate ** step
            return decay * S
        
        elif self.decay_mode == 'learned':
            # Per-neuron learned decay rates
            decay = self.decay_rates[layer_idx].unsqueeze(-1)  # (nh, N, 1)
            decay = torch.sigmoid(decay)  # Ensure in (0, 1)
            decay = decay ** step
            return S * decay
        
        elif self.decay_mode == 'rope':
            # Oscillatory decay: rotate state vectors
            return self._apply_rope_decay(S, step)
        
        else:
            raise ValueError(f"Unknown decay mode: {self.decay_mode}")
    
    def _apply_rope_decay(self, S: torch.Tensor, step: int = 1) -> torch.Tensor:
        """Apply RoPE-style rotational decay to state.
        
        This implements oscillatory dynamics where state vectors rotate
        in high-dimensional space, providing a form of temporal coding.
        
        Args:
            S: State matrix of shape (B, nh, N, D)
            step: Time steps elapsed
            
        Returns:
            Rotated state matrix
        """
        # Compute phases based on time step
        phases = self.rope_freqs * step  # (1, 1, N)
        phases = (phases % 1) * (2 * torch.pi)
        
        # Treat N dimension as complex pairs
        # Reshape to (B, nh, N//2, 2, D) for complex operations
        S_reshaped = S.reshape(*S.shape[:-2], -1, 2, S.shape[-1])
        
        # Convert to complex
        S_complex = torch.view_as_complex(S_reshaped.contiguous())
        
        # Apply rotation
        rotation = torch.exp(1j * phases.view(1, 1, -1, 1))
        S_rotated = S_complex * rotation
        
        # Convert back to real
        S_real = torch.view_as_real(S_rotated)
        return S_real.reshape(*S.shape)
    
    def compute_regularization_loss(
        self, 
        state: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute regularization loss for state matrices.
        
        Includes:
        - Spectral norm penalty (prevents explosion)
        - Sparsity encouragement (biological plausibility)
        
        Args:
            state: List of state tensors from all layers
            
        Returns:
            Scalar regularization loss
        """
        if not self.enable_regularization:
            return torch.tensor(0.0, device=state[0].device)
        
        total_loss = 0.0
        
        for S in state:
            # Spectral norm penalty: penalize large singular values
            # Use approximate method (Frobenius norm) for efficiency
            frobenius_norm = torch.norm(S, p='fro')
            target_norm = (S.size(-2) * S.size(-1)) ** 0.5  # Expected for normalized
            spectral_penalty = torch.relu(frobenius_norm - target_norm) ** 2
            
            # Sparsity penalty: encourage many near-zero synapses
            sparsity = (S.abs() < 0.01).float().mean()
            sparsity_penalty = (self.sparsity_target - sparsity) ** 2
            
            total_loss += self.spectral_penalty_weight * spectral_penalty
            total_loss += 0.001 * sparsity_penalty
        
        return total_loss / len(state)
    
    def get_state_statistics(self, state: List[torch.Tensor]) -> dict:
        """Compute diagnostic statistics about current state.
        
        Useful for monitoring training and debugging.
        
        Args:
            state: List of state tensors from all layers
            
        Returns:
            Dictionary of statistics
        """
        stats = {}
        
        for layer_idx, S in enumerate(state):
            # Basic statistics
            stats[f'layer_{layer_idx}_mean'] = S.mean().item()
            stats[f'layer_{layer_idx}_std'] = S.std().item()
            stats[f'layer_{layer_idx}_max'] = S.abs().max().item()
            
            # Sparsity (fraction of near-zero elements)
            sparsity = (S.abs() < 0.01).float().mean().item()
            stats[f'layer_{layer_idx}_sparsity'] = sparsity
            
            # Approximate spectral norm
            frobenius = torch.norm(S, p='fro').item()
            stats[f'layer_{layer_idx}_frobenius'] = frobenius
        
        # Aggregate statistics
        all_values = torch.cat([S.flatten() for S in state])
        stats['global_mean'] = all_values.mean().item()
        stats['global_std'] = all_values.std().item()
        stats['global_sparsity'] = (all_values.abs() < 0.01).float().mean().item()
        
        return stats
    
    def detach_states(self, state: List[torch.Tensor]) -> None:
        """Detach states from computation graph (for TBPTT).
        
        This is crucial for truncated backpropagation through time.
        Call this periodically during training to prevent memory issues.
        
        Args:
            state: List of state tensors to detach (modified in-place)
        """
        for S in state:
            S.detach_()