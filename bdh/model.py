"""
BDH (Biologically-inspired Dragon Hatchling) Language Model.

This module implements the core BDH architecture as described in the paper.

Phase 1 additions:
- Persistent synaptic state matrices
- True O(N²) linear attention with Hebbian updates
- State management across batches and layers
- O(1) generation per token

Future phases will add:
- Excitatory/Inhibitory circuits (Phase 3)
- Multi-timescale dynamics (Phase 4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from .config import BDHConfig
from .attention import Attention


class BDH(nn.Module):
    """BDH Language Model.
    
    Architecture:
    - Shared parameters across all layers (Universal Transformer style)
    - RoPE-based linear attention
    - ReLU-gated feedforward with Hebbian-like multiplication
    - Sparse positive activations emerge naturally
    - [Phase 1] Persistent state for true O(N²) attention
    
    Args:
        config: Model configuration
        
    Attributes:
        config: Stored configuration
        embed: Token embedding layer
        encoder: Projection to synaptic space (D -> N)
        encoder_v: Value projection for attention
        decoder: Projection back to neuron space (N -> D)
        attn: Attention mechanism (stateful or original)
        state_manager: Manages persistent state (if enabled)
        ln: Layer normalization (parameter-free)
        drop: Dropout layer
        lm_head: Language modeling head (D -> vocab_size)
    """

    def __init__(self, config: BDHConfig):
        super().__init__()
        assert config.vocab_size is not None, "vocab_size must be specified"
        self.config = config
        
        nh = config.n_head
        D = config.n_embd
        N = config.n_internal
        
        # Embedding
        self.embed = nn.Embedding(config.vocab_size, D)
        
        # Core transformation matrices (shared across layers)
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        
        # State management (Phase 1)
        self.use_persistent_state = getattr(config, 'use_persistent_state', False)
        
        if self.use_persistent_state:
            from .core.state import create_state_manager, StateConfig
            from .core.attention_stateful import create_attention_layer
            
            # Create state manager
            state_config = StateConfig(
                decay_rate=getattr(config, 'state_decay_rate', 0.99),
                decay_mode=getattr(config, 'state_decay_mode', 'exponential'),
            )
            self.state_manager = create_state_manager(config, state_config)
            
            # Create stateful attention
            self.attn = create_attention_layer(
                config,
                use_state=True,
                state_manager=self.state_manager
            )
            
            # Storage for state (will be initialized on first forward)
            self.S: Optional[List[torch.Tensor]] = None
        else:
            # Use original attention (Phase 0 fallback)
            self.attn = Attention(config)
            self.state_manager = None
            self.S = None
        
        # Normalization (parameter-free)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        
        # Regularization
        self.drop = nn.Dropout(config.dropout)
        
        # Language modeling head
        self.lm_head = nn.Parameter(torch.zeros((D, config.vocab_size)).normal_(std=0.02))
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize model weights.
        
        Args:
            module: Module to initialize
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def reset_state(self, batch_size: Optional[int] = None) -> None:
        """Reset persistent state.
        
        Call this at document boundaries during training/inference.
        
        Args:
            batch_size: If provided, initializes state; otherwise marks as uninitialized
        """
        if self.use_persistent_state:
            if batch_size is None:
                self.S = None
            else:
                device = next(self.parameters()).device
                dtype = next(self.parameters()).dtype
                self.S = self.state_manager.initialize_state(batch_size, device, dtype)
    
    def detach_state(self) -> None:
        """Detach state from computation graph (for TBPTT).
        
        Call this periodically during training to prevent memory issues.
        """
        if self.use_persistent_state and self.S is not None:
            self.state_manager.detach_states(self.S)
    
    def get_state(self) -> Optional[List[torch.Tensor]]:
        """Get current state for checkpointing.
        
        Returns:
            List of state tensors, one per layer, or None if not using state
        """
        if self.use_persistent_state and self.S is not None:
            return [s.detach().clone() for s in self.S]
        return None
    
    def set_state(self, state: List[torch.Tensor]) -> None:
        """Set state from checkpoint.
        
        Args:
            state: List of state tensors to restore
        """
        if self.use_persistent_state:
            self.S = [s.clone() for s in state]
    
    def get_state_statistics(self) -> Optional[dict]:
        """Get diagnostic statistics about current state.
        
        Returns:
            Dictionary of statistics or None if not using state
        """
        if self.use_persistent_state and self.S is not None:
            return self.state_manager.get_state_statistics(self.S)
        return None

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the model.
        
        Args:
            idx: Input token indices of shape (B, T)
            targets: Optional target indices of shape (B, T) for computing loss
            
        Returns:
            Tuple of:
                - logits: Predicted logits of shape (B, T, vocab_size)
                - loss: Optional scalar loss if targets provided
        
        Note:
            With persistent state, this is O(T × N²) per call.
            Without state, this is O(T² × d²) (original implementation).
        """
        C = self.config
        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = C.n_internal
        
        # Initialize state if needed
        if self.use_persistent_state:
            if self.S is None:
                self.S = self.state_manager.initialize_state(
                    B, idx.device, next(self.parameters()).dtype
                )
        
        # Embed tokens: (B, T) -> (B, 1, T, D)
        x = self.embed(idx).unsqueeze(1)
        
        # Initial layer norm (helps with training stability)
        x = self.ln(x)
        
        # Process through layers (shared parameters)
        for level in range(C.n_layer):
            # Project to synaptic space: (B, 1, T, D) -> (B, nh, T, N)
            x_latent = x @ self.encoder
            
            # Apply ReLU to get sparse activations
            x_sparse = F.relu(x_latent)
            
            # Attention mechanism
            if self.use_persistent_state:
                # Stateful attention: O(T × N²)
                yKV, self.S[level] = self.attn(
                    Q=x_sparse,
                    K=x_sparse,
                    V=x,
                    S_prev=self.S[level],
                    layer_idx=level
                )
            else:
                # Original attention: O(T²)
                yKV, _ = self.attn(
                    Q=x_sparse,
                    K=x_sparse,
                    V=x,
                    S_prev=None,
                    layer_idx=None
                )
            
            yKV = self.ln(yKV)
            
            # Project attention output to synaptic space for gating
            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            
            # Hebbian-like gating: multiply sparse activations
            # This implements "neurons that fire together wire together"
            xy_sparse = x_sparse * y_sparse
            
            # Apply dropout
            xy_sparse = self.drop(xy_sparse)
            
            # Project back to neuron space: (B, nh, T, N) -> (B, 1, T, D)
            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            )
            
            # Residual connection with layer norm
            y = self.ln(yMLP)
            x = self.ln(x + y)
        
        # Language modeling head: (B, 1, T, D) -> (B, T, vocab_size)
        logits = x.view(B, T, D) @ self.lm_head
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            
            # Add state regularization if using persistent state
            if self.use_persistent_state and self.training:
                state_reg = self.state_manager.compute_regularization_loss(self.S)
                loss = ce_loss + state_reg
            else:
                loss = ce_loss
        
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """Generate new tokens autoregressively.
        
        Args:
            idx: Starting token indices of shape (B, T)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k most likely tokens
            
        Returns:
            Generated sequence of shape (B, T + max_new_tokens)
            
        Note:
            With persistent state, this is O(1) per token after initialization.
            Without state, this is O(T²) per token (recomputes everything).
        """
        # If using state and it's not initialized, process the prompt first
        if self.use_persistent_state:
            if self.S is None or self.S[0].size(0) != idx.size(0):
                self.reset_state(batch_size=idx.size(0))
            
            # Process prompt to build initial state (only if needed)
            if idx.size(1) > 0:
                _, _ = self(idx)
        
        for _ in range(max_new_tokens):
            if self.use_persistent_state:
                # With state: only process last token (O(1) per token)
                idx_cond = idx[:, -1:]
            else:
                # Without state: need full context (O(T²) per token)
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Get logits
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Count number of parameters.
        
        Args:
            non_embedding: If True, exclude embedding parameters from count
            
        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed.weight.numel()
        
        # Note: State is not a parameter, it's a buffer
        # So it doesn't count toward parameter count
        
        return n_params