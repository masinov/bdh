"""
BDH (Biologically-inspired Dragon Hatchling) Language Model.
This module implements the core BDH architecture as described in the paper.
Phase 0 version maintains original implementation structure.
Future phases will add:

Persistent state (Phase 1)
Excitatory/Inhibitory circuits (Phase 3)
Multi-timescale dynamics (Phase 4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .config import BDHConfig
from .attention import Attention
class BDH(nn.Module):
    """BDH Language Model.
    Architecture:
    - Shared parameters across all layers (Universal Transformer style)
    - RoPE-based linear attention
    - ReLU-gated feedforward with Hebbian-like multiplication
    - Sparse positive activations emerge naturally

    Args:
        config: Model configuration
        
    Attributes:
        config: Stored configuration
        embed: Token embedding layer
        encoder: Projection to synaptic space (D -> N)
        encoder_v: Value projection for attention
        decoder: Projection back to neuron space (N -> D)
        attn: Attention mechanism
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
        
        # Attention mechanism
        self.attn = Attention(config)
        
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
        """
        C = self.config
        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = C.n_internal
        
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
            
            # Attention mechanism (currently O(T²), will be O(N²) in Phase 1)
            yKV = self.attn(
                Q=x_sparse,
                K=x_sparse,
                V=x
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
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        
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
            Currently recomputes for all tokens each step (O(T²) generation).
            Will be O(1) per token in Phase 1 with persistent state.
        """
        for _ in range(max_new_tokens):
            # Crop context if needed (current limitation)
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
        return n_params