"""
Configuration for BDH models.
This module defines the configuration dataclass for BDH models,
including factory methods for different model sizes.
"""
from dataclasses import dataclass
from typing import Literal

@dataclass
class BDHConfig:
    """Configuration for BDH model architecture and training.
    Attributes:
        n_layer: Number of layers (with shared parameters across layers)
        n_embd: Embedding dimension (neuron dimension D)
        n_head: Number of attention heads
        mlp_internal_dim_multiplier: Multiplier for internal MLP dimension (synaptic dimension)
        vocab_size: Size of vocabulary
        dropout: Dropout probability
        block_size: Maximum sequence length for training
        
    Note:
        The internal MLP dimension N is computed as:
        N = mlp_internal_dim_multiplier * n_embd // n_head
    """

    # Architecture
    n_layer: int = 6
    n_embd: int = 256
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    vocab_size: int = 256

    # Regularization
    dropout: float = 0.1

    # Training
    block_size: int = 512

    @property
    def n_internal(self) -> int:
        """Compute internal MLP dimension (synaptic dimension N)."""
        return self.mlp_internal_dim_multiplier * self.n_embd // self.n_head

    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.n_layer > 0, "n_layer must be positive"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"

    @classmethod
    def tiny(cls) -> "BDHConfig":
        """Tiny model configuration (~10M parameters).
        
        Good for quick experimentation and testing.
        """
        return cls(
            n_layer=4,
            n_embd=128,
            n_head=4,
            mlp_internal_dim_multiplier=128,
            vocab_size=256,
        )

    @classmethod
    def small(cls) -> "BDHConfig":
        """Small model configuration (~25M parameters).
        
        Default configuration for initial experiments.
        """
        return cls(
            n_layer=6,
            n_embd=256,
            n_head=4,
            mlp_internal_dim_multiplier=128,
            vocab_size=256,
        )

    @classmethod
    def medium(cls) -> "BDHConfig":
        """Medium model configuration (~100M parameters).
        
        For more serious experiments with better performance.
        """
        return cls(
            n_layer=8,
            n_embd=512,
            n_head=8,
            mlp_internal_dim_multiplier=128,
            vocab_size=256,
        )

    def get_num_params(self) -> int:
        """Estimate total number of parameters.
        
        Returns:
            Approximate number of trainable parameters
        """
        # Embedding table
        params = self.vocab_size * self.n_embd
        
        # Per layer (shared across all layers)
        nh = self.n_head
        D = self.n_embd
        N = self.n_internal
        
        # Encoder, encoder_v, decoder
        params += nh * D * N  # encoder
        params += nh * D * N  # encoder_v
        params += nh * N * D  # decoder
        
        # LM head
        params += D * self.vocab_size
        
        return params