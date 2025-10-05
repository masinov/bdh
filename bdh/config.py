"""
Configuration for BDH models.

This module defines the configuration dataclass for BDH models,
including factory methods for different model sizes.

Phase 1 additions:
- Persistent state configuration
- State decay modes and rates
- State regularization options
"""
from dataclasses import dataclass
from typing import Literal

@dataclass
class BDHConfig:
    """Configuration for BDH model architecture and training.
    
    Attributes:
        # Architecture
        n_layer: Number of layers (with shared parameters across layers)
        n_embd: Embedding dimension (neuron dimension D)
        n_head: Number of attention heads
        mlp_internal_dim_multiplier: Multiplier for internal MLP dimension (synaptic dimension)
        vocab_size: Size of vocabulary
        
        # Regularization
        dropout: Dropout probability
        
        # Training
        block_size: Maximum sequence length for training
        
        # Phase 1: Persistent State
        use_persistent_state: Whether to use persistent synaptic state
        state_decay_rate: Exponential decay factor γ ∈ (0, 1]. Typical: 0.99
        state_decay_mode: Type of temporal dynamics ('exponential', 'rope', 'learned')
        state_init_mode: How to initialize state ('zeros', 'warm')
        enable_state_regularization: Whether to apply state regularization during training
        state_spectral_penalty_weight: Weight for spectral norm regularization
        state_sparsity_target: Target sparsity level (fraction of near-zero synapses)
        
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
    
    # Phase 1: Persistent State
    use_persistent_state: bool = False
    state_decay_rate: float = 0.99
    state_decay_mode: Literal['exponential', 'rope', 'learned'] = 'exponential'
    state_init_mode: Literal['zeros', 'warm'] = 'zeros'
    enable_state_regularization: bool = True
    state_spectral_penalty_weight: float = 0.01
    state_sparsity_target: float = 0.95

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
        
        # Phase 1 validation
        if self.use_persistent_state:
            assert 0.0 < self.state_decay_rate <= 1.0, "state_decay_rate must be in (0, 1]"
            assert 0.0 <= self.state_sparsity_target <= 1.0, "state_sparsity_target must be in [0, 1]"
            assert self.state_spectral_penalty_weight >= 0.0, "penalty weight must be non-negative"

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
    
    @classmethod
    def with_persistent_state(cls, base_config: "BDHConfig" = None, **kwargs) -> "BDHConfig":
        """Create config with persistent state enabled.
        
        Args:
            base_config: Base configuration to modify (defaults to small())
            **kwargs: Additional state configuration overrides
            
        Returns:
            Config with use_persistent_state=True and optional overrides
        
        Example:
            config = BDHConfig.with_persistent_state(
                base_config=BDHConfig.small(),
                state_decay_rate=0.99,
                state_decay_mode='exponential'
            )
        """
        if base_config is None:
            base_config = cls.small()
        
        # Convert to dict, update with state settings
        config_dict = base_config.__dict__.copy()
        config_dict['use_persistent_state'] = True
        config_dict.update(kwargs)
        
        return cls(**config_dict)

    def get_num_params(self) -> int:
        """Estimate total number of parameters.
        
        Returns:
            Approximate number of trainable parameters
            
        Note:
            State matrices are buffers, not parameters, so they don't count here.
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
        
        # Learned decay rates (if using learned mode)
        if self.use_persistent_state and self.state_decay_mode == 'learned':
            params += self.n_layer * nh * N
        
        # LM head
        params += D * self.vocab_size
        
        return params