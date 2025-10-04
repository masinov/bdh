"""
Unit tests for BDH model components.
"""
import pytest
import torch
from bdh import BDH, BDHConfig, Attention


class TestBDHConfig:
    """Test configuration class."""

    def test_default_config(self):
        """Test default configuration."""
        config = BDHConfig()
        assert config.n_layer == 6
        assert config.n_embd == 256
        assert config.n_head == 4
        assert config.vocab_size == 256

    def test_tiny_config(self):
        """Test tiny model configuration."""
        config = BDHConfig.tiny()
        assert config.n_layer == 4
        assert config.n_embd == 128
        params = config.get_num_params()
        assert 5_000_000 < params < 15_000_000  # ~10M params

    def test_small_config(self):
        """Test small model configuration."""
        config = BDHConfig.small()
        params = config.get_num_params()
        assert 20_000_000 < params < 35_000_000  # ~25M params

    def test_internal_dim(self):
        """Test internal dimension computation."""
        config = BDHConfig(n_embd=256, n_head=4, mlp_internal_dim_multiplier=128)
        assert config.n_internal == 128 * 256 // 4

    def test_validation(self):
        """Test configuration validation."""
        with pytest.raises(AssertionError):
            BDHConfig(n_embd=255, n_head=4)  # Not divisible


class TestAttention:
    """Test attention mechanism."""

    def test_attention_forward(self):
        """Test attention forward pass."""
        config = BDHConfig.tiny()
        attn = Attention(config)
        
        B, nh, T, N = 2, 4, 10, 128
        D = 128
        
        Q = torch.randn(B, nh, T, N)
        K = Q  # Current implementation assumes K is Q
        V = torch.randn(B, nh, T, D)
        
        output = attn(Q, K, V)
        
        assert output.shape == (B, nh, T, D)

    def test_rope(self):
        """Test RoPE application."""
        phases = torch.randn(1, 1, 10, 64)
        v = torch.randn(2, 4, 10, 64)
        
        v_rotated = Attention.rope(phases, v)
        
        assert v_rotated.shape == v.shape
        # Check that rotation preserves norm (approximately)
        assert torch.allclose(v.norm(), v_rotated.norm(), rtol=1e-5)


class TestBDH:
    """Test full BDH model."""

    def test_model_creation(self):
        """Test model can be created."""
        config = BDHConfig.tiny()
        model = BDH(config)
        
        assert isinstance(model, torch.nn.Module)
        assert model.config == config

    def test_forward_pass(self):
        """Test forward pass."""
        config = BDHConfig.tiny()
        model = BDH(config)
        
        B, T = 2, 32
        idx = torch.randint(0, config.vocab_size, (B, T))
        
        logits, loss = model(idx)
        
        assert logits.shape == (B, T, config.vocab_size)
        assert loss is None  # No targets provided

    def test_forward_with_targets(self):
        """Test forward pass with loss computation."""
        config = BDHConfig.tiny()
        model = BDH(config)
        
        B, T = 2, 32
        idx = torch.randint(0, config.vocab_size, (B, T))
        targets = torch.randint(0, config.vocab_size, (B, T))
        
        logits, loss = model(idx, targets)
        
        assert logits.shape == (B, T, config.vocab_size)
        assert loss is not None
        assert loss.item() > 0

    def test_generation(self):
        """Test text generation."""
        config = BDHConfig.tiny()
        model = BDH(config)
        model.eval()
        
        B, T = 1, 10
        idx = torch.randint(0, config.vocab_size, (B, T))
        
        with torch.no_grad():
            output = model.generate(idx, max_new_tokens=20)
        
        assert output.shape == (B, T + 20)

    def test_num_params(self):
        """Test parameter counting."""
        config = BDHConfig.tiny()
        model = BDH(config)
        
        # Count manually
        manual_count = sum(p.numel() for p in model.parameters())
        method_count = model.get_num_params(non_embedding=False)
        
        assert manual_count == method_count

    def test_device_compatibility(self):
        """Test model works on different devices."""
        config = BDHConfig.tiny()
        
        # CPU
        model_cpu = BDH(config)
        idx = torch.randint(0, config.vocab_size, (2, 10))
        logits, _ = model_cpu(idx)
        assert logits.device.type == 'cpu'
        
        # GPU (if available)
        if torch.cuda.is_available():
            model_gpu = BDH(config).cuda()
            idx_gpu = idx.cuda()
            logits_gpu, _ = model_gpu(idx_gpu)
            assert logits_gpu.device.type == 'cuda'

    def test_gradient_flow(self):
        """Test gradients flow correctly."""
        config = BDHConfig.tiny()
        model = BDH(config)
        
        idx = torch.randint(0, config.vocab_size, (2, 10))
        targets = torch.randint(0, config.vocab_size, (2, 10))
        
        logits, loss = model(idx, targets)
        loss.backward()
        
        # Check that gradients exist for all parameters
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


class TestIntegration:
    """Integration tests."""

    def test_train_one_step(self):
        """Test one training step works."""
        config = BDHConfig.tiny()
        model = BDH(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        idx = torch.randint(0, config.vocab_size, (4, 32))
        targets = torch.randint(0, config.vocab_size, (4, 32))
        
        # Forward
        logits, loss = model(idx, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Loss should be finite
        assert torch.isfinite(loss).all()

    def test_overfitting_small_data(self):
        """Test model can overfit small dataset (sanity check)."""
        config = BDHConfig.tiny()
        config.n_layer = 2  # Smaller for speed
        model = BDH(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Create tiny dataset
        idx = torch.randint(0, config.vocab_size, (1, 32))
        targets = torch.randint(0, config.vocab_size, (1, 32))
        
        initial_loss = None
        
        # Train for 100 steps
        for i in range(100):
            logits, loss = model(idx, targets)
            
            if i == 0:
                initial_loss = loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        final_loss = loss.item()
        
        # Loss should decrease significantly
        assert final_loss < initial_loss * 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])