# BDH: Biologically-inspired Dragon Hatchling
A language model architecture that bridges transformers and brain-like computation. 
Alternative implementation of the paper "The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain"
https://arxiv.org/html/2509.26507v1

## Phase 0: Baseline Implementation

This is the Phase 0 baseline implementation, refactored from the original code with:

- Modular structure (separate config, model, attention)
- Comprehensive type hints and docstrings
- Test suite
- Training and generation scripts
- Baseline metrics tracking

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Quick Start

### Training

Train a small model (~25M params) on Tiny Shakespeare:
```bash
python scripts/train.py --model_size small --max_iters 5000
```

Train a tiny model (~10M params) for quick testing:
```bash
python scripts/train.py --model_size tiny --max_iters 2000
```

### Generation

Generate text from a trained model:
```bash
python scripts/generate.py \
    --checkpoint results/checkpoints/best_model.pt \
    --prompt "ROMEO:" \
    --max_new_tokens 500
```

### Testing

Run unit tests:
```bash
pytest tests/ -v
```

## Model Sizes

- **Tiny**: ~10M parameters (`n_embd=128`, `n_layer=4`)
- **Small**: ~25M parameters (`n_embd=256`, `n_layer=6`)
- **Medium**: ~100M parameters (`n_embd=512`, `n_layer=8`)

## Project Structure

```
bdh/
├── bdh/              # Core package
│   ├── config.py     # Configuration
│   ├── attention.py  # Attention mechanisms
│   ├── model.py      # BDH model
│   └── utils.py      # Utilities
├── scripts/          # Training scripts
│   ├── train.py      # Training script
│   └── generate.py   # Generation script
├── tests/            # Unit tests
└── results/          # Output directory
```

## Baseline Metrics

After running training, metrics will be saved to `results/baseline_metrics.md`.

**Expected performance on Tiny Shakespeare**:

- **Small model (25M)**: Val loss ~1.5–1.6 after 5K iterations (~1 hour)
- **Tiny model (10M)**: Val loss ~1.6–1.7 after 2K iterations (~20 minutes)

## Next Steps: Phase 1

Phase 1 will add:

- Persistent state matrices (the core BDH innovation)
- True O(N²) linear attention (removing O(T²) causal masking)
- State decay mechanisms
- Stateful training pipeline

This will enable:

- Infinite sequence length processing
- O(1) generation cost per token
- True biologically-inspired memory dynamics

## Citation

Based on the paper *"The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain"* (2025).

## License

MIT