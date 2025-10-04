"""
Training script for BDH on Tiny Shakespeare dataset.
This script:

Downloads Tiny Shakespeare if not present
Trains a BDH model
Tracks metrics (loss, tokens/sec, memory)
Saves checkpoints and generates samples
Documents baseline performance

Usage:
python scripts/train.py --model_size small --max_iters 5000
"""
import argparse
import os
import pickle
import requests
import torch
import torch.nn.functional as F
from contextlib import nullcontext
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bdh import BDH, BDHConfig, get_device, get_batch, estimate_loss, TrainingMonitor


def download_data(data_dir: str = 'data') -> str:
    """Download Tiny Shakespeare dataset if not present.
    Args:
        data_dir: Directory to store data
        
    Returns:
        Path to the downloaded file
    """
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, 'input.txt')

    if not os.path.exists(data_path):
        print("Downloading Tiny Shakespeare dataset...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        response = requests.get(url)
        with open(data_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Downloaded to {data_path}")
    else:
        print(f"Dataset already exists at {data_path}")

    return data_path


def prepare_data(data_path: str, data_dir: str = 'data') -> tuple:
    """Prepare training and validation data.
    Args:
        data_path: Path to input text file
        data_dir: Directory for processed data
        
    Returns:
        Tuple of (train_data, val_data, vocab_size)
    """
    # Check if preprocessed data exists
    train_path = os.path.join(data_dir, 'train.bin')
    val_path = os.path.join(data_dir, 'val.bin')
    meta_path = os.path.join(data_dir, 'meta.pkl')

    if all(os.path.exists(p) for p in [train_path, val_path, meta_path]):
        print("Loading preprocessed data...")
        train_data = torch.from_numpy(
            memmap(train_path, dtype='uint8', mode='r')
        ).long()
        val_data = torch.from_numpy(
            memmap(val_path, dtype='uint8', mode='r')
        ).long()
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        vocab_size = meta['vocab_size']
        print(f"Vocab size: {vocab_size}")
        return train_data, val_data, vocab_size

    # Load and process raw data
    print("Processing raw data...")
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Use character-level tokenization (simple)
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size}")

    # Create character mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(l):
        return ''.join([itos[i] for i in l])

    # Encode entire dataset
    data = torch.tensor(encode(text), dtype=torch.long)

    # Split into train/val (90/10)
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    # Save processed data
    print("Saving preprocessed data...")
    train_data.numpy().tofile(train_path)
    val_data.numpy().tofile(val_path)

    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    return train_data, val_data, vocab_size


def memmap(path, dtype, mode):
    """Simple wrapper around numpy memmap."""
    import numpy as np
    return np.memmap(path, dtype=dtype, mode=mode)


def main():
    parser = argparse.ArgumentParser(description='Train BDH on Tiny Shakespeare')
    parser.add_argument('--model_size', type=str, default='small',
                        choices=['tiny', 'small', 'medium'],
                        help='Model size (tiny=10M, small=25M, medium=100M)')
    parser.add_argument('--max_iters', type=int, default=5000,
                        help='Maximum number of training iterations')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--block_size', type=int, default=512,
                        help='Context length')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--eval_interval', type=int, default=500,
                        help='Evaluate every N iterations')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Log every N iterations')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--seed', type=int, default=1337,
                        help='Random seed')
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Setup device
    device = get_device()
    print(f"Using device: {device}")

    # Mixed precision setup
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = torch.amp.autocast(device_type=device.type, dtype=ptdtype) if 'cuda' in device.type else nullcontext()
    scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))

    print(f"Using dtype: {dtype}")

    # Download and prepare data
    data_path = download_data()
    train_data, val_data, vocab_size = prepare_data(data_path)

    print(f"\nDataset statistics:")
    print(f"  Train tokens: {len(train_data):,}")
    print(f"  Val tokens: {len(val_data):,}")
    print(f"  Vocabulary size: {vocab_size}")

    # Create model config
    if args.model_size == 'tiny':
        config = BDHConfig.tiny()
    elif args.model_size == 'small':
        config = BDHConfig.small()
    else:
        config = BDHConfig.medium()

    config.vocab_size = vocab_size
    config.block_size = args.block_size

    # Create model
    model = BDH(config).to(device)

    # Try to compile model (PyTorch 2.0+)
    try:
        print("Compiling model...")
        model = torch.compile(model)
        print("Model compiled successfully")
    except:
        print("Model compilation not available, continuing without it")

    # Print model info
    num_params = model.get_num_params(non_embedding=False)
    print(f"\nModel: BDH-{args.model_size}")
    print(f"  Parameters: {num_params:,}")
    print(f"  Layers: {config.n_layer}")
    print(f"  Embedding dim: {config.n_embd}")
    print(f"  Heads: {config.n_head}")
    print(f"  Internal dim: {config.n_internal}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.1
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)

    # Training monitor
    monitor = TrainingMonitor(args.max_iters, args.log_interval)

    # Training loop
    print(f"\n{'='*70}")
    print("Starting training...")
    print(f"{'='*70}\n")

    model.train()
    best_val_loss = float('inf')

    for iter_num in range(args.max_iters):
        # Evaluate periodically
        if iter_num % args.eval_interval == 0 and iter_num > 0:
            print("\nEvaluating...")
            train_loss = estimate_loss(
                model, train_data, config.block_size, 
                args.batch_size, eval_iters=200, device=device
            )
            val_loss = estimate_loss(
                model, val_data, config.block_size,
                args.batch_size, eval_iters=200, device=device
            )
            
            print(f"Step {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(args.output_dir, 'checkpoints', 'best_model.pt')
                torch.save({
                    'iter': iter_num,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'config': config,
                }, checkpoint_path)
                print(f"Saved best model (val_loss={val_loss:.4f})")
            
            model.train()
        
        # Get batch
        x, y = get_batch(train_data, config.block_size, args.batch_size, device)
        
        # Forward pass
        with ctx:
            logits, loss = model(x, y)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        # Log
        tokens_processed = args.batch_size * config.block_size
        monitor.log_iter(iter_num, loss.item(), tokens_processed)

    # Final evaluation
    print(f"\n{'='*70}")
    print("Final evaluation...")
    print(f"{'='*70}\n")

    train_loss = estimate_loss(
        model, train_data, config.block_size,
        args.batch_size, eval_iters=200, device=device
    )
    val_loss = estimate_loss(
        model, val_data, config.block_size,
        args.batch_size, eval_iters=200, device=device
    )

    print(f"Final train loss: {train_loss:.4f}")
    print(f"Final val loss: {val_loss:.4f}")

    # Generate sample
    print("\nGenerating sample...")
    model.eval()

    # Load meta for decoding
    meta_path = os.path.join('data', 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    itos = meta['itos']

    # Generate
    prompt = "ROMEO:"
    stoi = meta['stoi']
    start_ids = [stoi[c] for c in prompt]
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        y = model.generate(x, max_new_tokens=500, temperature=0.8, top_k=200)

    generated = ''.join([itos[i] for i in y[0].tolist()])
    print(generated)

    # Save final checkpoint
    final_checkpoint_path = os.path.join(args.output_dir, 'checkpoints', 'final_model.pt')
    torch.save({
        'iter': args.max_iters,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': config,
    }, final_checkpoint_path)
    print(f"\nSaved final checkpoint to {final_checkpoint_path}")

    # Save metrics report
    stats = monitor.finalize()

    report_path = os.path.join(args.output_dir, 'baseline_metrics.md')
    with open(report_path, 'w') as f:
        f.write("# BDH Baseline Metrics - Phase 0\n\n")
        f.write(f"## Model Configuration\n")
        f.write(f"- Size: {args.model_size}\n")
        f.write(f"- Parameters: {num_params:,}\n")
        f.write(f"- Layers: {config.n_layer}\n")
        f.write(f"- Embedding dim: {config.n_embd}\n")
        f.write(f"- Heads: {config.n_head}\n")
        f.write(f"- Internal dim: {config.n_internal}\n")
        f.write(f"- Block size: {config.block_size}\n\n")
        
        f.write(f"## Training Configuration\n")
        f.write(f"- Iterations: {args.max_iters}\n")
        f.write(f"- Batch size: {args.batch_size}\n")
        f.write(f"- Learning rate: {args.learning_rate}\n")
        f.write(f"- Device: {device}\n")
        f.write(f"- Dtype: {dtype}\n\n")
        
        f.write(f"## Results\n")
        f.write(f"- Final train loss: {train_loss:.4f}\n")
        f.write(f"- Final val loss: {val_loss:.4f}\n")
        f.write(f"- Best val loss: {best_val_loss:.4f}\n")
        f.write(f"- Training time: {stats['total_time_minutes']:.1f} minutes ({stats['total_time_hours']:.2f} hours)\n\n")
        
        f.write(f"## Sample Generation\n")
        f.write(f"Prompt: `{prompt}`\n\n")
        f.write("```\n")
        f.write(generated)
        f.write("\n```\n")

    print(f"\nMetrics report saved to {report_path}")
    print(f"\nTraining complete! Total time: {stats['total_time_minutes']:.1f} minutes")


if __name__ == '__main__':
    main()