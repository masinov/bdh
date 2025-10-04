"""
Text generation script for trained BDH models.
Usage:
python scripts/generate.py --checkpoint results/checkpoints/best_model.pt --prompt "ROMEO:"
"""
import argparse
import os
import pickle
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bdh import BDH, get_device


def main():
    parser = argparse.ArgumentParser(description='Generate text with BDH')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default='ROMEO:',
                        help='Text prompt to start generation')
    parser.add_argument('--max_new_tokens', type=int, default=500,
                        help='Number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=200,
                        help='Top-k sampling parameter')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (optional)')
    args = parser.parse_args()

    # Set seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Setup device
    device = get_device()
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']

    # Create model
    model = BDH(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model with {model.get_num_params():,} parameters")
    print(f"Trained for {checkpoint['iter']} iterations")
    print(f"Val loss: {checkpoint['val_loss']:.4f}")

    # Load meta for encoding/decoding
    meta_path = os.path.join('data', 'meta.pkl')
    if not os.path.exists(meta_path):
        print(f"Error: {meta_path} not found. Please run training first.")
        return

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    stoi = meta['stoi']
    itos = meta['itos']

    # Encode prompt
    try:
        start_ids = [stoi[c] for c in args.prompt]
    except KeyError as e:
        print(f"Error: Character {e} not in vocabulary")
        return

    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    # Generate
    print(f"\nGenerating {args.max_new_tokens} tokens...")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}")
    print(f"\n{'-'*70}\n")

    with torch.no_grad():
        y = model.generate(
            x,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )

    # Decode and print
    generated = ''.join([itos[i] for i in y[0].tolist()])
    print(generated)
    print(f"\n{'-'*70}\n")


if __name__ == '__main__':
    main()