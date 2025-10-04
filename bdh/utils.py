"""
Utility functions for BDH training and evaluation.
"""
import torch
import time
from typing import Optional

def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the best available device.
    Args:
        prefer_cuda: Whether to prefer CUDA over other devices
        
    Returns:
        torch.device object
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
    
def estimate_loss(
    model: torch.nn.Module,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_iters: int = 200,
    device: Optional[torch.device] = None
    ) -> float:
    """Estimate average loss on a dataset.
    Args:
        model: Model to evaluate
        data: Full dataset as 1D tensor
        block_size: Sequence length
        batch_size: Batch size
        eval_iters: Number of batches to average over
        device: Device to use (defaults to model's device)
        
    Returns:
        Average loss
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    losses = []

    with torch.no_grad():
        for _ in range(eval_iters):
            # Get random batch
            ix = torch.randint(len(data) - block_size, (batch_size,))
            x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
            y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
            
            # Forward pass
            _, loss = model(x, y)
            losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)

def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Get a random batch from data.
    Args:
        data: Full dataset as 1D tensor
        block_size: Sequence length
        batch_size: Batch size
        device: Device to put batch on
        
    Returns:
        Tuple of (inputs, targets)
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    if device.type == 'cuda':
        # Pin arrays for async GPU transfer
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    return x, y
class TrainingMonitor:
    """Monitor training progress and statistics.
    Tracks:
    - Loss
    - Tokens per second
    - Estimated time remaining
    - GPU memory usage (if available)
    """

    def __init__(self, total_iters: int, log_interval: int = 100):
        """Initialize monitor.
        
        Args:
            total_iters: Total number of training iterations
            log_interval: How often to log statistics
        """
        self.total_iters = total_iters
        self.log_interval = log_interval
        self.start_time = time.time()
        self.iter_start_time = time.time()
        self.loss_sum = 0.0
        self.loss_count = 0

    def log_iter(
        self,
        iter_num: int,
        loss: float,
        tokens_processed: int
    ) -> None:
        """Log iteration statistics.
        
        Args:
            iter_num: Current iteration number
            loss: Current loss value
            tokens_processed: Number of tokens processed this iteration
        """
        self.loss_sum += loss
        self.loss_count += 1
        
        if iter_num % self.log_interval == 0 and iter_num > 0:
            # Compute average loss
            avg_loss = self.loss_sum / self.loss_count
            
            # Compute tokens per second
            elapsed = time.time() - self.iter_start_time
            tokens_per_sec = (tokens_processed * self.log_interval) / elapsed
            
            # Estimate time remaining
            iters_remaining = self.total_iters - iter_num
            time_remaining = (elapsed / self.log_interval) * iters_remaining
            
            # GPU memory usage
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.max_memory_allocated() / 1024**3
                mem_reserved = torch.cuda.max_memory_reserved() / 1024**3
                mem_info = f"| GPU mem: {mem_allocated:.2f}/{mem_reserved:.2f} GB"
            else:
                mem_info = ""
            
            # Log
            print(
                f"iter {iter_num:6d}/{self.total_iters} | "
                f"loss {avg_loss:.4f} | "
                f"tok/sec {tokens_per_sec:8.0f} | "
                f"time remaining: {time_remaining/60:.1f}m {mem_info}"
            )
            
            # Reset counters
            self.loss_sum = 0.0
            self.loss_count = 0
            self.iter_start_time = time.time()
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

    def finalize(self) -> dict:
        """Get final statistics.
        
        Returns:
            Dictionary of final statistics
        """
        total_time = time.time() - self.start_time
        
        return {
            'total_time_minutes': total_time / 60,
            'total_time_hours': total_time / 3600,
        }