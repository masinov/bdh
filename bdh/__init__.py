
"""
BDH: Biologically-inspired Dragon Hatchling Language Model
A language model architecture that bridges transformers and brain-like computation.
"""
from .config import BDHConfig
from .model import BDH
from .attention import Attention
from .utils import get_device, estimate_loss, get_batch, TrainingMonitor
version = "0.1.0"
all = [
"BDH",
"BDHConfig",
"Attention",
"get_device",
"estimate_loss",
"get_batch",
"TrainingMonitor",
]
