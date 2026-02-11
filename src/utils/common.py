from __future__ import annotations
import os, random, time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Iterable, Tuple

import numpy as np
import torch

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class EarlyStopping:
    patience: int = 3
    min_delta: float = 0.0
    best: float = float("inf")
    bad_epochs: int = 0

    def step(self, val_loss: float) -> bool:
        """Return True if should stop."""
        improved = val_loss < (self.best - self.min_delta)
        if improved:
            self.best = val_loss
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def now_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def save_state_dict_bytes(state_dict: Dict[str, Any]) -> int:
    """Serialize state_dict to bytes and return size in bytes (approx model size)."""
    import io
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return buf.getbuffer().nbytes

def avg_inference_time_ms(model: torch.nn.Module, batch: Any, n_warmup: int = 20, n_runs: int = 200) -> float:
    """CPU inference time in ms."""
    import time
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(*batch) if isinstance(batch, (tuple, list)) else model(batch)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model(*batch) if isinstance(batch, (tuple, list)) else model(batch)
        t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / n_runs
