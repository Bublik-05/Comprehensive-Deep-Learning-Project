from __future__ import annotations
import time
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from src.utils.common import EarlyStopping, get_device

def _grad_norm(t: torch.Tensor) -> float:
    return float(torch.norm(t.detach(), p=2).cpu().item())

def evaluate(model: nn.Module, batches, device: torch.device) -> Tuple[float, float]:
    model.eval()
    losses = []
    y_true, y_pred = [], []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for b in batches:
            x = torch.tensor(b.x, device=device)
            m = torch.tensor(b.mask, device=device)
            y = torch.tensor(b.y, device=device)
            logits = model(x, m)
            loss = criterion(logits, y)
            losses.append(float(loss.item()))
            pred = logits.argmax(dim=-1).cpu().numpy().tolist()
            y_pred += pred
            y_true += y.cpu().numpy().tolist()
    return float(np.mean(losses)), float(accuracy_score(y_true, y_pred))

def train_one_run(model: nn.Module, train_batches, val_batches, *,
                  lr: float = 2e-3, weight_decay: float = 0.0, epochs: int = 10,
                  grad_norm_names: List[str] = None,
                  patience: int = 3, clip_grad: float = 1.0,
                  device: torch.device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    stopper = EarlyStopping(patience=patience, min_delta=0.0)

    history = []
    grad_rows = []

    t0 = time.perf_counter()
    best_state = None
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        y_true, y_pred = [], []

        # track average grad norm per selected parameter group
        accum = {name: [] for name in (grad_norm_names or [])}

        for b in train_batches:
            x = torch.tensor(b.x, device=device)
            m = torch.tensor(b.mask, device=device)
            y = torch.tensor(b.y, device=device)
            opt.zero_grad(set_to_none=True)
            logits = model(x, m)
            loss = criterion(logits, y)
            loss.backward()

            # gradient norms (for vanishing gradients analysis)
            if grad_norm_names:
                for name, p in model.named_parameters():
                    if (name in grad_norm_names) and (p.grad is not None):
                        accum[name].append(_grad_norm(p.grad))

            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()

            train_losses.append(float(loss.item()))
            pred = logits.argmax(dim=-1).detach().cpu().numpy().tolist()
            y_pred += pred
            y_true += y.detach().cpu().numpy().tolist()

        train_loss = float(np.mean(train_losses))
        train_acc = float(accuracy_score(y_true, y_pred))
        val_loss, val_acc = evaluate(model, val_batches, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc
        })

        if grad_norm_names:
            for name in grad_norm_names:
                vals = accum[name]
                grad_rows.append({
                    "step": epoch,
                    "name": name,
                    "grad_norm": float(np.mean(vals)) if len(vals) else 0.0
                })

        # early stopping
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if stopper.step(val_loss):
            break

    train_time = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    hist_df = pd.DataFrame(history)
    grad_df = pd.DataFrame(grad_rows) if grad_rows else pd.DataFrame(columns=["step","name","grad_norm"])
    return hist_df, grad_df, float(train_time), model
