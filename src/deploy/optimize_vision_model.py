from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torchvision import models

from src.vision.data import get_pneumoniamnist_loaders
from src.utils.common import ensure_dir, save_state_dict_bytes


def build_model(arch: str, num_classes: int = 2) -> nn.Module:
    if arch == "resnet18":
        m = models.resnet18(weights=None)
    elif arch == "resnet50":
        m = models.resnet50(weights=None)
    else:
        raise ValueError("arch must be resnet18 or resnet50")
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


@torch.no_grad()
def evaluate_vision(model: nn.Module, loader, device: torch.device) -> dict:
    model.eval()
    crit = nn.CrossEntropyLoss()

    losses = []
    y_true, y_pred, y_prob = [], [], []

    for x, y in loader:
        x = x.to(device)
        y = y.squeeze().long().to(device)

        logits = model(x)
        loss = crit(logits, y)
        losses.append(float(loss.item()))

        probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        pred = logits.argmax(dim=-1).detach().cpu().numpy()

        y_prob += probs.tolist()
        y_pred += pred.tolist()
        y_true += y.detach().cpu().numpy().tolist()

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    acc = float(accuracy_score(y_true_np, y_pred_np))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_np, y_pred_np, average="binary", zero_division=0
    )
    try:
        auc = float(roc_auc_score(y_true_np, np.array(y_prob)))
    except Exception:
        auc = float("nan")

    return {
        "loss": float(np.mean(losses)),
        "acc": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": auc,
    }


@torch.no_grad()
def avg_inference_time_ms(model: nn.Module, x: torch.Tensor, iters: int = 200, warmup: int = 20) -> float:
    model.eval()

    # warmup
    for _ in range(warmup):
        _ = model(x)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(x)
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / iters


def apply_pruning(model: nn.Module, amount: float = 0.3) -> nn.Module:
    """
    Unstructured L1 pruning on Conv2d and Linear layers.
    """
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="Path to a vision run folder (contains model.pt)")
    ap.add_argument("--arch", choices=["resnet18", "resnet50"], default="resnet18")
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--out_dir", type=str, default="runs/deploy")
    ap.add_argument("--prune_amount", type=float, default=0.3)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    device = torch.device("cpu")  # deployment benchmark on CPU

    # loaders (same dataset as Part 2)
    train_loader, val_loader, test_loader, info = get_pneumoniamnist_loaders(
        batch_size=args.batch_size, img_size=args.img_size
    )

    # load baseline model
    model = build_model(args.arch, num_classes=2).to(device)
    state = torch.load(os.path.join(args.run_dir, "model.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # take one batch for timing
    xb, yb = next(iter(test_loader))
    xb = xb.to(device)

    base_metrics = evaluate_vision(model, test_loader, device)
    base_time = avg_inference_time_ms(model, xb, iters=args.iters)
    base_size = save_state_dict_bytes(model.state_dict()) / (1024 * 1024)

    # pruning
    pruned = apply_pruning(model, amount=args.prune_amount).to(device)
    pruned.eval()
    pr_metrics = evaluate_vision(pruned, test_loader, device)
    pr_time = avg_inference_time_ms(pruned, xb, iters=args.iters)
    pr_size = save_state_dict_bytes(pruned.state_dict()) / (1024 * 1024)

    # dynamic quantization (Linear layers only) on CPU
    quant = torch.quantization.quantize_dynamic(pruned, {nn.Linear}, dtype=torch.qint8)
    quant.eval()
    q_metrics = evaluate_vision(quant, test_loader, device)
    q_time = avg_inference_time_ms(quant, xb, iters=args.iters)
    q_size = save_state_dict_bytes(quant.state_dict()) / (1024 * 1024)

    rows = [
        {
            "variant": "baseline",
            "size_mb": base_size,
            "time_ms": base_time,
            **{f"test_{k}": v for k, v in base_metrics.items()},
        },
        {
            "variant": f"pruned_{args.prune_amount}",
            "size_mb": pr_size,
            "time_ms": pr_time,
            **{f"test_{k}": v for k, v in pr_metrics.items()},
        },
        {
            "variant": f"pruned_{args.prune_amount}+quant_int8",
            "size_mb": q_size,
            "time_ms": q_time,
            **{f"test_{k}": v for k, v in q_metrics.items()},
        },
    ]
    df = pd.DataFrame(rows)

    out_run = os.path.join(args.out_dir, "vision_opt")
    ensure_dir(out_run)
    df.to_csv(os.path.join(out_run, "benchmark.csv"), index=False)

    # save optimized models
    torch.save(pruned.state_dict(), os.path.join(out_run, "pruned.pt"))
    torch.save(quant.state_dict(), os.path.join(out_run, "quantized.pt"))

    print(df.to_string(index=False))
    print(f"Saved benchmark to {out_run}")


if __name__ == "__main__":
    main()
