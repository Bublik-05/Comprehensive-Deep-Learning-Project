from __future__ import annotations
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
from torch.utils.data import DataLoader


from src.vision.data import get_pneumoniamnist_loaders
from src.utils.common import seed_everything, get_device, ensure_dir, now_id, count_params, save_state_dict_bytes, EarlyStopping
from src.utils.plotting import plot_curves

def build_model(arch: str, num_classes: int = 2) -> nn.Module:
    if arch == "resnet18":
        m = models.resnet18(weights=None)
    elif arch == "resnet50":
        m = models.resnet50(weights=None)
    else:
        raise ValueError("arch must be resnet18 or resnet50")
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def run_epoch(model, loader, device, opt=None):
    train = opt is not None
    model.train(train)
    crit = nn.CrossEntropyLoss()
    losses = []
    y_true, y_prob, y_pred = [], [], []

    for x, y in loader:
        x = x.to(device)
        # y is shape (B, 1) in MedMNIST
        y = y.squeeze().long().to(device)
        if train:
            opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = crit(logits, y)
        if train:
            loss.backward()
            opt.step()
        losses.append(float(loss.item()))
        probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        pred = logits.argmax(dim=-1).detach().cpu().numpy()
        y_prob += probs.tolist()
        y_pred += pred.tolist()
        y_true += y.detach().cpu().numpy().tolist()

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    acc = float(accuracy_score(y_true_np, y_pred_np))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true_np, y_pred_np, average="binary", zero_division=0)
    try:
        auc = float(roc_auc_score(y_true_np, np.array(y_prob)))
    except Exception:
        auc = float("nan")
    return float(np.mean(losses)), acc, float(prec), float(rec), float(f1), auc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["resnet18","resnet50"], required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--runs_dir", type=str, default="runs/vision")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()

    train_loader, val_loader, test_loader, info = get_pneumoniamnist_loaders(
        batch_size=args.batch_size, img_size=args.img_size
    )

    model = build_model(args.arch, num_classes=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stopper = EarlyStopping(patience=3)

    run_id = f"{args.arch}_{now_id()}"
    out_dir = os.path.join(args.runs_dir, run_id)
    ensure_dir(out_dir)

    params = count_params(model)
    t0 = time.perf_counter()
    best_state = None
    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_p, tr_r, tr_f1, tr_auc = run_epoch(model, train_loader, device, opt=opt)
        va_loss, va_acc, va_p, va_r, va_f1, va_auc = run_epoch(model, val_loader, device, opt=None)
        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "train_acc": tr_acc,
            "val_acc": va_acc
        })
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if stopper.step(va_loss):
            break

    train_time = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    te_loss, te_acc, te_p, te_r, te_f1, te_auc = run_epoch(model, test_loader, device, opt=None)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(out_dir, "history.csv"), index=False)
    plot_curves(hist_df, os.path.join(out_dir, "curves.png"), title=args.arch)

    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    size_mb = save_state_dict_bytes(model.state_dict()) / (1024*1024)

    metrics = {
        "arch": args.arch,
        "params": params,
        "train_time_sec": train_time,
        "test_loss": te_loss,
        "test_acc": te_acc,
        "precision": te_p,
        "recall": te_r,
        "f1": te_f1,
        "auc": te_auc,
        "model_size_mb": size_mb,
    }
    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)

    print(f"Saved to {out_dir}")
    print(pd.DataFrame([metrics]).to_string(index=False))

if __name__ == "__main__":
    main()
