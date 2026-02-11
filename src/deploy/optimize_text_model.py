from __future__ import annotations
import argparse
import glob
import os
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from src.text.data import try_load_sms_spam, normalize_dataset, train_val_test_split, make_batches
from src.text.models import TransformerClassifier, LSTMClassifier
from src.text.train_text import evaluate
from src.utils.common import get_device, ensure_dir, save_state_dict_bytes, avg_inference_time_ms

def find_best_experiment(runs_dir: str, model_prefix: str = "transformer") -> str:
    # pick newest run folder for that model, then best row in summary.csv
    candidates = sorted(glob.glob(os.path.join(runs_dir, f"{model_prefix}_*")))
    if not candidates:
        raise RuntimeError(f"No runs found under {runs_dir} for {model_prefix}. Run Part 1 training first.")
    run_dir = candidates[-1]
    summ = pd.read_csv(os.path.join(run_dir, "summary.csv"))
    best = summ.sort_values(["test_acc","val_acc"], ascending=False).iloc[0]
    exp_dir = os.path.join(run_dir, best["exp_dir"])
    return run_dir, exp_dir

def load_vocab(run_dir: str) -> dict:
    path = os.path.join(run_dir, "vocab.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def rebuild_dataset_batches(vocab: dict, max_len: int = 64, batch_size: int = 64, seed: int = 42):
    ds, _name = try_load_sms_spam()
    tk, lk = normalize_dataset(ds)
    split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
    texts = [ex[tk] for ex in split]
    labels_raw = [ex[lk] for ex in split]
    labels = []
    for y in labels_raw:
        if isinstance(y, (int, float)):
            labels.append(int(y))
        else:
            s = str(y).lower().strip()
            labels.append(1 if "spam" in s or s in {"1","true"} else 0)
    (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = train_val_test_split(texts, labels, seed=seed)
    test_batches = list(make_batches(te_x, te_y, vocab, batch_size=batch_size, max_len=max_len, shuffle=False, seed=seed))
    return test_batches

def apply_pruning(model: nn.Module, amount: float = 0.3) -> nn.Module:
    # Unstructured L1 pruning on all Linear layers
    for module in model.modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")  # make pruning permanent
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default="runs/text")
    ap.add_argument("--pick", type=str, default="best", help="best or path to experiment dir")
    ap.add_argument("--out_dir", type=str, default="runs/deploy")
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cpu")  # deployment benchmark on CPU
    ensure_dir(args.out_dir)

    if args.pick == "best":
        run_dir, exp_dir = find_best_experiment(args.runs_dir, model_prefix="transformer")
    else:
        exp_dir = args.pick
        run_dir = os.path.dirname(exp_dir)

    vocab = load_vocab(run_dir)
    test_batches = rebuild_dataset_batches(vocab, max_len=args.max_len, batch_size=args.batch_size, seed=args.seed)

    # Load baseline model
    model = TransformerClassifier(vocab_size=len(vocab), d_model=64, nhead=4, num_layers=3,
                                  dim_feedforward=128, dropout=0.1, num_classes=2, max_len=args.max_len)
    state = torch.load(os.path.join(exp_dir, "model.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    base_loss, base_acc = evaluate(model, test_batches, device)
    # Timing batch
    b0 = test_batches[0]
    xb = torch.tensor(b0.x, device=device)
    mb = torch.tensor(b0.mask, device=device)
    base_time = avg_inference_time_ms(model, (xb, mb))
    base_size = save_state_dict_bytes(model.state_dict()) / (1024*1024)

    # Prune
    pruned = apply_pruning(model, amount=0.3)
    pr_loss, pr_acc = evaluate(pruned, test_batches, device)
    pr_time = avg_inference_time_ms(pruned, (xb, mb))
    pr_size = save_state_dict_bytes(pruned.state_dict()) / (1024*1024)

    # Quantize dynamic (Linear layers only)
    quant = torch.quantization.quantize_dynamic(pruned, {nn.Linear}, dtype=torch.qint8)
    q_loss, q_acc = evaluate(quant, test_batches, device)
    q_time = avg_inference_time_ms(quant, (xb, mb))
    # quantized state_dict may be different; still serialize to estimate size
    q_size = save_state_dict_bytes(quant.state_dict()) / (1024*1024)

    rows = [
        {"variant": "baseline", "size_mb": base_size, "time_ms": base_time, "test_acc": base_acc, "test_loss": base_loss},
        {"variant": "pruned", "size_mb": pr_size, "time_ms": pr_time, "test_acc": pr_acc, "test_loss": pr_loss},
        {"variant": "pruned+quant_int8", "size_mb": q_size, "time_ms": q_time, "test_acc": q_acc, "test_loss": q_loss},
    ]
    df = pd.DataFrame(rows)
    out_run = os.path.join(args.out_dir, "text_opt")
    ensure_dir(out_run)
    df.to_csv(os.path.join(out_run, "benchmark.csv"), index=False)

    # Save models
    torch.save(pruned.state_dict(), os.path.join(out_run, "pruned.pt"))
    torch.save(quant.state_dict(), os.path.join(out_run, "quantized.pt"))

    print(df.to_string(index=False))
    print(f"Saved benchmark to {out_run}")

if __name__ == "__main__":
    main()
