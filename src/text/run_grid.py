from __future__ import annotations
import argparse
import os
import time
from typing import List, Dict

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.text.data import try_load_sms_spam, normalize_dataset, train_val_test_split, build_vocab, make_batches
from src.text.models import LSTMClassifier, TransformerClassifier
from src.text.train_text import train_one_run, evaluate
from src.utils.common import seed_everything, get_device, ensure_dir, now_id, count_params, save_state_dict_bytes
from src.utils.plotting import plot_curves, plot_grad_norms

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lstm","gru","transformer"], required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--runs_dir", type=str, default="runs/text")
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()

    ds, ds_name = try_load_sms_spam()
    tk, lk = normalize_dataset(ds)
    split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
    texts = [ex[tk] for ex in split]
    labels_raw = [ex[lk] for ex in split]

    # Make labels integer 0/1
    # Common cases: already 0/1, or strings "ham"/"spam"
    labels = []
    for y in labels_raw:
        if isinstance(y, (int, float)):
            labels.append(int(y))
        else:
            s = str(y).lower().strip()
            labels.append(1 if "spam" in s or s in {"1","true"} else 0)

    (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = train_val_test_split(texts, labels, seed=args.seed)
    vocab = build_vocab(tr_x, max_size=20000, min_freq=2)

    # Grid required by assignment
    dropouts = [0.1, 0.3, 0.5]
    weight_decays = [1e-5, 1e-4, 1e-3]

    run_id = f"{args.model}_{now_id()}_{ds_name.replace('/','_')}"
    out_dir = os.path.join(args.runs_dir, run_id)
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "dataset.txt"), "w", encoding="utf-8") as f:
        f.write(f"dataset={ds_name}\ntext_key={tk}\nlabel_key={lk}\n")

    # Save vocab to reproduce tokenization exactly
    import json
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as vf:
        json.dump(vocab, vf, ensure_ascii=False)

    results_rows = []

    for d in dropouts:
        for wd in weight_decays:
            exp_name = f"drop{d}_wd{wd}"
            exp_dir = os.path.join(out_dir, exp_name)
            ensure_dir(exp_dir)

            train_batches = list(make_batches(tr_x, tr_y, vocab, batch_size=args.batch_size, max_len=args.max_len, shuffle=True, seed=args.seed))
            val_batches = list(make_batches(va_x, va_y, vocab, batch_size=args.batch_size, max_len=args.max_len, shuffle=False, seed=args.seed))
            test_batches = list(make_batches(te_x, te_y, vocab, batch_size=args.batch_size, max_len=args.max_len, shuffle=False, seed=args.seed))

            if args.model in ["lstm","gru"]:
                model = LSTMClassifier(
                    vocab_size=len(vocab),
                    embed_dim=128,
                    hidden_dim=128,
                    num_layers=1,
                    dropout=d,
                    rnn_type="lstm" if args.model=="lstm" else "gru",
                    num_classes=2
                )
                grad_names = [
                    "embedding.weight",
                    "rnn.weight_ih_l0",
                    "rnn.weight_hh_l0",
                    "fc.weight"
                ]
            else:
                model = TransformerClassifier(
                    vocab_size=len(vocab),
                    d_model=64,
                    nhead=4,
                    num_layers=3,
                    dim_feedforward=128,
                    dropout=d,
                    num_classes=2,
                    max_len=args.max_len
                )
                grad_names = None  # required only for RNN/LSTM

            params = count_params(model)
            hist_df, grad_df, train_time, best_model = train_one_run(
                model,
                train_batches,
                val_batches,
                lr=args.lr,
                weight_decay=wd,
                epochs=args.epochs,
                grad_norm_names=grad_names,
                patience=3,
                clip_grad=1.0,
                device=device
            )

            # Evaluate
            tr_loss, tr_acc = evaluate(best_model, train_batches, device)
            va_loss, va_acc = evaluate(best_model, val_batches, device)
            te_loss, te_acc = evaluate(best_model, test_batches, device)
            overfit = tr_acc - va_acc

            # Save artifacts
            hist_df.to_csv(os.path.join(exp_dir, "history.csv"), index=False)
            plot_curves(hist_df, os.path.join(exp_dir, "curves.png"), title=f"{args.model} {exp_name}")

            if grad_df is not None and len(grad_df) and args.model in ["lstm","gru"]:
                grad_df.to_csv(os.path.join(exp_dir, "grad_norms.csv"), index=False)
                plot_grad_norms(grad_df, os.path.join(exp_dir, "grad_norms.png"),
                                title=f"Grad norms {args.model} {exp_name}")

            torch.save(best_model.state_dict(), os.path.join(exp_dir, "model.pt"))
            size_bytes = save_state_dict_bytes(best_model.state_dict())

            results_rows.append({
                "model": args.model,
                "dropout": d,
                "weight_decay": wd,
                "params": params,
                "train_time_sec": train_time,
                "train_acc": tr_acc,
                "val_acc": va_acc,
                "test_acc": te_acc,
                "overfit_severity": overfit,
                "model_size_mb": size_bytes / (1024*1024),
                "train_loss": tr_loss,
                "val_loss": va_loss,
                "test_loss": te_loss,
                "exp_dir": os.path.relpath(exp_dir, out_dir)
            })

    results = pd.DataFrame(results_rows).sort_values(["test_acc","val_acc"], ascending=False)
    results.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    # Write a short "best run" pointer
    best = results.iloc[0].to_dict()
    with open(os.path.join(out_dir, "best_run.txt"), "w", encoding="utf-8") as f:
        for k,v in best.items():
            f.write(f"{k}: {v}\n")

    print(f"Done. Results in: {out_dir}")
    print(results.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
