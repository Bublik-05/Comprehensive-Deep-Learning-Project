from __future__ import annotations
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

def plot_curves(history: pd.DataFrame, out_path: str, title: str = "") -> None:
    """history columns expected: epoch, train_loss, val_loss, train_acc, val_acc"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="train_loss")
    plt.plot(history["epoch"], history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title + " loss" if title else "loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path.replace(".png", "_loss.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(history["epoch"], history["train_acc"], label="train_acc")
    plt.plot(history["epoch"], history["val_acc"], label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title(title + " accuracy" if title else "accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path.replace(".png", "_acc.png"), dpi=200)
    plt.close()

def plot_grad_norms(grad_df: pd.DataFrame, out_path: str, title: str = "Gradient norms") -> None:
    """grad_df columns: step (or epoch), name, grad_norm"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    for name, g in grad_df.groupby("name"):
        plt.plot(g["step"], g["grad_norm"], label=name)
    plt.xlabel("step")
    plt.ylabel("grad norm (L2)")
    plt.title(title)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
