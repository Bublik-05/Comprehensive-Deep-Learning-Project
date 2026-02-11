from __future__ import annotations
import argparse
import glob
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt

from src.vision.data import get_pneumoniamnist_loaders
from src.utils.common import get_device

def build_model(arch: str) -> nn.Module:
    if arch == "resnet18":
        m = models.resnet18(weights=None)
    elif arch == "resnet50":
        m = models.resnet50(weights=None)
    else:
        raise ValueError("arch must be resnet18 or resnet50")
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m

def find_latest_run(runs_dir: str, arch: str) -> str:
    candidates = sorted(glob.glob(os.path.join(runs_dir, f"{arch}_*")))
    if not candidates:
        raise RuntimeError(f"No runs found in {runs_dir} for {arch}. Train first.")
    return candidates[-1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["resnet18","resnet50"], required=True)
    ap.add_argument("--runs_dir", type=str, default="runs/vision")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    device = get_device()
    run_dir = find_latest_run(args.runs_dir, args.arch)
    ckpt = os.path.join(run_dir, "model.pt")

    _, _, test_loader, info = get_pneumoniamnist_loaders(batch_size=args.batch_size, img_size=args.img_size)
    test_set = test_loader.dataset

    model = build_model(args.arch).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # sample indices
    idxs = random.sample(range(len(test_set)), k=min(args.n, len(test_set)))

    fig, axes = plt.subplots(nrows=len(idxs), ncols=1, figsize=(6, 2.2*len(idxs)))
    if len(idxs) == 1:
        axes = [axes]

    with torch.no_grad():
        for ax, i in zip(axes, idxs):
            x, y = test_set[i]
            y = int(y.squeeze())
            x_in = x.unsqueeze(0).to(device)
            logits = model(x_in)
            prob = torch.softmax(logits, dim=-1)[0,1].item()
            pred = int(torch.argmax(logits, dim=-1)[0].item())

            img = x[:1].repeat(3,1,1).permute(1,2,0).numpy()  # (H,W,3)
            ax.imshow(img, cmap="gray")
            ax.axis("off")
            ax.set_title(f"true={y} pred={pred} prob(pneumonia)={prob:.3f}")

    out_path = os.path.join(run_dir, f"predictions_{args.n}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
