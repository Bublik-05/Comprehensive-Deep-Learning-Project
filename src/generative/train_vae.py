from __future__ import annotations
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

import medmnist
from medmnist import INFO

from src.utils.common import seed_everything, get_device, ensure_dir, now_id

class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        # input: (B,1,64,64)
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),  # 32x32
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(), # 16x16
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),# 8x8
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),# 4x4
        )
        self.fc_mu = nn.Linear(256*4*4, latent_dim)
        self.fc_logvar = nn.Linear(256*4*4, latent_dim)
        self.fc_z = nn.Linear(latent_dim, 256*4*4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(), # 8x8
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),  # 16x16
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),   # 32x32
            nn.ConvTranspose2d(32, 1, 4, 2, 1),               # 64x64
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_z(z).view(z.size(0), 256, 4, 4)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def vae_loss(recon, x, mu, logvar):
    # BCE recon
    recon_loss = F.binary_cross_entropy(recon, x, reduction="mean")
    # KL divergence
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl, recon_loss, kl

def get_loader(batch_size: int = 128, img_size: int = 64):
    info = INFO["pneumoniamnist"]
    DataClass = getattr(medmnist, info["python_class"])
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
    ])
    train_set = DataClass(split="train", transform=tfm, download=True)
    val_set = DataClass(split="val", transform=tfm, download=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader

def save_samples(model, device, out_path: str, n: int = 20, latent_dim: int = 32):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n, latent_dim, device=device)
        x = model.decode(z).cpu().numpy()  # (n,1,H,W)
    cols = 5
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols*2, rows*2))
    for i in range(n):
        plt.subplot(rows, cols, i+1)
        plt.imshow(x[i,0], cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_losses(df: pd.DataFrame, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.plot(df["epoch"], df["train_total"], label="train_total")
    plt.plot(df["epoch"], df["val_total"], label="val_total")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(out_path.replace(".png","_total.png"), dpi=200); plt.close()

    plt.figure()
    plt.plot(df["epoch"], df["train_recon"], label="train_recon")
    plt.plot(df["epoch"], df["val_recon"], label="val_recon")
    plt.xlabel("epoch"); plt.ylabel("recon"); plt.legend(); plt.tight_layout()
    plt.savefig(out_path.replace(".png","_recon.png"), dpi=200); plt.close()

    plt.figure()
    plt.plot(df["epoch"], df["train_kl"], label="train_kl")
    plt.plot(df["epoch"], df["val_kl"], label="val_kl")
    plt.xlabel("epoch"); plt.ylabel("kl"); plt.legend(); plt.tight_layout()
    plt.savefig(out_path.replace(".png","_kl.png"), dpi=200); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--latent_dim", type=int, default=32)
    ap.add_argument("--runs_dir", type=str, default="runs/generative")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device()
    train_loader, val_loader = get_loader(batch_size=args.batch_size)

    model = ConvVAE(latent_dim=args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_id = f"vae_{now_id()}"
    out_dir = os.path.join(args.runs_dir, run_id)
    ensure_dir(out_dir)

    rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_tot = tr_rec = tr_kl = 0.0
        n_tr = 0
        for x, _y in train_loader:
            x = x.to(device)
            opt.zero_grad(set_to_none=True)
            recon, mu, logvar = model(x)
            loss, rec, kl = vae_loss(recon, x, mu, logvar)
            loss.backward()
            opt.step()
            bs = x.size(0)
            tr_tot += float(loss.item()) * bs
            tr_rec += float(rec.item()) * bs
            tr_kl += float(kl.item()) * bs
            n_tr += bs

        model.eval()
        va_tot = va_rec = va_kl = 0.0
        n_va = 0
        with torch.no_grad():
            for x, _y in val_loader:
                x = x.to(device)
                recon, mu, logvar = model(x)
                loss, rec, kl = vae_loss(recon, x, mu, logvar)
                bs = x.size(0)
                va_tot += float(loss.item()) * bs
                va_rec += float(rec.item()) * bs
                va_kl += float(kl.item()) * bs
                n_va += bs

        rows.append({
            "epoch": epoch,
            "train_total": tr_tot / n_tr,
            "train_recon": tr_rec / n_tr,
            "train_kl": tr_kl / n_tr,
            "val_total": va_tot / n_va,
            "val_recon": va_rec / n_va,
            "val_kl": va_kl / n_va,
        })
        print(f"epoch {epoch}: train_total={rows[-1]['train_total']:.4f} val_total={rows[-1]['val_total']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "losses.csv"), index=False)
    plot_losses(df, os.path.join(out_dir, "loss_curves.png"))
    torch.save(model.state_dict(), os.path.join(out_dir, "vae.pt"))

    save_samples(model, device, os.path.join(out_dir, "generated_samples.png"),
                 n=20, latent_dim=args.latent_dim)

    print(f"Saved to {out_dir}")

if __name__ == "__main__":
    main()
