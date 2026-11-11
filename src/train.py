"""Train FNO1d on the PDEBench 1D Burgers (nu=0.01) initial->final operator map.

Usage:
    python -m src.train --epochs 500
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from src.data import BurgersFNODataset, preprocess_and_cache, REPO_ROOT
from src.model import FNO1d

TRAIN_CKPT_DIR = os.path.join(REPO_ROOT, "checkpoints")  # scratch, gitignored


def relative_l2_loss(pred: torch.Tensor, target: torch.Tensor, norm_floor: float = 0.1) -> torch.Tensor:
    """Mean over the batch of ||pred - target||_2 / ||target||_2, flattened per sample.

    A handful of Burgers trajectories settle to an almost-constant profile by
    t=T_final, so after mean-centering their target norm is ~1e-3 (near zero).
    Without a floor on the denominator, those samples produce huge loss spikes
    that destabilize training; `norm_floor` (in normalized units, where the
    field has unit std) caps how small the denominator can get.
    """
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    diff_norm = torch.norm(pred - target, dim=1)
    target_norm = torch.norm(target, dim=1).clamp_min(norm_floor)
    return torch.mean(diff_norm / target_norm)


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, n_batches = 0.0, 0
    for inp, target in loader:
        inp, target = inp.to(device), target.to(device)
        with torch.set_grad_enabled(training):
            pred = model(inp)
            loss = relative_l2_loss(pred, target)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--step-size", type=int, default=100, help="LR decay step (epochs)")
    parser.add_argument("--gamma", type=float, default=0.5, help="LR decay factor")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    payload = preprocess_and_cache()
    train_ds = BurgersFNODataset(payload, "train")
    val_ds = BurgersFNODataset(payload, "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = FNO1d(modes=args.modes, width=args.width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    os.makedirs(TRAIN_CKPT_DIR, exist_ok=True)
    history = []
    best_val = float("inf")
    best_path = os.path.join(TRAIN_CKPT_DIR, "fno_best.pt")

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device, optimizer=None)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            # Write to a temp file then atomically rename, so a crash/disk-full
            # mid-write can never corrupt the last good checkpoint.
            tmp_path = best_path + ".tmp"
            torch.save({
                "model_state": model.state_dict(),
                "modes": args.modes,
                "width": args.width,
                "mean": payload["mean"],
                "std": payload["std"],
                "resolution": payload["resolution"],
                "epoch": epoch,
                "val_loss": val_loss,
            }, tmp_path)
            os.replace(tmp_path, best_path)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"epoch {epoch:4d}/{args.epochs}  train_rel_l2 {train_loss:.4f}  "
                  f"val_rel_l2 {val_loss:.4f}  best_val {best_val:.4f}  elapsed {elapsed:.1f}s")

    with open(os.path.join(TRAIN_CKPT_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"done. best val relative L2 = {best_val:.4f}, checkpoint at {best_path}")


if __name__ == "__main__":
    main()
