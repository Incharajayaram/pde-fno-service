"""Train FNO2d on PDEBench 2D Darcy Flow (beta=1.0): nu(x,y) -> pressure(x,y).

Usage:
    python -m src.train_darcy --epochs 500
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from src.data_darcy import DarcyFNODataset, preprocess_and_cache, REPO_ROOT
from src.model_fno2d import FNO2d
from src.train import relative_l2_loss  # same loss, dataset-agnostic

TRAIN_CKPT_DIR = os.path.join(REPO_ROOT, "checkpoints_darcy")  # scratch, gitignored


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
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--modes1", type=int, default=12)
    parser.add_argument("--modes2", type=int, default=12)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--step-size", type=int, default=100, help="LR decay step (epochs)")
    parser.add_argument("--gamma", type=float, default=0.5, help="LR decay factor")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    payload = preprocess_and_cache()
    train_ds = DarcyFNODataset(payload, "train")
    val_ds = DarcyFNODataset(payload, "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = FNO2d(modes1=args.modes1, modes2=args.modes2, width=args.width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    os.makedirs(TRAIN_CKPT_DIR, exist_ok=True)
    history = []
    best_val = float("inf")
    best_path = os.path.join(TRAIN_CKPT_DIR, "fno2d_darcy_best.pt")

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device, optimizer=None)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            tmp_path = best_path + ".tmp"
            torch.save({
                "model_state": model.state_dict(),
                "modes1": args.modes1,
                "modes2": args.modes2,
                "width": args.width,
                "nu_mean": payload["nu_mean"], "nu_std": payload["nu_std"],
                "p_mean": payload["p_mean"], "p_std": payload["p_std"],
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
