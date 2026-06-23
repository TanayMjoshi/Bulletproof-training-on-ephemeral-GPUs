"""
pytorch_reference.py — the pattern in real PyTorch
==================================================

This mirrors the article's code one-to-one for a PyTorch training loop:
full-state checkpoint (model + optimizer + scheduler + epoch + RNG), atomic
save, resume-from-epoch, and the `weights_only` gotcha.

It trains a tiny MLP on synthetic data, so it runs on CPU in a few seconds with
no dataset download.

    pip install torch
    python examples/pytorch_reference.py            # trains, checkpoints each epoch
    CRASH_AT=5 python examples/pytorch_reference.py  # "dies" at epoch 5
    python examples/pytorch_reference.py            # resumes and finishes

Requires: torch. For a dependency-free demo of the same ideas, see quickstart.py.
"""

import json
import os
import random
from pathlib import Path

import torch
import torch.nn as nn

WORK_DIR = Path(__file__).resolve().parents[1] / ".runs" / "mlp"
CKPT = WORK_DIR / "checkpoint.pt"
EPOCHS = 15
CRASH_AT = int(os.environ.get("CRASH_AT", "0"))


# --- 2. atomic write --------------------------------------------------------
def save_atomic(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)              # atomic on the same filesystem


# --- 1. checkpoint the WHOLE state ------------------------------------------
def make_checkpoint(epoch, model, optimizer, scheduler, best) -> dict:
    return {
        "epochs_done": epoch + 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_metric": best,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "python": random.getstate(),
        },
    }


def restore_rng(rng: dict) -> None:
    torch.set_rng_state(rng["torch"])
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    random.setstate(rng["python"])


def main() -> None:
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # synthetic regression problem
    X = torch.randn(512, 16, device=device)
    true_w = torch.randn(16, 1, device=device)
    y = X @ true_w + 0.1 * torch.randn(512, 1, device=device)

    model = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = nn.MSELoss()
    best_metric = float("inf")

    # --- 5. resume == continue, not restart --------------------------------
    start_epoch = 0
    if CKPT.exists():
        # weights_only=False because the checkpoint holds non-tensor RNG state.
        # Only ever do this for files YOU wrote and trust.
        ck = torch.load(CKPT, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        best_metric = ck["best_metric"]
        start_epoch = ck["epochs_done"]
        restore_rng(ck["rng"])
        print(f"[resume] continuing from epoch {start_epoch + 1}")
    else:
        print("[start] fresh run")

    for epoch in range(start_epoch, EPOCHS):   # note: start_epoch, not 0
        if CRASH_AT and epoch == CRASH_AT:
            print(f"\n   b o o m  — machine reclaimed at epoch {epoch}\n")
            os._exit(1)

        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        best_metric = min(best_metric, loss.item())
        lr = scheduler.get_last_lr()[0]
        print(f"[save] epoch {epoch + 1}/{EPOCHS}  loss={loss.item():.5f}  lr={lr:.5f}")
        save_atomic(CKPT, make_checkpoint(epoch, model, optimizer, scheduler, best_metric))

    # --- 3. done marker: write results LAST, then drop the checkpoint ------
    (WORK_DIR / "results.json").write_text(
        json.dumps({"epochs": EPOCHS, "best_metric": best_metric}, indent=2))
    CKPT.unlink(missing_ok=True)
    print(f"\n[done] best loss = {best_metric:.5f}")


if __name__ == "__main__":
    main()
