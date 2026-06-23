"""
quickstart.py — run this, kill it, run it again. Watch it resume.
=================================================================

A dependency-light demo (just NumPy) of the resumable-training pattern. It
fits a tiny linear-regression model with plain gradient descent, checkpointing
the WHOLE state (weights, optimizer momentum, epoch, RNG) every epoch to a
durable directory you choose.

Try it:

    python examples/quickstart.py                 # trains to completion
    python examples/quickstart.py                 # instantly: already complete
    rm -rf .runs && \
        CRASH_AT=6 python examples/quickstart.py  # "dies" at epoch 6
    python examples/quickstart.py                 # resumes from epoch 6, finishes

Because the RNG is checkpointed too, the final weights are identical whether or
not the run was interrupted — that's reproducibility surviving the break.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

# make src/ importable without installing anything
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from resumable_trainer import CheckpointedTrainer  # noqa: E402

WORK_DIR = Path(__file__).resolve().parents[1] / ".runs" / "linreg"
EPOCHS = 15
LR = 0.03
MOMENTUM = 0.6
CRASH_AT = int(os.environ.get("CRASH_AT", "0"))  # 0 = never crash


def make_data(seed: int = 0, n: int = 256, d: int = 8):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    true_w = rng.standard_normal(d)
    y = X @ true_w + 0.1 * rng.standard_normal(n)
    return X, y, true_w


def build_initial_state(d: int) -> dict:
    # state is a plain, picklable dict that WE own
    return {
        "w": np.zeros(d),          # model weights
        "v": np.zeros(d),          # optimizer momentum (the bit people forget)
    }


def main() -> None:
    X, y, _ = make_data()
    d = X.shape[1]

    trainer = CheckpointedTrainer(work_dir=WORK_DIR, max_epochs=EPOCHS,
                                  save_every=1)

    def train_one_epoch(epoch: int, state: dict) -> dict:
        # Simulate a machine that vanishes mid-run (e.g. spot reclamation).
        if CRASH_AT and epoch == CRASH_AT:
            print(f"\n   b o o m  — machine reclaimed at epoch {epoch}\n")
            os._exit(1)

        w, v = state["w"], state["v"]
        grad = X.T @ (X @ w - y) / len(y)      # MSE gradient
        v[:] = MOMENTUM * v - LR * grad        # momentum update (in place)
        w[:] = w + v
        loss = float(np.mean((X @ w - y) ** 2))
        time.sleep(0.15)                       # make it long enough to Ctrl-C
        # weights ride along in the metrics so they land in results.json too
        return {"loss": round(loss, 6), "w_head": np.round(w[:3], 6).tolist()}

    state = build_initial_state(d)
    results = trainer.fit(state, train_one_epoch)

    m = results.get("metrics", {})
    print(f"\nfinal loss: {m.get('loss', 'n/a')}")
    print(f"weights[:3]: {m.get('w_head', 'n/a')}")


if __name__ == "__main__":
    main()
