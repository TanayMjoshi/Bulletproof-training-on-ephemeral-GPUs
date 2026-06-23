# Bulletproof training on ephemeral GPUs

> Make ML training runs **resumable** and **idempotent** on free or pre-emptible
> compute (Colab, Kaggle, spot instances) — so a disconnect costs **seconds, not hours**.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Dependencies](https://img.shields.io/badge/core-zero%20dependencies-brightgreen)
![License](https://img.shields.io/badge/license-MIT-black)

A small, runnable reference implementation of a pattern I use for every long
training run. The core toolkit has **no framework dependency** — it runs with
the standard library — and there's a PyTorch version that mirrors it one-to-one.

**▶ Interactive walkthrough:** https://resumable-ml-training.vercel.app

---

## Run it in 30 seconds

No GPU, no dataset, no heavy install. This trains a tiny model, and you can
kill it mid-run and watch it resume exactly where it left off:

```bash
git clone https://github.com/TanayMjoshi/Bulletproof-training-on-ephemeral-GPUs.git
cd Bulletproof-training-on-ephemeral-GPUs
pip install numpy

python examples/quickstart.py            # trains to completion, checkpointing each epoch
python examples/quickstart.py            # instant: "already complete" (idempotent)

rm -rf .runs
CRASH_AT=8 python examples/quickstart.py # simulates the machine dying at epoch 8
python examples/quickstart.py            # → [resume] continuing from epoch 9 … finishes
```

Because the RNG is checkpointed too, the interrupted run produces **byte-identical**
final weights to an uninterrupted one. A disconnect becomes a non-event.

Prefer real PyTorch? `pip install torch && python examples/pytorch_reference.py`
mirrors the article's code exactly (optimizer + scheduler + RNG, atomic save,
the `weights_only` gotcha).

## The five ideas

The full write-up is in [`content/article-canonical.md`](content/article-canonical.md);
the short version:

1. **Checkpoint the *whole* state, not just the weights.** A resume needs the
   optimizer state, LR-scheduler position, epoch counter, early-stop counter,
   and RNG — not only `model.state_dict()`.
2. **Write atomically.** Save to a temp file, then `os.replace()`. A rename on
   the same filesystem is atomic, so a crash mid-write can't corrupt the file or
   destroy the previous good checkpoint.
3. **A "done marker" makes a whole sweep idempotent.** Write the results file
   *only* on full completion; re-launching the batch skips finished runs and
   resumes the interrupted one — never duplicating work.
4. **Put state where it outlives the machine.** Not the node's local scratch
   disk (wiped on recycle) — a cloud bucket, mounted drive, NAS, or external SSD.
5. **Resume means *continue*, not restart.** On startup, load the checkpoint and
   loop from the saved epoch. Smoke test: kill it, restart, and confirm the
   learning rate picks up smoothly instead of snapping back to its initial value.

## What's in here

```
.
├── src/
│   └── resumable_trainer.py     # the toolkit: atomic save, full-state checkpoint,
│                                #   done-marker sweeps, resume — zero dependencies
├── examples/
│   ├── quickstart.py            # NumPy demo: run it, kill it, watch it resume
│   └── pytorch_reference.py     # the same pattern in real PyTorch (matches the article)
├── docs/
│   └── index.html               # the interactive walkthrough (deployed on Vercel)
├── assets/                      # diagram (PNG + SVG)
├── content/                     # the article + every platform version + posting strategy
└── PUBLISH.md                   # how to push this and deploy the walkthrough
```

## Using the toolkit in your own code

```python
from resumable_trainer import CheckpointedTrainer

trainer = CheckpointedTrainer(work_dir="/mnt/durable/run-001", max_epochs=100)

def train_one_epoch(epoch, state):
    # mutate `state` (a plain picklable dict you own) in place
    ...
    return {"loss": loss}   # small metrics dict

trainer.fit(state, train_one_epoch)   # resumes automatically if interrupted
```

Point `work_dir` at durable storage. Re-running after any disconnect resumes
from the last checkpoint; re-running after completion is a no-op.

## License

MIT © Tanay Joshi — see [LICENSE](LICENSE). Use it, fork it, ship it.
