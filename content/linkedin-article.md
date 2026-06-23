# I stopped losing training runs to GPU disconnects. Here's the exact pattern.

*By Tanay Joshi*

*Originally published on [Dev.to →](https://dev.to/tanay_joshi_04/never-lose-a-training-run-again-a-checkpoint-and-resume-playbook-for-ephemeral-gpus-2m1j)*

Free GPUs are one of the best deals in machine learning — right up until the session disconnects at epoch 90 of 100 and every epoch since you last looked is gone. You start again from zero.

I used to lose hours this way. Now a disconnect costs me seconds. The difference isn't a fancier GPU or a keep-alive browser hack — it's making training **resumable** and the pipeline **idempotent**, so an interruption is a non-event. Here's the whole pattern, generic enough to drop into any training loop.

---

**1. Checkpoint the *whole* state, not just the weights.**

Saving `model.state_dict()` alone isn't enough to resume. You also need the optimizer state (Adam's momentum and variance), the LR scheduler position, the epoch counter, your best-so-far / early-stopping counters, and the RNG state. Miss these and a "resume" silently restarts your learning rate, re-runs finished epochs, and breaks reproducibility. Save all of it, every epoch. The cost is milliseconds.

```python
def make_checkpoint(epoch, model, optimizer, scheduler, best, rng):
    return {
        "epochs_done": epoch + 1,
        "model":       model.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "scheduler":   scheduler.state_dict(),
        "best_metric": best,
        "rng": {
            "torch":  torch.get_rng_state(),
            "numpy":  np.random.get_state(),
            "python": random.getstate(),
        },
    }
```

**2. Write atomically, or risk a corrupted checkpoint.**

If the machine dies *while you're writing the checkpoint*, you get a half-written file — and you've now destroyed the good checkpoint you were overwriting. Fix: write to a temp file, then `os.replace()` it into place. A rename on the same filesystem is atomic — you always have either the complete old file or the complete new one, never a torn mix.

```python
def save_atomic(path, state):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)   # atomic on the same filesystem
```

**3. Use a "done marker" to make a whole sweep idempotent.**

Write the final results file *only* when a run fully completes. Then your launcher can re-run the entire batch after any disconnect: finished runs are skipped, the interrupted one resumes, nothing is duplicated. It's the same idea `make` uses — declare the output, do the work only if it's missing.

```python
def run_one(run_dir, cfg):
    if (run_dir / "results.json").exists():           # done marker
        return load(run_dir / "results.json")         # skip finished work
    # ... train (resuming from checkpoint if present) ...
    save_atomic(run_dir / "results.json", results)    # write marker LAST
```

**4. Put the state where it outlives the machine.**

The #1 mistake is checkpointing to the node's local scratch disk, which is wiped the instant the runtime recycles. Your state has to live somewhere external: a cloud bucket or mounted drive, *or* a local SSD / NAS you control. A clean trick: keep code on the fast local disk, but symlink the output directory to durable storage. Local-disk speed for reads, durable state for writes.

```bash
# code stays on fast local disk; outputs live on durable storage
ln -s /mnt/persistent/my_project/outputs  ./outputs
```

**5. Resume means *continue*, not restart.**

On startup, check for a checkpoint and continue from the saved epoch — `range(start_epoch, num_epochs)`, not `range(0, ...)`. Good smoke test: start a run, kill it mid-training, restart, and confirm the learning rate picks up smoothly instead of jumping back to its initial value. If the LR is continuous, your state survived correctly.

```python
start_epoch = 0
if checkpoint_path.exists():
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    scheduler.load_state_dict(ck["scheduler"])
    start_epoch = ck["epochs_done"]
    restore_rng(ck["rng"])

for epoch in range(start_epoch, num_epochs):   # note: start_epoch, not 0
    train_one_epoch(...)
    save_atomic(checkpoint_path, make_checkpoint(epoch, ...))
```

---

Once this is in place, a dropped session stops being a disaster and becomes a shrug. You reconnect, re-run the cheap setup, and watch it print `[resume] continuing from epoch …`. No lost epochs. No duplicate runs. No keep-alive hacks.

Ephemeral compute gives you all the upside of free GPUs with almost none of the fragility — you just have to engineer for the disconnect instead of fearing it.

The full version — with runnable code for every step (the atomic-save helper, the done-marker launcher, the resume loop, and the gotchas that cost me real runs) — is on Dev.to, and the complete toolkit is on GitHub. Links below.

*What's the worst training run you've ever lost? Curious whether I'm the only one who learned this the hard way.*

---

**Found this useful?**

I write about the unglamorous engineering that makes ML actually ship.

- 🔗 Interactive walkthrough — https://resumable-ml-training.vercel.app
- 💻 Runnable code (MIT) — https://github.com/TanayMjoshi/Bulletproof-training-on-ephemeral-GPUs
- 💼 LinkedIn — https://www.linkedin.com/in/tanay-joshi-2a3bba1ab/
- 🐦 X / Twitter — https://x.com/MysteryMan60934

If this saved you a training run, a ⭐ on the repo or a follow means a lot.

---

**#MachineLearning #MLEngineering #DeepLearning #PyTorch #MLOps**
