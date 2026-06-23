# Never lose a training run again: a checkpoint-and-resume playbook for ephemeral GPUs

*A practical pattern for training models on free or pre-emptible compute (Colab, Kaggle, spot instances) that disconnect without warning — so a dropped session costs you seconds, not hours.*

---

## The 2 a.m. disconnect

If you have ever trained a model on a free GPU, you know the feeling. You kick off a long run, check back later, and the session is gone. The notebook is disconnected, the runtime recycled, and every epoch since the last time you looked has evaporated. You start again from zero.

Free and pre-emptible compute is fantastic — but it is *ephemeral*. The machine can vanish at any moment: idle timeouts, usage caps, spot-instance reclamation. Fighting this with keep-alive hacks is treating the symptom. The real fix is to make your training **resumable** and your pipeline **idempotent**, so an interruption is a non-event.

Here is the pattern I now use for every long run. It rests on five ideas.

---

## 1. Checkpoint the *whole* state — not just the weights

The most common mistake is saving only `model.state_dict()`. That is not enough to resume training. If you reload only the weights and start a fresh optimizer, you lose:

- the **optimizer** state (Adam's moment estimates — momentum and variance),
- the **learning-rate scheduler** position (so the LR jumps back to its starting value),
- the **epoch counter** (so you re-run epochs you already did),
- the **best-so-far** tracking and early-stopping counter,
- the **RNG state** (so the run is no longer reproducible across the break).

A resumable checkpoint captures all of it:

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
            "cuda":   torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy":  np.random.get_state(),
            "python": random.getstate(),
        },
    }
```

Save this **every epoch**. The cost is milliseconds; the payoff is never losing more than one epoch.

---

## 2. Write atomically — or risk a corrupted checkpoint

Here is a subtle trap: if the machine dies *while you are writing the checkpoint*, you get a half-written, unreadable file — and now you have lost everything, including the good checkpoint you just overwrote.

The fix is the **write-temp-then-rename** trick. A rename on the same filesystem is atomic: the checkpoint file is either the complete old version or the complete new version, never a torn mix.

```python
import os

def save_atomic(path, state):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)   # atomic on the same filesystem
```

This one helper has saved me more than once.

---

## 3. A "done marker" makes the entire job idempotent

Resuming one run is good. Resuming a *sweep* of many runs is better. If you train across several configurations, seeds, or datasets, you want to re-launch the whole batch and have it automatically skip everything already finished and resume only the one that was interrupted.

The trick is a **done marker**: write the final results file (metrics, summary) only when a run fully completes. Then the launcher logic is trivial:

```python
def run_one(run_dir, cfg):
    if (run_dir / "results.json").exists():      # done marker
        print(f"[skip] already complete: {run_dir.name}")
        return load(run_dir / "results.json")
    # ... train (resuming from checkpoint if present) ...
    save_atomic(run_dir / "results.json", results) # write marker LAST
    (run_dir / "checkpoint.pt").unlink(missing_ok=True)  # done → drop checkpoint
```

Now your orchestration loop is fully restartable:

```python
for seed in SEEDS:
    for config in CONFIGS:
        run_one(run_dir_for(seed, config), config)
```

Re-run it after any disconnect. Finished work is skipped, interrupted work resumes, nothing is ever duplicated. This is the same principle that build tools like `make` use: declare the output, and only do the work if the output is missing.

---

## 4. Put the state where it outlives the machine

A checkpoint is only useful if it survives the thing that died. The number-one mistake on ephemeral compute is writing checkpoints to the node's **local scratch disk** — which is wiped the instant the runtime is recycled. Your checkpoints must live on storage that is *external* to the compute.

You have two families of options.

**Cloud / network storage** (best for ephemeral cloud GPUs):
- A mounted cloud drive (Google Drive, Dropbox, OneDrive, iCloud Drive).
- An object-store bucket (S3, GCS, R2, Azure Blob) you `sync` to after each epoch.
- A network filesystem (NFS/SMB) on a persistent volume.

**Local / self-hosted storage** (best when *you* own the machine, or for hybrid setups):
- An external SSD/HDD, or a second internal disk that is not part of the ephemeral root.
- A home server or NAS the training box can reach over the LAN.
- Your laptop: periodically `rsync`/`scp` the checkpoint directory back from a remote box, so a copy always exists on hardware you control.

A clean trick that works in both worlds: keep your **code on the fast local disk** but **symlink the checkpoint/output directory to persistent storage**. You get local-disk speed for reads and durable state for writes.

```bash
# code stays on fast local disk; outputs live on durable storage
ln -s /mnt/persistent/my_project/outputs  ./outputs
```

The principle is storage-agnostic: *the checkpoint must outlive the compute node.* Cloud bucket, mounted drive, NAS, or an external SSD on your desk — any of them works, as long as it is not the disk that gets wiped.

---

## 5. Resume means *continue*, not restart

With state saved durably, the training loop just checks for a checkpoint on startup and continues:

```python
start_epoch = 0
if checkpoint_path.exists():
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    scheduler.load_state_dict(ck["scheduler"])
    best_metric = ck["best_metric"]
    start_epoch = ck["epochs_done"]
    restore_rng(ck["rng"])
    print(f"[resume] continuing from epoch {start_epoch + 1}")

for epoch in range(start_epoch, num_epochs):   # note: start_epoch, not 0
    train_one_epoch(...)
    save_atomic(checkpoint_path, make_checkpoint(epoch, ...))
```

A good smoke test: start a run, kill it mid-training, restart it, and confirm the log shows `[resume] continuing from epoch N` — with the learning rate picking up smoothly where it left off, not jumping back to its initial value. If the LR is continuous, your optimizer and scheduler state survived correctly.

---

## Gotchas I hit (so you don't have to)

- **The `weights_only` pickle trap.** Recent PyTorch (2.6+) defaults `torch.load(..., weights_only=True)`, which *refuses* to load checkpoints containing non-tensor objects — like the NumPy/Python RNG state above. For your own trusted checkpoint files, pass `weights_only=False`. (Never do this for files from untrusted sources.)
- **Append your logs, don't truncate.** On resume, open the log file in append mode so you keep the full history across restarts instead of overwriting it.
- **Mind early-stopping state.** If you track "epochs since last improvement," checkpoint that counter too — otherwise a resume silently resets your patience.
- **Setup cells will re-run, and that's fine.** When a runtime is recycled, the interpreter's memory is gone — you *cannot* avoid re-importing libraries or re-mounting storage. Make those steps cheap and idempotent (mount only if not mounted; install only if missing; cache any preprocessed data to durable storage). The goal is not to skip setup; it is to make the *expensive* work resumable.

---

## The payoff

Once this is in place, the disconnect stops being a disaster and becomes a shrug. You reconnect, re-run the cheap setup, hit "go," and watch it print `[resume] continuing from epoch …`. No lost epochs. No duplicate runs. No keep-alive browser hacks.

A quick checklist to make any training job bulletproof:

- [ ] Checkpoint model **+ optimizer + scheduler + epoch + RNG**, every epoch
- [ ] Write checkpoints **atomically** (temp file → rename)
- [ ] Write a **done marker** only on full completion; skip finished runs
- [ ] Store state on something that **outlives the compute node** (cloud or local)
- [ ] On startup, **resume from the saved epoch**, not from zero
- [ ] Test it: interrupt, restart, confirm a clean resume

Ephemeral compute is one of the best deals in machine learning. With a resumable, idempotent pipeline, you get all of its upside and almost none of its fragility.

---

### Want the short version for a feed?

**LinkedIn:**
> Free GPUs are amazing — until they disconnect at epoch 90 of 100 and you lose everything. The fix isn't keep-alive hacks. It's making training *resumable* and *idempotent*: checkpoint the full state (model + optimizer + scheduler + epoch + RNG) every epoch, write it atomically to storage that outlives the machine (cloud bucket *or* a local SSD), and use a "done marker" so a relaunch skips finished work and resumes the rest. A dropped session becomes a shrug, not a disaster. Full write-up below. 👇

**X / thread starter:**
> Lost a training run to a Colab disconnect? Stop using keep-alive hacks.
> Make training resumable instead. 5 ideas that turn a dropped GPU into a non-event 🧵
