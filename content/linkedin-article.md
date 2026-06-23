# I stopped losing training runs to GPU disconnects. Here's the exact pattern.

*By Tanay Joshi*

Free GPUs are one of the best deals in machine learning — right up until the session disconnects at epoch 90 of 100 and every epoch since you last looked is gone. You start again from zero.

I used to lose hours this way. Now a disconnect costs me seconds. The difference isn't a fancier GPU or a keep-alive browser hack — it's making training **resumable** and the pipeline **idempotent**, so an interruption is a non-event. Here's the whole pattern, generic enough to drop into any training loop.

---

**1. Checkpoint the *whole* state, not just the weights.**

Saving `model.state_dict()` alone isn't enough to resume. You also need the optimizer state (Adam's momentum and variance), the LR scheduler position, the epoch counter, your best-so-far / early-stopping counters, and the RNG state. Miss these and a "resume" silently restarts your learning rate, re-runs finished epochs, and breaks reproducibility. Save all of it, every epoch. The cost is milliseconds.

**2. Write atomically, or risk a corrupted checkpoint.**

If the machine dies *while you're writing the checkpoint*, you get a half-written file — and you've now destroyed the good checkpoint you were overwriting. Fix: write to a temp file, then `os.replace()` it into place. A rename on the same filesystem is atomic — you always have either the complete old file or the complete new one, never a torn mix.

**3. Use a "done marker" to make a whole sweep idempotent.**

Write the final results file *only* when a run fully completes. Then your launcher can re-run the entire batch after any disconnect: finished runs are skipped, the interrupted one resumes, nothing is duplicated. It's the same idea `make` uses — declare the output, do the work only if it's missing.

**4. Put the state where it outlives the machine.**

The #1 mistake is checkpointing to the node's local scratch disk, which is wiped the instant the runtime recycles. Your state has to live somewhere external: a cloud bucket or mounted drive, *or* a local SSD / NAS you control. A clean trick: keep code on the fast local disk, but symlink the output directory to durable storage. Local-disk speed for reads, durable state for writes.

**5. Resume means *continue*, not restart.**

On startup, check for a checkpoint and continue from the saved epoch — `range(start_epoch, num_epochs)`, not `range(0, ...)`. Good smoke test: start a run, kill it mid-training, restart, and confirm the learning rate picks up smoothly instead of jumping back to its initial value. If the LR is continuous, your state survived correctly.

---

Once this is in place, a dropped session stops being a disaster and becomes a shrug. You reconnect, re-run the cheap setup, and watch it print `[resume] continuing from epoch …`. No lost epochs. No duplicate runs. No keep-alive hacks.

Ephemeral compute gives you all the upside of free GPUs with almost none of the fragility — you just have to engineer for the disconnect instead of fearing it.

I wrote up the full version with code for each step (atomic save helper, the done-marker launcher, the resume loop, and the gotchas that cost me real runs). Link in the comments. 👇

*What's the worst training run you've ever lost? Curious whether I'm the only one who learned this the hard way.*

---

**#MachineLearning #MLEngineering #DeepLearning #PyTorch #MLOps**
