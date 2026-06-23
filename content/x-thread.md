# X / Twitter thread (8 tweets)

**1/**
Lost a training run to a Colab disconnect at epoch 90 of 100?

Stop reaching for keep-alive hacks.

Make training *resumable* instead — so a dropped GPU costs seconds, not hours.

The pattern, in 7 tweets 🧵

**2/**
Checkpoint the *whole* state, not just the weights.

`model.state_dict()` alone won't resume training. You also need:
• optimizer state (Adam's momentum + variance)
• LR scheduler position
• epoch counter
• best-so-far / early-stop counters
• RNG state

Save it every epoch. Costs ms.

**3/**
Write checkpoints atomically.

If the machine dies mid-write, you get a half-written file — and you just destroyed the good checkpoint you were overwriting.

Fix: write to a temp file, then os.replace() it.

A rename on the same filesystem is atomic. Old file or new file, never torn.

**4/**
Make a whole sweep idempotent with a "done marker."

Write the results file ONLY when a run fully completes.

Now you can re-launch the entire batch after any disconnect:
finished runs skip, the interrupted one resumes, nothing duplicates.

Same idea `make` uses.

**5/**
Put state where it outlives the machine.

#1 mistake: checkpointing to the node's local scratch disk — wiped the instant the runtime recycles.

It has to live external: a cloud bucket / mounted drive, OR a local SSD or NAS you control.

**6/**
Nice trick: keep code on the fast local disk, symlink the output dir to durable storage.

`ln -s /mnt/persistent/outputs ./outputs`

Local-disk speed for reads, durable state for writes. Works for cloud and self-hosted setups alike.

**7/**
Resume means *continue*, not restart.

On startup, load the checkpoint and run `range(start_epoch, num_epochs)` — not from 0.

Smoke test: kill a run mid-training, restart, and check the LR picks up smoothly instead of jumping back to its initial value. Continuous LR = state survived.

**8/**
Once it's in place, a disconnect stops being a disaster and becomes a shrug.

Reconnect, re-run the cheap setup, watch it print `[resume] continuing from epoch …`.

No lost epochs. No duplicate runs. No keep-alive hacks.

Full write-up w/ code 👉 [LINK]

---

*Reply-bait alt for tweet 8 ending:* What's the worst run you've ever lost to a disconnect?
