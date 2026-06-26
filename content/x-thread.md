# X / Twitter thread (8 tweets + pinned reply)

Post to your main timeline, then pin it. Link lives in the last tweet only.
Tweet 1 is the hook that decides whether anyone reads on, so it stands alone.

---

**1/**
Lost a training run to a Colab disconnect at epoch 90 of 100?

Stop reaching for keep-alive hacks.

Make training resumable instead, so a dropped GPU costs seconds, not hours.

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

If the machine dies mid-write, you get a half-written file, and you just destroyed the good checkpoint you were overwriting.

Fix: write to a temp file, then os.replace() it.

A rename on the same filesystem is atomic. Old file or new file, never torn.

**4/**
Make a whole sweep idempotent with a "done marker."

Write the results file ONLY when a run fully completes.

Now you can re-launch the entire batch after any disconnect: finished runs skip, the interrupted one resumes, nothing duplicates.

Same idea `make` uses.

**5/**
Put state where it outlives the machine.

#1 mistake: checkpointing to the node's local scratch disk, wiped the instant the runtime recycles.

It has to live external: a cloud bucket / mounted drive, OR a local SSD or NAS you control.

**6/**
Nice trick: keep code on the fast local disk, symlink the output dir to durable storage.

`ln -s /mnt/persistent/outputs ./outputs`

Local-disk speed for reads, durable state for writes. Works for cloud and self-hosted setups alike.

**7/**
Resume means *continue*, not restart.

On startup, load the checkpoint and run `range(start_epoch, num_epochs)`, not from 0.

Smoke test: kill a run mid-training, restart, and check the LR picks up smoothly instead of jumping back to its starting value. Continuous LR = state survived.

**8/**
Once it's in place, a disconnect stops being a disaster and becomes a shrug.

I built an interactive version where you can break a training run yourself and watch it resume:
https://resumable-ml-training.vercel.app

Runnable code (clone it, kill it, watch it resume):
https://github.com/TanayMjoshi/Bulletproof-training-on-ephemeral-GPUs

**Pinned reply (post right after the thread):**
Full written version with all the code 👇
https://dev.to/tanay_joshi_04/never-lose-a-training-run-again-a-checkpoint-and-resume-playbook-for-ephemeral-gpus-2m1j

---

## Posting strategy

- **Post to your main timeline first**, then pin the thread to your profile for a week.
- **Links only in tweet 8 + the reply.** Mid-thread links cut reach.
- About 30 min after posting, **reply to your own thread** with the diagram image (extra surface, and it bumps the thread).
- Then **share the thread into 1–2 X Communities** (e.g. "Machine Learning", "Build in Public") for topical reach. Your timeline is the brand-builder; communities are the amplifier.
- **Hashtags:** keep to 1–2, only on the last tweet: #MachineLearning #PyTorch
- Reply to anyone who engages in the first hour. Early replies compound reach.

*(Reply-bait option to end tweet 8 instead of the links, if you want more comments: "What's the worst run you've ever lost to a disconnect?" — then put the links in the pinned reply.)*
