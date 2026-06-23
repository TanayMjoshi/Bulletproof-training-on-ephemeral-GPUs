# LinkedIn feed post (~150 words)

Free GPUs are amazing — until the session disconnects at epoch 90 of 100 and every epoch since you last looked is gone. 😅

For a while I treated that as the cost of free compute. It isn't. The fix isn't a keep-alive browser hack — it's making training *resumable* and *idempotent* so a disconnect is a non-event.

The pattern, in five moves:

→ Checkpoint the *full* state every epoch — model + optimizer + scheduler + epoch + RNG, not just the weights
→ Write it atomically (temp file → rename) so a mid-write crash can't corrupt it
→ Use a "done marker" so re-launching a sweep skips finished work and resumes the rest
→ Store it somewhere that outlives the machine — a cloud bucket *or* a local SSD
→ On startup, resume from the saved epoch, not from zero

A dropped session becomes a shrug, not a disaster.

Full write-up with code 👉 [LINK]

#MachineLearning #MLOps #PyTorch #DeepLearning
