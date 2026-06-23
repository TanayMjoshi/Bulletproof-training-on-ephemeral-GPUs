"""
resumable_trainer.py
====================

A tiny, framework-agnostic toolkit for making training runs *resumable* and
*idempotent* on ephemeral compute (Colab, Kaggle, spot instances) — so a
disconnect costs seconds, not hours.

It implements the five ideas from the article:

  1. Checkpoint the WHOLE state, every epoch (not just the weights).
  2. Write checkpoints atomically (temp file -> os.replace).
  3. A "done marker" makes a whole sweep idempotent.
  4. Store state where it outlives the machine (you choose the directory).
  5. Resume == continue, not restart.

There is no hard dependency on a deep-learning framework: state is serialized
with ``pickle`` by default, so the whole thing runs with the standard library
(+ NumPy if you use it in your own code). If you train with PyTorch, pass
``save_fn=torch.save`` / ``load_fn=torch.load`` — see ``examples/`` for both.

Author: Tanay Joshi · MIT License
"""

from __future__ import annotations

import json
import os
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:  # NumPy is optional — we only touch it if it's importable.
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None


# --------------------------------------------------------------------------- #
# 2. Atomic writes
# --------------------------------------------------------------------------- #
def save_atomic(path: os.PathLike | str, obj: Any,
                save_fn: Callable[[Any, Any], None] | None = None) -> None:
    """Serialize ``obj`` to ``path`` atomically.

    Writes to ``<path>.tmp`` first, then ``os.replace`` renames it into place.
    A rename on the same filesystem is atomic, so a crash mid-write can never
    leave a half-written file *or* destroy the previous good checkpoint.

    ``save_fn(obj, file_handle)`` lets you swap in e.g. ``torch.save``.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as fh:
        if save_fn is None:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            save_fn(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())  # make sure bytes hit the disk before the rename
    os.replace(tmp, path)      # atomic on the same filesystem


def load_blob(path: os.PathLike | str,
              load_fn: Callable[[Any], Any] | None = None) -> Any:
    """Inverse of :func:`save_atomic`."""
    with open(path, "rb") as fh:
        return pickle.load(fh) if load_fn is None else load_fn(fh)


# --------------------------------------------------------------------------- #
# 1. (part of) Whole-state checkpointing — capture/restore RNG
# --------------------------------------------------------------------------- #
def capture_rng() -> Dict[str, Any]:
    """Snapshot every RNG we know about so a resume is bit-for-bit reproducible."""
    rng: Dict[str, Any] = {"python": random.getstate()}
    if _np is not None:
        rng["numpy"] = _np.random.get_state()
    try:  # only if torch is installed
        import torch
        rng["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            rng["cuda"] = torch.cuda.get_rng_state_all()
    except Exception:
        pass
    return rng


def restore_rng(rng: Optional[Dict[str, Any]]) -> None:
    """Restore RNG state captured by :func:`capture_rng`."""
    if not rng:
        return
    random.setstate(rng["python"])
    if _np is not None and "numpy" in rng:
        _np.random.set_state(rng["numpy"])
    if "torch" in rng:
        try:
            import torch
            torch.set_rng_state(rng["torch"])
            if "cuda" in rng and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# 1. + 5. Whole-state checkpointing and resume-not-restart
# --------------------------------------------------------------------------- #
@dataclass
class CheckpointedTrainer:
    """Drives a training loop that survives the machine vanishing under it.

    You own ``state`` (a plain, picklable ``dict`` holding your weights,
    optimizer moments, scheduler position, best metric, early-stop counter —
    whatever you need). The trainer owns the boring-but-critical parts:
    capturing the epoch counter and RNG, saving the whole thing atomically
    every ``save_every`` epochs, resuming from the last checkpoint on startup,
    and writing a done marker when the run completes.

    Parameters
    ----------
    work_dir:
        Directory for ``checkpoint.pt`` and ``results.json``. Point this at
        DURABLE storage (a mounted drive, a bucket sync dir, an external SSD) —
        not the node's local scratch disk, which is wiped on recycle.
    max_epochs:
        Total epochs for a complete run.
    save_every:
        Checkpoint cadence in epochs (default 1 — the cost is milliseconds).
    save_fn / load_fn:
        Optional serializers, e.g. ``torch.save`` / ``torch.load``.
    verbose:
        Print ``[resume]`` / ``[save]`` / ``[done]`` lines.
    """

    work_dir: os.PathLike | str
    max_epochs: int
    save_every: int = 1
    save_fn: Callable[[Any, Any], None] | None = None
    load_fn: Callable[[Any], Any] | None = None
    verbose: bool = True
    _ckpt: Path = field(init=False)
    _done: Path = field(init=False)

    def __post_init__(self) -> None:
        self.work_dir = Path(self.work_dir)
        self._ckpt = self.work_dir / "checkpoint.pt"
        self._done = self.work_dir / "results.json"

    # -- public API -------------------------------------------------------- #
    def is_complete(self) -> bool:
        """True if the done marker exists (run already finished)."""
        return self._done.exists()

    def fit(self,
            state: Dict[str, Any],
            train_one_epoch: Callable[[int, Dict[str, Any]], Dict[str, Any]]
            ) -> Dict[str, Any]:
        """Run (or resume) training to completion and return the results dict.

        ``train_one_epoch(epoch, state)`` does exactly one epoch of work,
        mutating ``state`` in place, and returns a small metrics ``dict``.
        """
        # 3. done marker: if this run already finished, skip instantly.
        if self.is_complete():
            if self.verbose:
                print(f"[skip] already complete: {self.work_dir}")
            return load_json(self._done)

        # 5. resume: continue from the saved epoch, not from zero.
        start_epoch = 0
        last_metrics: Dict[str, Any] = {}
        if self._ckpt.exists():
            ck = load_blob(self._ckpt, self.load_fn)
            state.clear()
            state.update(ck["state"])
            start_epoch = ck["epochs_done"]
            last_metrics = ck.get("metrics", {})
            restore_rng(ck.get("rng"))
            if self.verbose:
                print(f"[resume] continuing from epoch {start_epoch + 1}")
        elif self.verbose:
            print("[start] no checkpoint found — fresh run")

        for epoch in range(start_epoch, self.max_epochs):
            last_metrics = train_one_epoch(epoch, state) or {}
            if (epoch + 1) % self.save_every == 0 or epoch + 1 == self.max_epochs:
                self._save(epoch, state, last_metrics)

        # 3. write the done marker LAST, then drop the now-useless checkpoint.
        results = {"epochs": self.max_epochs, "metrics": last_metrics}
        save_atomic(self._done, _as_jsonable(results),
                    save_fn=_json_save)
        try:  # dropping the now-useless checkpoint is an optimization, not critical
            self._ckpt.unlink(missing_ok=True)
        except OSError:
            pass
        if self.verbose:
            print(f"[done] wrote {self._done.name}")
        return results

    # -- internals --------------------------------------------------------- #
    def _save(self, epoch: int, state: Dict[str, Any],
              metrics: Dict[str, Any]) -> None:
        payload = {
            "epochs_done": epoch + 1,   # next run starts here
            "state": state,
            "metrics": metrics,
            "rng": capture_rng(),
        }
        save_atomic(self._ckpt, payload, self.save_fn)
        if self.verbose:
            print(f"[save] epoch {epoch + 1}/{self.max_epochs} "
                  f"checkpointed ({metrics})")


# --------------------------------------------------------------------------- #
# 3. Idempotent sweeps via the done marker
# --------------------------------------------------------------------------- #
def run_sweep(runs: Dict[str, Dict[str, Any]],
              run_fn: Callable[[Path, Dict[str, Any]], Dict[str, Any]],
              root: os.PathLike | str) -> Dict[str, Dict[str, Any]]:
    """Run many configs idempotently.

    ``runs`` maps a unique run name -> its config. ``run_fn(run_dir, cfg)``
    trains one config to completion (typically by building a
    :class:`CheckpointedTrainer` rooted at ``run_dir``). Re-invoking
    ``run_sweep`` after any disconnect skips finished runs, resumes the
    interrupted one, and never duplicates work.
    """
    root = Path(root)
    out: Dict[str, Dict[str, Any]] = {}
    for name, cfg in runs.items():
        run_dir = root / name
        marker = run_dir / "results.json"
        if marker.exists():
            print(f"[skip] {name}")
            out[name] = load_json(marker)
            continue
        out[name] = run_fn(run_dir, cfg)
    return out


# --------------------------------------------------------------------------- #
# small JSON helpers
# --------------------------------------------------------------------------- #
def _json_save(obj: Any, fh) -> None:
    fh.write(json.dumps(obj, indent=2).encode("utf-8"))


def load_json(path: os.PathLike | str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _as_jsonable(obj: Any) -> Any:
    """Best-effort coercion so results.json is human-readable."""
    if _np is not None and isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _as_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_jsonable(v) for v in obj]
    if _np is not None and isinstance(obj, (_np.floating, _np.integer)):
        return obj.item()
    return obj


__all__ = [
    "save_atomic", "load_blob", "capture_rng", "restore_rng",
    "CheckpointedTrainer", "run_sweep", "load_json",
]
