"""
checkpointing.py
────────────────
Atomic checkpoint manager. Unchanged in design from V1; updated types.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class CheckpointConfig:
    output_dir: str | Path
    keep_last_k: int = 3
    save_every_n_chunks: int = 20
    best_metric: str = "val_loss"
    best_metric_mode: str = "min"


class CheckpointManager:
    LATEST = "latest.pt"
    BEST = "best.pt"
    META = "ckpt_meta.json"

    def __init__(self, config: CheckpointConfig):
        self.cfg = config
        self.out = Path(config.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self._best_value: Optional[float] = None
        self._saved: list[Path] = []
        self._meta = self._load_meta()
        logger.info("CheckpointManager | dir=%s | keep_last=%d", self.out, config.keep_last_k)

    def should_save(self, chunk_idx: int) -> bool:
        return (chunk_idx + 1) % self.cfg.save_every_n_chunks == 0

    def save(
        self,
        epoch: int,
        chunk: int,
        model,
        optimizer,
        scheduler=None,
        batch_ctrl=None,
        scheduler_state: Optional[dict] = None,
        metric: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        t0 = time.perf_counter()
        payload = {
            "epoch": epoch,
            "chunk": chunk,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if scheduler is not None:
            payload["lr_scheduler"] = scheduler.state_dict()
        if batch_ctrl is not None:
            payload["batch_ctrl"] = batch_ctrl.state_dict()
        if scheduler_state is not None:
            payload["chunk_scheduler"] = scheduler_state
        if metric is not None:
            payload["metric"] = metric
        if extra:
            payload.update(extra)

        name = f"ckpt_e{epoch:04d}_c{chunk:06d}.pt"
        path = self.out / name
        self._atomic_save(payload, path)
        shutil.copy2(path, self.out / self.LATEST)

        self._saved.append(path)
        while len(self._saved) > self.cfg.keep_last_k:
            old = self._saved.pop(0)
            if old.exists():
                old.unlink()

        is_best = self._is_best(metric)
        if is_best:
            shutil.copy2(path, self.out / self.BEST)
            logger.info("New best | %s=%.4f", self.cfg.best_metric, metric)

        self._save_meta(epoch, chunk, metric)
        logger.info("Checkpoint saved | %s | %.2fs", name, time.perf_counter() - t0)
        return path

    def load_latest(self) -> Optional[Dict[str, Any]]:
        p = self.out / self.LATEST
        return torch.load(p, map_location="cpu", weights_only=False) if p.exists() else None

    def resume_info(self) -> Dict[str, Any]:
        return {"epoch": self._meta.get("epoch", 0), "chunk": self._meta.get("chunk", 0)}

    def _atomic_save(self, payload: dict, path: Path):
        fd, tmp = tempfile.mkstemp(dir=self.out, suffix=".tmp")
        os.close(fd)
        try:
            torch.save(payload, tmp)
            shutil.move(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            raise

    def _is_best(self, metric: Optional[float]) -> bool:
        if metric is None:
            return False
        if self._best_value is None:
            self._best_value = metric
            return True
        if self.cfg.best_metric_mode == "min" and metric < self._best_value:
            self._best_value = metric
            return True
        if self.cfg.best_metric_mode == "max" and metric > self._best_value:
            self._best_value = metric
            return True
        return False

    def _load_meta(self) -> dict:
        p = self.out / self.META
        return json.loads(p.read_text()) if p.exists() else {}

    def _save_meta(self, epoch: int, chunk: int, metric: Optional[float]):
        (self.out / self.META).write_text(json.dumps({
            "epoch": epoch, "chunk": chunk,
            "metric": metric, "best": self._best_value,
            "ts": time.time(),
        }, indent=2))
