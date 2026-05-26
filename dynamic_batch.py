"""
dynamic_batch.py
────────────────
VRAM-aware dynamic batch controller. Unchanged in core logic from V1,
updated to use torch.amp (not deprecated torch.cuda.amp).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    initial_batch_size: int = 8
    min_batch_size: int = 1
    max_batch_size: int = 64
    target_effective_batch_size: int = 32
    grow_after_clean_steps: int = 200
    grow_factor: float = 1.25
    vram_headroom: float = 0.20
    max_grad_norm: float = 0.1


class DynamicBatchController:
    """VRAM-aware batch controller with OOM recovery and grad accumulation."""

    def __init__(self, config: Optional[BatchConfig] = None):
        self.cfg = config or BatchConfig()
        self._batch_size = self.cfg.initial_batch_size
        self._clean_steps = 0
        self._oom_count = 0
        self._total_steps = 0
        self._grad_accum = self._compute_accum()

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def grad_accum_steps(self) -> int:
        return self._grad_accum

    @property
    def effective_batch_size(self) -> int:
        return self._batch_size * self._grad_accum

    def safe_step(
        self,
        model,
        images: torch.Tensor,
        targets: list,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.amp.GradScaler] = None,
        device: torch.device = torch.device("cpu"),
        is_last_accum: bool = True,
    ) -> Optional[float]:
        try:
            images = images.to(device, non_blocking=True)

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss_dict = model.compute_loss(images, targets)
                    loss = sum(loss_dict.values()) / self._grad_accum
                scaler.scale(loss).backward()
                if is_last_accum:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.cfg.max_grad_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss_dict = model.compute_loss(images, targets)
                loss = sum(loss_dict.values()) / self._grad_accum
                loss.backward()
                if is_last_accum:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.cfg.max_grad_norm
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            self._clean_steps += 1
            self._total_steps += 1
            self._maybe_grow()
            return loss.item() * self._grad_accum

        except torch.cuda.OutOfMemoryError:
            return self._handle_oom(optimizer)

    def _handle_oom(self, optimizer) -> None:
        self._oom_count += 1
        self._clean_steps = 0
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        old = self._batch_size
        self._batch_size = max(self.cfg.min_batch_size, self._batch_size // 2)
        self._grad_accum = self._compute_accum()
        logger.warning(
            "CUDA OOM #%d | batch %d → %d | accum=%d | eff=%d",
            self._oom_count, old, self._batch_size,
            self._grad_accum, self.effective_batch_size,
        )
        return None

    def _maybe_grow(self):
        if self._clean_steps < self.cfg.grow_after_clean_steps:
            return
        if self._batch_size >= self.cfg.max_batch_size:
            return
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            if free / total < self.cfg.vram_headroom:
                return
        old = self._batch_size
        self._batch_size = min(self.cfg.max_batch_size,
                               int(self._batch_size * self.cfg.grow_factor))
        self._grad_accum = self._compute_accum()
        self._clean_steps = 0
        logger.info(
            "Batch grow %d → %d | accum=%d | eff=%d",
            old, self._batch_size, self._grad_accum, self.effective_batch_size,
        )

    def _compute_accum(self) -> int:
        target = self.cfg.target_effective_batch_size
        if target <= 0 or target <= self._batch_size:
            return 1
        return math.ceil(target / self._batch_size)

    def state_dict(self) -> dict:
        return {
            "batch_size": self._batch_size,
            "grad_accum": self._grad_accum,
            "clean_steps": self._clean_steps,
            "oom_count": self._oom_count,
            "total_steps": self._total_steps,
        }

    def load_state_dict(self, sd: dict):
        self._batch_size = sd.get("batch_size", self._batch_size)
        self._grad_accum = sd.get("grad_accum", self._grad_accum)
        self._clean_steps = sd.get("clean_steps", 0)
        self._oom_count = sd.get("oom_count", 0)
        self._total_steps = sd.get("total_steps", 0)
        logger.info(
            "BatchController restored | bs=%d accum=%d total_steps=%d",
            self._batch_size, self._grad_accum, self._total_steps,
        )
