"""
trainer.py
──────────
Core training loop. Operates on one AsyncChunk at a time.
"""

from __future__ import annotations

import gc
import logging
import time
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from async_pipeline import AsyncChunk, DataLoaderFactory
from dataset import COCOChunkDataset, build_category_id_map, build_train_transforms
from dynamic_batch import DynamicBatchController
from model_wrapper import RFDETRModelWrapper

logger = logging.getLogger(__name__)


def train_one_chunk(
    chunk: AsyncChunk,
    model: RFDETRModelWrapper,
    optimizer: torch.optim.Optimizer,
    batch_ctrl: DynamicBatchController,
    loader_factory: DataLoaderFactory,
    device: torch.device,
    img_size: int = 636,
    scaler: Optional[torch.amp.GradScaler] = None,
    log_every_n: int = 20,
    max_oom_retries: int = 3,
) -> Dict[str, float]:
    """
    Train model for one full pass over a chunk's DataLoader.

    Returns metric dict: avg_loss, num_batches, oom_retries, elapsed_s, imgs_per_s
    """
    model.train()

    # Build dataset for this chunk
    cat_map = build_category_id_map(chunk.categories)
    transforms = build_train_transforms(img_size)
    dataset = COCOChunkDataset(
        images=chunk.images,
        ann_index=chunk.ann_index,
        img_dir=_find_img_dir(chunk),
        category_id_map=cat_map,
        img_size=img_size,
        transforms=transforms,
    )

    total_loss = 0.0
    num_batches = 0
    oom_retries = 0
    t_start = time.perf_counter()

    for attempt in range(max_oom_retries + 1):
        loader = loader_factory.build(
            dataset,
            batch_size=batch_ctrl.batch_size,
            collate_fn=COCOChunkDataset.collate_fn,
            shuffle=True,
        )

        total_loss = 0.0
        num_batches = 0
        accum_step = 0
        got_oom = False

        for images, targets in loader:
            is_last_accum = (accum_step + 1) >= batch_ctrl.grad_accum_steps

            loss = batch_ctrl.safe_step(
                model=model,
                images=images,
                targets=targets,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                is_last_accum=is_last_accum,
            )

            if loss is None:
                oom_retries += 1
                got_oom = True
                logger.warning(
                    "OOM in chunk %d (attempt %d) | new bs=%d",
                    chunk.chunk_idx, attempt + 1, batch_ctrl.batch_size,
                )
                break

            accum_step = (accum_step + 1) % batch_ctrl.grad_accum_steps
            total_loss += loss
            num_batches += 1

            if num_batches % log_every_n == 0:
                elapsed = time.perf_counter() - t_start
                ips = num_batches * batch_ctrl.batch_size / max(elapsed, 1e-6)
                logger.info(
                    "E%d C%d | batch %d | loss=%.4f | %.1f img/s | bs=%d accum=%d",
                    chunk.epoch, chunk.chunk_idx, num_batches,
                    total_loss / num_batches, ips,
                    batch_ctrl.batch_size, batch_ctrl.grad_accum_steps,
                )

        if not got_oom:
            break
        if attempt >= max_oom_retries:
            logger.error("Max OOM retries (%d) reached — skipping chunk", max_oom_retries)
            break

    elapsed = time.perf_counter() - t_start
    avg_loss = total_loss / max(num_batches, 1)
    ips = num_batches * batch_ctrl.batch_size / max(elapsed, 1e-6)

    logger.info(
        "Chunk %d/%d done | avg_loss=%.4f | %d batches | %.1fs | %.1f img/s",
        chunk.chunk_idx + 1, chunk.total_chunks,
        avg_loss, num_batches, elapsed, ips,
    )

    # Explicit cleanup
    del dataset
    gc.collect()

    return {
        "avg_loss": avg_loss,
        "num_batches": num_batches,
        "oom_retries": oom_retries,
        "elapsed_s": elapsed,
        "imgs_per_s": ips,
    }


def _find_img_dir(chunk: AsyncChunk) -> str:
    """
    Placeholder — in production this is passed via config.
    The actual img_dir is set in main.py and passed to trainer.
    """
    return ""  # overridden in main.py


# ── Convenience wrapper that takes img_dir explicitly ────────────────────────

class ChunkTrainer:
    """Stateless trainer that wraps train_one_chunk with config."""

    def __init__(
        self,
        model: RFDETRModelWrapper,
        optimizer: torch.optim.Optimizer,
        batch_ctrl: DynamicBatchController,
        loader_factory: DataLoaderFactory,
        device: torch.device,
        img_dir: str,
        img_size: int = 636,
        scaler: Optional[torch.amp.GradScaler] = None,
        log_every_n: int = 20,
    ):
        self.model = model
        self.optimizer = optimizer
        self.batch_ctrl = batch_ctrl
        self.loader_factory = loader_factory
        self.device = device
        self.img_dir = img_dir
        self.img_size = img_size
        self.scaler = scaler
        self.log_every_n = log_every_n

    def train(self, chunk: AsyncChunk) -> Dict[str, float]:
        model = self.model
        cat_map = build_category_id_map(chunk.categories)
        transforms = build_train_transforms(self.img_size)
        dataset = COCOChunkDataset(
            images=chunk.images,
            ann_index=chunk.ann_index,
            img_dir=self.img_dir,
            category_id_map=cat_map,
            img_size=self.img_size,
            transforms=transforms,
        )

        total_loss = 0.0
        num_batches = 0
        oom_retries = 0
        t_start = time.perf_counter()

        for attempt in range(4):  # max 3 OOM retries
            loader = self.loader_factory.build(
                dataset,
                batch_size=self.batch_ctrl.batch_size,
                collate_fn=COCOChunkDataset.collate_fn,
                shuffle=True,
            )
            total_loss = 0.0
            num_batches = 0
            accum_step = 0
            got_oom = False

            for images, targets in loader:
                is_last_accum = (accum_step + 1) >= self.batch_ctrl.grad_accum_steps
                loss = self.batch_ctrl.safe_step(
                    model=model,
                    images=images,
                    targets=targets,
                    optimizer=self.optimizer,
                    scaler=self.scaler,
                    device=self.device,
                    is_last_accum=is_last_accum,
                )
                if loss is None:
                    oom_retries += 1
                    got_oom = True
                    break
                accum_step = (accum_step + 1) % self.batch_ctrl.grad_accum_steps
                total_loss += loss
                num_batches += 1
                if num_batches % self.log_every_n == 0:
                    elapsed = time.perf_counter() - t_start
                    ips = num_batches * self.batch_ctrl.batch_size / max(elapsed, 1e-6)
                    logger.info(
                        "E%d C%d | batch %d/%d | loss=%.4f | %.1f img/s | bs=%d accum=%d",
                        chunk.epoch, chunk.chunk_idx, num_batches,
                        len(loader), total_loss / num_batches, ips,
                        self.batch_ctrl.batch_size, self.batch_ctrl.grad_accum_steps,
                    )
            if not got_oom:
                break

        elapsed = time.perf_counter() - t_start
        avg_loss = total_loss / max(num_batches, 1)
        ips = num_batches * self.batch_ctrl.batch_size / max(elapsed, 1e-6)
        logger.info(
            "Chunk %d/%d done | avg_loss=%.4f | %d batches | %.1fs | %.1f img/s",
            chunk.chunk_idx + 1, chunk.total_chunks,
            avg_loss, num_batches, elapsed, ips,
        )
        del dataset
        gc.collect()
        return {
            "avg_loss": avg_loss,
            "num_batches": num_batches,
            "oom_retries": oom_retries,
            "elapsed_s": elapsed,
            "imgs_per_s": ips,
        }
