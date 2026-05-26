# -*- coding: utf-8 -*-
"""
main.py  — RF-DETR V2 Streaming Training Pipeline
──────────────────────────────────────────────────

Fixes V1's "Too many open files" crash + adds:
  - True async producer-consumer pipeline
  - Throughput-aware chunk adaptation
  - Validation loop every N epochs
  - Cleaner RF-DETR integration

Usage
-----
  python main.py
  python main.py --resume
  python main.py --epochs 100 --chunk-size 500 --batch-size 8 --amp
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
from pathlib import Path

import psutil
import torch

# ── Insert project root on path ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from async_pipeline import AsyncChunkPipeline, DataLoaderFactory
from checkpointing import CheckpointConfig, CheckpointManager
from chunk_scheduler_v2 import ChunkSchedulerV2, SchedulerConfig
from dynamic_batch import BatchConfig, DynamicBatchController
from model_wrapper import RFDETRModelWrapper, validate_img_size
from streaming_coco_v2 import StreamingCOCOParserV2
from trainer import ChunkTrainer
from validation import Validator

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_JSON  = Path("/media/wi/ssd_hub/Avinash_work/dataset/train.json")
VAL_JSON    = Path("/media/wi/ssd_hub/Avinash_work/dataset/val.json")
TRAIN_IMGS  = Path("/media/wi/ssd_hub/Avinash_work/dataset/images/train")
VAL_IMGS    = Path("/media/wi/ssd_hub/Avinash_work/dataset/images/val")
OUTPUT_ROOT = Path("/media/wi/ssd_hub/Avinash_work/rfdetr_v2_outputs")

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path, level: int = logging.INFO) -> None:
    """Set up logging configuration with file and console handlers."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "training.log"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("RF-DETR V2 Streaming Trainer")
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--chunk-size",   type=int,   default=500)
    p.add_argument("--batch-size",   type=int,   default=8)
    p.add_argument("--eff-batch",    type=int,   default=32)
    p.add_argument("--img-size",     type=int,   default=636,
                   help="Must be divisible by 12 (DINOv2 patch size). Default: 636=53x12")
    p.add_argument("--num-workers",  type=int,   default=4)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--amp",          action="store_true")
    p.add_argument("--resume",       action="store_true")
    p.add_argument("--pretrained",   type=str,   default=None)
    p.add_argument("--output-dir",   type=str,   default=str(OUTPUT_ROOT))
    p.add_argument("--train-json",   type=str,   default=str(TRAIN_JSON))
    p.add_argument("--val-json",     type=str,   default=str(VAL_JSON))
    p.add_argument("--train-imgs",   type=str,   default=str(TRAIN_IMGS))
    p.add_argument("--val-imgs",     type=str,   default=str(VAL_IMGS))
    p.add_argument("--validate-every", type=int, default=5,
                   help="Run validation every N epochs")
    p.add_argument("--ram-high",     type=float, default=0.78)
    p.add_argument("--ram-low",      type=float, default=0.55)
    p.add_argument("--save-every",   type=int,   default=20)
    p.add_argument("--queue-depth",  type=int,   default=2,
                   help="Async pipeline queue depth (chunks pre-loaded)")
    p.add_argument("--mp-context",   type=str,   default="forkserver",
                   choices=["forkserver", "spawn", "fork"],
                   help="DataLoader multiprocessing context. 'forkserver' prevents FD leaks.")
    p.add_argument("--debug",        action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)

    setup_logging(
        out_dir / "logs",
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    logger.info("=" * 65)
    logger.info("RF-DETR V2 Streaming Training Pipeline")
    logger.info("=" * 65)
    logger.info("Args: %s", vars(args))

    # Validate img_size
    args.img_size = validate_img_size(args.img_size)

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info(
            "GPU: %s | VRAM: %.1f GB",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    # ── Set ulimit for file descriptors ───────────────────────────────────────
    # Increase FD limit to avoid "Too many open files" with many DataLoader workers
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(hard, max(soft, 65536))
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        logger.info("FD limit set: %d (was %d, hard=%d)", target, soft, hard)
    except Exception as e:
        logger.warning("Could not set FD limit: %s", e)

    # ── Step 1: Parse training metadata (streaming) ───────────────────────────
    logger.info("Parsing train.json (streaming) ...")
    t0 = time.perf_counter()
    train_parser = StreamingCOCOParserV2(args.train_json, skip_crowd=True)
    train_parser.build_index()
    parse_elapsed = time.perf_counter() - t0

    num_classes = len(train_parser.categories)
    image_list = list(train_parser.images.values())
    logger.info(
        "Train parse done in %.1fs | %d images | %d categories",
        parse_elapsed, len(image_list), num_classes,
    )
    logger.info(
        "RAM after train parse+GC: %.1f%% | %.1f GB free",
        psutil.virtual_memory().percent,
        psutil.virtual_memory().available / 1e9,
    )
    gc.collect()

    # ── Step 2: Parse val metadata ────────────────────────────────────────────
    val_parser = None
    if Path(args.val_json).exists():
        logger.info("Parsing val.json (streaming) ...")
        val_parser = StreamingCOCOParserV2(args.val_json, skip_crowd=True)
        val_parser.build_index()
        logger.info("Val parse done | %d images", len(val_parser.images))
        gc.collect()

    # ── Step 3: Checkpointing ─────────────────────────────────────────────────
    ckpt_mgr = CheckpointManager(CheckpointConfig(
        output_dir=out_dir / "checkpoints",
        save_every_n_chunks=args.save_every,
        keep_last_k=3,
    ))

    start_epoch = 0
    start_chunk = 0
    saved_state = None

    if args.resume:
        saved_state = ckpt_mgr.load_latest()
        if saved_state is not None:
            info = ckpt_mgr.resume_info()
            start_epoch = info["epoch"]
            start_chunk = info["chunk"] + 1
            logger.info("Resuming from epoch=%d chunk=%d", start_epoch, start_chunk)
        else:
            logger.warning("--resume but no checkpoint found; starting fresh")

    # ── Step 4: Free parse RAM then init model ────────────────────────────────
    gc.collect()
    logger.info(
        "Pre-model RAM: %.1f%% | %.1f GB free",
        psutil.virtual_memory().percent,
        psutil.virtual_memory().available / 1e9,
    )

    model = RFDETRModelWrapper(
        num_classes=num_classes,
        img_size=args.img_size,
        pretrained_weights=args.pretrained,
    )
    model.to(device)

    # ── Step 5: Optimizer + LR scheduler ─────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    # ── Step 6: Batch controller ──────────────────────────────────────────────
    batch_ctrl = DynamicBatchController(BatchConfig(
        initial_batch_size=args.batch_size,
        target_effective_batch_size=args.eff_batch,
    ))

    # ── Step 7: AMP ───────────────────────────────────────────────────────────
    scaler = None
    if args.amp and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
        logger.info("AMP enabled (fp16)")

    # ── Step 8: Restore from checkpoint ──────────────────────────────────────
    if saved_state is not None:
        model.load_state_dict(saved_state["model"])
        optimizer.load_state_dict(saved_state["optimizer"])
        if "lr_scheduler" in saved_state:
            lr_scheduler.load_state_dict(saved_state["lr_scheduler"])
        if "batch_ctrl" in saved_state:
            batch_ctrl.load_state_dict(saved_state["batch_ctrl"])
        logger.info("All states restored from checkpoint")

    # ── Step 9: Chunk scheduler ────────────────────────────────────────────────
    sched_cfg = SchedulerConfig(
        initial_chunk_size=args.chunk_size,
        ram_high_watermark=args.ram_high,
        ram_low_watermark=args.ram_low,
    )
    chunk_scheduler = ChunkSchedulerV2(image_list, train_parser.categories, sched_cfg)
    if saved_state and "chunk_scheduler" in saved_state:
        chunk_scheduler.load_state_dict(saved_state["chunk_scheduler"])

    # ── Step 10: Async pipeline ───────────────────────────────────────────────
    pipeline = AsyncChunkPipeline(
        parser=train_parser,
        image_list=image_list,
        chunk_size=chunk_scheduler.chunk_size,
        shuffle=sched_cfg.shuffle,
        seed=sched_cfg.seed,
        queue_depth=args.queue_depth,
    )

    # ── Step 11: DataLoader factory (solves FD leak) ──────────────────────────
    loader_factory = DataLoaderFactory(
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2,
        multiprocessing_context=args.mp_context,
    )

    # ── Step 12: Trainer ──────────────────────────────────────────────────────
    chunk_trainer = ChunkTrainer(
        model=model,
        optimizer=optimizer,
        batch_ctrl=batch_ctrl,
        loader_factory=loader_factory,
        device=device,
        img_dir=str(args.train_imgs),
        img_size=args.img_size,
        scaler=scaler,
        log_every_n=20,
    )

    # ── Step 13: Validator ─────────────────────────────────────────────────────
    validator = None
    if val_parser is not None:
        validator = Validator(
            val_parser=val_parser,
            val_img_dir=args.val_imgs,
            num_classes=num_classes,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=min(args.num_workers, 2),
            chunk_size=300,
            device=device,
        )

    # ── Step 14: Training loop ─────────────────────────────────────────────────
    logger.info("Starting training | epochs=%d | device=%s", args.epochs, device)
    total_start = time.perf_counter()

    for epoch in range(start_epoch, args.epochs):
        logger.info("━━━ EPOCH %d / %d ━━━", epoch + 1, args.epochs)

        # Start async producer for this epoch
        pipeline.update_chunk_size(chunk_scheduler.chunk_size)
        pipeline.start_epoch(epoch=epoch, chunk_size=chunk_scheduler.chunk_size)

        chunk_idx_global = 0
        for chunk in pipeline:
            # Skip already-trained chunks when resuming
            if epoch == start_epoch and chunk.chunk_idx < start_chunk:
                chunk_idx_global += 1
                continue

            logger.info(
                "Epoch %d | Chunk %d/%d | %d imgs | %d anns | load=%.1fs",
                epoch, chunk.chunk_idx + 1, chunk.total_chunks,
                chunk.num_images, chunk.num_annotations, chunk.load_time_s,
            )

            # Train this chunk
            train_t0 = time.perf_counter()
            metrics = chunk_trainer.train(chunk)
            train_elapsed = time.perf_counter() - train_t0

            # Adapt chunk size based on throughput + RAM
            chunk_scheduler.adapt(
                load_time_s=chunk.load_time_s,
                train_time_s=train_elapsed,
                num_images=chunk.num_images,
            )
            pipeline.update_chunk_size(chunk_scheduler.chunk_size)

            # Checkpoint
            if ckpt_mgr.should_save(chunk.chunk_idx):
                ckpt_mgr.save(
                    epoch=epoch,
                    chunk=chunk.chunk_idx,
                    model=model,
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    batch_ctrl=batch_ctrl,
                    scheduler_state=chunk_scheduler.state_dict(),
                    metric=metrics.get("avg_loss"),
                )

            chunk_idx_global += 1

        # End of epoch
        lr_scheduler.step()
        logger.info(
            "Epoch %d done | LR=%.2e | avg_ips=%.1f",
            epoch,
            lr_scheduler.get_last_lr()[0],
            chunk_scheduler._throughput.avg_imgs_per_sec(),
        )
        chunk_scheduler.log_stats()

        # Validation
        if validator is not None and (epoch + 1) % args.validate_every == 0:
            logger.info("Running validation ...")
            val_metrics = validator.run(model, epoch=epoch)
            logger.info("Val metrics: %s", {k: f"{v:.4f}" for k, v in val_metrics.items()})
            # Save best model by val_loss
            ckpt_mgr.save(
                epoch=epoch,
                chunk=0,
                model=model,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                batch_ctrl=batch_ctrl,
                scheduler_state=chunk_scheduler.state_dict(),
                metric=val_metrics.get("val_loss"),
            )

        # Reset start_chunk after first epoch
        start_chunk = 0

    # ── Final save ────────────────────────────────────────────────────────────
    ckpt_mgr.save(
        epoch=args.epochs - 1, chunk=0,
        model=model, optimizer=optimizer,
        scheduler=lr_scheduler, batch_ctrl=batch_ctrl,
        scheduler_state=chunk_scheduler.state_dict(),
    )

    pipeline.shutdown()
    loader_factory.shutdown()
    chunk_scheduler.shutdown()

    total = time.perf_counter() - total_start
    logger.info("Training complete in %.1f hours", total / 3600)


if __name__ == "__main__":
    main()
