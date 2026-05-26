# Chunkwise Training — RF-DETR V2 Streaming Pipeline

A production-ready, memory-safe training pipeline for fine-tuning **RF-DETR SegNano** on large custom COCO datasets.

Built to solve two critical V1 failures:
- `OSError: Too many open files` from DataLoader FD accumulation
- Silent `DummyCriterion` (loss = 1.0 constant) from class-count mismatch

---

## Key Features

| Feature | Detail |
|---|---|
| **Async producer-consumer** | Background thread pre-loads next chunk while GPU trains current |
| **3-layer OOM protection** | VRAM (dynamic batch) + RAM (chunk sizing) + FD (DataLoader factory) |
| **True streaming COCO** | ~30 MB index vs V1's ~2.4 GB full annotation store |
| **Adaptive chunk scheduling** | RAM watermarks + load/train ratio auto-tune chunk size |
| **Atomic checkpointing** | Crash-safe saves; keeps last-K + best-by-val-loss |
| **SetCriterion fix** | Direct construction — no silent fallback to DummyCriterion |

---

## Architecture

```
main.py
  ├── StreamingCOCOParserV2    ← streaming annotation loader (~30 MB RAM)
  ├── RFDETRModelWrapper       ← LWDETR model + SetCriterion (direct build)
  ├── ChunkSchedulerV2         ← adaptive chunk sizing (RAM + throughput)
  ├── AsyncChunkPipeline       ← producer-consumer; overlaps I/O with GPU
  ├── DataLoaderFactory        ← per-chunk DataLoaders (FD leak fix)
  ├── ChunkTrainer             ← training loop with OOM retry
  ├── DynamicBatchController   ← VRAM-aware batch size
  ├── Validator                ← streaming val + COCO mAP
  └── CheckpointManager        ← atomic save/load
```

---

## OOM Resolution

### Layer 1 — VRAM (`DynamicBatchController`)
Catches `torch.cuda.OutOfMemoryError` → halves batch size → doubles grad accumulation → effective batch stays constant.

```
Start:     bs=8,  accum=4,  effective=32
OOM hit:   bs=4,  accum=8,  effective=32  ✓
OOM again: bs=2,  accum=16, effective=32  ✓
After 200 clean steps → grows back
```

### Layer 2 — RAM (`ChunkSchedulerV2`)
Background `RAMMonitor` thread samples every 2s. After each chunk:
```
RAM > 78%  →  chunk_size × 0.75   (fewer images in RAM at once)
RAM < 55% and fast load  →  chunk_size × 1.20
load/train ratio > 0.8   →  chunk_size × 0.75   (I/O bottleneck)
```

### Layer 3 — File Descriptors (`DataLoaderFactory`)
```python
DataLoader(
    persistent_workers=False,           # workers die after each loader
    multiprocessing_context="forkserver", # no FD inheritance from parent
)
# + explicit _cleanup() before each new DataLoader
# + ulimit raised to 65536 at startup
```

---

## Quick Start

### Requirements
```bash
pip install -r requirements.txt
```

### Dataset structure (COCO format)
```
dataset/
├── train.json
├── val.json
└── images/
    ├── train/
    └── val/
```

### Run training
```bash
python main.py \
  --train-json /path/to/train.json \
  --val-json   /path/to/val.json \
  --train-imgs /path/to/images/train \
  --val-imgs   /path/to/images/val \
  --output-dir /path/to/outputs \
  --epochs 100 \
  --chunk-size 500 \
  --batch-size 8 \
  --eff-batch 32 \
  --img-size 636 \
  --amp \
  --validate-every 5
```

### Resume from checkpoint
```bash
python main.py --resume --output-dir /path/to/outputs
```

---

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 100 | Total training epochs |
| `--chunk-size` | 500 | Images per chunk (auto-adapts) |
| `--batch-size` | 8 | GPU batch size (auto-adapts on OOM) |
| `--eff-batch` | 32 | Target effective batch (controls grad accum) |
| `--img-size` | 636 | Must be divisible by 12 (DINOv2 patch size) |
| `--num-workers` | 4 | DataLoader workers |
| `--lr` | 1e-4 | Learning rate (AdamW) |
| `--weight-decay` | 1e-4 | AdamW weight decay |
| `--amp` | False | Enable FP16 mixed precision |
| `--resume` | False | Resume from latest checkpoint |
| `--validate-every` | 5 | Run validation every N epochs |
| `--ram-high` | 0.78 | RAM watermark to shrink chunks |
| `--ram-low` | 0.55 | RAM watermark to grow chunks |
| `--save-every` | 20 | Save checkpoint every N chunks |
| `--queue-depth` | 2 | Async pipeline pre-load depth |
| `--mp-context` | forkserver | DataLoader multiprocessing context |

---

## Output Structure

```
outputs/
├── checkpoints/
│   ├── ckpt_e0000_c000020.pt   ← periodic saves
│   ├── latest.pt               ← most recent checkpoint
│   ├── best.pt                 ← best validation loss
│   └── ckpt_meta.json          ← epoch, chunk, metric info
└── logs/
    └── training.log
```

---

## Module Reference

| File | Purpose |
|---|---|
| `main.py` | Entry point — orchestrates all 14 init steps + training loop |
| `streaming_coco_v2.py` | Memory-efficient COCO parser (ijson + byte-offset index) |
| `model_wrapper.py` | RF-DETR wrapper + deterministic SetCriterion build |
| `async_pipeline.py` | Async producer-consumer pipeline + DataLoader factory |
| `chunk_scheduler_v2.py` | Adaptive chunk sizing (RAM monitor + throughput tracker) |
| `dynamic_batch.py` | VRAM-aware batch controller with OOM recovery |
| `dataset.py` | Per-chunk PyTorch Dataset + transforms + target builder |
| `trainer.py` | Single-chunk training loop with OOM retry |
| `validation.py` | Streaming validation + COCO mAP (pycocotools) |
| `checkpointing.py` | Atomic checkpoint manager (last-K + best tracking) |

---

## License

MIT
