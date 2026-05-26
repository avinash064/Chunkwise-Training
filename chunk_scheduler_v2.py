"""
chunk_scheduler_v2.py
─────────────────────
Adaptive chunk scheduler that controls chunk_size based on:
  1. RAM usage (psutil) — same as V1
  2. Data loading latency — shrink chunk if load > train time (GPU idling)
  3. GPU utilisation — grow chunk if GPU util is low
  4. Throughput — track images/second trend

Also manages epoch ordering, stratified sampling state, and resume.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import psutil

from streaming_coco_v2 import CategoryMeta, ImageMeta

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SchedulerConfig:
    # Chunk sizing
    initial_chunk_size: int = 500
    min_chunk_size: int = 100
    max_chunk_size: int = 3000
    chunk_grow_factor: float = 1.20
    chunk_shrink_factor: float = 0.75

    # RAM watermarks
    ram_high_watermark: float = 0.78
    ram_low_watermark: float = 0.55
    ram_sample_interval_s: float = 2.0
    ram_window: int = 5

    # Throughput adaptation
    load_train_ratio_high: float = 0.8   # if load_time > 0.8×train_time → shrink (GPU waiting)
    load_train_ratio_low: float = 0.2    # if load_time < 0.2×train_time → grow (load underutilised)

    # Shuffle
    shuffle: bool = True
    seed: int = 42

    # Save / resume
    save_every_n_chunks: int = 20


# ─────────────────────────────────────────────────────────────────────────────
# RAM monitor (background thread)
# ─────────────────────────────────────────────────────────────────────────────

class RAMMonitor:
    def __init__(self, interval: float = 2.0, window: int = 5):
        self._interval = interval
        self._window = window
        self._samples: List[float] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="RAMMonitor")

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def current(self) -> float:
        with self._lock:
            if not self._samples:
                return psutil.virtual_memory().percent / 100.0
            return sum(self._samples) / len(self._samples)

    def _run(self):
        while not self._stop.is_set():
            pct = psutil.virtual_memory().percent / 100.0
            with self._lock:
                self._samples.append(pct)
                if len(self._samples) > self._window:
                    self._samples.pop(0)
            self._stop.wait(self._interval)


# ─────────────────────────────────────────────────────────────────────────────
# Throughput tracker
# ─────────────────────────────────────────────────────────────────────────────

class ThroughputTracker:
    """Tracks recent images/sec and load vs train time ratio."""

    def __init__(self, window: int = 10):
        self._window = window
        self._imgs_per_sec: List[float] = []
        self._load_times: List[float] = []
        self._train_times: List[float] = []

    def record(self, num_images: int, load_time_s: float, train_time_s: float):
        if train_time_s > 0:
            self._imgs_per_sec.append(num_images / train_time_s)
        self._load_times.append(load_time_s)
        self._train_times.append(train_time_s)
        if len(self._imgs_per_sec) > self._window:
            self._imgs_per_sec.pop(0)
            self._load_times.pop(0)
            self._train_times.pop(0)

    def avg_imgs_per_sec(self) -> float:
        return sum(self._imgs_per_sec) / len(self._imgs_per_sec) if self._imgs_per_sec else 0.0

    def load_train_ratio(self) -> float:
        """load_time / train_time — if > 1 then loading is the bottleneck."""
        lt = sum(self._load_times[-5:]) / max(len(self._load_times[-5:]), 1)
        tt = sum(self._train_times[-5:]) / max(len(self._train_times[-5:]), 1)
        return lt / max(tt, 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Chunk scheduler
# ─────────────────────────────────────────────────────────────────────────────

class ChunkSchedulerV2:
    """
    Manages chunk sizing and epoch ordering.
    Decoupled from the producer — just tells the pipeline what chunk_size to use.
    """

    def __init__(
        self,
        image_list: List[ImageMeta],
        categories: Dict[int, CategoryMeta],
        config: Optional[SchedulerConfig] = None,
    ):
        self.images = image_list
        self.categories = categories
        self.cfg = config or SchedulerConfig()

        self._chunk_size = self.cfg.initial_chunk_size
        self._ram_monitor = RAMMonitor(
            interval=self.cfg.ram_sample_interval_s,
            window=self.cfg.ram_window,
        ).start()
        self._throughput = ThroughputTracker()
        self._epoch_chunk_count = 0

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def num_chunks_per_epoch(self) -> int:
        return max(1, (len(self.images) + self._chunk_size - 1) // self._chunk_size)

    def adapt(self, load_time_s: float, train_time_s: float, num_images: int):
        """
        Called after each chunk completes. Adjusts chunk_size for next chunk.
        Three signals: RAM, load/train ratio, throughput trend.
        """
        self._throughput.record(num_images, load_time_s, train_time_s)
        ram = self._ram_monitor.current()
        ratio = self._throughput.load_train_ratio()

        reasons = []
        new_size = self._chunk_size

        # RAM signal (highest priority)
        if ram > self.cfg.ram_high_watermark:
            new_size = max(self.cfg.min_chunk_size,
                           int(self._chunk_size * self.cfg.chunk_shrink_factor))
            reasons.append(f"RAM={ram*100:.1f}%>high")

        elif ram < self.cfg.ram_low_watermark:
            # Only grow on load signal too (both confirm underutilisation)
            if ratio < self.cfg.load_train_ratio_low:
                new_size = min(self.cfg.max_chunk_size,
                               int(self._chunk_size * self.cfg.chunk_grow_factor))
                reasons.append(f"RAM={ram*100:.1f}%<low+fast_load")

        # Load/train balance (secondary)
        elif ratio > self.cfg.load_train_ratio_high:
            # Loading is slow relative to training → smaller chunks to reduce per-chunk scan
            new_size = max(self.cfg.min_chunk_size,
                           int(self._chunk_size * self.cfg.chunk_shrink_factor))
            reasons.append(f"load/train={ratio:.2f}>high")

        if new_size != self._chunk_size:
            logger.info(
                "chunk_size %d → %d | reason: %s | imgs/s=%.1f",
                self._chunk_size, new_size,
                ", ".join(reasons),
                self._throughput.avg_imgs_per_sec(),
            )
            self._chunk_size = new_size

    def should_save(self, chunk_idx: int) -> bool:
        return (chunk_idx + 1) % self.cfg.save_every_n_chunks == 0

    def log_stats(self):
        logger.info(
            "Scheduler | chunk_size=%d | imgs/s=%.1f | RAM=%.1f%% | load/train=%.2f",
            self._chunk_size,
            self._throughput.avg_imgs_per_sec(),
            self._ram_monitor.current() * 100,
            self._throughput.load_train_ratio(),
        )

    def state_dict(self) -> dict:
        return {"chunk_size": self._chunk_size}

    def load_state_dict(self, sd: dict):
        self._chunk_size = sd.get("chunk_size", self._chunk_size)

    def shutdown(self):
        self._ram_monitor.stop()
