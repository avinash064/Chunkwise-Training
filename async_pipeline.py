"""
async_pipeline.py
─────────────────
Asynchronous producer-consumer pipeline that overlaps:
  - Chunk annotation loading (I/O-bound, CPU)
  - GPU forward/backward (compute-bound)

Architecture:
  ChunkProducer (background thread)
    → reads annotations for next chunk while GPU trains current chunk
    → puts Chunk objects into a bounded queue

  ChunkConsumer (main training thread)
    → pulls Chunk from queue
    → builds Dataset + DataLoader
    → runs train_one_chunk()

The queue depth is configurable (default=2):
  depth=1 → always one chunk pre-loaded (standard prefetch)
  depth=2 → two chunks buffered (absorbs I/O spikes)

The key fix for V1's "Too many open files" bug:
  DataLoader is rebuilt each chunk with persistent_workers=False.
  Workers are explicitly terminated before the next DataLoader is created.
  This prevents shared memory FD accumulation across chunk boundaries.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from streaming_coco_v2 import AnnotationRecord, ImageMeta, StreamingCOCOParserV2

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Chunk data container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AsyncChunk:
    epoch: int
    chunk_idx: int
    total_chunks: int
    images: List[ImageMeta]
    ann_index: Dict[int, List[AnnotationRecord]]
    categories: dict
    load_time_s: float = 0.0
    is_sentinel: bool = False           # True = producer signals end of epoch

    @property
    def num_images(self) -> int:
        return len(self.images)

    @property
    def num_annotations(self) -> int:
        return sum(len(v) for v in self.ann_index.values())


_SENTINEL = AsyncChunk(
    epoch=-1, chunk_idx=-1, total_chunks=-1,
    images=[], ann_index={}, categories={},
    is_sentinel=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Producer (background thread)
# ─────────────────────────────────────────────────────────────────────────────

class ChunkProducer(threading.Thread):
    """
    Background thread that pre-loads chunk annotations while the GPU trains.

    Thread safety: only writes to self._queue. Main thread only reads.
    Errors in the producer are caught and propagated via an error queue.
    """

    def __init__(
        self,
        parser: StreamingCOCOParserV2,
        image_list: List[ImageMeta],
        chunk_size: int,
        epoch: int,
        shuffle: bool,
        seed: int,
        queue_depth: int = 2,
    ):
        super().__init__(daemon=True, name=f"ChunkProducer-E{epoch}")
        self._parser = parser
        self._image_list = image_list
        self._chunk_size = chunk_size
        self._epoch = epoch
        self._shuffle = shuffle
        self._seed = seed
        self._queue: queue.Queue[AsyncChunk] = queue.Queue(maxsize=queue_depth + 1)
        self._error_queue: queue.Queue[Exception] = queue.Queue()
        self._stop_event = threading.Event()
        self._produced = 0

    @property
    def chunk_queue(self) -> queue.Queue:
        return self._queue

    @property
    def error_queue(self) -> queue.Queue:
        return self._error_queue

    def stop(self):
        self._stop_event.set()

    def run(self):
        try:
            import random
            ordered = list(self._image_list)
            if self._shuffle:
                rng = random.Random(self._seed + self._epoch)
                rng.shuffle(ordered)

            total_chunks = max(1, (len(ordered) + self._chunk_size - 1) // self._chunk_size)

            for chunk_idx, start in enumerate(range(0, len(ordered), self._chunk_size)):
                if self._stop_event.is_set():
                    logger.info("Producer stopped at chunk %d", chunk_idx)
                    break

                chunk_imgs = ordered[start: start + self._chunk_size]
                chunk_ids = {img.id for img in chunk_imgs}

                t0 = time.perf_counter()
                chunk_anns = self._parser.load_annotations_for_images(chunk_ids)
                load_time = time.perf_counter() - t0

                chunk = AsyncChunk(
                    epoch=self._epoch,
                    chunk_idx=chunk_idx,
                    total_chunks=total_chunks,
                    images=chunk_imgs,
                    ann_index=chunk_anns,
                    categories=self._parser.categories,
                    load_time_s=load_time,
                )

                logger.debug(
                    "Producer: chunk %d/%d loaded | %d imgs | %.1fs",
                    chunk_idx + 1, total_chunks, len(chunk_imgs), load_time,
                )

                # Block until consumer has space in queue
                while not self._stop_event.is_set():
                    try:
                        self._queue.put(chunk, timeout=1.0)
                        self._produced += 1
                        break
                    except queue.Full:
                        continue

            # Signal end of epoch with retry loop
            if not self._stop_event.is_set():
                while True:
                    try:
                        self._queue.put(_SENTINEL, timeout=1.0)
                        logger.info("Sentinel pushed to queue (shutdown)")
                        break
                    except queue.Full:
                        continue

        except Exception as e:
            logger.error("ChunkProducer error: %s", e, exc_info=True)
            self._error_queue.put(e)
            # Put sentinel so consumer doesn't hang (retry with backoff)
            for attempt in range(10):
                try:
                    self._queue.put(_SENTINEL, timeout=1.0)
                    logger.info("Sentinel pushed to queue (error recovery)")
                    break
                except queue.Full:
                    continue


# ─────────────────────────────────────────────────────────────────────────────
# Async pipeline manager
# ─────────────────────────────────────────────────────────────────────────────

class AsyncChunkPipeline:
    """
    Manages the producer-consumer pipeline for one training epoch.

    Usage
    -----
    pipeline = AsyncChunkPipeline(parser, image_list, chunk_size=500)
    pipeline.start_epoch(epoch=0)

    for chunk in pipeline:
        # chunk is AsyncChunk, pre-loaded while previous chunk was training
        train(chunk)

    pipeline.shutdown()
    """

    def __init__(
        self,
        parser: StreamingCOCOParserV2,
        image_list: List[ImageMeta],
        chunk_size: int,
        shuffle: bool = True,
        seed: int = 42,
        queue_depth: int = 2,
    ):
        self._parser = parser
        self._image_list = image_list
        self._chunk_size = chunk_size
        self._shuffle = shuffle
        self._seed = seed
        self._queue_depth = queue_depth
        self._producer: Optional[ChunkProducer] = None
        self._current_epoch = 0
        self._load_times: List[float] = []

    def start_epoch(self, epoch: int, chunk_size: Optional[int] = None):
        """Start a new producer thread for the given epoch."""
        self._stop_current_producer()
        if chunk_size is not None:
            self._chunk_size = chunk_size
        self._current_epoch = epoch
        self._producer = ChunkProducer(
            parser=self._parser,
            image_list=self._image_list,
            chunk_size=self._chunk_size,
            epoch=epoch,
            shuffle=self._shuffle,
            seed=self._seed,
            queue_depth=self._queue_depth,
        )
        self._producer.start()
        logger.info(
            "AsyncChunkPipeline: epoch %d started | chunk_size=%d | queue_depth=%d",
            epoch, self._chunk_size, self._queue_depth,
        )

    def __iter__(self):
        """Yield AsyncChunk objects until the sentinel is received."""
        if self._producer is None:
            raise RuntimeError("Call start_epoch() before iterating")

        while True:
            # Check for producer errors
            if not self._producer.error_queue.empty():
                err = self._producer.error_queue.get()
                raise RuntimeError(f"ChunkProducer failed: {err}") from err

            try:
                chunk = self._producer.chunk_queue.get(timeout=300.0)
            except queue.Empty:
                raise RuntimeError(
                    "ChunkProducer timed out after 300s — possible deadlock"
                )

            if chunk.is_sentinel:
                break

            self._load_times.append(chunk.load_time_s)
            yield chunk

    def avg_load_time_s(self) -> float:
        if not self._load_times:
            return 0.0
        return sum(self._load_times) / len(self._load_times)

    def update_chunk_size(self, new_size: int):
        """Update chunk size for next epoch (takes effect on next start_epoch())."""
        self._chunk_size = new_size

    def shutdown(self):
        self._stop_current_producer()
        logger.info("AsyncChunkPipeline shut down")

    def _stop_current_producer(self):
        if self._producer is not None and self._producer.is_alive():
            self._producer.stop()
            self._producer.join(timeout=10)
            if self._producer.is_alive():
                logger.warning("Producer thread did not stop cleanly")
            self._producer = None


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory — solves the "Too many open files" V1 bug
# ─────────────────────────────────────────────────────────────────────────────

class DataLoaderFactory:
    """
    Creates and destroys DataLoaders cleanly per chunk.

    V1 Bug: persistent_workers=True caused shared memory FDs to accumulate
    across chunk boundaries. With 4 workers and 244 chunks, this hit the
    OS file descriptor limit (ulimit -n).

    V2 Fix:
    1. persistent_workers=False — workers are terminated after each DataLoader
       is garbage collected.
    2. Explicit _cleanup() call before creating the next DataLoader.
    3. multiprocessing_context="forkserver" on Linux to avoid FD inheritance.
    4. Optional: use shared_memory backend (POSIX shm) with explicit cleanup.
    """

    def __init__(
        self,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        multiprocessing_context: str = "forkserver",
    ):
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.mp_context = multiprocessing_context
        self._current_loader = None

    def build(
        self,
        dataset,
        batch_size: int,
        collate_fn=None,
        shuffle: bool = True,
    ):
        """Build a new DataLoader, cleaning up the previous one first."""
        import torch
        self._cleanup()

        import torch.utils.data as tud
        loader = tud.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory and torch.cuda.is_available(),
            # CRITICAL: persistent_workers=False prevents FD accumulation
            persistent_workers=False,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            drop_last=True,
            collate_fn=collate_fn,
            # forkserver avoids FD inheritance from parent process
            multiprocessing_context=self.mp_context if self.num_workers > 0 else None,
        )
        self._current_loader = loader
        return loader

    def _cleanup(self):
        """Explicitly terminate workers and release shared memory."""
        if self._current_loader is not None:
            try:
                # Calling ._iterator._shutdown_workers() if it exists
                it = getattr(self._current_loader, "_iterator", None)
                if it is not None:
                    shutdown = getattr(it, "_shutdown_workers", None)
                    if shutdown:
                        shutdown()
            except Exception:
                pass
            self._current_loader = None
            import gc
            gc.collect()

    def shutdown(self):
        self._cleanup()
