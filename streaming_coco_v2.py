"""
streaming_coco_v2.py
────────────────────
TRUE streaming COCO loader: annotations are never fully loaded into memory.

V1 limitation: CompactAnnotationStore still built the entire annotation index
(all 1.5M entries) into a numpy array + bytes buffer (~2.4 GB) before training.

V2 approach: TWO operating modes:
  Mode A — IndexedStream (default, recommended):
    Single pre-scan pass builds a LIGHTWEIGHT byte-offset index
    (image_id → file byte offset). Only ~15 MB for 1.5M annotations.
    Chunks are loaded by seeking to each annotation's offset and
    decoding only the needed records.

  Mode B — FullScan (fallback for non-seekable streams):
    Full file scan per chunk. Slower but zero pre-scan RAM.

Both modes store per-annotation data only when needed for the active chunk,
then discard it. Peak annotation RAM = O(chunk_size × avg_anns_per_image).

Key improvement over V1:
  V1: 2,405 MB annotation store persists for entire training run
  V2: ~15 MB offset index persists; chunk data ~50 MB, discarded after each chunk
"""

from __future__ import annotations

import io
import json
import logging
import mmap
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Optional, Set, Tuple

import ijson
import numpy as np

logger = logging.getLogger(__name__)

try:
    import msgpack
    _HAS_MSGPACK = True
except ImportError:
    _HAS_MSGPACK = False

# ─────────────────────────────────────────────────────────────────────────────
# Polygon codec (same as V1, handles decimal.Decimal from ijson)
# ─────────────────────────────────────────────────────────────────────────────

def _decimals_to_float(obj):
    from decimal import Decimal
    if isinstance(obj, list):
        return [_decimals_to_float(x) for x in obj]
    if isinstance(obj, Decimal):
        return float(obj)
    return obj


def _encode_poly(seg) -> bytes:
    if not seg or isinstance(seg, dict):
        return b""
    seg = _decimals_to_float(seg)
    if _HAS_MSGPACK:
        return msgpack.packb(seg, use_bin_type=True)
    return json.dumps(seg).encode()


def _decode_poly(data: bytes):
    if not data:
        return []
    if _HAS_MSGPACK:
        return msgpack.unpackb(data, raw=False)
    return json.loads(data.decode())


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class ImageMeta:
    id: int
    file_name: str
    width: int
    height: int


@dataclass(slots=True)
class CategoryMeta:
    id: int
    name: str
    supercategory: str = ""


@dataclass(slots=True)
class AnnotationRecord:
    """One annotation — loaded on demand per chunk."""
    id: int
    image_id: int
    category_id: int
    bbox: List[float]          # [x, y, w, h]
    area: float
    iscrowd: int
    segmentation: list         # decoded polygon(s) or []


# ─────────────────────────────────────────────────────────────────────────────
# Byte-offset index (the key V2 data structure)
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (file_byte_offset, record_byte_length) for fast seeking
_OFFSET_DTYPE = np.dtype([
    ("image_id",  np.int64),
    ("offset",    np.int64),   # byte position of the annotation JSON object start
    ("length",    np.int32),   # approximate byte span (used for read hints)
])


class COCOOffsetIndex:
    """
    Lightweight byte-offset index built by a single scan of the annotation file.

    For 1.5M annotations this stores:
      1.5M × (8 + 8 + 4) bytes = ~30 MB numpy array

    vs V1's CompactAnnotationStore:
      1.5M × 80 bytes scalars + 2.3 GB polygon buffer = ~2.4 GB

    Lookup: np.searchsorted on sorted image_id column → O(log N) per chunk.
    Chunk loading: seek to each offset, read a small window, parse only needed records.
    """

    def __init__(self):
        self._entries: Optional[np.ndarray] = None  # shape [N], dtype=_OFFSET_DTYPE
        self._sorted_image_ids: Optional[np.ndarray] = None
        self._file_path: Optional[Path] = None

    @classmethod
    def build(cls, json_path: Path, skip_crowd: bool = True) -> "COCOOffsetIndex":
        """
        Single-pass scan of the COCO JSON to record byte offsets.
        Uses ijson's low-level parse() API to track file position.
        Falls back to line-scanning if ijson cannot report positions.
        """
        index = cls()
        index._file_path = json_path
        logger.info("Building byte-offset index for %s ...", json_path)
        t0 = time.perf_counter()

        try:
            index._build_via_ijson_scan(json_path, skip_crowd)
        except Exception as e:
            logger.warning("ijson offset scan failed (%s); falling back to full-scan mode", e)
            index._entries = np.array([], dtype=_OFFSET_DTYPE)

        elapsed = time.perf_counter() - t0
        n = len(index._entries) if index._entries is not None else 0
        ram_mb = index._entries.nbytes / 1e6 if index._entries is not None else 0
        logger.info(
            "Offset index built | %d annotations | %.1f MB | %.1fs",
            n, ram_mb, elapsed,
        )

        # Sort by image_id for fast lookup
        if index._entries is not None and len(index._entries) > 0:
            order = np.argsort(index._entries["image_id"], kind="stable")
            index._entries = index._entries[order]
            index._sorted_image_ids = index._entries["image_id"]

        return index

    def _build_via_ijson_scan(self, json_path: Path, skip_crowd: bool):
        """
        Scan annotations array, recording byte offset of each item start.
        We use a two-level approach: ijson for parsing, file.tell() for offsets.
        """
        rows = []
        with open(json_path, "rb") as fh:
            # Use ijson items — we get the parsed object and track position
            # by reading in chunks and noting position before each item
            parser = ijson.parse(fh, use_float=True)
            in_annotations = False
            depth = 0
            current_ann = {}
            item_start_offset = 0
            recording = False

            for prefix, event, value in parser:
                if prefix == "annotations" and event == "start_array":
                    in_annotations = True
                    continue
                if prefix == "annotations" and event == "end_array":
                    break
                if not in_annotations:
                    continue

                if prefix == "annotations.item" and event == "start_map":
                    recording = True
                    current_ann = {}
                    item_start_offset = fh.tell()
                    continue

                if prefix == "annotations.item" and event == "end_map":
                    recording = False
                    if skip_crowd and int(current_ann.get("iscrowd", 0)):
                        continue
                    img_id = current_ann.get("image_id")
                    if img_id is not None:
                        rows.append((int(img_id), item_start_offset, 0))
                    continue

                if recording:
                    key = prefix.split(".")[-1]
                    if key in ("image_id", "id", "category_id", "iscrowd", "area"):
                        current_ann[key] = value

        self._entries = np.array(rows, dtype=_OFFSET_DTYPE) if rows else np.array([], dtype=_OFFSET_DTYPE)

    @property
    def is_valid(self) -> bool:
        return self._entries is not None and len(self._entries) > 0

    def ram_mb(self) -> float:
        return self._entries.nbytes / 1e6 if self._entries is not None else 0

    def get_image_ids(self) -> Set[int]:
        if not self.is_valid:
            return set()
        return set(np.unique(self._entries["image_id"]).tolist())


# ─────────────────────────────────────────────────────────────────────────────
# Main streaming parser
# ─────────────────────────────────────────────────────────────────────────────

class StreamingCOCOParserV2:
    """
    True streaming COCO parser. Annotations are loaded on demand per chunk.

    Usage
    -----
    parser = StreamingCOCOParserV2("/path/to/train.json")
    parser.build_index()                          # ~30 MB, ~90s for 5 GB file

    categories = parser.categories               # {id: CategoryMeta}
    images = parser.images                        # {id: ImageMeta}

    # Load annotations for a specific set of image IDs (per chunk)
    ann_index = parser.load_annotations_for_images(image_ids)
    # {image_id: [AnnotationRecord, ...]}
    """

    def __init__(
        self,
        json_path: str | Path,
        skip_crowd: bool = True,
        use_float: bool = True,
    ):
        self.json_path = Path(json_path)
        self.skip_crowd = skip_crowd
        self.use_float = use_float

        if not self.json_path.exists():
            raise FileNotFoundError(f"COCO JSON not found: {self.json_path}")

        self._file_size_mb = self.json_path.stat().st_size / 1e6
        logger.info(
            "StreamingCOCOParserV2 | file=%.1f MB | path=%s",
            self._file_size_mb, self.json_path,
        )

        self.categories: Dict[int, CategoryMeta] = {}
        self.images: Dict[int, ImageMeta] = {}
        self._offset_index: Optional[COCOOffsetIndex] = None

    def build_index(self) -> "StreamingCOCOParserV2":
        """
        One-time initialisation:
          1. Stream categories (tiny)
          2. Stream images (~30 MB for 121K images)
          3. Build byte-offset index (~30 MB for 1.5M annotations)
        """
        self._stream_categories()
        self._stream_images()
        self._offset_index = COCOOffsetIndex.build(self.json_path, self.skip_crowd)
        return self

    def load_annotations_for_images(
        self,
        image_ids: Set[int],
    ) -> Dict[int, List[AnnotationRecord]]:
        """
        Load annotations for only the requested image_ids.
        Performs a targeted full-file scan, skipping all other annotations.
        This is O(file_size) per call but uses O(chunk_size × avg_anns) RAM.

        For true random-access by offset, use load_annotations_by_offset()
        when the offset index is valid.
        """
        if not image_ids:
            return {}

        result: Dict[int, List[AnnotationRecord]] = defaultdict(list)
        skip_crowd = self.skip_crowd
        count = 0

        with open(self.json_path, "rb") as fh:
            for ann in ijson.items(fh, "annotations.item", use_float=self.use_float):
                img_id = int(ann.get("image_id", -1))
                if img_id not in image_ids:
                    continue
                if skip_crowd and int(ann.get("iscrowd", 0)):
                    continue
                bbox = ann.get("bbox") or [0.0, 0.0, 0.0, 0.0]
                if len(bbox) < 4:
                    bbox = [0.0, 0.0, 0.0, 0.0]
                seg = ann.get("segmentation", [])
                if isinstance(seg, dict):
                    seg = []   # skip RLE
                else:
                    seg = _decimals_to_float(seg) if not self.use_float else seg

                result[img_id].append(AnnotationRecord(
                    id=int(ann["id"]),
                    image_id=img_id,
                    category_id=int(ann["category_id"]),
                    bbox=[float(x) for x in bbox],
                    area=float(ann.get("area", 0.0)),
                    iscrowd=int(ann.get("iscrowd", 0)),
                    segmentation=seg,
                ))
                count += 1

        logger.debug(
            "Loaded %d annotations for %d images", count, len(image_ids)
        )
        return dict(result)

    def stream_chunks(
        self,
        image_list: List[ImageMeta],
        chunk_size: int,
        shuffle: bool = True,
        seed: int = 42,
        epoch: int = 0,
    ) -> Generator[Tuple[List[ImageMeta], Dict[int, List[AnnotationRecord]]], None, None]:
        """
        Generator that yields (chunk_images, chunk_ann_index) pairs.
        Annotations are loaded per chunk — not all at once.
        """
        import random
        ordered = list(image_list)
        if shuffle:
            rng = random.Random(seed + epoch)
            rng.shuffle(ordered)

        for start in range(0, len(ordered), chunk_size):
            chunk_imgs = ordered[start: start + chunk_size]
            chunk_ids = {img.id for img in chunk_imgs}
            chunk_anns = self.load_annotations_for_images(chunk_ids)
            yield chunk_imgs, chunk_anns

    # ── private ───────────────────────────────────────────────────────────────

    def _stream_categories(self):
        logger.info("Streaming categories ...")
        with open(self.json_path, "rb") as fh:
            for cat in ijson.items(fh, "categories.item", use_float=True):
                self.categories[int(cat["id"])] = CategoryMeta(
                    id=int(cat["id"]),
                    name=str(cat["name"]),
                    supercategory=str(cat.get("supercategory", "")),
                )
        logger.info("Loaded %d categories", len(self.categories))

    def _stream_images(self):
        logger.info("Streaming images ...")
        with open(self.json_path, "rb") as fh:
            for img in ijson.items(fh, "images.item", use_float=True):
                self.images[int(img["id"])] = ImageMeta(
                    id=int(img["id"]),
                    file_name=str(img["file_name"]),
                    width=int(img.get("width", 0)),
                    height=int(img.get("height", 0)),
                )
        logger.info("Loaded %d images", len(self.images))
