"""
dataset.py
──────────
Lightweight per-chunk PyTorch Dataset + stratified chunk sampler.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms as T

from streaming_coco_v2 import AnnotationRecord, CategoryMeta, ImageMeta

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────

def build_train_transforms(img_size: int = 636) -> T.Compose:
    return T.Compose([
        T.ToPILImage(),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_val_transforms(img_size: int = 636) -> T.Compose:
    return T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Polygon → mask
# ─────────────────────────────────────────────────────────────────────────────

def polygon_to_mask(segmentation: list, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if not segmentation or isinstance(segmentation, dict):
        return mask
    for poly in segmentation:
        if len(poly) < 6:
            continue
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], color=1)
    return mask


def build_category_id_map(categories: Dict[int, CategoryMeta]) -> Dict[int, int]:
    return {cat_id: idx for idx, cat_id in enumerate(sorted(categories.keys()))}


# ─────────────────────────────────────────────────────────────────────────────
# Target builder
# ─────────────────────────────────────────────────────────────────────────────

def build_target(
    anns: List[AnnotationRecord],
    orig_h: int,
    orig_w: int,
    new_h: int,
    new_w: int,
    category_id_map: Dict[int, int],
) -> dict:
    boxes, labels, masks, areas, crowds = [], [], [], [], []
    sx = new_w / max(orig_w, 1)
    sy = new_h / max(orig_h, 1)

    for ann in anns:
        cat_idx = category_id_map.get(ann.category_id)
        if cat_idx is None:
            continue
        if ann.bbox and len(ann.bbox) == 4:
            x, y, bw, bh = ann.bbox
            cx = (x * sx + bw * sx / 2) / new_w
            cy = (y * sy + bh * sy / 2) / new_h
            nw = bw * sx / new_w
            nh = bh * sy / new_h
            boxes.append([cx, cy, nw, nh])
        else:
            boxes.append([0.0, 0.0, 0.0, 0.0])
        labels.append(cat_idx)
        areas.append(ann.area)
        crowds.append(ann.iscrowd)
        raw = polygon_to_mask(ann.segmentation, orig_h, orig_w)
        if raw.shape != (new_h, new_w):
            raw = cv2.resize(raw, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        masks.append(raw.astype(bool))

    if boxes:
        return {
            "boxes":   torch.tensor(boxes,  dtype=torch.float32),
            "labels":  torch.tensor(labels, dtype=torch.long),
            "masks":   torch.tensor(np.stack(masks), dtype=torch.bool),
            "area":    torch.tensor(areas,  dtype=torch.float32),
            "iscrowd": torch.tensor(crowds, dtype=torch.long),
        }
    return {
        "boxes":   torch.zeros((0, 4), dtype=torch.float32),
        "labels":  torch.zeros((0,),   dtype=torch.long),
        "masks":   torch.zeros((0, new_h, new_w), dtype=torch.bool),
        "area":    torch.zeros((0,),   dtype=torch.float32),
        "iscrowd": torch.zeros((0,),   dtype=torch.long),
    }


# ─────────────────────────────────────────────────────────────────────────────
# COCOChunkDataset
# ─────────────────────────────────────────────────────────────────────────────

class COCOChunkDataset(Dataset):
    """
    Stateless per-chunk Dataset. No global state, no caching.
    """

    def __init__(
        self,
        images: List[ImageMeta],
        ann_index: Dict[int, List[AnnotationRecord]],
        img_dir: str | Path,
        category_id_map: Dict[int, int],
        img_size: int = 636,
        transforms: Optional[T.Compose] = None,
    ):
        self._images = images
        self._ann_index = ann_index
        self._img_dir = Path(img_dir)
        self._cat_map = category_id_map
        self._img_size = img_size
        self._transforms = transforms

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        meta = self._images[idx]
        anns = self._ann_index.get(meta.id, [])

        # Load image
        img_path = self._img_dir / meta.file_name
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            logger.warning("Cannot read: %s", img_path)
            h = meta.height or self._img_size
            w = meta.width or self._img_size
            img_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = img_rgb.shape[:2]

        if self._transforms is not None:
            img_tensor = self._transforms(img_rgb)
        else:
            img_r = cv2.resize(img_rgb, (self._img_size, self._img_size))
            img_tensor = torch.from_numpy(img_r).permute(2, 0, 1).float() / 255.0

        target = build_target(
            anns, orig_h, orig_w,
            self._img_size, self._img_size,
            self._cat_map,
        )
        target["image_id"] = torch.tensor([meta.id], dtype=torch.long)
        return img_tensor, target

    @staticmethod
    def collate_fn(batch):
        images, targets = zip(*batch)
        return torch.stack(images, 0), list(targets)


# ─────────────────────────────────────────────────────────────────────────────
# Stratified chunk sampler
# ─────────────────────────────────────────────────────────────────────────────

class StratifiedChunkSampler:
    """
    Assigns images to chunks such that each chunk has approximately
    uniform class distribution, reducing training bias from chunking.

    Usage
    -----
    sampler = StratifiedChunkSampler(images, ann_index, categories, chunk_size)
    for chunk_imgs in sampler.chunks():
        ...
    """

    def __init__(
        self,
        images: List[ImageMeta],
        ann_index: Dict[int, List[AnnotationRecord]],
        categories: Dict[int, CategoryMeta],
        chunk_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self._images = images
        self._ann_index = ann_index
        self._categories = categories
        self._chunk_size = chunk_size
        self._shuffle = shuffle
        self._seed = seed

    def chunks(self, epoch: int = 0) -> Iterator[List[ImageMeta]]:
        """Yield image lists with stratified class distribution."""
        import random
        rng = random.Random(self._seed + epoch)

        # Group images by dominant category
        cat_buckets: Dict[int, List[ImageMeta]] = {c: [] for c in self._categories}
        uncategorised: List[ImageMeta] = []

        for img in self._images:
            anns = self._ann_index.get(img.id, [])
            if not anns:
                uncategorised.append(img)
                continue
            # Dominant category = most frequent in this image
            from collections import Counter
            cat_counts = Counter(a.category_id for a in anns)
            dominant = cat_counts.most_common(1)[0][0]
            if dominant in cat_buckets:
                cat_buckets[dominant].append(img)
            else:
                uncategorised.append(img)

        # Shuffle each bucket
        if self._shuffle:
            for bucket in cat_buckets.values():
                rng.shuffle(bucket)
            rng.shuffle(uncategorised)

        # Interleave buckets into chunks
        # Simple round-robin: take one image from each category per round
        cat_iters = {c: iter(imgs) for c, imgs in cat_buckets.items() if imgs}
        all_iterators = list(cat_iters.values()) + [iter(uncategorised)]

        chunk: List[ImageMeta] = []
        exhausted = set()

        while len(exhausted) < len(all_iterators):
            for i, it in enumerate(all_iterators):
                if i in exhausted:
                    continue
                try:
                    img = next(it)
                    chunk.append(img)
                    if len(chunk) == self._chunk_size:
                        yield chunk
                        chunk = []
                except StopIteration:
                    exhausted.add(i)

        if chunk:
            yield chunk
