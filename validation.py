"""
validation.py
─────────────
Memory-safe validation loop.
Streams val.json per chunk using StreamingCOCOParserV2 (no full load).
Computes COCO mAP if pycocotools is available, otherwise reports loss.
"""

from __future__ import annotations

import gc
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from async_pipeline import DataLoaderFactory
from dataset import COCOChunkDataset, build_category_id_map, build_val_transforms
from model_wrapper import RFDETRModelWrapper
from streaming_coco_v2 import StreamingCOCOParserV2

logger = logging.getLogger(__name__)

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    _HAS_COCO_TOOLS = True
except ImportError:
    _HAS_COCO_TOOLS = False
    logger.warning("pycocotools not installed — COCO mAP evaluation unavailable")


class Validator:
    """
    Streaming validator: loads val annotations per chunk, never all at once.

    Usage
    -----
    validator = Validator(val_parser, val_img_dir, num_classes=20)
    metrics = validator.run(model, epoch=0, max_chunks=None)
    # returns {"val_loss": ..., "AP": ..., "AP50": ..., ...}
    """

    def __init__(
        self,
        val_parser: StreamingCOCOParserV2,
        val_img_dir: str | Path,
        num_classes: int,
        img_size: int = 636,
        batch_size: int = 8,
        num_workers: int = 2,
        chunk_size: int = 300,
        device: torch.device = torch.device("cpu"),
    ):
        self.parser = val_parser
        self.img_dir = Path(val_img_dir)
        self.num_classes = num_classes
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.device = device
        self._cat_map = build_category_id_map(val_parser.categories)
        self._loader_factory = DataLoaderFactory(
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2,
        )

    @torch.no_grad()
    def run(
        self,
        model: RFDETRModelWrapper,
        epoch: int = 0,
        max_chunks: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on the validation set.
        Returns dict of metrics (val_loss always present; COCO metrics if pycocotools available).
        """
        model.eval()
        t_start = time.perf_counter()

        all_losses: List[float] = []
        all_predictions: List[dict] = []    # for COCO eval
        all_targets_for_coco: List[dict] = []

        image_list = list(self.parser.images.values())
        transforms = build_val_transforms(self.img_size)
        chunks_processed = 0

        for start in range(0, len(image_list), self.chunk_size):
            if max_chunks is not None and chunks_processed >= max_chunks:
                break

            chunk_imgs = image_list[start: start + self.chunk_size]
            chunk_ids = {img.id for img in chunk_imgs}
            chunk_anns = self.parser.load_annotations_for_images(chunk_ids)

            dataset = COCOChunkDataset(
                images=chunk_imgs,
                ann_index=chunk_anns,
                img_dir=self.img_dir,
                category_id_map=self._cat_map,
                img_size=self.img_size,
                transforms=transforms,
            )

            loader = self._loader_factory.build(
                dataset,
                batch_size=self.batch_size,
                collate_fn=COCOChunkDataset.collate_fn,
                shuffle=False,
            )

            for images, targets in loader:
                images = images.to(self.device)

                # Compute loss
                try:
                    with torch.amp.autocast("cuda") if self.device.type == "cuda" else _nullctx():
                        loss_dict = model.compute_loss(images, targets)
                    loss = sum(loss_dict.values()).item()
                    all_losses.append(loss)
                except Exception as e:
                    logger.debug("Val loss failed: %s", e)

                # Collect predictions for COCO eval
                if _HAS_COCO_TOOLS:
                    try:
                        outputs = model.forward_eval(images)
                        preds = _decode_predictions(outputs, targets, self._cat_map)
                        all_predictions.extend(preds)
                    except Exception as e:
                        logger.debug("Prediction decode failed: %s", e)

            del dataset
            gc.collect()
            chunks_processed += 1

        # Aggregate metrics
        metrics: Dict[str, float] = {}
        metrics["val_loss"] = sum(all_losses) / max(len(all_losses), 1)
        metrics["val_chunks"] = float(chunks_processed)
        elapsed = time.perf_counter() - t_start
        metrics["val_time_s"] = elapsed

        # COCO mAP
        if _HAS_COCO_TOOLS and all_predictions:
            coco_metrics = self._compute_coco_map(all_predictions)
            metrics.update(coco_metrics)

        logger.info(
            "Validation E%d | val_loss=%.4f | AP=%.3f | %.1fs | %d chunks",
            epoch,
            metrics["val_loss"],
            metrics.get("AP", 0.0),
            elapsed,
            chunks_processed,
        )
        return metrics

    def _compute_coco_map(self, predictions: List[dict]) -> Dict[str, float]:
        """Run COCOeval on collected predictions."""
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(predictions, f)
                pred_path = f.name

            # Build gt COCO object from val parser
            gt_data = {
                "images": [{"id": img.id} for img in self.parser.images.values()],
                "categories": [{"id": c.id, "name": c.name} for c in self.parser.categories.values()],
                "annotations": [],
            }
            coco_gt = COCO()
            coco_gt.dataset = gt_data
            coco_gt.createIndex()

            coco_dt = coco_gt.loadRes(pred_path)
            evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            stats = evaluator.stats  # [AP, AP50, AP75, APs, APm, APl, ...]
            return {
                "AP": float(stats[0]),
                "AP50": float(stats[1]),
                "AP75": float(stats[2]),
                "APs": float(stats[3]),
                "APm": float(stats[4]),
                "APl": float(stats[5]),
            }
        except Exception as e:
            logger.warning("COCO eval failed: %s", e)
            return {}


def _decode_predictions(outputs: dict, targets: list, cat_map: Dict[int, int]) -> List[dict]:
    """
    Convert model outputs to COCO prediction format.
    Assumes outputs["pred_boxes"] (cx,cy,w,h normalised) and outputs["pred_logits"].
    """
    predictions = []
    reverse_cat_map = {v: k for k, v in cat_map.items()}

    pred_logits = outputs.get("pred_logits")
    pred_boxes = outputs.get("pred_boxes")
    if pred_logits is None or pred_boxes is None:
        return predictions

    scores = torch.softmax(pred_logits, dim=-1)[..., :-1]  # drop bg class
    for i, target in enumerate(targets):
        img_id = int(target["image_id"][0]) if "image_id" in target else i
        sc = scores[i]
        bx = pred_boxes[i]
        conf, cls = sc.max(dim=-1)
        for j in range(len(conf)):
            if float(conf[j]) < 0.05:
                continue
            cx, cy, w, h = bx[j].tolist()
            x = (cx - w / 2) * 640
            y = (cy - h / 2) * 640
            predictions.append({
                "image_id": img_id,
                "category_id": reverse_cat_map.get(int(cls[j]), 1),
                "bbox": [x, y, w * 640, h * 640],
                "score": float(conf[j]),
            })
    return predictions


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): pass
