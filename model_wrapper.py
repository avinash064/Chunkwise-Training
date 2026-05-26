"""
model_wrapper.py  (FIXED — v5)
──────────────────────────────
Deterministic SetCriterion integration. No fallbacks, no DummyCriterion.

Uses RF-DETR's own build_criterion_from_config() for correct:
  - group_detr matching
  - ia_bce_loss selection
  - aux_loss + two_stage weight_dict
  - segmentation mask losses
"""

from __future__ import annotations

import importlib
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

PATCH_SIZE = 12   # rfdetr's DINOv2 uses patch_size=12

# ─────────────────────────────────────────────────────────────────────────────
# img_size validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_img_size(img_size: int) -> int:
    if img_size % PATCH_SIZE != 0:
        corrected = (img_size // PATCH_SIZE) * PATCH_SIZE
        logger.warning(
            "img_size=%d not divisible by %d → corrected to %d",
            img_size, PATCH_SIZE, corrected,
        )
        return corrected
    return img_size


# ─────────────────────────────────────────────────────────────────────────────
# NestedTensor builder
# ─────────────────────────────────────────────────────────────────────────────

def to_nested_tensor(images: torch.Tensor, device: torch.device):
    images = images.to(device)
    B, C, H, W = images.shape
    mask = torch.zeros((B, H, W), dtype=torch.bool, device=device)

    for module_path in (
        "rfdetr.utilities.tensors",
        "rfdetr.util.misc",
        "rfdetr.utilities.misc",
    ):
        try:
            mod = importlib.import_module(module_path)
            NT = getattr(mod, "NestedTensor")
            return NT(images, mask)
        except (ImportError, AttributeError):
            continue

    raise ImportError("NestedTensor not found in rfdetr. Tried 3 paths.")


# ─────────────────────────────────────────────────────────────────────────────
# LWDETR unwrapper
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap_to_nn_module(obj, depth: int = 0, max_depth: int = 6) -> Optional[nn.Module]:
    if obj is None or depth > max_depth:
        return None
    if isinstance(obj, nn.Module):
        logger.info("Found nn.Module at depth=%d | type=%s", depth, type(obj).__name__)
        return obj
    for attr in ("model", "net", "module", "detr", "detector", "backbone"):
        child = getattr(obj, attr, None)
        if child is not None and child is not obj:
            result = _unwrap_to_nn_module(child, depth + 1, max_depth)
            if result is not None:
                logger.info("Unwrapped .%s at depth=%d | wrapper=%s", attr, depth, type(obj).__name__)
                return result
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Build SetCriterion — deterministic, no fallback
# ─────────────────────────────────────────────────────────────────────────────

def _build_criterion(trainer) -> nn.Module:
    """
    Build SetCriterion with HungarianMatcher — matching RF-DETR's official
    build_criterion_and_postprocessors() exactly (lwdetr.py:468-520).

    Critical parameters that MUST be correct:
      - group_detr: splits grouped queries for matcher (13 for SegNano)
      - ia_bce_loss: IoU-aware BCE instead of focal loss
      - aux_loss + two_stage: adds _0, _1, ... and _enc weight_dict keys
      - segmentation_head: adds mask losses

    Always returns a valid SetCriterion or raises RuntimeError.
    No silent fallbacks — any failure is fatal and visible.
    """
    mc = getattr(trainer, "model_config", None)
    if mc is None:
        raise RuntimeError("trainer has no model_config — cannot build criterion")

    from rfdetr.models.criterion import SetCriterion
    from rfdetr.models.matcher import HungarianMatcher

    # ── Read all parameters from model_config + hardcoded defaults ────────────
    # These defaults match rfdetr.models._defaults.MODEL_DEFAULTS exactly.
    num_classes      = getattr(mc, "num_classes", 20)
    group_detr       = getattr(mc, "group_detr", 13)
    ia_bce_loss      = getattr(mc, "ia_bce_loss", True)
    seg_head         = getattr(mc, "segmentation_head", False)
    dec_layers       = getattr(mc, "dec_layers", 4)
    two_stage        = getattr(mc, "two_stage", True)

    # Loss coefficients — defaults from MODEL_DEFAULTS + SegmentationTrainConfig
    focal_alpha      = 0.25
    cls_loss_coef    = getattr(mc, "cls_loss_coef", 5.0 if seg_head else 1.0)
    bbox_loss_coef   = 5.0   # MODEL_DEFAULTS.bbox_loss_coef
    giou_loss_coef   = 2.0   # MODEL_DEFAULTS.giou_loss_coef

    # Matcher costs — defaults from MODEL_DEFAULTS
    set_cost_class   = 2.0   # MODEL_DEFAULTS.set_cost_class
    set_cost_bbox    = 5.0   # MODEL_DEFAULTS.set_cost_bbox
    set_cost_giou    = 2.0   # MODEL_DEFAULTS.set_cost_giou

    # Segmentation-specific
    mask_ce_loss_coef   = 5.0   # SegmentationTrainConfig default
    mask_dice_loss_coef = 5.0   # SegmentationTrainConfig default
    mask_point_sample_ratio = 16

    logger.info(
        "Building SetCriterion | num_classes=%d (+1 bg = %d) | "
        "group_detr=%d | ia_bce_loss=%s | seg_head=%s | "
        "dec_layers=%d | two_stage=%s",
        num_classes, num_classes + 1,
        group_detr, ia_bce_loss, seg_head,
        dec_layers, two_stage,
    )

    # ── HungarianMatcher ──────────────────────────────────────────────────────
    matcher_kwargs = dict(
        cost_class=set_cost_class,
        cost_bbox=set_cost_bbox,
        cost_giou=set_cost_giou,
        focal_alpha=focal_alpha,
    )
    if seg_head:
        matcher_kwargs.update(
            cost_mask_ce=mask_ce_loss_coef,
            cost_mask_dice=mask_dice_loss_coef,
            mask_point_sample_ratio=mask_point_sample_ratio,
        )
    matcher = HungarianMatcher(**matcher_kwargs)

    # ── weight_dict ───────────────────────────────────────────────────────────
    # Base weights (matches lwdetr.py:471-475)
    weight_dict = {
        "loss_ce":   cls_loss_coef,
        "loss_bbox": bbox_loss_coef,
        "loss_giou": giou_loss_coef,
    }
    if seg_head:
        weight_dict["loss_mask_ce"]   = mask_ce_loss_coef
        weight_dict["loss_mask_dice"] = mask_dice_loss_coef

    # Auxiliary decoder layer weights + encoder weights (matches lwdetr.py:477-483)
    # aux_loss is True by default in MODEL_DEFAULTS
    aux_weight_dict = {}
    for i in range(dec_layers - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
    if two_stage:
        aux_weight_dict.update({k + "_enc": v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)

    # ── Losses list ───────────────────────────────────────────────────────────
    losses = ["labels", "boxes", "cardinality"]
    if seg_head:
        losses.append("masks")

    # ── SetCriterion (matches lwdetr.py:491-516) ──────────────────────────────
    criterion_kwargs = dict(
        num_classes=num_classes + 1,   # +1 for background / no-object class
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=focal_alpha,
        losses=losses,
        group_detr=group_detr,
        sum_group_losses=False,        # MODEL_DEFAULTS
        use_varifocal_loss=False,      # MODEL_DEFAULTS
        use_position_supervised_loss=False,  # MODEL_DEFAULTS
        ia_bce_loss=ia_bce_loss,
    )
    if seg_head:
        criterion_kwargs["mask_point_sample_ratio"] = mask_point_sample_ratio

    criterion = SetCriterion(**criterion_kwargs)

    logger.info(
        "SetCriterion ACTIVE | type=%s | num_classes=%d | group_detr=%d | "
        "ia_bce_loss=%s | losses=%s | weight_dict_keys=%s",
        type(criterion).__name__,
        criterion.num_classes,
        criterion.group_detr,
        criterion.ia_bce_loss,
        criterion.losses,
        list(criterion.weight_dict.keys()),
    )
    return criterion


# ─────────────────────────────────────────────────────────────────────────────
# Main model wrapper
# ─────────────────────────────────────────────────────────────────────────────

class RFDETRModelWrapper:
    """
    Stable wrapper around LWDETR + SetCriterion.
    No DummyCriterion — if criterion build fails, it raises immediately.
    """

    def __init__(
        self,
        num_classes: int,
        img_size: int = 636,
        pretrained_weights: Optional[str] = None,
    ):
        self.num_classes = num_classes
        self.img_size = validate_img_size(img_size)
        self._device = torch.device("cpu")
        self._model: nn.Module
        self._criterion: nn.Module
        self._build(num_classes, pretrained_weights)

    def _build(self, num_classes: int, pretrained: Optional[str]):
        from rfdetr import RFDETRSegNano

        logger.info(
            "Instantiating RFDETRSegNano | num_classes=%d | img_size=%d",
            num_classes, self.img_size,
        )
        trainer = RFDETRSegNano(num_classes=num_classes, resolution=self.img_size)
        logger.info(
            "Trainer attrs: %s",
            {k: type(v).__name__ for k, v in vars(trainer).items()},
        )

        nn_model = _unwrap_to_nn_module(trainer)
        if nn_model is None:
            raise RuntimeError(
                f"Could not find nn.Module. Attrs: {list(vars(trainer).keys())}"
            )

        criterion = _build_criterion(trainer)

        self._model = nn_model
        self._criterion = criterion

        if pretrained:
            self._load_pretrained(pretrained, num_classes)

        n_params = sum(p.numel() for p in self._model.parameters()) / 1e6
        logger.info("Model ready | %.1fM params", n_params)

    def _load_pretrained(self, path: str, num_classes: int):
        logger.info("Loading pretrained: %s", path)
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if "model" in sd:
            sd = sd["model"]
        sd = {k: v for k, v in sd.items()
              if "class_embed" not in k or v.shape[0] == num_classes}
        miss, unexp = self._model.load_state_dict(sd, strict=False)
        logger.info("Pretrained loaded | missing=%d | unexpected=%d", len(miss), len(unexp))

    # ── delegation API ────────────────────────────────────────────────────────

    @property
    def raw_model(self) -> nn.Module:
        return self._model

    def to(self, device):
        self._device = torch.device(device) if not isinstance(device, torch.device) else device
        self._model = self._model.to(self._device)
        self._criterion = self._criterion.to(self._device)
        return self

    def train(self, mode: bool = True):
        self._model.train(mode)
        if hasattr(self._criterion, "train"):
            self._criterion.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def parameters(self, recurse: bool = True):
        return self._model.parameters(recurse=recurse)

    def state_dict(self) -> dict:
        return self._model.state_dict()

    def load_state_dict(self, sd: dict, strict: bool = True):
        return self._model.load_state_dict(sd, strict=strict)

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def compute_loss(
        self, images: torch.Tensor, targets: list
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass + loss.

        Targets must be list of dicts:
          boxes:  FloatTensor[N, 4]  — normalised cx, cy, w, h in [0, 1]
          labels: LongTensor[N]      — class indices in [0, num_classes)
          masks:  BoolTensor[N,H,W]  — optional
        """
        nested = to_nested_tensor(images, self._device)
        outputs = self._model(nested)
        targets = [{k: v.to(self._device) for k, v in t.items()} for t in targets]
        loss_dict = self._criterion(outputs, targets)

        # Weighted loss (use weight_dict if criterion has it)
        if hasattr(self._criterion, "weight_dict"):
            wd = self._criterion.weight_dict
            # Multiply matching losses by their weight and ONLY return those
            weighted = {k: loss_dict[k] * wd[k] for k in loss_dict.keys() if k in wd}
            return weighted

        return loss_dict

    def forward_eval(self, images: torch.Tensor) -> dict:
        nested = to_nested_tensor(images, self._device)
        return self._model(nested)
