#!/usr/bin/env python3
"""
Debug why criterion building fails in model_wrapper._build_criterion
"""
import sys
import logging
import traceback
import importlib

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Get the model_config
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("Step 1: Creating RFDETRSegNano to get model_config")
print("=" * 70)

from rfdetr import RFDETRSegNano
trainer = RFDETRSegNano(num_classes=20, resolution=636)
mc = trainer.model_config

print(f"model_config type: {type(mc).__name__}")
print(f"num_classes: {getattr(mc, 'num_classes', 'NOT FOUND')}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Test Path A from model_wrapper._build_criterion
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Step 2: Testing Path A (_namespace_from_configs + build_model)")
print("=" * 70)

for model_module in ("rfdetr.models", "rfdetr.models.lwdetr"):
    print(f"\nTrying {model_module}...")
    try:
        from rfdetr._namespace import _namespace_from_configs
        from rfdetr.config import TrainConfig
        print("  ✓ Imports succeeded")
        
        ns = _namespace_from_configs(mc, TrainConfig())
        print(f"  ✓ _namespace_from_configs created ns with num_classes={getattr(ns, 'num_classes', 'NOT FOUND')}")
        
        # Override num_classes
        ns.num_classes = mc.num_classes
        print(f"  ✓ Overrode ns.num_classes to {ns.num_classes}")
        
        mod = importlib.import_module(model_module)
        bm = getattr(mod, "build_model", None)
        if bm is None:
            print(f"  ✗ build_model not found in {model_module}")
            continue
        
        _, criterion, _ = bm(ns)
        print(f"  ✓ Path A SUCCESS: {type(criterion).__name__}")
        sys.exit(0)
        
    except Exception as e:
        print(f"  ✗ Path A failed: {type(e).__name__}: {e}")
        # Don't print full traceback, just continue

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Test Path B from model_wrapper._build_criterion  
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Step 3: Testing Path B (direct SetCriterion + HungarianMatcher)")
print("=" * 70)

for crit_module in ("rfdetr.models.criterion", "rfdetr.models.lwdetr", "rfdetr.models.set_criterion"):
    print(f"\nTrying {crit_module}...")
    try:
        mod = importlib.import_module(crit_module)
        SetCriterion = getattr(mod, "SetCriterion", None)
        HungarianMatcher = getattr(mod, "HungarianMatcher", None)
        
        if SetCriterion is None:
            print(f"  ✗ SetCriterion not in {crit_module}")
            continue
        if HungarianMatcher is None:
            print(f"  ✗ HungarianMatcher not in {crit_module}")
            continue
        
        print(f"  ✓ Found both SetCriterion and HungarianMatcher")
        
        # Try to create HungarianMatcher
        print(f"  Creating HungarianMatcher...")
        try:
            matcher = HungarianMatcher(
                cost_class=float(getattr(mc, "set_cost_class", 2.0)),
                cost_bbox=float(getattr(mc, "set_cost_bbox", 5.0)),
                cost_giou=float(getattr(mc, "set_cost_giou", 2.0)),
            )
            print(f"    ✓ HungarianMatcher created")
        except Exception as e:
            print(f"    ✗ HungarianMatcher creation failed: {e}")
            raise
        
        # Build weight_dict
        print(f"  Building weight_dict...")
        weight_dict = {
            "loss_ce":   float(getattr(mc, "cls_loss_coef",  2.0)),
            "loss_bbox": float(getattr(mc, "bbox_loss_coef", 5.0)),
            "loss_giou": float(getattr(mc, "giou_loss_coef", 2.0)),
        }
        print(f"    weight_dict (base): {weight_dict}")
        
        # Aux loss weights
        num_layers = int(getattr(mc, "dec_layers", 6))
        print(f"    num_layers: {num_layers}")
        for i in range(num_layers - 1):
            for k, v in list(weight_dict.items()):
                if not any(f"_{j}" in k for j in range(num_layers)):
                    weight_dict[f"{k}_{i}"] = v
        print(f"    weight_dict (with aux): {weight_dict}")
        
        # Try to create SetCriterion
        losses = ["labels", "boxes"]
        focal_alpha = getattr(mc, "focal_alpha", None)
        
        print(f"  Creating SetCriterion (focal_alpha={focal_alpha})...")
        if focal_alpha is not None:
            print(f"    Trying WITH focal_alpha...")
            try:
                criterion = SetCriterion(
                    num_classes=20,
                    matcher=matcher,
                    weight_dict=weight_dict,
                    losses=losses,
                    focal_alpha=float(focal_alpha),
                )
                print(f"    ✓ SetCriterion created WITH focal_alpha")
            except TypeError as te:
                print(f"    ✗ TypeError with focal_alpha: {te}")
                print(f"    Trying WITHOUT focal_alpha...")
                criterion = SetCriterion(
                    num_classes=20,
                    matcher=matcher,
                    weight_dict=weight_dict,
                    losses=losses,
                )
                print(f"    ✓ SetCriterion created WITHOUT focal_alpha")
        else:
            print(f"    Trying WITHOUT focal_alpha (since focal_alpha is None)...")
            criterion = SetCriterion(
                num_classes=20,
                matcher=matcher,
                weight_dict=weight_dict,
                losses=losses,
            )
            print(f"    ✓ SetCriterion created")
        
        print(f"  ✓ Path B SUCCESS: {type(criterion).__name__}")
        print(f"\nSetCriterion attributes:")
        print(f"  - weight_dict: {getattr(criterion, 'weight_dict', 'NOT FOUND')}")
        print(f"  - num_classes: {getattr(criterion, 'num_classes', 'NOT FOUND')}")
        sys.exit(0)
        
    except Exception as e:
        print(f"  ✗ Path B ({crit_module}) failed:")
        traceback.print_exc()
        print()

print("\n" + "=" * 70)
print("ALL PATHS FAILED")
print("=" * 70)
