#!/usr/bin/env python3
"""
debug_criterion.py
──────────────────
Self-contained. Run from your rfdetr_v2 directory:
    python debug_criterion.py

Prints EXACTLY why SetCriterion fails and which import paths work.
"""
import sys, traceback, logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")

NUM_CLASSES = 20
IMG_SIZE    = 636

print("\n" + "="*60)
print("STEP 1: Build trainer + get model_config")
print("="*60)
from rfdetr import RFDETRSegNano
trainer = RFDETRSegNano(num_classes=NUM_CLASSES, resolution=IMG_SIZE)
mc = trainer.model_config
print(f"model_config type : {type(mc).__name__}")
print(f"model_config attrs: {[a for a in dir(mc) if not a.startswith('_')]}")

# Show key values
for attr in ("num_classes","dec_layers","set_cost_class","set_cost_bbox",
             "set_cost_giou","cls_loss_coef","bbox_loss_coef","giou_loss_coef",
             "focal_alpha","num_queries"):
    print(f"  mc.{attr} = {getattr(mc, attr, 'NOT FOUND')}")

print("\n" + "="*60)
print("STEP 2: Try _namespace_from_configs")
print("="*60)
try:
    from rfdetr._namespace import _namespace_from_configs
    from rfdetr.config import TrainConfig
    ns = _namespace_from_configs(mc, TrainConfig())
    print(f"namespace type: {type(ns).__name__}")
    print(f"namespace attrs: {[a for a in dir(ns) if not a.startswith('_')]}")
    for attr in ("num_classes","dec_layers","device"):
        print(f"  ns.{attr} = {getattr(ns, attr, 'NOT FOUND')}")
    print("SUCCESS: _namespace_from_configs works")
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()

print("\n" + "="*60)
print("STEP 3: Try rfdetr.models.build_model(ns)")
print("="*60)
try:
    from rfdetr._namespace import _namespace_from_configs
    from rfdetr.config import TrainConfig
    ns = _namespace_from_configs(mc, TrainConfig())
    ns.num_classes = NUM_CLASSES          # <-- override key field
    from rfdetr.models import build_model
    model_out, criterion, postproc = build_model(ns)
    print(f"SUCCESS: criterion={type(criterion).__name__}")
    print(f"  weight_dict={getattr(criterion,'weight_dict',None)}")
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()

print("\n" + "="*60)
print("STEP 4: Try rfdetr.models.lwdetr.build_model(ns)")
print("="*60)
try:
    from rfdetr._namespace import _namespace_from_configs
    from rfdetr.config import TrainConfig
    ns = _namespace_from_configs(mc, TrainConfig())
    ns.num_classes = NUM_CLASSES
    from rfdetr.models.lwdetr import build_model
    model_out, criterion, postproc = build_model(ns)
    print(f"SUCCESS: criterion={type(criterion).__name__}")
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()

print("\n" + "="*60)
print("STEP 5: Scan ALL rfdetr submodules for SetCriterion / HungarianMatcher")
print("="*60)
import rfdetr, pkgutil, importlib
for importer, mod_name, ispkg in pkgutil.walk_packages(
        rfdetr.__path__, prefix="rfdetr.", onerror=lambda x: None):
    try:
        mod = importlib.import_module(mod_name)
        members = [n for n in dir(mod)
                   if "criterion" in n.lower() or "matcher" in n.lower()
                   or "SetCriterion" in n or "Hungarian" in n]
        if members:
            print(f"  {mod_name}: {members}")
    except Exception:
        pass

print("\n" + "="*60)
print("STEP 6: Try direct SetCriterion import from every found location")
print("="*60)
for mod_path in ["rfdetr.models.criterion", "rfdetr.models.lwdetr",
                 "rfdetr.models.matcher", "rfdetr.models"]:
    try:
        mod = importlib.import_module(mod_path)
        SC  = getattr(mod, "SetCriterion",    None)
        HM  = getattr(mod, "HungarianMatcher", None)
        print(f"  {mod_path}: SetCriterion={SC is not None}  HungarianMatcher={HM is not None}")
        if SC and HM:
            # Try to instantiate
            try:
                matcher = HM(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
                weight_dict = {"loss_ce":2.0,"loss_bbox":5.0,"loss_giou":2.0}
                crit = SC(num_classes=NUM_CLASSES, matcher=matcher,
                          weight_dict=weight_dict, losses=["labels","boxes"])
                print(f"    INSTANTIATION SUCCESS: {type(crit).__name__}")
            except Exception as e:
                print(f"    INSTANTIATION FAILED: {e}")
                traceback.print_exc()
    except Exception as e:
        print(f"  {mod_path}: import failed — {e}")

print("\n" + "="*60)
print("STEP 7: Walk trainer for any nn.Module that looks like a criterion")
print("="*60)
import torch.nn as nn, inspect
def walk(obj, prefix="trainer", depth=0):
    if depth > 3: return
    for name in dir(obj):
        if name.startswith("__"): continue
        try:
            val = getattr(obj, name)
        except Exception:
            continue
        if isinstance(val, nn.Module):
            print(f"  {prefix}.{name} = {type(val).__name__}")
            if depth < 2:
                walk(val, f"{prefix}.{name}", depth+1)

walk(trainer)
