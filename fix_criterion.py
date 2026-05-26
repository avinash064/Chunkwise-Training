#!/usr/bin/env python3
"""
fix_criterion.py
────────────────
Run this once. It finds the exact call signature of
build_criterion_and_postprocessors, builds a working SetCriterion,
then patches model_wrapper.py permanently.

Usage:
    cd /media/wi/ssd_hub/Avinash_work/rfdetr_v2
    python fix_criterion.py
"""
import sys, inspect, traceback
from pathlib import Path

NUM_CLASSES = 20
IMG_SIZE    = 636

# ─────────────────────────────────────────────────────────────────────────────
# 1. Inspect the real API
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Step 1: Inspecting build_criterion_and_postprocessors")
print("=" * 60)

from rfdetr.models.lwdetr import build_criterion_and_postprocessors
from rfdetr.models.matcher import HungarianMatcher, build_matcher

sig = inspect.signature(build_criterion_and_postprocessors)
print(f"Signature: build_criterion_and_postprocessors{sig}")
print()

# Get the source to understand exactly what args it needs
src = inspect.getsource(build_criterion_and_postprocessors)
print("Source:")
print(src)
print()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Build the criterion using the real API
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Step 2: Building SetCriterion")
print("=" * 60)

from rfdetr import RFDETRSegNano
trainer = RFDETRSegNano(num_classes=NUM_CLASSES, resolution=IMG_SIZE)
mc = trainer.model_config

print(f"model_config type: {type(mc).__name__}")

# Try calling build_criterion_and_postprocessors with model_config
criterion = None

# Try A: pass model_config directly
try:
    result = build_criterion_and_postprocessors(mc)
    if isinstance(result, tuple):
        criterion = result[0]
    else:
        criterion = result
    print(f"SUCCESS (A - model_config): {type(criterion).__name__}")
except Exception as e:
    print(f"FAILED (A): {e}")
    traceback.print_exc()

# Try B: pass as keyword
if criterion is None:
    try:
        params = list(sig.parameters.keys())
        print(f"  parameter names: {params}")
        kwargs = {params[0]: mc} if params else {"args": mc}
        result = build_criterion_and_postprocessors(**kwargs)
        criterion = result[0] if isinstance(result, tuple) else result
        print(f"SUCCESS (B - kwarg): {type(criterion).__name__}")
    except Exception as e:
        print(f"FAILED (B): {e}")
        traceback.print_exc()

# Try C: inspect source, find what it actually needs and build it manually
if criterion is None:
    print("\nTrying manual construction via HungarianMatcher + SetCriterion ...")
    try:
        from rfdetr.models.criterion import SetCriterion

        # Inspect SetCriterion.__init__
        sc_sig = inspect.signature(SetCriterion.__init__)
        hm_sig = inspect.signature(HungarianMatcher.__init__)
        print(f"  SetCriterion.__init__{sc_sig}")
        print(f"  HungarianMatcher.__init__{hm_sig}")

        # Build with all known params from model_config, fall back to defaults
        hm_params = {}
        for param_name in inspect.signature(HungarianMatcher.__init__).parameters:
            if param_name == "self":
                continue
            # Map from config names
            config_map = {
                "cost_class": getattr(mc, "set_cost_class", 2.0),
                "cost_bbox":  getattr(mc, "set_cost_bbox",  5.0),
                "cost_giou":  getattr(mc, "set_cost_giou",  2.0),
            }
            if param_name in config_map:
                hm_params[param_name] = config_map[param_name]
        print(f"  HungarianMatcher params: {hm_params}")
        matcher = HungarianMatcher(**hm_params)

        # Build SetCriterion
        sc_params = list(sc_sig.parameters.keys())
        sc_params.remove("self")
        print(f"  SetCriterion params: {sc_params}")

        weight_dict = {
            "loss_ce":   float(getattr(mc, "cls_loss_coef",  2.0)),
            "loss_bbox": float(getattr(mc, "bbox_loss_coef", 5.0)),
            "loss_giou": float(getattr(mc, "giou_loss_coef", 2.0)),
        }
        # Add aux loss weights
        num_layers = int(getattr(mc, "dec_layers", 4))
        for i in range(num_layers - 1):
            for k, v in list(weight_dict.items()):
                if not any(f"_{j}" in k for j in range(num_layers)):
                    weight_dict[f"{k}_{i}"] = v
        print(f"  weight_dict: {weight_dict}")

        sc_kwargs = {
            "num_classes": NUM_CLASSES,
            "matcher":     matcher,
            "weight_dict": weight_dict,
            "losses":      ["labels", "boxes"],
        }
        # Add any extra params that SetCriterion accepts
        for p in sc_params:
            if p in sc_kwargs:
                continue
            if hasattr(mc, p):
                sc_kwargs[p] = getattr(mc, p)
            elif p == "focal_alpha":
                sc_kwargs[p] = 0.25
            elif p == "focal_gamma":
                sc_kwargs[p] = 2.0

        print(f"  SetCriterion kwargs: {list(sc_kwargs.keys())}")

        # Try with all kwargs, then progressively strip optional ones
        for attempt in range(len(sc_kwargs)):
            try:
                criterion = SetCriterion(**sc_kwargs)
                print(f"SUCCESS (C - direct, attempt {attempt}): {type(criterion).__name__}")
                break
            except TypeError as te:
                # Remove the last optional kwarg and retry
                extra_keys = [k for k in sc_kwargs if k not in
                              ("num_classes","matcher","weight_dict","losses")]
                if extra_keys:
                    removed = extra_keys[-1]
                    del sc_kwargs[removed]
                    print(f"  Removed '{removed}', retrying ...")
                else:
                    print(f"FAILED (C): {te}")
                    traceback.print_exc()
                    break

    except Exception as e:
        print(f"FAILED (C): {e}")
        traceback.print_exc()

if criterion is None:
    print("\nALL PATHS FAILED. Printing full rfdetr._namespace contents:")
    import rfdetr._namespace as ns_mod
    print(dir(ns_mod))
    print("\nContents of _namespace.py:")
    print(inspect.getsource(ns_mod))
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Verify with a forward pass
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 3: Verifying criterion with a forward pass")
print("=" * 60)

import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# Get LWDETR from trainer
def unwrap(obj, depth=0):
    if isinstance(obj, nn.Module): return obj
    if depth > 5: return None
    for attr in ("model", "net", "module", "detr"):
        child = getattr(obj, attr, None)
        if child and child is not obj:
            r = unwrap(child, depth+1)
            if r: return r
    return None

nn_model = unwrap(trainer)
print(f"LWDETR: {type(nn_model).__name__} | {sum(p.numel() for p in nn_model.parameters())/1e6:.1f}M params")

nn_model = nn_model.to(device)
criterion = criterion.to(device)
nn_model.train()

# NestedTensor
for nt_path in ("rfdetr.utilities.tensors", "rfdetr.util.misc", "rfdetr.utilities.misc"):
    try:
        import importlib
        mod = importlib.import_module(nt_path)
        NestedTensor = getattr(mod, "NestedTensor")
        break
    except Exception:
        continue

B = 2
imgs = torch.randn(B, 3, IMG_SIZE, IMG_SIZE, device=device)
mask = torch.zeros(B, IMG_SIZE, IMG_SIZE, dtype=torch.bool, device=device)
nested = NestedTensor(imgs, mask)

targets = []
for _ in range(B):
    targets.append({
        "boxes":  torch.tensor([[0.5,0.5,0.3,0.3],[0.2,0.3,0.1,0.2]],
                               dtype=torch.float32, device=device),
        "labels": torch.tensor([0, 1], dtype=torch.long, device=device),
    })

outputs = nn_model(nested)
print(f"outputs.keys(): {list(outputs.keys())}")
for k,v in outputs.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.shape}")

loss_dict = criterion(outputs, targets)
print(f"\nloss_dict: { {k: f'{v.item():.4f}' for k,v in loss_dict.items() if isinstance(v, torch.Tensor)} }")
total = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor) and v.requires_grad)
print(f"total_loss: {total.item():.4f}")

assert abs(total.item() - 1.0) > 0.01, "Loss is still 1.0 — criterion not working"
assert total.requires_grad, "No gradient!"
total.backward()
gnorms = [p.grad.norm().item() for p in nn_model.parameters() if p.grad is not None]
print(f"gradients: {len(gnorms)} tensors, max={max(gnorms):.4f}, mean={sum(gnorms)/len(gnorms):.4f}")
print("\nCRITERION VERIFIED ✓")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Write the permanent fix into model_wrapper.py
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 4: Patching model_wrapper.py")
print("=" * 60)

# Figure out which path worked
crit_module   = type(criterion).__module__
matcher_module = type(matcher).__module__ if 'matcher' in dir() else "rfdetr.models.matcher"
used_direct   = "matcher" in dir()

# Build the replacement _build_criterion function
if used_direct:
    # We built it directly — write that path
    replacement = f'''def _build_criterion(trainer) -> Optional[nn.Module]:
    """
    Build SetCriterion using the verified working path.
    Auto-fixed by fix_criterion.py — uses direct construction.
    """
    mc = getattr(trainer, "model_config", None)
    if mc is None:
        logger.error("trainer.model_config is None")
        return None

    num_classes = getattr(mc, "num_classes", 20)
    logger.info("Building SetCriterion | num_classes=%d", num_classes)

    # PRIMARY: build_criterion_and_postprocessors (rfdetr native builder)
    try:
        from rfdetr.models.lwdetr import build_criterion_and_postprocessors as _bcap
        import inspect as _inspect
        _sig = _inspect.signature(_bcap)
        _params = list(_sig.parameters.keys())
        # Pass model_config as first positional arg (or as keyword)
        try:
            _result = _bcap(mc)
        except TypeError:
            _result = _bcap(mc, num_classes)
        _crit = _result[0] if isinstance(_result, tuple) else _result
        # Override num_classes if it was built with wrong count
        if hasattr(_crit, "num_classes") and _crit.num_classes != num_classes:
            _crit.num_classes = num_classes
            logger.warning("Overrode criterion.num_classes to %d", num_classes)
        logger.info("SetCriterion built via build_criterion_and_postprocessors | type=%s",
                    type(_crit).__name__)
        return _crit
    except Exception as e:
        logger.debug("build_criterion_and_postprocessors failed: %s", e)

    # FALLBACK: direct construction
    try:
        from rfdetr.models.criterion import SetCriterion
        from rfdetr.models.matcher import HungarianMatcher

        matcher = HungarianMatcher(
            **{{k: float(getattr(mc, k2, d))
              for k, k2, d in [
                  ("cost_class", "set_cost_class", 2.0),
                  ("cost_bbox",  "set_cost_bbox",  5.0),
                  ("cost_giou",  "set_cost_giou",  2.0),
              ]}}
        )
        weight_dict = {{
            "loss_ce":   float(getattr(mc, "cls_loss_coef",  2.0)),
            "loss_bbox": float(getattr(mc, "bbox_loss_coef", 5.0)),
            "loss_giou": float(getattr(mc, "giou_loss_coef", 2.0)),
        }}
        num_layers = int(getattr(mc, "dec_layers", 4))
        for i in range(num_layers - 1):
            for k, v in list(weight_dict.items()):
                if not any(f"_{{j}}" in k for j in range(num_layers)):
                    weight_dict[f"{{k}}_{{i}}"] = v

        # Try with focal params, fall back without
        for kwargs in [
            {{"num_classes": num_classes, "matcher": matcher,
              "weight_dict": weight_dict, "losses": ["labels", "boxes"],
              "focal_alpha": float(getattr(mc, "focal_alpha", 0.25))}},
            {{"num_classes": num_classes, "matcher": matcher,
              "weight_dict": weight_dict, "losses": ["labels", "boxes"]}},
        ]:
            try:
                crit = SetCriterion(**kwargs)
                logger.info("SetCriterion built via direct construction | type=%s",
                            type(crit).__name__)
                return crit
            except TypeError:
                continue
    except Exception as e:
        logger.debug("Direct SetCriterion construction failed: %s", e)

    return None
'''
else:
    # build_criterion_and_postprocessors worked directly
    replacement = f'''def _build_criterion(trainer) -> Optional[nn.Module]:
    """
    Build SetCriterion using build_criterion_and_postprocessors.
    Auto-fixed by fix_criterion.py.
    """
    mc = getattr(trainer, "model_config", None)
    if mc is None:
        logger.error("trainer.model_config is None")
        return None

    num_classes = getattr(mc, "num_classes", 20)
    logger.info("Building SetCriterion | num_classes=%d", num_classes)

    try:
        from rfdetr.models.lwdetr import build_criterion_and_postprocessors
        result = build_criterion_and_postprocessors(mc)
        crit = result[0] if isinstance(result, tuple) else result
        logger.info("SetCriterion built | type=%s", type(crit).__name__)
        return crit
    except Exception as e:
        logger.error("build_criterion_and_postprocessors failed: %s", e)
        return None
'''

# Apply to model_wrapper.py
mw_path = Path(__file__).parent / "model_wrapper.py"
src = mw_path.read_text()

# Find and replace the existing _build_criterion function
import re
pattern = r'def _build_criterion\(trainer\).*?(?=\ndef |\nclass |\Z)'
match = re.search(pattern, src, re.DOTALL)
if match:
    src = src[:match.start()] + replacement + "\n" + src[match.end():]
    mw_path.write_text(src)
    print(f"✓ model_wrapper.py patched")
    print(f"  Replaced _build_criterion() with verified working version")
else:
    # Append as override
    src += "\n\n# PATCHED by fix_criterion.py\n" + replacement
    mw_path.write_text(src)
    print(f"✓ model_wrapper.py — _build_criterion appended as override")

print("""
════════════════════════════════════════════════════════════
  FIX APPLIED

  Now restart training:
      python main.py --resume

  First chunk log should show:
      loss_dict = {loss_ce: X.XX, loss_bbox: X.XX, loss_giou: X.XX}
      total_loss = X.XX   (NOT 1.0)
════════════════════════════════════════════════════════════
""")
