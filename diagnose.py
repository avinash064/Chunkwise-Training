#!/usr/bin/env python3
"""
diagnose.py
───────────
Run this ONCE before applying fixes. It will print the exact cause
of loss=1.0 and show the full state of the model, criterion, targets,
and gradient flow.

Usage:
    python diagnose.py

It builds the model, runs one forward pass on a synthetic batch,
and prints a full diagnostic report. Takes < 30 seconds.
"""
import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

# ─────────────────────────────────────────────────────────────────────────────
SEP = "=" * 65

def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def ok(msg):   print(f"  ✓  {msg}")
def warn(msg): print(f"  ⚠  {msg}")
def err(msg):  print(f"  ✗  {msg}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Import model wrapper
# ─────────────────────────────────────────────────────────────────────────────
section("1. Importing model_wrapper")
try:
    from model_wrapper import RFDETRModelWrapper, DummyCriterion, validate_img_size
    ok("model_wrapper imported")
except Exception as e:
    err(f"Import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Build model
# ─────────────────────────────────────────────────────────────────────────────
section("2. Building RFDETRModelWrapper")
NUM_CLASSES = 20
IMG_SIZE = 636
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  device = {device}")

try:
    model = RFDETRModelWrapper(num_classes=NUM_CLASSES, img_size=IMG_SIZE)
    model.to(device)
    ok(f"Model built | {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
except Exception as e:
    err(f"Model build failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Check criterion type
# ─────────────────────────────────────────────────────────────────────────────
section("3. Criterion Diagnosis (MOST IMPORTANT)")
crit = model._criterion
crit_type = type(crit).__name__
print(f"  criterion type = {crit_type}")
print(f"  criterion module = {type(crit).__module__}")

if isinstance(crit, DummyCriterion):
    err("DummyCriterion is ACTIVE — this is the cause of loss=1.0")
    err("SetCriterion was never built. See section 4 for the fix path.")
else:
    ok(f"Real criterion active: {crit_type}")
    # Print its weight attributes if it's SetCriterion
    for attr in ("weight_dict", "losses", "num_classes"):
        val = getattr(crit, attr, None)
        if val is not None:
            print(f"    criterion.{attr} = {val}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Try to build criterion manually and show which path works
# ─────────────────────────────────────────────────────────────────────────────
section("4. Criterion Build Path Diagnosis")

def try_path(label, fn):
    try:
        result = fn()
        if result is not None:
            ok(f"{label} → SUCCESS | type={type(result).__name__}")
            return result
        else:
            warn(f"{label} → returned None")
    except Exception as e:
        err(f"{label} → FAILED: {e}")
    return None

mc = getattr(getattr(model, "_model", None), "__class__", None)
trainer_mc = None

# Re-instantiate trainer to get model_config
print("  Re-instantiating RFDETRSegNano to inspect model_config ...")
try:
    from rfdetr import RFDETRSegNano
    _trainer = RFDETRSegNano(num_classes=NUM_CLASSES, resolution=IMG_SIZE)
    trainer_mc = getattr(_trainer, "model_config", None)
    print(f"  model_config type = {type(trainer_mc).__name__}")
    print(f"  model_config attrs = {[a for a in dir(trainer_mc) if not a.startswith('_')][:20]}")
except Exception as e:
    err(f"Could not re-instantiate trainer: {e}")

if trainer_mc is not None:
    working_criterion = None

    working_criterion = working_criterion or try_path(
        "Path A: _namespace_from_configs + rfdetr.models.build_model",
        lambda: _path_a(trainer_mc)
    )
    working_criterion = working_criterion or try_path(
        "Path B: rfdetr.models.lwdetr.build_model",
        lambda: _path_b(trainer_mc)
    )
    working_criterion = working_criterion or try_path(
        "Path C: rfdetr.models.criterion direct import",
        lambda: _path_c(trainer_mc)
    )
    working_criterion = working_criterion or try_path(
        "Path D: inspect trainer internals",
        lambda: _path_d(_trainer)
    )

    if working_criterion is None:
        err("ALL criterion paths failed. Running deeper inspection ...")
        _inspect_rfdetr_modules()


def _path_a(mc):
    from rfdetr._namespace import _namespace_from_configs
    from rfdetr.config import TrainConfig
    ns = _namespace_from_configs(mc, TrainConfig())
    from rfdetr.models import build_model
    _, criterion, _ = build_model(ns)
    return criterion

def _path_b(mc):
    from rfdetr._namespace import _namespace_from_configs
    from rfdetr.config import TrainConfig
    ns = _namespace_from_configs(mc, TrainConfig())
    from rfdetr.models.lwdetr import build_model
    _, criterion, _ = build_model(ns)
    return criterion

def _path_c(mc):
    # Try direct SetCriterion construction by reading model_config fields
    num_classes = getattr(mc, "num_classes", NUM_CLASSES)
    try:
        from rfdetr.models.criterion import SetCriterion, HungarianMatcher
        matcher = HungarianMatcher(
            cost_class=getattr(mc, "set_cost_class", 2.0),
            cost_bbox=getattr(mc, "set_cost_bbox", 5.0),
            cost_giou=getattr(mc, "set_cost_giou", 2.0),
        )
        weight_dict = {
            "loss_ce": getattr(mc, "cls_loss_coef", 2.0),
            "loss_bbox": getattr(mc, "bbox_loss_coef", 5.0),
            "loss_giou": getattr(mc, "giou_loss_coef", 2.0),
        }
        losses = ["labels", "boxes"]
        criterion = SetCriterion(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
        )
        return criterion
    except Exception as e:
        raise e

def _path_d(trainer):
    # Walk all attributes looking for SetCriterion
    import inspect
    for name, obj in inspect.getmembers(trainer):
        if name.startswith("_"):
            continue
        if isinstance(obj, nn.Module) and "criterion" in type(obj).__name__.lower():
            return obj
    # Check if trainer.model.model has a build method that returns criterion
    inner = getattr(getattr(trainer, "model", None), "model", None)
    if inner is not None:
        for name in ("criterion", "losses"):
            v = getattr(inner, name, None)
            if v is not None and isinstance(v, nn.Module):
                return v
    return None

def _inspect_rfdetr_modules():
    """Show all available modules in rfdetr package."""
    import rfdetr, pkgutil, importlib
    print("\n  Available rfdetr submodules:")
    for importer, name, ispkg in pkgutil.walk_packages(
        rfdetr.__path__, prefix="rfdetr.", onerror=lambda x: None
    ):
        print(f"    {name}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Check model train/eval mode
# ─────────────────────────────────────────────────────────────────────────────
section("5. Model Mode")
model.train()
training = model._model.training
if training:
    ok("Model is in train() mode")
else:
    err("Model is in EVAL mode — gradients won't flow correctly")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Check frozen parameters
# ─────────────────────────────────────────────────────────────────────────────
section("6. Parameter Requires-Grad Check")
total = 0
frozen = 0
for name, p in model._model.named_parameters():
    total += 1
    if not p.requires_grad:
        frozen += 1
        warn(f"FROZEN: {name}")

if frozen == 0:
    ok(f"All {total} parameter tensors have requires_grad=True")
else:
    err(f"{frozen}/{total} parameters are FROZEN — gradients will not flow")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Synthetic forward pass + target format check
# ─────────────────────────────────────────────────────────────────────────────
section("7. Synthetic Forward Pass")

B, C, H, W = 2, 3, IMG_SIZE, IMG_SIZE
images = torch.randn(B, C, H, W).to(device)

# Build DETR-format targets (correct format)
targets = []
for i in range(B):
    N = 3  # annotations per image
    targets.append({
        "boxes":   torch.tensor([[0.5, 0.5, 0.4, 0.4],
                                  [0.2, 0.3, 0.1, 0.2],
                                  [0.7, 0.6, 0.2, 0.3]], dtype=torch.float32).to(device),
        "labels":  torch.tensor([0, 1, 2], dtype=torch.long).to(device),
        "masks":   torch.zeros(N, H, W, dtype=torch.bool).to(device),
    })

print(f"  Input images: {images.shape}")
print(f"  Targets: {len(targets)} items")
print(f"  Target[0] boxes: {targets[0]['boxes']}")
print(f"  Target[0] labels: {targets[0]['labels']}")

try:
    from model_wrapper import to_nested_tensor
    nested = to_nested_tensor(images, device)
    ok(f"NestedTensor created: {type(nested).__name__}")
except Exception as e:
    err(f"NestedTensor creation failed: {e}")
    traceback.print_exc()

try:
    model.train()
    with torch.no_grad():
        nested2 = to_nested_tensor(images, device)
        outputs = model._model(nested2)
    print(f"\n  outputs.keys() = {list(outputs.keys())}")
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"    outputs['{k}'].shape = {v.shape}")
        elif isinstance(v, list):
            print(f"    outputs['{k}'] = list of {len(v)}")
    ok("Forward pass succeeded")
except Exception as e:
    err(f"Forward pass FAILED: {e}")
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# 8. Loss computation check
# ─────────────────────────────────────────────────────────────────────────────
section("8. Loss Computation Check")
try:
    model.train()
    nested3 = to_nested_tensor(images, device)
    outputs = model._model(nested3)

    loss_dict = model._criterion(outputs, targets)
    print(f"  loss_dict = {loss_dict}")
    total_loss = sum(loss_dict.values())
    print(f"  total loss = {total_loss.item():.6f}")

    if abs(total_loss.item() - 1.0) < 1e-5:
        err("Loss is EXACTLY 1.0 — DummyCriterion confirmed")
    elif total_loss.item() == 0.0:
        err("Loss is ZERO — targets may be empty or criterion is broken")
    else:
        ok(f"Loss is non-trivial: {total_loss.item():.4f}")

    # Check if loss has gradient
    if total_loss.requires_grad:
        ok("Loss has requires_grad=True — backward() can run")
    else:
        err("Loss has requires_grad=FALSE — backward() will fail silently")

except Exception as e:
    err(f"Loss computation FAILED: {e}")
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# 9. Gradient flow check
# ─────────────────────────────────────────────────────────────────────────────
section("9. Gradient Flow Check")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
optimizer.zero_grad()

try:
    model.train()
    nested4 = to_nested_tensor(images, device)
    outputs = model._model(nested4)
    loss_dict = model._criterion(outputs, targets)
    loss = sum(loss_dict.values())
    loss.backward()

    # Sample gradient norms
    grad_norms = []
    zero_grad_params = []
    for name, p in model._model.named_parameters():
        if p.grad is not None:
            norm = p.grad.data.norm(2).item()
            grad_norms.append(norm)
            if norm < 1e-10:
                zero_grad_params.append(name)
        else:
            zero_grad_params.append(f"{name} (no grad)")

    if grad_norms:
        print(f"  Total params with grad: {len(grad_norms)}")
        print(f"  Grad norm mean: {sum(grad_norms)/len(grad_norms):.6f}")
        print(f"  Grad norm max:  {max(grad_norms):.6f}")
        print(f"  Grad norm min:  {min(grad_norms):.6f}")
        if max(grad_norms) < 1e-7:
            err("ALL gradients are near-zero — model is not learning")
        elif len(zero_grad_params) > len(grad_norms) // 2:
            warn(f"{len(zero_grad_params)} params have zero/no grad")
        else:
            ok("Gradients flowing normally")
    else:
        err("No gradients found — backward() did not propagate")

    if zero_grad_params[:5]:
        print(f"  Zero/no-grad params (first 5): {zero_grad_params[:5]}")

except Exception as e:
    err(f"Gradient check FAILED: {e}")
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# 10. Target format validation
# ─────────────────────────────────────────────────────────────────────────────
section("10. Target Format Validation")

def validate_targets(targets, img_size):
    issues = []
    for i, t in enumerate(targets):
        boxes = t.get("boxes")
        labels = t.get("labels")
        masks = t.get("masks")

        if boxes is None:
            issues.append(f"target[{i}] missing 'boxes'")
            continue
        if labels is None:
            issues.append(f"target[{i}] missing 'labels'")

        if len(boxes) == 0:
            issues.append(f"target[{i}] has EMPTY boxes — criterion may skip")
            continue

        if boxes.shape[-1] != 4:
            issues.append(f"target[{i}] boxes has wrong shape: {boxes.shape}")

        # Check values are in [0,1] (normalised cx,cy,w,h)
        if boxes.max() > 1.01:
            issues.append(f"target[{i}] boxes NOT normalised: max={boxes.max():.2f} — must be in [0,1]")
        if boxes.min() < -0.01:
            issues.append(f"target[{i}] boxes have negative values: min={boxes.min():.2f}")

        if boxes.dtype != torch.float32:
            issues.append(f"target[{i}] boxes dtype={boxes.dtype}, expected float32")
        if labels is not None and labels.dtype != torch.long:
            issues.append(f"target[{i}] labels dtype={labels.dtype}, expected torch.long")

        # Check for NaN/Inf
        if torch.isnan(boxes).any():
            issues.append(f"target[{i}] boxes contain NaN")
        if torch.isinf(boxes).any():
            issues.append(f"target[{i}] boxes contain Inf")

    return issues

issues = validate_targets(targets, IMG_SIZE)
if not issues:
    ok("All synthetic targets are valid")
else:
    for issue in issues:
        err(issue)

print("\n" + SEP)
print("  DIAGNOSIS COMPLETE")
print(SEP)
print()
print("  Next step: run  python apply_fix.py  to patch the issue")
print()
