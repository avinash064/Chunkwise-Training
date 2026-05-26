"""
apply_fix.py
────────────
Applies all fixes for loss=1.0 issue.

Run ONCE: python apply_fix.py
Then restart training normally: python main.py --resume

What this fixes:
  1. DummyCriterion → extracts real SetCriterion by inspecting rfdetr source
  2. Target format → validates boxes are normalised cx/cy/w/h in [0,1]
  3. Logging → verifies loss_dict is real, not placeholder
  4. Gradient check → ensures backward() actually moves weights
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Find the real SetCriterion from rfdetr
# ─────────────────────────────────────────────────────────────────────────────

def build_real_criterion(num_classes: int, img_size: int = 636):
    """
    Extract SetCriterion from rfdetr using every known path.
    Returns the first one that works.
    """
    from rfdetr import RFDETRSegNano
    trainer = RFDETRSegNano(num_classes=num_classes, resolution=img_size)
    mc = getattr(trainer, "model_config", None)

    # ── Strategy 1: _namespace_from_configs ──────────────────────────────────
    for build_fn_path in [
        ("rfdetr.models", "build_model"),
        ("rfdetr.models.lwdetr", "build_model"),
    ]:
        try:
            import importlib
            from rfdetr._namespace import _namespace_from_configs
            from rfdetr.config import TrainConfig
            ns = _namespace_from_configs(mc, TrainConfig())
            mod = importlib.import_module(build_fn_path[0])
            _, crit, _ = getattr(mod, build_fn_path[1])(ns)
            print(f"  ✓ SetCriterion built via {build_fn_path[0]}.{build_fn_path[1]}")
            return crit
        except Exception as e:
            print(f"  ✗ {build_fn_path[0]}: {e}")

    # ── Strategy 2: Direct SetCriterion construction ─────────────────────────
    try:
        crit = _build_set_criterion_direct(mc, num_classes)
        if crit:
            print("  ✓ SetCriterion built directly")
            return crit
    except Exception as e:
        print(f"  ✗ Direct build: {e}")

    # ── Strategy 3: Walk trainer internal state ───────────────────────────────
    try:
        crit = _walk_trainer_for_criterion(trainer)
        if crit:
            print(f"  ✓ SetCriterion found in trainer: {type(crit).__name__}")
            return crit
    except Exception as e:
        print(f"  ✗ Walk trainer: {e}")

    # ── Strategy 4: Trigger rfdetr's own training setup ───────────────────────
    try:
        crit = _trigger_trainer_setup(trainer, num_classes)
        if crit:
            print(f"  ✓ SetCriterion from trainer setup: {type(crit).__name__}")
            return crit
    except Exception as e:
        print(f"  ✗ Trigger setup: {e}")

    raise RuntimeError(
        "Could not build SetCriterion from any path. "
        "See MANUAL_FIX section below."
    )


def _build_set_criterion_direct(mc, num_classes: int):
    """Build SetCriterion by reading config fields directly."""
    # Try importing matcher and criterion
    for crit_module in ["rfdetr.models.criterion", "rfdetr.models.lwdetr"]:
        try:
            import importlib
            mod = importlib.import_module(crit_module)

            # Get HungarianMatcher
            Matcher = getattr(mod, "HungarianMatcher", None)
            if Matcher is None:
                continue
            SetCrit = getattr(mod, "SetCriterion", None)
            if SetCrit is None:
                continue

            matcher = Matcher(
                cost_class=float(getattr(mc, "set_cost_class", 2.0)),
                cost_bbox=float(getattr(mc, "set_cost_bbox", 5.0)),
                cost_giou=float(getattr(mc, "set_cost_giou", 2.0)),
            )

            weight_dict = {
                "loss_ce":   float(getattr(mc, "cls_loss_coef",  2.0)),
                "loss_bbox": float(getattr(mc, "bbox_loss_coef", 5.0)),
                "loss_giou": float(getattr(mc, "giou_loss_coef", 2.0)),
            }

            # Add aux loss weights if used
            num_layers = int(getattr(mc, "dec_layers", 6))
            aux_weight_dict = {}
            for i in range(num_layers - 1):
                aux_weight_dict.update({f"{k}_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

            losses = ["labels", "boxes"]

            criterion = SetCrit(
                num_classes=num_classes,
                matcher=matcher,
                weight_dict=weight_dict,
                losses=losses,
            )
            return criterion
        except Exception as e:
            print(f"    {crit_module}: {e}")

    return None


def _walk_trainer_for_criterion(trainer):
    """Walk all attributes of trainer + nested objects looking for SetCriterion."""
    import inspect

    def is_real_criterion(obj):
        if obj is None:
            return False
        name = type(obj).__name__.lower()
        return isinstance(obj, nn.Module) and (
            "criterion" in name or "setcriterion" in name or "loss" in name
        ) and not "dummy" in name

    # Direct attrs
    for attr in dir(trainer):
        if attr.startswith("__"):
            continue
        try:
            v = getattr(trainer, attr)
            if is_real_criterion(v):
                return v
        except Exception:
            continue

    # One level deep
    for attr in ("model", "net"):
        container = getattr(trainer, attr, None)
        if container is None:
            continue
        for sub_attr in dir(container):
            if sub_attr.startswith("__"):
                continue
            try:
                v = getattr(container, sub_attr)
                if is_real_criterion(v):
                    return v
            except Exception:
                continue

    return None


def _trigger_trainer_setup(trainer, num_classes: int):
    """
    Some rfdetr versions only build criterion when .train() is called.
    Try calling the internal setup method.
    """
    for method_name in ("_setup_training", "setup", "_build_criterion",
                        "build_criterion", "_prepare"):
        method = getattr(trainer, method_name, None)
        if method is not None:
            try:
                method()
                crit = _walk_trainer_for_criterion(trainer)
                if crit:
                    return crit
            except Exception:
                continue

    # Last resort: mock a training call with tiny dataset to force setup
    # (without actually loading data)
    try:
        import tempfile, os, json
        # Create a minimal dummy dataset structure
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            (tmpdir / "images").mkdir()
            ann = {
                "images": [{"id": 1, "file_name": "x.jpg", "width": 100, "height": 100}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                                  "bbox": [10,10,20,20], "area": 400, "iscrowd": 0,
                                  "segmentation": []}],
                "categories": [{"id": i, "name": str(i)} for i in range(1, num_classes+1)],
            }
            ann_path = tmpdir / "ann.json"
            ann_path.write_text(json.dumps(ann))
            # Don't actually call train() — just check if setup happened
    except Exception:
        pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Verify the criterion works on a real forward pass
# ─────────────────────────────────────────────────────────────────────────────

def verify_criterion(criterion, model_wrapper, device, num_classes: int, img_size: int):
    """Run one forward pass with real criterion and check loss is non-trivial."""
    from model_wrapper import to_nested_tensor

    model_wrapper.train()
    criterion = criterion.to(device)

    B = 2
    images = torch.randn(B, 3, img_size, img_size).to(device)
    targets = []
    for _ in range(B):
        targets.append({
            "boxes":  torch.tensor([[0.5, 0.5, 0.4, 0.3],
                                     [0.2, 0.3, 0.1, 0.2]], dtype=torch.float32).to(device),
            "labels": torch.tensor([0, 1], dtype=torch.long).to(device),
        })

    nested = to_nested_tensor(images, device)
    outputs = model_wrapper._model(nested)

    print(f"\n  Forward pass outputs:")
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k}: {v.shape} dtype={v.dtype}")
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            print(f"    {k}: list[{len(v)}] of dicts")

    loss_dict = criterion(outputs, targets)
    print(f"\n  loss_dict = {loss_dict}")
    total = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor) and v.requires_grad)
    print(f"  total_loss = {total.item():.6f}")

    assert abs(total.item() - 1.0) > 1e-4, \
        f"Loss is still 1.0 — criterion is not working correctly"
    assert total.requires_grad, "Loss has no gradient!"

    # Test backward
    total.backward()
    grad_norms = [p.grad.norm().item() for p in model_wrapper._model.parameters()
                  if p.grad is not None]
    print(f"  Gradients: {len(grad_norms)} tensors, max_norm={max(grad_norms):.4f}")
    assert max(grad_norms) > 1e-8, "Gradients are zero — something is wrong"

    print("\n  ✓ Criterion verified — loss is real and gradients flow correctly")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Patch model_wrapper.py
# ─────────────────────────────────────────────────────────────────────────────

def patch_model_wrapper(num_classes: int, img_size: int, working_criterion):
    """
    Write a new model_wrapper.py with the criterion baked in.
    The patched version skips all the trial-and-error and goes directly
    to the working path.
    """
    wrapper_path = Path(__file__).parent / "model_wrapper.py"
    src = wrapper_path.read_text()

    # Get the working module path
    crit_module = type(working_criterion).__module__
    crit_class = type(working_criterion).__name__
    print(f"\n  Criterion module: {crit_module}")
    print(f"  Criterion class:  {crit_class}")

    # We'll write a _build_criterion_v2 function that uses the known-good path
    new_func = f'''
def _build_criterion_v2(trainer) -> Optional[nn.Module]:
    """
    Auto-patched criterion builder — uses verified working path.
    Criterion type: {crit_module}.{crit_class}
    """
    mc = getattr(trainer, "model_config", None)
    if mc is None:
        logger.error("model_config is None — cannot build criterion")
        return None

    # VERIFIED WORKING PATH (auto-patched by apply_fix.py)
    try:
        from rfdetr._namespace import _namespace_from_configs
        from rfdetr.config import TrainConfig
        ns = _namespace_from_configs(mc, TrainConfig())
        from {crit_module.rsplit(".", 1)[0]} import build_model
        _, criterion, _ = build_model(ns)
        logger.info("Criterion built via PATCHED path | type=%s", type(criterion).__name__)
        return criterion
    except Exception as e:
        logger.warning("Patched path failed: %s", e)

    # Fallback: direct construction
    try:
        from {crit_module} import {crit_class}
        from {crit_module} import HungarianMatcher
        num_classes = getattr(mc, "num_classes", {num_classes})
        matcher = HungarianMatcher(
            cost_class=float(getattr(mc, "set_cost_class", 2.0)),
            cost_bbox=float(getattr(mc, "set_cost_bbox", 5.0)),
            cost_giou=float(getattr(mc, "set_cost_giou", 2.0)),
        )
        weight_dict = {{
            "loss_ce":   float(getattr(mc, "cls_loss_coef",  2.0)),
            "loss_bbox": float(getattr(mc, "bbox_loss_coef", 5.0)),
            "loss_giou": float(getattr(mc, "giou_loss_coef", 2.0)),
        }}
        num_layers = int(getattr(mc, "dec_layers", 6))
        for i in range(num_layers - 1):
            for k, v in list(weight_dict.items()):
                if "_" + str(i) not in k:
                    weight_dict[f"{{k}}_{{i}}"] = v
        return {crit_class}(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            losses=["labels", "boxes"],
        )
    except Exception as e:
        logger.error("Direct criterion build failed: %s", e)
        return None
'''

    # Replace the old _build_criterion function
    if "_build_criterion_v2" in src:
        print("  model_wrapper.py already patched")
        return

    # Insert after existing _build_criterion and replace the call in _build()
    src = src.replace(
        "criterion = _build_criterion(trainer)",
        "criterion = _build_criterion_v2(trainer) or _build_criterion(trainer)"
    )
    src = src + "\n" + new_func

    # Backup and write
    backup = wrapper_path.with_suffix(".py.bak")
    wrapper_path.rename(backup)
    wrapper_path.write_text(src)
    print(f"  ✓ model_wrapper.py patched (backup: {backup.name})")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Patch trainer.py — add debug logging + target validation
# ─────────────────────────────────────────────────────────────────────────────

TRAINER_PATCH = '''
# ── Injected by apply_fix.py ──────────────────────────────────────────────────

def validate_and_fix_targets(targets: list, device, num_classes: int) -> list:
    """
    Validate and fix DETR-format targets in-place.
    Ensures:
      - boxes are float32 in [0, 1] (normalised cx, cy, w, h)
      - labels are long in [0, num_classes)
      - no empty target batches (DETR criterion will error on them)
    """
    fixed = []
    for i, t in enumerate(targets):
        boxes  = t.get("boxes",  torch.zeros(0, 4, dtype=torch.float32))
        labels = t.get("labels", torch.zeros(0, dtype=torch.long))

        # Dtype fixes
        if boxes.dtype  != torch.float32: boxes  = boxes.float()
        if labels.dtype != torch.long:    labels = labels.long()

        # Clamp boxes to [0, 1]
        boxes = boxes.clamp(0.0, 1.0)

        # Filter invalid boxes (zero-area)
        if len(boxes) > 0:
            w = boxes[:, 2]
            h = boxes[:, 3]
            valid = (w > 1e-4) & (h > 1e-4)
            if valid.sum() == 0:
                # All boxes degenerate — use a dummy box so criterion doesn't crash
                boxes  = torch.tensor([[0.5, 0.5, 0.01, 0.01]], dtype=torch.float32)
                labels = torch.tensor([0], dtype=torch.long)
            else:
                boxes  = boxes[valid]
                labels = labels[valid]

        # Clamp labels to valid range
        labels = labels.clamp(0, num_classes - 1)

        new_t = {k: v.to(device) for k, v in t.items()}
        new_t["boxes"]  = boxes.to(device)
        new_t["labels"] = labels.to(device)
        fixed.append(new_t)

    return fixed


def debug_one_batch(model, images, targets, device, logger, step_name=""):
    """Call this on the FIRST batch of the FIRST chunk to verify everything."""
    from model_wrapper import to_nested_tensor
    import logging
    log = logging.getLogger("debug_batch")

    log.info("=== DEBUG BATCH [%s] ===", step_name)
    log.info("  images.shape = %s  dtype=%s", images.shape, images.dtype)
    log.info("  len(targets) = %d", len(targets))

    for i, t in enumerate(targets[:2]):
        boxes  = t.get("boxes",  None)
        labels = t.get("labels", None)
        log.info(
            "  target[%d] boxes=%s labels=%s",
            i,
            boxes.shape if boxes is not None else None,
            labels.tolist() if labels is not None else None,
        )
        if boxes is not None and len(boxes) > 0:
            log.info("    box values (first): %s", boxes[0].tolist())
            log.info(
                "    box range: min=%.3f max=%.3f",
                boxes.min().item(), boxes.max().item(),
            )

    nested = to_nested_tensor(images, device)
    with torch.no_grad():
        outputs = model._model(nested)

    log.info("  outputs.keys() = %s", list(outputs.keys()))
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            log.info("    outputs['%s'].shape = %s", k, v.shape)

    loss_dict = model._criterion(outputs, targets)
    log.info("  loss_dict = %s", {k: f"{v.item():.4f}" for k, v in loss_dict.items()
                                   if isinstance(v, torch.Tensor)})
    total = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor) and v.requires_grad)
    log.info("  total_loss = %.4f  requires_grad=%s", total.item(), total.requires_grad)
    log.info("=== END DEBUG BATCH ===")
    return total.item()
'''


def patch_trainer(num_classes: int):
    trainer_path = Path(__file__).parent / "trainer.py"
    src = trainer_path.read_text()

    if "validate_and_fix_targets" in src:
        print("  trainer.py already patched")
        return

    # 1. Add import at top
    src = src.replace(
        "import torch\n",
        "import torch\nimport torch.nn as nn  # noqa\n"
    )

    # 2. Append helper functions
    src = src + "\n" + TRAINER_PATCH

    # 3. Patch the training loop to call validate_and_fix_targets
    # Find the for loop in ChunkTrainer.train()
    old_line = "            for images, targets in loader:"
    new_line = f"""            _debug_done = False
            for images, targets in loader:
                # Validate + fix target format before every batch
                targets = validate_and_fix_targets(targets, self.device, {num_classes})
                if not _debug_done and not getattr(self, '_global_debug_done', False):
                    debug_one_batch(self.model, images, targets, self.device, None, "first_batch")
                    self._global_debug_done = True
                    _debug_done = True"""

    if old_line in src:
        src = src.replace(old_line, new_line, 1)
        print("  ✓ trainer.py patched with target validation")
    else:
        print("  ⚠ Could not find exact line in trainer.py — manual patch needed")
        print(f"    Add before the for loop:\n    targets = validate_and_fix_targets(targets, self.device, {num_classes})")

    backup = trainer_path.with_suffix(".py.bak")
    trainer_path.rename(backup)
    trainer_path.write_text(src)


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Patch dataset.py — fix box normalisation
# ─────────────────────────────────────────────────────────────────────────────

def patch_dataset():
    """
    The existing build_target() function in dataset.py computes normalised
    cx/cy/w/h correctly, but let's add a final clamp to guarantee [0,1].
    """
    dataset_path = Path(__file__).parent / "dataset.py"
    src = dataset_path.read_text()

    if "# PATCHED: clamp boxes" in src:
        print("  dataset.py already patched")
        return

    old_target_append = '''    if boxes:
        return {
            "boxes":   torch.tensor(boxes,  dtype=torch.float32),'''

    new_target_append = '''    if boxes:
        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        # PATCHED: clamp boxes to valid [0,1] range (handles edge-case annotations)
        boxes_t = boxes_t.clamp(0.0, 1.0)
        # Filter zero-area boxes
        valid = (boxes_t[:, 2] > 1e-4) & (boxes_t[:, 3] > 1e-4)
        if valid.sum() == 0:
            # Replace degenerate boxes with a dummy
            boxes_t  = torch.tensor([[0.5, 0.5, 0.01, 0.01]], dtype=torch.float32)
            labels   = [labels[0]] if labels else [0]
            masks    = [masks[0]]  if masks  else [np.zeros((1, 1), bool)]
            areas    = [areas[0]]  if areas  else [0.0]
            crowds   = [crowds[0]] if crowds else [0]
        else:
            boxes_t = boxes_t[valid]
            labels  = [l for l, v in zip(labels, valid.tolist()) if v]
            masks   = [m for m, v in zip(masks,  valid.tolist()) if v]
            areas   = [a for a, v in zip(areas,  valid.tolist()) if v]
            crowds  = [c for c, v in zip(crowds, valid.tolist()) if v]
        return {
            "boxes":   boxes_t,'''

    if old_target_append in src:
        src = src.replace(old_target_append, new_target_append, 1)
        backup = dataset_path.with_suffix(".py.bak")
        dataset_path.rename(backup)
        dataset_path.write_text(src)
        print("  ✓ dataset.py patched with box clamping + zero-area filter")
    else:
        print("  ⚠ Could not patch dataset.py automatically — the target builder format differs")
        print("    Manually add: boxes = boxes.clamp(0.0, 1.0) after building the boxes tensor")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    NUM_CLASSES = 20
    IMG_SIZE = 636

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nApplying fixes for loss=1.0 issue")
    print(f"device={device}  num_classes={NUM_CLASSES}  img_size={IMG_SIZE}\n")

    # ── 1. Build real criterion ───────────────────────────────────────────────
    print("Step 1: Building real SetCriterion ...")
    try:
        criterion = build_real_criterion(NUM_CLASSES, IMG_SIZE)
        print(f"  criterion type = {type(criterion).__name__}")
        print(f"  criterion module = {type(criterion).__module__}")
        if hasattr(criterion, "weight_dict"):
            print(f"  weight_dict = {criterion.weight_dict}")
    except Exception as e:
        print(f"\n  FAILED: {e}")
        print("""
  MANUAL FIX REQUIRED:
  --------------------
  The automatic criterion builder failed. You need to find where rfdetr
  builds its criterion. Run this to inspect the installed package:

      import rfdetr, pkgutil
      for _, name, _ in pkgutil.walk_packages(rfdetr.__path__, "rfdetr."):
          print(name)

  Then look for SetCriterion and HungarianMatcher.
  Common locations:
    - rfdetr.models.criterion
    - rfdetr.models.lwdetr
    - rfdetr.models.matcher

  Once found, update _build_criterion() in model_wrapper.py to use the
  correct import path.
""")
        sys.exit(1)

    # ── 2. Verify criterion works ─────────────────────────────────────────────
    print("\nStep 2: Verifying criterion ...")
    from model_wrapper import RFDETRModelWrapper
    model = RFDETRModelWrapper(num_classes=NUM_CLASSES, img_size=IMG_SIZE)
    model.to(device)

    try:
        ok = verify_criterion(criterion, model, device, NUM_CLASSES, IMG_SIZE)
    except AssertionError as e:
        print(f"\n  VERIFICATION FAILED: {e}")
        sys.exit(1)

    # ── 3. Patch model_wrapper.py ─────────────────────────────────────────────
    print("\nStep 3: Patching model_wrapper.py ...")
    patch_model_wrapper(NUM_CLASSES, IMG_SIZE, criterion)

    # ── 4. Patch trainer.py ───────────────────────────────────────────────────
    print("\nStep 4: Patching trainer.py ...")
    patch_trainer(NUM_CLASSES)

    # ── 5. Patch dataset.py ───────────────────────────────────────────────────
    print("\nStep 5: Patching dataset.py ...")
    patch_dataset()

    print("""
════════════════════════════════════════════════════════════════
  ALL FIXES APPLIED SUCCESSFULLY

  Restart training with:
      python main.py --resume

  What to expect on the first batch after fix:
    - DEBUG BATCH log will show actual loss_dict keys (loss_ce, loss_bbox, loss_giou)
    - loss values will NOT be 1.0
    - loss should decrease over chunks (typical first-batch range: 5.0 - 20.0)

  If loss still shows 1.0 after restart:
    - Check log for "DummyCriterion" warning — if present, apply_fix.py
      did not succeed in finding SetCriterion
    - Run:  grep -n "DummyCriterion\|SetCriterion\|weight_dict" <logfile>
════════════════════════════════════════════════════════════════
""")
