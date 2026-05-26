#!/usr/bin/env python3
"""Test if model_wrapper.py now builds SetCriterion correctly."""
import sys
sys.path.insert(0, '/run/user/1000/gvfs/sftp:host=192.168.31.177,user=wi/media/wi/ssd_hub/Avinash_work/rfdetr_v2')

import logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

try:
    from model_wrapper import RFDETRModelWrapper, DummyCriterion
    
    print("\n" + "="*70)
    print("TESTING FIXED model_wrapper.py")
    print("="*70)
    
    wrapper = RFDETRModelWrapper(num_classes=20, img_size=636)
    
    print("\n" + "="*70)
    print("TEST RESULT")
    print("="*70)
    
    if isinstance(wrapper._criterion, DummyCriterion):
        print("✗ FAILED: DummyCriterion still ACTIVE → loss=1.0")
        print("  The fix did not work. Run python diagnose.py")
        sys.exit(1)
    else:
        print(f"✓ SUCCESS: SetCriterion is ACTIVE")
        print(f"  Criterion type: {type(wrapper._criterion).__name__}")
        if hasattr(wrapper._criterion, "weight_dict"):
            print(f"  weight_dict: {wrapper._criterion.weight_dict}")
        print("\n✓ FIX VERIFIED - Training can now resume!")
        sys.exit(0)
        
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
