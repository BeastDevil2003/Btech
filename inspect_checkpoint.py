"""
Checkpoint Architecture Inspector
===================================
Load and analyze the saved checkpoint to determine:
1. What model architecture was used
2. What resolution (14x14 vs 28x28)
3. Compare with current code architecture
"""

import torch
import os
from collections import defaultdict

def inspect_checkpoint():
    checkpoint_path = "checkpoints/best_model.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return
    
    print("\n" + "="*80)
    print("CHECKPOINT ARCHITECTURE ANALYSIS")
    print("="*80)
    
    # Count layers by component
    components = defaultdict(list)
    for key in state_dict.keys():
        if "backbone" in key:
            components["Backbone"].append(key)
        elif "spatial_branch" in key:
            components["Spatial Branch"].append(key)
        elif "frequency_branch" in key:
            components["Frequency Branch"].append(key)
        elif "noise_branch" in key:
            components["Noise Branch"].append(key)
        elif "cgaf" in key or "fusion" in key:
            components["CGAF Fusion"].append(key)
        elif "temporal" in key:
            components["Temporal Model"].append(key)
        elif "classification_head" in key:
            components["Classification Head"].append(key)
        elif "localization_head" in key:
            components["Localization Head"].append(key)
        else:
            components["Other"].append(key)
    
    print("\n📊 COMPONENT BREAKDOWN:")
    print("-" * 80)
    for comp, layers in sorted(components.items()):
        print(f"  {comp:<30}: {len(layers):>3} layers")
    
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"\n  {'TOTAL PARAMETERS':<30}: {total_params:>10,}")
    
    # Analyze key layers to determine resolution
    print("\n" + "="*80)
    print("RESOLUTION DETECTION (14×14 vs 28×28)")
    print("="*80)
    
    # Check spatial branch adapter
    spatial_adapter_keys = [k for k in state_dict.keys() if "spatial_branch" in k and "adapter" in k]
    if spatial_adapter_keys:
        print(f"\n🔍 Spatial Branch Adapter:")
        for key in sorted(spatial_adapter_keys)[:3]:
            shape = state_dict[key].shape
            print(f"  {key:<50}: {shape}")
    
    # Check frequency branch adapter
    freq_adapter_keys = [k for k in state_dict.keys() if "frequency_branch" in k and "adapter" in k]
    if freq_adapter_keys:
        print(f"\n🔍 Frequency Branch Adapter:")
        for key in sorted(freq_adapter_keys)[:3]:
            shape = state_dict[key].shape
            print(f"  {key:<50}: {shape}")
    
    # Check localization head to determine decoder steps
    loc_head_keys = sorted([k for k in state_dict.keys() if "localization_head" in k])
    if loc_head_keys:
        print(f"\n🔍 Localization Head (Decoder) Layers:")
        up_layers = [k for k in loc_head_keys if "up" in k]
        for i, key in enumerate(sorted(up_layers), 1):
            shape = state_dict[key].shape if len(state_dict[key].shape) > 1 else "bias"
            print(f"  Step {i}: {key:<45}: {shape}")
        
        print(f"\n  Total decoder steps: {len(up_layers)}")
        if len(up_layers) == 4:
            print(f"  ➜ Interpretation: 4 steps (14×14 → 28 → 56 → 112 → 224)")
            print(f"     This checkpoint is from OLD 14×14 ARCHITECTURE")
        elif len(up_layers) == 3:
            print(f"  ➜ Interpretation: 3 steps (28×28 → 56 → 112 → 224)")
            print(f"     This checkpoint is from NEW 28×28 ARCHITECTURE")
    
    # Analyze CGAF fusion weights
    cgaf_keys = [k for k in state_dict.keys() if "cgaf" in k or "fusion" in k]
    if cgaf_keys:
        print(f"\n🔍 CGAF Fusion Weights:")
        for key in sorted(cgaf_keys)[:5]:
            shape = state_dict[key].shape
            print(f"  {key:<50}: {shape}")
    
    # Show sample tensor values
    print("\n" + "="*80)
    print("SAMPLE PARAMETER VALUES (First 5 keys)")
    print("="*80)
    for i, (key, tensor) in enumerate(list(state_dict.items())[:5]):
        print(f"\n{i+1}. {key}")
        print(f"   Shape: {tensor.shape}")
        print(f"   Dtype: {tensor.dtype}")
        print(f"   Min/Max: [{tensor.min():.4f}, {tensor.max():.4f}]")
        print(f"   Mean: {tensor.mean():.4f}, Std: {tensor.std():.4f}")

def compare_with_current_model():
    """Load current model and compare architectures"""
    print("\n\n" + "="*80)
    print("CURRENT CODE ARCHITECTURE")
    print("="*80)
    
    try:
        from models.afag_net_v3 import AFAGNetV3
        model = AFAGNetV3()
        current_state = model.state_dict()
        
        print(f"\n📊 COMPONENT BREAKDOWN (Current Code):")
        print("-" * 80)
        
        components = defaultdict(list)
        for key in current_state.keys():
            if "backbone" in key:
                components["Backbone"].append(key)
            elif "spatial_branch" in key:
                components["Spatial Branch"].append(key)
            elif "frequency_branch" in key:
                components["Frequency Branch"].append(key)
            elif "noise_branch" in key:
                components["Noise Branch"].append(key)
            elif "cgaf" in key or "fusion" in key:
                components["CGAF Fusion"].append(key)
            elif "temporal" in key:
                components["Temporal Model"].append(key)
            elif "classification_head" in key:
                components["Classification Head"].append(key)
            elif "localization_head" in key:
                components["Localization Head"].append(key)
            else:
                components["Other"].append(key)
        
        for comp, layers in sorted(components.items()):
            print(f"  {comp:<30}: {len(layers):>3} layers")
        
        total_params = sum(p.numel() for p in current_state.values())
        print(f"\n  {'TOTAL PARAMETERS':<30}: {total_params:>10,}")
        
        # Check decoder steps
        loc_head_keys = sorted([k for k in current_state.keys() if "localization_head" in k])
        up_layers = [k for k in loc_head_keys if "up" in k]
        
        print(f"\n🔍 Localization Head (Decoder) - Current Code:")
        for i, key in enumerate(sorted(up_layers), 1):
            shape = current_state[key].shape if len(current_state[key].shape) > 1 else "bias"
            print(f"  Step {i}: {key:<45}: {shape}")
        
        print(f"\n  Total decoder steps: {len(up_layers)}")
        if len(up_layers) == 3:
            print(f"  ➜ Current code: 28×28 ARCHITECTURE (3 steps)")
        elif len(up_layers) == 4:
            print(f"  ➜ Current code: 14×14 ARCHITECTURE (4 steps)")
        
    except Exception as e:
        print(f"❌ Error loading current model: {e}")

def detailed_comparison():
    """Detailed comparison of checkpoint vs current code"""
    print("\n\n" + "="*80)
    print("ARCHITECTURE COMPATIBILITY CHECK")
    print("="*80)
    
    checkpoint_path = "checkpoints/best_model.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    try:
        checkpoint_state = torch.load(checkpoint_path, map_location="cpu")
        from models.afag_net_v3 import AFAGNetV3
        
        current_model = AFAGNetV3()
        current_state = current_model.state_dict()
        
        print("\n🔄 LAYER COMPATIBILITY:")
        print("-" * 80)
        
        # Find matching, missing, and extra layers
        matching = []
        missing = []
        shape_mismatch = []
        extra = []
        
        for key in checkpoint_state.keys():
            if key not in current_state:
                missing.append(key)
            elif checkpoint_state[key].shape != current_state[key].shape:
                shape_mismatch.append((key, checkpoint_state[key].shape, current_state[key].shape))
            else:
                matching.append(key)
        
        for key in current_state.keys():
            if key not in checkpoint_state:
                extra.append(key)
        
        print(f"\n✅ Matching layers:      {len(matching)}")
        print(f"❌ Missing layers:       {len(missing)}")
        print(f"⚠️  Shape mismatches:    {len(shape_mismatch)}")
        print(f"🆕 New in current code:  {len(extra)}")
        
        print(f"\n{'TOTAL CHECKPOINT PARAMS:':<30} {len(checkpoint_state)}")
        print(f"{'TOTAL CURRENT CODE PARAMS:':<30} {len(current_state)}")
        
        if len(shape_mismatch) > 0:
            print(f"\n⚠️  SHAPE MISMATCHES DETECTED:")
            print("-" * 80)
            for key, old_shape, new_shape in shape_mismatch[:10]:
                print(f"  {key:<50}")
                print(f"    Checkpoint: {old_shape}")
                print(f"    Current:    {new_shape}")
        
        if len(missing) > 0:
            print(f"\n❌ LAYERS MISSING FROM CHECKPOINT:")
            print("-" * 80)
            for key in missing[:10]:
                print(f"  {key}")
            if len(missing) > 10:
                print(f"  ... and {len(missing)-10} more")
        
        if len(extra) > 0:
            print(f"\n🆕 NEW LAYERS IN CURRENT CODE:")
            print("-" * 80)
            for key in extra[:10]:
                print(f"  {key}")
            if len(extra) > 10:
                print(f"  ... and {len(extra)-10} more")
        
        # Verdict
        print("\n" + "="*80)
        print("VERDICT:")
        print("="*80)
        
        if len(shape_mismatch) == 0 and len(missing) == 0 and len(extra) == 0:
            print("✅ CHECKPOINT IS FULLY COMPATIBLE with current code!")
            print("   All layers match perfectly.")
        elif len(shape_mismatch) > 0:
            print("⚠️  CHECKPOINT HAS ARCHITECTURE MISMATCHES")
            print(f"   {len(shape_mismatch)} layers have shape differences")
            print("   Safe loading will load only compatible parameters")
            compatible_count = len(matching)
            total_ckpt = len(checkpoint_state)
            print(f"   Expected compatible load: {compatible_count}/{total_ckpt} ({100*compatible_count/total_ckpt:.1f}%)")
        else:
            print("✅ CHECKPOINT IS COMPATIBLE with current code")
        
    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("\n🔍 DEEPFAKE MODEL CHECKPOINT ANALYZER\n")
    print("This script inspects your saved checkpoint and determines:")
    print("  1. What model architecture was saved")
    print("  2. What resolution (14×14 vs 28×28)")
    print("  3. Compatibility with current code\n")
    
    inspect_checkpoint()
    compare_with_current_model()
    detailed_comparison()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")
