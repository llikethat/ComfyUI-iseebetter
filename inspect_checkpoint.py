"""
Script to inspect iSeeBetter checkpoint structure.
Run this in ComfyUI to see exactly what keys and shapes are in your checkpoint.

Usage: Place in custom_nodes folder and run once, or run from Python:
    python inspect_checkpoint.py /path/to/your/model.pth
"""

import torch
import sys

def inspect_checkpoint(path):
    print(f"Loading checkpoint: {path}")
    state_dict = torch.load(path, map_location='cpu')
    
    # Handle DataParallel wrapper
    if any(k.startswith('module.') for k in state_dict.keys()):
        print("Note: Checkpoint uses DataParallel wrapper (module. prefix)")
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
        state_dict = new_state_dict
    
    print(f"\nTotal keys: {len(state_dict)}")
    print("\n" + "="*80)
    print("KEY STRUCTURE AND SHAPES:")
    print("="*80)
    
    # Group by prefix
    prefixes = {}
    for key in sorted(state_dict.keys()):
        prefix = key.split('.')[0]
        if prefix not in prefixes:
            prefixes[prefix] = []
        prefixes[prefix].append(key)
    
    for prefix in sorted(prefixes.keys()):
        print(f"\n--- {prefix} ---")
        for key in prefixes[prefix]:
            shape = tuple(state_dict[key].shape)
            print(f"  {key}: {shape}")
    
    # Try to detect parameters
    print("\n" + "="*80)
    print("DETECTED PARAMETERS:")
    print("="*80)
    
    # Detect scale factor from kernel sizes
    for key in state_dict.keys():
        if 'deconv.weight' in key:
            shape = state_dict[key].shape
            if len(shape) == 4:
                kernel = shape[2]
                if kernel == 6:
                    print(f"Scale factor: 2 (kernel={kernel} in {key})")
                elif kernel == 8:
                    print(f"Scale factor: 4 (kernel={kernel} in {key})")
                elif kernel == 12:
                    print(f"Scale factor: 8 (kernel={kernel} in {key})")
                break
    
    # Detect nFrames from output layer
    if 'output.conv.weight' in state_dict:
        in_ch = state_dict['output.conv.weight'].shape[1]
        print(f"Output layer input channels: {in_ch}")
        print(f"If feat=64, nFrames = {in_ch // 64 + 1}")
    
    # Detect PReLU type
    for key in state_dict.keys():
        if '.act.weight' in key:
            shape = state_dict[key].shape
            if len(shape) == 1:
                if shape[0] == 1:
                    print(f"PReLU type: shared (num_parameters=1)")
                else:
                    print(f"PReLU type: per-channel (num_parameters={shape[0]})")
            break

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_checkpoint(sys.argv[1])
    else:
        print("Usage: python inspect_checkpoint.py /path/to/model.pth")
