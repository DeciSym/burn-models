#!/usr/bin/env python3
import torch
import sys
sys.path.insert(0, '/home/aac/src/transformers/src')

# Check if mamba2 dependencies are available
try:
    from transformers.utils.import_utils import is_causal_conv1d_available, is_mamba_2_ssm_available
    print(f"is_causal_conv1d_available: {is_causal_conv1d_available()}")
    print(f"is_mamba_2_ssm_available: {is_mamba_2_ssm_available()}")
    
    # Check what imports are actually available
    try:
        from mamba_ssm.ops.triton.selective_state_update import selective_state_update
        print("selective_state_update: Available")
    except ImportError as e:
        print(f"selective_state_update: Not available - {e}")
        
    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined
        print("mamba_chunk_scan_combined: Available")
        print("mamba_split_conv1d_scan_combined: Available")
    except ImportError as e:
        print(f"mamba_ssm ops: Not available - {e}")
        
    try:
        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
        print("causal_conv1d_fn: Available")
        print("causal_conv1d_update: Available")
    except ImportError as e:
        print(f"causal_conv1d: Not available - {e}")
        
    # Check is_fast_path_available
    from transformers.models.mamba2.modeling_mamba2 import is_fast_path_available
    print(f"\nis_fast_path_available: {is_fast_path_available}")
    
    # Check if CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        
    # Test which forward method would be used
    from transformers import Mamba2Config, Mamba2Model
    config = Mamba2Config.from_pretrained("AntonV/mamba2-130m-hf")
    model = Mamba2Model(config)
    
    if torch.cuda.is_available():
        model = model.cuda()
        device_type = model.embeddings.weight.device.type
        print(f"\nModel on device: {model.embeddings.weight.device}")
        print(f"Device type: {device_type}")
        print(f"'cuda' in device_type: {'cuda' in device_type}")
        
        # Check what forward method would be used
        mixer = model.layers[0].mixer
        print(f"\nMixer in_proj device: {mixer.in_proj.weight.device}")
        print(f"Would use cuda_kernels_forward: {is_fast_path_available and 'cuda' in mixer.in_proj.weight.device.type}")
    else:
        print("\nCUDA not available - would use torch_forward")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()