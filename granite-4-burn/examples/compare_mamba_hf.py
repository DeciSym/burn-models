#!/usr/bin/env python3
"""
Compare our Mamba implementation with HuggingFace's to understand B and C matrices
"""

import torch
from transformers.models.granite_moe_hybrid import GraniteMoeHybridMamba


def analyze_mamba_layer():
    """Analyze how HuggingFace Granite Mamba layer handles B and C matrices"""
    
    # Configuration from our config.json
    config = {
        "hidden_size": 1536,
        "mamba_expand": 2,
        "mamba_d_conv": 4,
        "mamba_d_state": 128,
        "mamba_d_head": 64,
        "mamba_n_heads": 48,
        "mamba_conv_bias": True,
        "mamba_proj_bias": False,
    }
    
    # Create a Mamba layer
    mamba_intermediate = config["mamba_expand"] * config["hidden_size"]  # 3072
    
    print("Mamba configuration:")
    print(f"  hidden_size: {config['hidden_size']}")
    print(f"  mamba_intermediate: {mamba_intermediate}")
    print(f"  mamba_d_state: {config['mamba_d_state']}")
    print(f"  mamba_n_heads: {config['mamba_n_heads']}")
    print(f"  mamba_d_head: {config['mamba_d_head']}")
    
    # Check the actual Granite implementation
    import inspect
    try:
        # Try to get the GraniteMoeHybridMamba source
        source = inspect.getsource(GraniteMoeHybridMamba)
        print("\nFound GraniteMoeHybridMamba source")
        
        # Look for x_proj weight definition
        if "x_proj" in source:
            print("\nFound x_proj in source - this is likely used to compute B and C")
            
        # Look for selective scan usage
        if "selective_scan" in source or "ssm" in source:
            print("Found selective scan or SSM in source")
            
    except Exception as e:
        print(f"Could not get source: {e}")
        print("Let's check the model weights instead")
    
    # Load model to examine weights
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            "ibm-granite/granite-4.0-tiny-preview",
            device_map="cpu",
            torch_dtype=torch.bfloat16
        )
        
        # Get first Mamba layer (layer 0)
        first_layer = model.model.layers[0]
        if hasattr(first_layer, 'temporal_block'):
            mamba = first_layer.temporal_block
            print("\nMamba layer weights:")
            for name, param in mamba.named_parameters():
                print(f"  {name}: {param.shape}")
                
            # Check for x_proj specifically
            if hasattr(mamba, 'x_proj'):
                x_proj_weight = mamba.x_proj.weight
                print(f"\nx_proj weight shape: {x_proj_weight.shape}")
                # Expected: [dt_rank + 2*d_state, mamba_intermediate]
                dt_rank = 96  # Based on dt_out_channels = 48
                expected_out = dt_rank + 2 * config["mamba_d_state"]
                print(f"Expected x_proj output dim: {expected_out} (dt_rank={dt_rank} + 2*d_state={2*config['mamba_d_state']})")
                
                if x_proj_weight.shape[0] == expected_out:
                    print("x_proj is used to compute dt, B, and C matrices!")
                    
    except Exception as e:
        print(f"Could not load model: {e}")


def create_simple_test():
    """Create a simple test case to understand the computation"""
    
    batch = 1
    seq_len = 2
    d_inner = 4
    d_state = 2
    
    # Create simple inputs
    x = torch.randn(batch, seq_len, d_inner)
    delta = torch.ones(batch, seq_len, d_inner) * 0.5
    a_log = torch.ones(d_inner) * -1.0  # A = exp(-1) ≈ 0.368
    
    # In HuggingFace, B and C are computed from x through x_proj
    # x_proj output is split into: dt, B, C
    # Let's simulate this:
    
    # Normally x_proj would compute these, but for testing let's create them
    B = torch.randn(batch, seq_len, d_state) * 0.1
    C = torch.randn(batch, seq_len, d_state) * 0.1
    D = torch.ones(d_inner) * 0.5
    
    print("\nSimple test case:")
    print(f"x shape: {x.shape}")
    print(f"B shape: {B.shape}")
    print(f"C shape: {C.shape}")
    print(f"delta shape: {delta.shape}")
    print(f"a_log shape: {a_log.shape}")
    print(f"D shape: {D.shape}")
    
    # Manual SSM computation
    A = torch.exp(a_log)
    h = torch.zeros(batch, d_inner, d_state)
    outputs = []
    
    for t in range(seq_len):
        x_t = x[:, t]  # [batch, d_inner]
        delta_t = delta[:, t]  # [batch, d_inner]
        B_t = B[:, t]  # [batch, d_state]
        C_t = C[:, t]  # [batch, d_state]
        
        # Discretize A: A_bar = exp(delta * log(A))
        A_bar = torch.exp(delta_t.unsqueeze(-1) * a_log.unsqueeze(0).unsqueeze(-1))
        
        # Update state: h = A_bar * h + delta * B * x
        # Correct: h has shape [batch, d_inner, d_state]
        h = A_bar * h
        
        # Add input contribution
        # x_t: [batch, d_inner]
        # B_t: [batch, d_state]
        # We need to compute outer product for each batch
        delta_B_x = delta_t.unsqueeze(-1) * x_t.unsqueeze(-1) * B_t.unsqueeze(1)
        h = h + delta_B_x
        
        # Compute output: y = C * h + D * x
        # h: [batch, d_inner, d_state]
        # C_t: [batch, d_state]
        y_t = torch.sum(h * C_t.unsqueeze(1), dim=-1) + D.unsqueeze(0) * x_t
        
        outputs.append(y_t)
        
    output = torch.stack(outputs, dim=1)
    print(f"\nFinal output shape: {output.shape}")
    print(f"Final h shape: {h.shape}")
    

if __name__ == "__main__":
    print("Analyzing Mamba implementation...")
    analyze_mamba_layer()
    
    print("\n" + "="*50 + "\n")
    create_simple_test()