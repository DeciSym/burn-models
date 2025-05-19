use burn::prelude::*;
use burn_models::model::mamba_selective_scan_fixed::SelectiveScan;

#[test]
fn test_selective_scan_dimensions() {
    type B = burn::backend::NdArray;
    let device = <B as Backend>::Device::default();
    
    // Set up dimensions to match Granite-4 configuration
    let batch_size = 1;
    let seq_len = 5;
    let d_inner = 2048; // mamba_intermediate
    let d_state = 16;   // mamba_d_state
    
    // Create test tensors
    let x = Tensor::<B, 3>::zeros([batch_size, seq_len, d_inner], &device);
    let delta = Tensor::<B, 3>::zeros([batch_size, seq_len, d_inner], &device);
    let a_log = Tensor::<B, 1>::zeros([d_inner], &device);
    let b = Tensor::<B, 3>::zeros([batch_size, seq_len, d_state], &device);
    let c = Tensor::<B, 3>::zeros([batch_size, seq_len, d_state], &device);
    let d = Tensor::<B, 1>::zeros([d_inner], &device);
    
    // Call forward
    let result = SelectiveScan::forward(x, delta, a_log, b, c, d);
    
    // Check output shape
    assert_eq!(result.dims(), [batch_size, seq_len, d_inner]);
}