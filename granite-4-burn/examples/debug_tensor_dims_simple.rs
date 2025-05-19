use burn::prelude::*;
use burn_ndarray::{NdArray, NdArrayDevice};

fn main() {
    let device = NdArrayDevice::default();
    let batch_size = 2;
    let d_inner = 3;
    let d_state = 4;

    // Test sum_dim output shape
    let h = Tensor::<NdArray, 3>::zeros([batch_size, d_inner, d_state], &device);
    let c_expanded = Tensor::<NdArray, 3>::ones([batch_size, d_inner, d_state], &device);
    let y_from_h = (h * c_expanded).sum_dim(2);
    println!("y_from_h shape after sum_dim: {:?}", y_from_h.dims());

    // Test element-wise multiplication
    let x_t = Tensor::<NdArray, 2>::ones([batch_size, d_inner], &device);
    let d = Tensor::<NdArray, 1>::ones([d_inner], &device);
    let d_expanded = d.reshape([1, d_inner]).expand([batch_size, d_inner]);
    println!("d_expanded shape: {:?}", d_expanded.dims());
    
    let y_from_x = d_expanded * x_t.clone();
    println!("y_from_x shape: {:?}", y_from_x.dims());
    
    // Now try to add them
    println!("Attempting to add {:?} + {:?}", y_from_h.dims(), y_from_x.dims());
}