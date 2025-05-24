use burn::backend::libtorch::LibTorchDevice;

/// Automatically selects the best available device (GPU if available, otherwise CPU)
pub fn auto_device() -> LibTorchDevice {
    // Try to check if CUDA is available using burn-tch
    #[cfg(feature = "tch-gpu")]
    {
        // When tch-gpu feature is enabled, try to use GPU
        match std::panic::catch_unwind(|| {
            // Try to create a CUDA device - if it fails, CUDA is not available
            let _test = LibTorchDevice::Cuda(0);
            true
        }) {
            Ok(_) => {
                println!("CUDA is available, using GPU");
                return LibTorchDevice::Cuda(0);
            }
            Err(_) => {
                println!("CUDA not available, using CPU");
                return LibTorchDevice::Cpu;
            }
        }
    }
    
    #[cfg(not(feature = "tch-gpu"))]
    {
        println!("GPU support not compiled, using CPU");
        LibTorchDevice::Cpu
    }
}

/// Create a device based on preference with automatic fallback
pub fn get_device(prefer_gpu: bool) -> LibTorchDevice {
    if prefer_gpu {
        auto_device()
    } else {
        println!("Using CPU");
        LibTorchDevice::Cpu
    }
}