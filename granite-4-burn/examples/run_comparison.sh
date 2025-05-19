#!/bin/bash

# Script to run HuggingFace capture then Burn comparison

echo "Step 1: Capturing HuggingFace outputs..."
cd /home/aac/projects/burn-models/granite-4-burn/examples
python3 capture_hf_outputs.py

echo -e "\nStep 2: Running Burn layer comparison..."
cd /home/aac/projects/burn-models/granite-4-burn
cargo run --example debug_layer_comparison --features tch-gpu

echo -e "\nComparison complete! Check the output above for differences."