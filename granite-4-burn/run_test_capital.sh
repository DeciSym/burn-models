#!/bin/bash

# Run the capital test with a 5-minute timeout
echo "Running capital of France test with 5-minute timeout..."

# Set environment variables
export RUST_BACKTRACE=1

# Run the test with timeout
timeout --preserve-status 300s cargo run --release --features tch-gpu --example test_capital_paris

exit_code=$?

if [ $exit_code -eq 124 ]; then
    echo "Test timed out after 5 minutes"
elif [ $exit_code -eq 0 ]; then
    echo "Test completed successfully"
else
    echo "Test failed with exit code: $exit_code"
fi

exit $exit_code