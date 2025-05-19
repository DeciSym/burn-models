#!/bin/bash

# Download tokenizer files from HuggingFace
cd src/model

# Download tokenizer.json
wget https://huggingface.co/ibm-granite/granite-4.0-tiny-preview/raw/main/tokenizer.json

# Download tokenizer_config.json
wget https://huggingface.co/ibm-granite/granite-4.0-tiny-preview/raw/main/tokenizer_config.json

echo "Tokenizer files downloaded successfully!"