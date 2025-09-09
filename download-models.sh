#!/bin/bash
set -e

echo "🚀 Starting model download..."
MODEL_DIR="/app/models"
MODEL_FILE="$MODEL_DIR/gte-Qwen2-1.5B-instruct-f16.gguf"
MODEL_URL="https://huggingface.co/gaianet/gte-Qwen2-1.5B-instruct-GGUF/resolve/main/gte-Qwen2-1.5B-instruct-f16.gguf"

# Create models directory if it doesn't exist
mkdir -p $MODEL_DIR

# Check if model already exists
if [ -f "$MODEL_FILE" ]; then
    echo "✅ Model already exists at $MODEL_FILE"
    echo "📊 Model size: $(du -h $MODEL_FILE | cut -f1)"
else
    echo "📥 Downloading model from $MODEL_URL..."
    
    # Download with resume support and progress
    wget -q --show-progress --continue -O "$MODEL_FILE" "$MODEL_URL"
    
    if [ $? -eq 0 ]; then
        echo "✅ Model downloaded successfully!"
        echo "📊 Model size: $(du -h $MODEL_FILE | cut -f1)"
    else
        echo "❌ Model download failed!"
        exit 1
    fi
fi

# Install WasmEdge if not present
if ! command -v wasmedge &> /dev/null; then
    echo "🔧 Installing WasmEdge..."
    curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install_v2.sh | bash -s -- -p /app/.wasmedge
    export PATH="/app/.wasmedge/bin:$PATH"
    echo 'export PATH="/app/.wasmedge/bin:$PATH"' >> ~/.bashrc
fi

echo "✅ Setup completed successfully!"