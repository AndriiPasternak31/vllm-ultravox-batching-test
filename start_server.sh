#!/bin/bash
# Start vLLM server with Ultravox model

# Use /workspace for HuggingFace cache if available (more disk space)
if [ -d "/workspace" ]; then
    export HF_HOME=/workspace/.hf_home
fi

echo "Starting vLLM server with Ultravox (1B model)..."
echo "This may take a few minutes to download the model on first run."
echo ""

vllm serve fixie-ai/ultravox-v0_5-llama-3_2-1b \
    --trust-remote-code \
    --max-model-len 8192 \
    --host 0.0.0.0 \
    --port 8000
