#!/bin/bash
# Start vLLM server with Ultravox model

echo "Starting vLLM server with Ultravox..."
echo "This may take a few minutes to download the model on first run."
echo ""

vllm serve fixie-ai/ultravox-v0_6-llama-3_1-8b \
    --trust-remote-code \
    --max-model-len 8192 \
    --host 0.0.0.0 \
    --port 8000
