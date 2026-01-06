#!/bin/bash
# Install vLLM with the variable-length audio batching fix

set -e

echo "=========================================="
echo "Installing vLLM with audio batching fix"
echo "=========================================="

# Step 1: Install pre-built vLLM (avoids building from source)
echo ""
echo "Step 1: Installing pre-built vLLM..."
pip install vllm

# Step 2: Clone the fix and copy only the changed Python files
echo ""
echo "Step 2: Applying the fix..."
TEMP_DIR=$(mktemp -d)
git clone --depth 1 -b fix/ultravox-variable-length-audio-batching \
    https://github.com/AndriiPasternak31/vllm.git "$TEMP_DIR/vllm-fix"

# Find where vllm is installed
VLLM_PATH=$(python -c "import vllm; print(vllm.__path__[0])")
echo "vLLM installed at: $VLLM_PATH"

# Copy the fixed files
echo "Copying fixed files..."
cp "$TEMP_DIR/vllm-fix/vllm/multimodal/inputs.py" "$VLLM_PATH/multimodal/inputs.py"
cp "$TEMP_DIR/vllm-fix/vllm/model_executor/models/ultravox.py" "$VLLM_PATH/model_executor/models/ultravox.py"

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo "=========================================="
echo "Done! vLLM installed with audio fix."
echo "=========================================="
echo ""
echo "Verify the fix is applied:"
echo "  python -c \"from vllm.multimodal.inputs import MultiModalListField; print('Fix applied!')\""
