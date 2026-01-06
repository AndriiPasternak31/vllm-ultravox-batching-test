#!/bin/bash
# Install vLLM with the variable-length audio batching fix

set -e

echo "=========================================="
echo "Installing vLLM with audio batching fix"
echo "=========================================="

# Step 1: Clone the fix branch first to check its vLLM version
echo ""
echo "Step 1: Cloning fix branch..."
TEMP_DIR=$(mktemp -d)
git clone --depth 1 -b fix/ultravox-batching-v2 \
    https://github.com/AndriiPasternak31/vllm.git "$TEMP_DIR/vllm-fix"

# Step 2: Install the SAME version of vLLM as the fix branch
echo ""
echo "Step 2: Installing vLLM from fix branch..."
cd "$TEMP_DIR/vllm-fix"

# Try to install without building (Python-only mode)
export VLLM_USE_PRECOMPILED=1
pip install -e . --no-build-isolation 2>/dev/null || {
    echo "Editable install failed, trying regular install..."
    # Fall back to installing pre-built vLLM and copying files
    pip install vllm

    VLLM_PATH=$(python -c "import vllm; print(vllm.__path__[0])")
    echo "vLLM installed at: $VLLM_PATH"

    echo "Copying fixed multimodal directory..."
    cp -r "$TEMP_DIR/vllm-fix/vllm/multimodal/"* "$VLLM_PATH/multimodal/"
    cp "$TEMP_DIR/vllm-fix/vllm/model_executor/models/ultravox.py" "$VLLM_PATH/model_executor/models/ultravox.py"
}

cd -

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo "=========================================="
echo "Done! vLLM installed with audio fix."
echo "=========================================="
echo ""
echo "Verify the fix is applied:"
echo '  python -c "from vllm.multimodal.inputs import MultiModalListField; print(\"Fix applied\")"'
