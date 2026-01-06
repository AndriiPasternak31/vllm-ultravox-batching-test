#!/bin/bash
# Install vLLM with the variable-length audio batching fix

set -e

echo "=========================================="
echo "Installing vLLM with audio batching fix"
echo "=========================================="

# Step 1: Install pre-built vLLM first
echo ""
echo "Step 1: Installing pre-built vLLM..."
pip uninstall vllm -y 2>/dev/null || true
pip install vllm

VLLM_PATH=$(python -c "import vllm; print(vllm.__path__[0])")
echo "vLLM installed at: $VLLM_PATH"

# Step 2: Clone the fix branch
echo ""
echo "Step 2: Cloning fix branch..."
TEMP_DIR=$(mktemp -d)
git clone --depth 1 -b fix/ultravox-batching-v0.13 \
    https://github.com/AndriiPasternak31/vllm.git "$TEMP_DIR/vllm-fix"

# Step 3: Copy fixed files over the installed vLLM
echo ""
echo "Step 3: Applying fix patches..."
cp "$TEMP_DIR/vllm-fix/vllm/multimodal/inputs.py" "$VLLM_PATH/multimodal/inputs.py"
cp "$TEMP_DIR/vllm-fix/vllm/model_executor/models/ultravox.py" "$VLLM_PATH/model_executor/models/ultravox.py"

echo "Patched files:"
echo "  - $VLLM_PATH/multimodal/inputs.py"
echo "  - $VLLM_PATH/model_executor/models/ultravox.py"

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo "=========================================="
echo "Done! vLLM installed with audio fix."
echo "=========================================="
echo ""
echo "Verify the fix is applied:"
echo '  python -c "from vllm.multimodal.inputs import MultiModalListField; print(\"Fix applied\")"'
