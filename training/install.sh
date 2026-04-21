set -e

echo "============================================================"
echo "Elara — Environment Setup"
echo "============================================================"

pip install --upgrade pip setuptools wheel

# Unsloth — no-deps prevents overwriting existing PyTorch/CUDA
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --no-deps

# xformers — pre-built wheel for CUDA 12.x
pip install xformers --index-url https://download.pytorch.org/whl/cu121 --no-deps

# Training stack — isolated from PyTorch
pip install peft==0.10.0 --no-deps
pip install trl==0.8.6 --no-deps
pip install accelerate==0.29.3 --no-deps

# Required runtime dependencies (safe, no PyTorch conflict)
pip install datasets transformers sentencepiece protobuf \
    bitsandbytes triton packaging einops scipy

# Fix tyro version requirement from trl
pip install "tyro>=0.5.11"

# Unsloth companion package
pip install unsloth_zoo

echo ""
echo "============================================================"
echo "Verifying installation..."
echo "============================================================"

python -c "
import torch, unsloth, xformers
print('PyTorch :', torch.__version__)
print('CUDA    :', torch.version.cuda)
print('GPU     :', torch.cuda.get_device_name(0))
print('Unsloth : OK')
print('xformers:', xformers.__version__)
"

echo ""
echo "Environment setup complete."
