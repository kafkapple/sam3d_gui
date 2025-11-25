#!/bin/bash
# SAM 3D GUI Final Environment Setup
# Carefully designed to avoid dependency conflicts

set -e  # Exit on error

echo "==========================================="
echo "SAM 3D GUI Environment Setup (Final)"
echo "==========================================="

# 1. Remove existing environment
echo -e "\n[1/8] Removing existing sam3d_gui environment..."
conda env remove -n sam3d_gui -y 2>/dev/null || true
echo "✅ Old environment removed"

# 2. Create fresh environment
echo -e "\n[2/8] Creating fresh conda environment..."
conda create -n sam3d_gui python=3.10 -y
echo "✅ Fresh environment created"

# 3. Install PyTorch CUDA 11.8 (PINNED VERSION)
echo -e "\n[3/8] Installing PyTorch 2.0.0 + CUDA 11.8..."
conda run -n sam3d_gui pip install \
    torch==2.0.0 \
    torchvision==0.15.0 \
    torchaudio==2.0.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch
TORCH_VERSION=$(conda run -n sam3d_gui python -c "import torch; print(torch.__version__)" 2>/dev/null)
if [[ "$TORCH_VERSION" != "2.0.0+cu118" ]]; then
    echo "❌ PyTorch version mismatch: $TORCH_VERSION"
    exit 1
fi
echo "✅ PyTorch 2.0.0+cu118 verified"

# 4. Fix NumPy version (must be < 2.0 for Kaolin)
echo -e "\n[4/8] Fixing NumPy version..."
conda run -n sam3d_gui pip install "numpy<2"
echo "✅ NumPy 1.x installed"

# 5. Install Kaolin
echo -e "\n[5/8] Installing Kaolin 0.17.0..."
echo "⚠️  This will take 10-15 minutes to compile..."
conda run -n sam3d_gui pip install ninja setuptools wheel cython packaging
export TORCH_CUDA_ARCH_LIST="8.6"
export FORCE_CUDA=1
conda run -n sam3d_gui pip install --no-build-isolation \
    git+https://github.com/NVIDIAGameWorks/kaolin.git@v0.17.0

# Verify PyTorch hasn't changed
TORCH_VERSION=$(conda run -n sam3d_gui python -c "import torch; print(torch.__version__)" 2>/dev/null)
if [[ "$TORCH_VERSION" != "2.0.0+cu118" ]]; then
    echo "❌ PyTorch was upgraded by Kaolin to $TORCH_VERSION!"
    exit 1
fi
echo "✅ Kaolin installed, PyTorch still 2.0.0+cu118"

# 6. Install gsplat
echo -e "\n[6/8] Installing gsplat..."
conda run -n sam3d_gui pip install --no-build-isolation \
    git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7

# Verify PyTorch
TORCH_VERSION=$(conda run -n sam3d_gui python -c "import torch; print(torch.__version__)" 2>/dev/null)
if [[ "$TORCH_VERSION" != "2.0.0+cu118" ]]; then
    echo "❌ PyTorch was upgraded by gsplat to $TORCH_VERSION!"
    exit 1
fi
echo "✅ gsplat installed, PyTorch still 2.0.0+cu118"

# 7. Install SAM 3D dependencies (WITHOUT LIGHTNING)
echo -e "\n[7/8] Installing SAM 3D inference dependencies..."
conda run -n sam3d_gui pip install \
    loguru \
    timm \
    optree \
    astor \
    --no-deps

# Install their dependencies carefully
conda run -n sam3d_gui pip install \
    huggingface_hub \
    safetensors \
    pyyaml

echo "✅ SAM 3D dependencies installed"

# 8. Install pytorch3d
echo -e "\n[8/8] Installing pytorch3d..."
export TORCH_CUDA_ARCH_LIST='8.6'
export FORCE_CUDA=1
conda run -n sam3d_gui pip install --no-build-isolation \
    'git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7'

# Final verification
TORCH_VERSION=$(conda run -n sam3d_gui python -c "import torch; print(torch.__version__)" 2>/dev/null)
if [[ "$TORCH_VERSION" != "2.0.0+cu118" ]]; then
    echo "❌ PyTorch was upgraded to $TORCH_VERSION!"
    exit 1
fi
echo "✅ pytorch3d installed, PyTorch still 2.0.0+cu118"

# Install remaining requirements
echo -e "\n[9/9] Installing other requirements..."
conda run -n sam3d_gui pip install -r requirements.txt || echo "⚠️  Some packages failed, but core is installed"

# Force NumPy back to 1.x if it was upgraded
conda run -n sam3d_gui pip install "numpy<2" --force-reinstall

echo -e "\n==========================================="
echo "🎉 Environment setup complete!"
echo "==========================================="
echo ""
echo "✅ Installed:"
echo "  - PyTorch 2.0.0 + CUDA 11.8"
echo "  - NumPy 1.x"
echo "  - Kaolin 0.17.0"
echo "  - gsplat"
echo "  - pytorch3d 0.7.7"
echo "  - SAM 3D dependencies (NO Lightning)"
echo ""
echo "📋 Next steps:"
echo "  1. conda activate sam3d_gui"
echo "  2. Test: python -c 'import torch; print(torch.cuda.is_available())'"
echo "  3. bash run.sh"
echo "==========================================="
