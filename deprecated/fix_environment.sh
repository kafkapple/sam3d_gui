#!/bin/bash
# SAM 3D GUI Environment Fix Script

set -e  # Exit on error

echo "==================================="
echo "SAM 3D GUI Environment Fix"
echo "==================================="

# Check if in conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "❌ Not in conda environment. Activating sam3d_gui..."
    eval "$(conda shell.bash hook)"
    conda activate sam3d_gui

    if [ $? -ne 0 ]; then
        echo "❌ Failed to activate conda environment 'sam3d_gui'"
        echo "Please run: conda activate sam3d_gui"
        exit 1
    fi
    echo "✅ Activated conda environment: sam3d_gui"
elif [ "$CONDA_DEFAULT_ENV" != "sam3d_gui" ]; then
    echo "⚠️  Currently in conda environment: $CONDA_DEFAULT_ENV"
    echo "Switching to sam3d_gui..."
    conda activate sam3d_gui
    echo "✅ Activated conda environment: sam3d_gui"
else
    echo "✅ Already in conda environment: $CONDA_DEFAULT_ENV"
fi

# 0. Remove conflicting packages
echo -e "\n[0/4] Removing conflicting packages..."
pip uninstall -y pytorch3d 2>/dev/null || true
echo "✅ Conflicting packages removed"

# 1. Reinstall PyTorch with CUDA 11.8 using pip (avoid conda conflicts)
echo -e "\n[1/4] Reinstalling PyTorch with CUDA 11.8..."
pip uninstall -y torch torchvision torchaudio
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118

# 2. Verify PyTorch CUDA
echo -e "\n[2/4] Verifying PyTorch CUDA installation..."
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA device:', torch.cuda.get_device_name(0))
else:
    print('❌ CUDA not available!')
    exit(1)
"
echo "✅ PyTorch CUDA verified"

# 3. Reinstall Kaolin (rebuild from source for PyTorch 2.0.0 + CUDA 11.8)
echo -e "\n[3/5] Reinstalling Kaolin from GitHub..."
echo "⚠️  This may take 10-15 minutes to compile..."
pip uninstall -y kaolin 2>/dev/null || true
# Install build dependencies for Kaolin
pip install ninja setuptools wheel cython packaging
# Rebuild from source to match current PyTorch
export TORCH_CUDA_ARCH_LIST="8.6"  # RTX 3060
export FORCE_CUDA=1
# Install from GitHub without build isolation (so it can see PyTorch)
pip install --no-build-isolation git+https://github.com/NVIDIAGameWorks/kaolin.git@v0.17.0
echo "✅ Kaolin rebuilt and installed"

# 4. Install gsplat
echo -e "\n[4/5] Installing gsplat..."
# Pre-install build dependencies
pip install ninja setuptools wheel
echo "✅ Build dependencies installed"

# Set CUDA architecture for RTX 3060 (compute capability 8.6)
export TORCH_CUDA_ARCH_LIST="8.6"
echo "✅ CUDA architecture set: $TORCH_CUDA_ARCH_LIST"

# Install gsplat without build isolation (to access installed torch)
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7
echo "✅ gsplat installed"

# 5. Test SAM3D imports
echo -e "\n[5/5] Testing SAM3D imports..."
python -c "
import sys
from pathlib import Path

# Add SAM3D paths
sam3d_path = Path('/home/joon/dev/sam3d_gui/external/sam-3d-objects')
sys.path.insert(0, str(sam3d_path))
sys.path.insert(0, str(sam3d_path / 'notebook'))

try:
    from inference import Inference, load_image
    print('✅ SAM 3D Inference imported successfully!')
except ImportError as e:
    print(f'❌ SAM 3D import failed: {e}')
    import traceback
    traceback.print_exc()
"

# 6. Save conda environment info
echo -e "\n[6/6] Saving conda environment info..."
conda env export > /home/joon/dev/sam3d_gui/environment_sam3d_gui.yml
echo "✅ Environment saved to environment_sam3d_gui.yml"

echo -e "\n==================================="
echo "✅ Environment fix complete!"
echo ""
echo "Summary:"
echo "  - PyTorch 2.0.0 + CUDA 11.8"
echo "  - Kaolin 0.17.0"
echo "  - gsplat (latest)"
echo "  - Environment: environment_sam3d_gui.yml"
echo ""
echo "Please restart your application"
echo "==================================="
