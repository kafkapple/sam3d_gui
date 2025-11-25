#!/bin/bash
# SAM 3D GUI Environment Recreation Script

set -e  # Exit on error

echo "==================================="
echo "SAM 3D GUI Environment Recreation"
echo "==================================="

# 1. Remove existing environment
echo -e "\n[1/7] Removing existing sam3d_gui environment..."
conda env remove -n sam3d_gui -y 2>/dev/null || true
echo "✅ Old environment removed"

# 2. Create fresh environment
echo -e "\n[2/7] Creating fresh conda environment..."
conda create -n sam3d_gui python=3.10 -y
echo "✅ Fresh environment created"

# 3. Activate and install PyTorch CUDA 11.8
echo -e "\n[3/7] Installing PyTorch 2.0.0 + CUDA 11.8..."
conda run -n sam3d_gui pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118

# 4. Verify CUDA
echo -e "\n[4/7] Verifying PyTorch CUDA..."
conda run -n sam3d_gui python -c "
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

# 5. Install Kaolin from GitHub
echo -e "\n[5/7] Installing Kaolin 0.17.0..."
echo "⚠️  This may take 10-15 minutes to compile..."
conda run -n sam3d_gui pip install ninja setuptools wheel cython packaging
export TORCH_CUDA_ARCH_LIST="8.6"
export FORCE_CUDA=1
conda run -n sam3d_gui pip install --no-build-isolation git+https://github.com/NVIDIAGameWorks/kaolin.git@v0.17.0
echo "✅ Kaolin installed"

# 6. Install gsplat
echo -e "\n[6/7] Installing gsplat..."
conda run -n sam3d_gui pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7
echo "✅ gsplat installed"

# 7. Install other requirements
echo -e "\n[7/7] Installing other requirements..."
conda run -n sam3d_gui pip install -r requirements.txt
echo "✅ Requirements installed"

# Save environment
echo -e "\nSaving conda environment..."
conda env export -n sam3d_gui > environment_sam3d_gui.yml
echo "✅ Environment saved to environment_sam3d_gui.yml"

echo -e "\n==================================="
echo "✅ Environment recreation complete!"
echo ""
echo "Summary:"
echo "  - PyTorch 2.0.0 + CUDA 11.8"
echo "  - Kaolin 0.17.0"
echo "  - gsplat (latest)"
echo "  - All requirements installed"
echo ""
echo "To use the environment:"
echo "  conda activate sam3d_gui"
echo "  bash run.sh"
echo "==================================="
