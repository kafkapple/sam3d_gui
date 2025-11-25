#!/bin/bash
# SAM 3D GUI - Environment Diagnostic Script
# 새로운 서버에서 setup.sh 실행 전 필수 환경 정보 수집

echo "=========================================="
echo "SAM 3D GUI - Environment Diagnostic"
echo "=========================================="
echo ""
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Hostname: $(hostname)"
echo ""

# ==========================================
# 1. Operating System Information
# ==========================================
echo "=========================================="
echo "1. Operating System"
echo "=========================================="
if [ -f /etc/os-release ]; then
    cat /etc/os-release
else
    echo "OS: $(uname -s)"
    echo "Kernel: $(uname -r)"
fi
echo ""

# ==========================================
# 2. CPU Information
# ==========================================
echo "=========================================="
echo "2. CPU Information"
echo "=========================================="
echo "CPU Model: $(grep "model name" /proc/cpuinfo | head -1 | cut -d':' -f2 | xargs)"
echo "CPU Cores: $(nproc)"
echo "Architecture: $(uname -m)"
echo ""

# ==========================================
# 3. Memory Information
# ==========================================
echo "=========================================="
echo "3. Memory Information"
echo "=========================================="
free -h
echo ""

# ==========================================
# 4. GPU Information
# ==========================================
echo "=========================================="
echo "4. GPU Information"
echo "=========================================="

# Check if nvidia-smi exists
if command -v nvidia-smi &> /dev/null; then
    echo "✅ nvidia-smi found"
    echo ""
    nvidia-smi --query-gpu=gpu_name,memory.total,driver_version,compute_cap --format=csv
    echo ""
    echo "Full nvidia-smi output:"
    nvidia-smi
else
    echo "❌ nvidia-smi not found - No NVIDIA GPU or driver not installed"
fi
echo ""

# ==========================================
# 5. CUDA Information
# ==========================================
echo "=========================================="
echo "5. CUDA Information"
echo "=========================================="

# Check nvcc
if command -v nvcc &> /dev/null; then
    echo "✅ nvcc found"
    nvcc --version
    echo ""
    echo "CUDA Location: $(which nvcc)"
    echo "CUDA Path: $(dirname $(dirname $(which nvcc)))"
else
    echo "❌ nvcc not found"
    echo ""
    echo "Searching for CUDA installations..."
    if [ -d /usr/local/cuda ]; then
        echo "Found: /usr/local/cuda"
        ls -la /usr/local/cuda/bin/nvcc 2>/dev/null || echo "  nvcc not in /usr/local/cuda/bin"
    fi
    if [ -d /usr/local/cuda-11.8 ]; then
        echo "Found: /usr/local/cuda-11.8"
        ls -la /usr/local/cuda-11.8/bin/nvcc 2>/dev/null || echo "  nvcc not in /usr/local/cuda-11.8/bin"
    fi
    if [ -d /usr/local/cuda-11.7 ]; then
        echo "Found: /usr/local/cuda-11.7"
    fi
    if [ -d /usr/local/cuda-12.0 ]; then
        echo "Found: /usr/local/cuda-12.0"
    fi
fi

# CUDA environment variables
echo ""
echo "CUDA Environment Variables:"
echo "  CUDA_HOME: ${CUDA_HOME:-Not set}"
echo "  CUDA_PATH: ${CUDA_PATH:-Not set}"
echo "  LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-Not set}"
echo ""

# ==========================================
# 6. Python & Conda Information
# ==========================================
echo "=========================================="
echo "6. Python & Conda"
echo "=========================================="

# Check Python
if command -v python &> /dev/null; then
    echo "✅ python found"
    python --version
    echo "Location: $(which python)"
else
    echo "❌ python not found"
fi
echo ""

if command -v python3 &> /dev/null; then
    echo "✅ python3 found"
    python3 --version
    echo "Location: $(which python3)"
else
    echo "❌ python3 not found"
fi
echo ""

# Check Conda
if command -v conda &> /dev/null; then
    echo "✅ conda found"
    conda --version
    echo "Location: $(which conda)"
    echo ""
    echo "Conda Info:"
    conda info | grep -E "conda version|python version|platform|base environment|envs directories"
    echo ""
    echo "Existing Environments:"
    conda env list
else
    echo "❌ conda not found"
fi
echo ""

# ==========================================
# 7. GCC & Compiler Information
# ==========================================
echo "=========================================="
echo "7. Compiler Information"
echo "=========================================="

if command -v gcc &> /dev/null; then
    echo "✅ gcc found"
    gcc --version | head -1
    echo "Location: $(which gcc)"
else
    echo "❌ gcc not found"
fi
echo ""

if command -v g++ &> /dev/null; then
    echo "✅ g++ found"
    g++ --version | head -1
    echo "Location: $(which g++)"
else
    echo "❌ g++ not found"
fi
echo ""

if command -v make &> /dev/null; then
    echo "✅ make found"
    make --version | head -1
else
    echo "❌ make not found"
fi
echo ""

# ==========================================
# 8. Git & Git LFS
# ==========================================
echo "=========================================="
echo "8. Git & Version Control"
echo "=========================================="

if command -v git &> /dev/null; then
    echo "✅ git found"
    git --version
    echo "Location: $(which git)"
else
    echo "❌ git not found"
fi
echo ""

if command -v git-lfs &> /dev/null; then
    echo "✅ git-lfs found"
    git-lfs --version
else
    echo "❌ git-lfs not found (needed for SAM 3D checkpoint download)"
fi
echo ""

# ==========================================
# 9. System Libraries (for PyAV)
# ==========================================
echo "=========================================="
echo "9. System Libraries"
echo "=========================================="

echo "Checking FFmpeg libraries (required for PyAV):"
echo ""

# libavcodec
if ldconfig -p | grep -q libavcodec; then
    echo "✅ libavcodec found"
    ldconfig -p | grep libavcodec | head -1
else
    echo "❌ libavcodec not found"
fi

# libavformat
if ldconfig -p | grep -q libavformat; then
    echo "✅ libavformat found"
    ldconfig -p | grep libavformat | head -1
else
    echo "❌ libavformat not found"
fi

# libavutil
if ldconfig -p | grep -q libavutil; then
    echo "✅ libavutil found"
    ldconfig -p | grep libavutil | head -1
else
    echo "❌ libavutil not found"
fi

# libswscale
if ldconfig -p | grep -q libswscale; then
    echo "✅ libswscale found"
    ldconfig -p | grep libswscale | head -1
else
    echo "❌ libswscale not found"
fi

# libswresample
if ldconfig -p | grep -q libswresample; then
    echo "✅ libswresample found"
    ldconfig -p | grep libswresample | head -1
else
    echo "❌ libswresample not found"
fi

echo ""

# Check pkg-config
if command -v pkg-config &> /dev/null; then
    echo "✅ pkg-config found"
    echo ""
    echo "FFmpeg version (via pkg-config):"
    pkg-config --modversion libavcodec 2>/dev/null || echo "  libavcodec: Not found"
    pkg-config --modversion libavformat 2>/dev/null || echo "  libavformat: Not found"
    pkg-config --modversion libavutil 2>/dev/null || echo "  libavutil: Not found"
else
    echo "❌ pkg-config not found"
fi
echo ""

# ==========================================
# 10. Disk Space
# ==========================================
echo "=========================================="
echo "10. Disk Space"
echo "=========================================="
df -h /home
echo ""
echo "Current directory: $(pwd)"
du -sh . 2>/dev/null || echo "Cannot calculate directory size"
echo ""

# ==========================================
# 11. Network Connectivity
# ==========================================
echo "=========================================="
echo "11. Network Connectivity"
echo "=========================================="

echo "Testing connectivity to required repositories:"
echo ""

# GitHub
if ping -c 1 github.com &> /dev/null; then
    echo "✅ GitHub accessible"
else
    echo "❌ GitHub not accessible"
fi

# HuggingFace
if ping -c 1 huggingface.co &> /dev/null; then
    echo "✅ HuggingFace accessible"
else
    echo "❌ HuggingFace not accessible"
fi

# PyPI
if ping -c 1 pypi.org &> /dev/null; then
    echo "✅ PyPI accessible"
else
    echo "❌ PyPI not accessible"
fi

echo ""

# ==========================================
# 12. Existing Installation Check
# ==========================================
echo "=========================================="
echo "12. Existing Installation Check"
echo "=========================================="

# Check if sam3d_gui environment exists
if conda env list | grep -q "sam3d_gui"; then
    echo "⚠️  Conda environment 'sam3d_gui' already exists"
    echo ""
    echo "Details:"
    conda list -n sam3d_gui | grep -E "^torch |^torchvision |^kaolin |^pytorch3d" 2>/dev/null || echo "  Key packages not found"
else
    echo "✅ No existing 'sam3d_gui' environment"
fi
echo ""

# Check project directory
if [ -d "$(pwd)/external/sam-3d-objects" ]; then
    echo "✅ SAM 3D Objects submodule exists"
    ls -lh $(pwd)/external/sam-3d-objects/checkpoints/hf/*.ckpt 2>/dev/null | wc -l | xargs echo "  Checkpoints found:"
else
    echo "ℹ️  SAM 3D Objects submodule not yet cloned"
fi
echo ""

# ==========================================
# 13. Recommendations
# ==========================================
echo "=========================================="
echo "13. Installation Recommendations"
echo "=========================================="
echo ""

# Check CUDA compatibility
if command -v nvidia-smi &> /dev/null; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)

    echo "GPU Configuration:"
    echo "  Driver Version: $DRIVER_VERSION"
    echo "  CUDA Compute Capability: $CUDA_ARCH"
    echo ""

    # Recommend CUDA architecture
    if [[ "$CUDA_ARCH" == "8.0" ]]; then
        echo "✅ Detected Ampere architecture (8.0) - A100/A6000 class"
        echo "   Recommended: export TORCH_CUDA_ARCH_LIST=\"8.0\""
    elif [[ "$CUDA_ARCH" == "8.6" ]]; then
        echo "✅ Detected Ampere architecture (8.6) - RTX 30xx class"
        echo "   Recommended: export TORCH_CUDA_ARCH_LIST=\"8.6\""
    elif [[ "$CUDA_ARCH" == "8.9" ]]; then
        echo "✅ Detected Ada architecture (8.9) - RTX 40xx class"
        echo "   Recommended: export TORCH_CUDA_ARCH_LIST=\"8.9\""
    elif [[ "$CUDA_ARCH" == "7.5" ]]; then
        echo "✅ Detected Turing architecture (7.5) - RTX 20xx/T4 class"
        echo "   Recommended: export TORCH_CUDA_ARCH_LIST=\"7.5\""
    else
        echo "ℹ️  Detected compute capability: $CUDA_ARCH"
        echo "   Set TORCH_CUDA_ARCH_LIST accordingly"
    fi
    echo ""
fi

# Check for missing dependencies
echo "Required System Packages Check:"
echo ""

MISSING_PACKAGES=()

if ! command -v gcc &> /dev/null; then
    MISSING_PACKAGES+=("build-essential (for gcc)")
fi

if ! ldconfig -p | grep -q libavcodec; then
    MISSING_PACKAGES+=("ffmpeg or libavcodec-dev")
fi

if ! command -v git-lfs &> /dev/null; then
    MISSING_PACKAGES+=("git-lfs")
fi

if [ ${#MISSING_PACKAGES[@]} -eq 0 ]; then
    echo "✅ All required system packages appear to be installed"
else
    echo "⚠️  Missing packages detected:"
    for pkg in "${MISSING_PACKAGES[@]}"; do
        echo "  - $pkg"
    done
    echo ""
    echo "Install with:"
    echo "  sudo apt update"
    echo "  sudo apt install build-essential ffmpeg git-lfs"
fi
echo ""

# PyAV specific recommendation
if ! ldconfig -p | grep -q libavcodec; then
    echo "⚠️  FFmpeg libraries not found"
    echo "   PyAV (av package) will fail to install without these."
    echo ""
    echo "   Solution 1 (System packages):"
    echo "     sudo apt install ffmpeg libavcodec-dev libavformat-dev libavutil-dev"
    echo ""
    echo "   Solution 2 (Conda packages):"
    echo "     conda install -c conda-forge av"
    echo ""
fi

# ==========================================
# Summary
# ==========================================
echo "=========================================="
echo "14. Diagnostic Summary"
echo "=========================================="
echo ""

echo "GPU Ready:        $(command -v nvidia-smi &> /dev/null && echo "✅ Yes" || echo "❌ No")"
echo "CUDA Available:   $(command -v nvcc &> /dev/null && echo "✅ Yes" || echo "❌ No")"
echo "Conda Ready:      $(command -v conda &> /dev/null && echo "✅ Yes" || echo "❌ No")"
echo "Compiler Ready:   $(command -v gcc &> /dev/null && echo "✅ Yes" || echo "❌ No")"
echo "FFmpeg Ready:     $(ldconfig -p | grep -q libavcodec && echo "✅ Yes" || echo "❌ No")"
echo "Git LFS Ready:    $(command -v git-lfs &> /dev/null && echo "✅ Yes" || echo "❌ No")"
echo ""

echo "=========================================="
echo "Diagnostic Complete"
echo "=========================================="
echo ""
echo "Save this output and share it for troubleshooting:"
echo "  ./check_environment.sh > environment_report.txt 2>&1"
echo ""
