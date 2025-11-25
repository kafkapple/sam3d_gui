#!/bin/bash
# SAM 3D GUI - Enhanced Environment Diagnostic Script
# 시스템 CUDA 감지 및 심볼릭 링크 활용 권장

echo "=========================================="
echo "SAM 3D GUI - Environment Diagnostic v2"
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
# 2. GPU Information
# ==========================================
echo "=========================================="
echo "2. GPU Information"
echo "=========================================="

if command -v nvidia-smi &> /dev/null; then
    echo "✅ nvidia-smi found"
    echo ""
    nvidia-smi --query-gpu=gpu_name,memory.total,driver_version,compute_cap --format=csv
    echo ""

    # Store for later use
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
    GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | head -1)
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
else
    echo "❌ nvidia-smi not found"
    COMPUTE_CAP="unknown"
fi
echo ""

# ==========================================
# 3. CUDA Information (Enhanced)
# ==========================================
echo "=========================================="
echo "3. CUDA Information (All Installations)"
echo "=========================================="

# List all CUDA installations
echo "📁 Scanning /usr/local/ for CUDA installations:"
echo ""

CUDA_INSTALLS=()
if ls -d /usr/local/cuda* 2>/dev/null; then
    for cuda_dir in /usr/local/cuda*; do
        if [ -d "$cuda_dir" ]; then
            echo "  📦 $cuda_dir"

            # Check if symlink
            if [ -L "$cuda_dir" ]; then
                TARGET=$(readlink -f "$cuda_dir")
                echo "     Type: Symlink → $TARGET"
            else
                echo "     Type: Directory"
            fi

            # Try to get version
            if [ -f "$cuda_dir/bin/nvcc" ]; then
                NVCC_VERSION=$($cuda_dir/bin/nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | tr -d ',')
                echo "     nvcc: $NVCC_VERSION"
                CUDA_INSTALLS+=("$cuda_dir:$NVCC_VERSION")
            else
                echo "     nvcc: Not found"
            fi

            # Check library path
            if [ -d "$cuda_dir/lib64" ]; then
                LIB_SIZE=$(du -sh "$cuda_dir/lib64" 2>/dev/null | awk '{print $1}')
                echo "     lib64: $LIB_SIZE"
            fi

            echo ""
        fi
    done
else
    echo "  ❌ No CUDA installations found in /usr/local/"
    echo ""
fi

# Current nvcc in PATH
echo "🔍 Current nvcc (in PATH):"
if command -v nvcc &> /dev/null; then
    echo "  ✅ Found"
    nvcc --version | grep "release" | awk '{print "     Version:", $5}'
    NVCC_PATH=$(which nvcc)
    CURRENT_CUDA=$(dirname $(dirname $NVCC_PATH))
    echo "     Location: $NVCC_PATH"
    echo "     CUDA Root: $CURRENT_CUDA"
else
    echo "  ❌ Not in PATH"
    CURRENT_CUDA="none"
fi
echo ""

# CUDA environment variables
echo "🌍 CUDA Environment Variables:"
echo "  CUDA_HOME: ${CUDA_HOME:-Not set}"
echo "  CUDA_PATH: ${CUDA_PATH:-Not set}"
echo "  LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-Not set}"
echo ""

# ==========================================
# 4. CUDA Selection Recommendations
# ==========================================
echo "=========================================="
echo "4. CUDA Selection Strategy"
echo "=========================================="
echo ""

# Analyze available CUDA versions
HAVE_CUDA_118=false
HAVE_CUDA_117=false
HAVE_CUDA_12X=false
CUDA_118_PATH=""
CUDA_117_PATH=""

for install in "${CUDA_INSTALLS[@]}"; do
    path="${install%%:*}"
    version="${install##*:}"

    if [[ "$version" == "11.8" ]]; then
        HAVE_CUDA_118=true
        CUDA_118_PATH="$path"
    elif [[ "$version" == "11.7" ]]; then
        HAVE_CUDA_117=true
        CUDA_117_PATH="$path"
    elif [[ "$version" =~ ^12\. ]]; then
        HAVE_CUDA_12X=true
    fi
done

echo "💡 Recommendation for PyTorch 2.0.0 + CUDA:"
echo ""

if [ "$HAVE_CUDA_118" = true ]; then
    echo "  ✅ OPTIMAL: CUDA 11.8 found at $CUDA_118_PATH"
    echo ""
    echo "  🎯 Best approach: Use system CUDA 11.8 via symlink"
    echo "     Advantages:"
    echo "       - No additional download (~5GB saved)"
    echo "       - Faster setup (no Conda CUDA install)"
    echo "       - Same stability as Conda CUDA"
    echo ""
    echo "  📋 Setup command:"
    echo "     export CUDA_HOME=$CUDA_118_PATH"
    echo "     export PATH=\$CUDA_HOME/bin:\$PATH"
    echo "     export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
    echo ""
    echo "  🔧 For Conda environment (recommended):"
    echo "     # In setup script, create symlink:"
    echo "     ln -sf $CUDA_118_PATH \$CONDA_PREFIX/cuda"
    echo "     export CUDA_HOME=\$CONDA_PREFIX/cuda"
    echo ""
elif [ "$HAVE_CUDA_117" = true ]; then
    echo "  ⚠️  ACCEPTABLE: CUDA 11.7 found at $CUDA_117_PATH"
    echo ""
    echo "  🎯 Options:"
    echo "     Option 1: Use system CUDA 11.7 (works but not optimal)"
    echo "        - Install PyTorch 2.0.0+cu117"
    echo "        - export CUDA_HOME=$CUDA_117_PATH"
    echo ""
    echo "     Option 2: Install Conda CUDA 11.8 (recommended)"
    echo "        - conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit"
    echo "        - Better compatibility with PyTorch 2.0.0"
    echo ""
else
    echo "  ⚠️  No CUDA 11.7/11.8 found"
    echo ""
    if [ "$HAVE_CUDA_12X" = true ]; then
        echo "  Found CUDA 12.x, but PyTorch 2.0.0 uses CUDA 11.x"
    fi
    echo "  🎯 Recommendation: Install Conda CUDA 11.8"
    echo "     conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit"
    echo ""
fi

# ==========================================
# 5. Setup Script Recommendation
# ==========================================
echo "=========================================="
echo "5. Setup Script Selection"
echo "=========================================="
echo ""

if [ "$HAVE_CUDA_118" = true ]; then
    echo "✅ Recommended: setup_system_cuda.sh (NEW - to be created)"
    echo "   Reason: System CUDA 11.8 available, no need for Conda CUDA"
    echo "   Benefits:"
    echo "     - Saves ~5GB disk space"
    echo "     - Faster installation (10-15 min saved)"
    echo "     - Uses system-optimized CUDA"
    echo ""
    echo "   Command: ./setup_system_cuda.sh --cuda-path=$CUDA_118_PATH"
    echo ""
elif [ "$HAVE_CUDA_117" = true ]; then
    echo "⚠️  Recommended: setup_conda_cuda.sh"
    echo "   Reason: System CUDA 11.7, better to use Conda CUDA 11.8"
    echo "   Alternative: setup_cu117.sh (use system CUDA 11.7)"
    echo ""
    echo "   Command: ./setup_conda_cuda.sh"
    echo ""
else
    echo "📦 Recommended: setup_conda_cuda.sh"
    echo "   Reason: No suitable system CUDA found"
    echo ""
    echo "   Command: ./setup_conda_cuda.sh"
    echo ""
fi

# ==========================================
# 6. Python & Conda
# ==========================================
echo "=========================================="
echo "6. Python & Conda"
echo "=========================================="

if command -v conda &> /dev/null; then
    echo "✅ conda found"
    conda --version
    echo ""

    # Check for existing sam3d_gui environment
    if conda env list | grep -q "^sam3d_gui "; then
        echo "⚠️  Existing 'sam3d_gui' environment found"
        echo ""
        echo "Installed packages:"
        conda run -n sam3d_gui pip list 2>/dev/null | grep -E "^torch |^kaolin |^pytorch3d " || echo "  Key packages not found"
    else
        echo "✅ No existing 'sam3d_gui' environment"
    fi
else
    echo "❌ conda not found"
fi
echo ""

# ==========================================
# 7. Diagnostic Summary
# ==========================================
echo "=========================================="
echo "7. Diagnostic Summary"
echo "=========================================="
echo ""

echo "GPU:              $GPU_NAME"
echo "Compute Cap:      $COMPUTE_CAP"
echo "Driver:           ${DRIVER_VERSION:-N/A}"
echo "System CUDA:      $([ "$HAVE_CUDA_118" = true ] && echo "11.8 ✅" || ([ "$HAVE_CUDA_117" = true ] && echo "11.7 ⚠️" || echo "Not optimal ❌"))"
echo "Current nvcc:     $(command -v nvcc &> /dev/null && echo "Found" || echo "Not in PATH")"
echo "Conda:            $(command -v conda &> /dev/null && echo "Installed ✅" || echo "Not found ❌")"
echo ""

if [ "$HAVE_CUDA_118" = true ]; then
    echo "🎯 Recommended Setup: System CUDA 11.8 (symlink approach)"
    echo "   Estimated Time: 20-25 minutes"
    echo "   Disk Space: ~10GB (no Conda CUDA)"
elif [ "$HAVE_CUDA_117" = true ]; then
    echo "🎯 Recommended Setup: Conda CUDA 11.8"
    echo "   Estimated Time: 30-35 minutes"
    echo "   Disk Space: ~15GB (includes Conda CUDA)"
else
    echo "🎯 Recommended Setup: Conda CUDA 11.8 (required)"
    echo "   Estimated Time: 30-35 minutes"
    echo "   Disk Space: ~15GB"
fi
echo ""

echo "=========================================="
echo "Diagnostic Complete"
echo "=========================================="
echo ""
echo "Save this output:"
echo "  ./check_environment_v2.sh > environment_report.txt 2>&1"
echo ""
