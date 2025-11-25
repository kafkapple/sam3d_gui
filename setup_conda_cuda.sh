#!/bin/bash
# SAM 3D GUI - Conda CUDA 11.8 통합 환경 설정
# 시스템 CUDA와 독립적으로 Conda 환경 내부에 CUDA 11.8 설치
set -e

# ==========================================
# 프로젝트 루트 경로 설정
# ==========================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT"

echo "============================================="
echo "SAM 3D GUI - Conda CUDA 11.8 환경 설정"
echo "============================================="
echo "프로젝트 루트: $PROJECT_ROOT"
echo ""
echo "🔧 설정 방식:"
echo "  - Conda 환경 내부에 CUDA 11.8 설치"
echo "  - 시스템 CUDA (11.7)와 독립적으로 운영"
echo "  - PyTorch 2.0.0 + CUDA 11.8 최적 조합"
echo ""

# ==========================================
# 0. NVIDIA Driver 확인
# ==========================================
echo "[0/8] NVIDIA Driver 확인..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi를 찾을 수 없습니다."
    echo "   NVIDIA GPU 드라이버가 설치되어 있는지 확인하세요."
    exit 1
fi

DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | head -1)

echo "✅ GPU 감지됨:"
echo "   GPU: $GPU_NAME"
echo "   Driver: $DRIVER_VERSION"
echo "   Compute Capability: $COMPUTE_CAP"
echo ""

# Driver 버전 확인 (CUDA 11.8은 driver >= 450 필요)
DRIVER_MAJOR=$(echo $DRIVER_VERSION | cut -d. -f1)
if [ "$DRIVER_MAJOR" -lt 450 ]; then
    echo "⚠️  경고: CUDA 11.8은 드라이버 버전 450 이상 필요"
    echo "   현재 버전: $DRIVER_VERSION"
    echo "   계속 진행하시겠습니까? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "설치 취소됨"
        exit 1
    fi
fi

# ==========================================
# 1. Conda 환경 생성
# ==========================================
echo "[1/8] Conda 환경 생성..."
if conda env list | grep -q "^sam3d_gui "; then
    echo "⚠️  기존 sam3d_gui 환경 발견"
    echo "   삭제하고 재생성하시겠습니까? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "기존 환경 제거 중..."
        conda env remove -n sam3d_gui -y
    else
        echo "기존 환경 유지. 업데이트 모드로 진행합니다."
        UPDATE_MODE=true
    fi
fi

if [ "$UPDATE_MODE" != true ]; then
    conda create -n sam3d_gui python=3.10 -y
    echo "✅ Python 3.10 환경 생성 완료"
fi

# ==========================================
# 2. CUDA Toolkit 11.8 설치 (Conda)
# ==========================================
echo ""
echo "[2/8] CUDA Toolkit 11.8 설치 (Conda)..."
echo "   약 3-5GB 디스크 공간 필요"
echo ""

conda run -n sam3d_gui conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit -y

# CUDA 환경변수 설정 확인
echo ""
echo "✅ CUDA Toolkit 11.8 설치 완료"
echo ""
echo "Conda 환경 내 CUDA 경로:"
CONDA_ENV_PATH=$(conda run -n sam3d_gui python -c "import sys; print(sys.prefix)")
echo "  CUDA: $CONDA_ENV_PATH/pkgs/cuda-toolkit"
echo ""

# ==========================================
# 3. PyTorch 2.0.0 + CUDA 11.8 설치
# ==========================================
echo ""
echo "[3/8] PyTorch 2.0.0 + CUDA 11.8 설치..."
conda run -n sam3d_gui pip install \
    torch==2.0.0 \
    torchvision==0.15.0 \
    torchaudio==2.0.0 \
    --index-url https://download.pytorch.org/whl/cu118

# 검증
TORCH_VERSION=$(conda run -n sam3d_gui python -c "import torch; print(torch.__version__)" 2>/dev/null)
CUDA_AVAILABLE=$(conda run -n sam3d_gui python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
CUDA_VERSION=$(conda run -n sam3d_gui python -c "import torch; print(torch.version.cuda)" 2>/dev/null)

if [[ "$TORCH_VERSION" != "2.0.0+cu118" ]]; then
    echo "❌ PyTorch 설치 실패: $TORCH_VERSION"
    exit 1
fi

if [[ "$CUDA_AVAILABLE" != "True" ]]; then
    echo "⚠️  경고: CUDA를 사용할 수 없습니다."
    echo "   GPU 드라이버를 확인하세요."
fi

echo "✅ PyTorch 2.0.0+cu118 설치 완료"
echo "   CUDA Available: $CUDA_AVAILABLE"
echo "   CUDA Version: $CUDA_VERSION"

# ==========================================
# 4. NumPy 버전 고정
# ==========================================
echo ""
echo "[4/8] NumPy 1.x 설치..."
conda run -n sam3d_gui pip install "numpy<2"
echo "✅ NumPy 1.x 설치 완료"

# ==========================================
# 5. Kaolin 설치
# ==========================================
echo ""
echo "[5/8] Kaolin 0.17.0 설치 (15-20분 소요)..."
conda run -n sam3d_gui pip install ninja setuptools wheel cython packaging

# GPU architecture 자동 감지 및 설정
if [[ "$COMPUTE_CAP" == "8.6" ]]; then
    ARCH_LIST="8.6"
    echo "   Architecture: RTX 30xx / A6000 (8.6)"
elif [[ "$COMPUTE_CAP" == "8.0" ]]; then
    ARCH_LIST="8.0"
    echo "   Architecture: A100 / A6000 (8.0)"
elif [[ "$COMPUTE_CAP" == "8.9" ]]; then
    ARCH_LIST="8.9"
    echo "   Architecture: RTX 40xx (8.9)"
elif [[ "$COMPUTE_CAP" == "7.5" ]]; then
    ARCH_LIST="7.5"
    echo "   Architecture: RTX 20xx / T4 (7.5)"
else
    ARCH_LIST="$COMPUTE_CAP"
    echo "   Architecture: $COMPUTE_CAP (auto-detected)"
fi

export TORCH_CUDA_ARCH_LIST="$ARCH_LIST"
export FORCE_CUDA=1

# Conda CUDA 경로 명시
export CUDA_HOME="$CONDA_ENV_PATH"
export PATH="$CONDA_ENV_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:$LD_LIBRARY_PATH"

echo "   CUDA_HOME: $CUDA_HOME"
echo "   Compiling for architecture: $ARCH_LIST"
echo ""

conda run -n sam3d_gui bash -c "
export TORCH_CUDA_ARCH_LIST='$ARCH_LIST'
export FORCE_CUDA=1
export CUDA_HOME='$CONDA_ENV_PATH'
export PATH='$CONDA_ENV_PATH/bin:\$PATH'
pip install --no-build-isolation git+https://github.com/NVIDIAGameWorks/kaolin.git@v0.17.0
"

# PyTorch 버전 재확인
TORCH_VERSION=$(conda run -n sam3d_gui python -c "import torch; print(torch.__version__)" 2>/dev/null)
if [[ "$TORCH_VERSION" != "2.0.0+cu118" ]]; then
    echo "❌ Kaolin이 PyTorch를 변경했습니다: $TORCH_VERSION"
    exit 1
fi
echo "✅ Kaolin 설치 완료, PyTorch 2.0.0 유지됨"

# ==========================================
# 6. pytorch3d, gsplat 설치
# ==========================================
echo ""
echo "[6/8] pytorch3d 및 기타 3D 라이브러리 설치..."

# gsplat
conda run -n sam3d_gui bash -c "
export TORCH_CUDA_ARCH_LIST='$ARCH_LIST'
export FORCE_CUDA=1
export CUDA_HOME='$CONDA_ENV_PATH'
export PATH='$CONDA_ENV_PATH/bin:\$PATH'
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7
"

# pytorch3d
conda run -n sam3d_gui bash -c "
export TORCH_CUDA_ARCH_LIST='$ARCH_LIST'
export FORCE_CUDA=1
export CUDA_HOME='$CONDA_ENV_PATH'
export PATH='$CONDA_ENV_PATH/bin:\$PATH'
pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7'
"

echo "✅ pytorch3d, gsplat 설치 완료"

# ==========================================
# 7. PyAV (av) 설치 - Conda
# ==========================================
echo ""
echo "[7/8] PyAV (av) 설치 - Conda로 FFmpeg 포함..."
conda run -n sam3d_gui conda install -c conda-forge av -y
echo "✅ PyAV 설치 완료"

# ==========================================
# 8. 기타 의존성 설치
# ==========================================
echo ""
echo "[8/8] 기타 의존성 설치..."

# SAM 3D 의존성 (Lightning 제외)
conda run -n sam3d_gui pip install \
    loguru timm optree astor \
    huggingface_hub safetensors pyyaml \
    --no-deps

conda run -n sam3d_gui pip install \
    huggingface_hub safetensors pyyaml

# SAM 3D 추가 의존성 (av 제외, spconv-cu118 사용)
conda run -n sam3d_gui pip install \
    spconv-cu118==2.3.8 \
    xatlas roma einops-exts \
    decord open3d trimesh \
    pyvista pymeshfix pyrender \
    python-igraph \
    easydict point-cloud-utils polyscope \
    plyfile gdown rootutils \
    git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b

# NumPy 버전 재확인
conda run -n sam3d_gui pip install "numpy<2" --force-reinstall

# Web UI
conda run -n sam3d_gui pip install -r "$PROJECT_ROOT/requirements.txt" || echo "⚠️ 일부 패키지 설치 실패 (핵심은 완료)"

echo "✅ 기타 의존성 설치 완료"

# ==========================================
# 모델 다운로드
# ==========================================
echo ""
echo "============================================="
echo "모델 체크포인트 다운로드"
echo "============================================="

CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints"
mkdir -p "$CHECKPOINT_DIR"

# SAM2 체크포인트
SAM2_DIR="$CHECKPOINT_DIR/sam2"
mkdir -p "$SAM2_DIR"

if [ -f "$SAM2_DIR/sam2_hiera_large.pt" ]; then
    echo "✅ SAM2 체크포인트 이미 존재"
else
    echo "SAM2 체크포인트 다운로드 중..."
    cd "$SAM2_DIR"
    wget -q --show-progress https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
    echo "✅ SAM2 체크포인트 다운로드 완료"
    cd "$PROJECT_ROOT"
fi

# SAM 3D 체크포인트
SAM3D_DIR="$PROJECT_ROOT/external/sam-3d-objects/checkpoints/hf"

if [ ! -d "$PROJECT_ROOT/external/sam-3d-objects" ]; then
    echo "⚠️  SAM 3D submodule이 없습니다."
    echo "다음 명령으로 초기화하세요:"
    echo "  git submodule update --init --recursive"
else
    echo "✅ SAM 3D submodule (PyTorch 2.0 호환 버전) 발견"
fi

if [ -d "$SAM3D_DIR" ] && [ "$(ls -A $SAM3D_DIR/*.ckpt 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "✅ SAM 3D 체크포인트 이미 존재"
else
    echo ""
    echo "⚠️  SAM 3D 체크포인트가 없습니다."
    echo "다음 스크립트로 다운로드하세요:"
    echo "  ./download_sam3d.sh"
fi

# ==========================================
# 설정 파일 업데이트
# ==========================================
CONFIG_FILE="$PROJECT_ROOT/config/model_config.yaml"
if [ -f "$CONFIG_FILE" ]; then
    echo ""
    echo "설정 파일 업데이트 중..."
    sed -i "s|/home/[^/]*/dev/sam3d_gui|$PROJECT_ROOT|g" "$CONFIG_FILE"
    echo "✅ 설정 파일 업데이트 완료"
fi

# ==========================================
# 환경 검증
# ==========================================
echo ""
echo "============================================="
echo "🔍 환경 검증"
echo "============================================="

echo ""
echo "PyTorch 정보:"
conda run -n sam3d_gui python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA Available: {torch.cuda.is_available()}')
print(f'  CUDA Version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

echo ""
echo "설치된 주요 패키지:"
conda run -n sam3d_gui pip list | grep -E "torch|kaolin|pytorch3d|av"

# ==========================================
# 완료 메시지
# ==========================================
echo ""
echo "============================================="
echo "🎉 환경 설정 완료! (Conda CUDA 11.8)"
echo "============================================="
echo ""
echo "✅ 설치 완료:"
echo "  - Python 3.10"
echo "  - CUDA Toolkit 11.8 (Conda 환경 내부)"
echo "  - PyTorch 2.0.0 + CUDA 11.8"
echo "  - NumPy 1.x"
echo "  - Kaolin 0.17.0"
echo "  - pytorch3d 0.7.7"
echo "  - gsplat"
echo "  - PyAV (av) via Conda"
echo "  - SAM 3D dependencies"
echo "  - SAM2 checkpoint"
echo ""
echo "🖥️ GPU 최적화:"
echo "  - GPU: $GPU_NAME"
echo "  - Compute Capability: $COMPUTE_CAP"
echo "  - CUDA Architecture: $ARCH_LIST"
echo ""
echo "🔧 CUDA 설정:"
echo "  - 시스템 CUDA: 시스템에 설치된 버전 (독립적)"
echo "  - Conda CUDA: 11.8 (sam3d_gui 환경 전용)"
echo "  - 위치: $CONDA_ENV_PATH"
echo ""
echo "📋 다음 단계:"
echo "  1. SAM 3D 체크포인트 다운로드 (아직 안 한 경우):"
echo "     ./download_sam3d.sh"
echo ""
echo "  2. 환경 활성화 및 테스트:"
echo "     conda activate sam3d_gui"
echo "     python -c 'import torch; print(torch.cuda.is_available())'"
echo ""
echo "  3. 웹 인터페이스 실행:"
echo "     ./run.sh"
echo ""
echo "============================================="
