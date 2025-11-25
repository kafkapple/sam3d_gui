#!/bin/bash
# SAM 3D GUI - 통합 환경 설정 및 모델 다운로드
# CUDA 11.8 호환, 상대 경로 기반, A6000 호환
set -e

# ==========================================
# 프로젝트 루트 경로 설정 (상대 경로 기반)
# ==========================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT"

echo "============================================="
echo "SAM 3D GUI 통합 환경 설정"
echo "============================================="
echo "프로젝트 루트: $PROJECT_ROOT"
echo ""

# ==========================================
# 1. Conda 환경 생성
# ==========================================
echo "[1/6] Conda 환경 생성..."
if conda env list | grep -q "^sam3d_gui "; then
    echo "기존 sam3d_gui 환경 발견. 제거 후 재생성합니다."
    conda env remove -n sam3d_gui -y
fi

conda create -n sam3d_gui python=3.10 -y
echo "✅ Python 3.10 환경 생성 완료"

# ==========================================
# 2. PyTorch 2.0.0 + CUDA 11.8 설치
# ==========================================
echo ""
echo "[2/6] PyTorch 2.0.0 + CUDA 11.8 설치..."
conda run -n sam3d_gui pip install \
    torch==2.0.0 \
    torchvision==0.15.0 \
    torchaudio==2.0.0 \
    --index-url https://download.pytorch.org/whl/cu118

# 검증
TORCH_VERSION=$(conda run -n sam3d_gui python -c "import torch; print(torch.__version__)" 2>/dev/null)
if [[ "$TORCH_VERSION" != "2.0.0+cu118" ]]; then
    echo "❌ PyTorch 설치 실패: $TORCH_VERSION"
    exit 1
fi
echo "✅ PyTorch 2.0.0+cu118 설치 완료"

# ==========================================
# 3. NumPy 버전 고정 (Kaolin 요구사항)
# ==========================================
echo ""
echo "[3/6] NumPy 1.x 설치..."
conda run -n sam3d_gui pip install "numpy<2"
echo "✅ NumPy 1.x 설치 완료"

# ==========================================
# 4. Kaolin 설치 (CUDA 11.8 호환)
# ==========================================
echo ""
echo "[4/6] Kaolin 0.17.0 설치 (15-20분 소요)..."
conda run -n sam3d_gui pip install ninja setuptools wheel cython packaging

# GPU architecture 설정 (A6000, RTX 3060 모두 지원)
export TORCH_CUDA_ARCH_LIST="8.0;8.6"  # A6000=8.0, RTX 3060=8.6
export FORCE_CUDA=1

conda run -n sam3d_gui pip install --no-build-isolation \
    git+https://github.com/NVIDIAGameWorks/kaolin.git@v0.17.0

# PyTorch 버전 재확인
TORCH_VERSION=$(conda run -n sam3d_gui python -c "import torch; print(torch.__version__)" 2>/dev/null)
if [[ "$TORCH_VERSION" != "2.0.0+cu118" ]]; then
    echo "❌ Kaolin이 PyTorch를 변경했습니다: $TORCH_VERSION"
    exit 1
fi
echo "✅ Kaolin 설치 완료, PyTorch 2.0.0 유지됨"

# ==========================================
# 5. pytorch3d, gsplat 설치
# ==========================================
echo ""
echo "[5/6] pytorch3d 및 기타 3D 라이브러리 설치..."

# gsplat
conda run -n sam3d_gui pip install --no-build-isolation \
    git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7

# pytorch3d
export TORCH_CUDA_ARCH_LIST='8.0;8.6'
export FORCE_CUDA=1
conda run -n sam3d_gui pip install --no-build-isolation \
    'git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7'

echo "✅ pytorch3d, gsplat 설치 완료"

# ==========================================
# 6. 기타 의존성 설치
# ==========================================
echo ""
echo "[6/6] 기타 의존성 설치..."

# SAM 3D 의존성 (Lightning 제외)
conda run -n sam3d_gui pip install \
    loguru timm optree astor \
    huggingface_hub safetensors pyyaml \
    --no-deps

conda run -n sam3d_gui pip install \
    huggingface_hub safetensors pyyaml

# SAM 3D 추가 의존성
conda run -n sam3d_gui pip install \
    spconv-cu118==2.3.8 \
    xatlas roma einops-exts \
    av decord open3d trimesh \
    pyvista pymeshfix pyrender \
    python-igraph \
    easydict point-cloud-utils polyscope \
    plyfile gdown rootutils \
    git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b

# NumPy 버전 재확인 및 고정
conda run -n sam3d_gui pip install "numpy<2" --force-reinstall

# Web UI 및 기타 도구
conda run -n sam3d_gui pip install -r "$PROJECT_ROOT/requirements.txt" || echo "⚠️ 일부 패키지 설치 실패 (핵심은 완료)"

echo "✅ 기타 의존성 설치 완료"

# ==========================================
# 모델 다운로드
# ==========================================
echo ""
echo "============================================="
echo "모델 체크포인트 다운로드"
echo "============================================="

# 체크포인트 디렉토리 생성 (프로젝트 루트 기준)
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints"
mkdir -p "$CHECKPOINT_DIR"

# SAM2 체크포인트 다운로드
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

# SAM 3D 체크포인트 다운로드
SAM3D_DIR="$PROJECT_ROOT/external/sam-3d-objects/checkpoints/hf"

if [ ! -d "$PROJECT_ROOT/external/sam-3d-objects" ]; then
    echo "⚠️  SAM 3D submodule이 없습니다."
    echo "다음 명령으로 초기화하세요:"
    echo "  git submodule update --init --recursive"
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
# 설정 파일 업데이트 (상대 경로로)
# ==========================================
CONFIG_FILE="$PROJECT_ROOT/config/model_config.yaml"
if [ -f "$CONFIG_FILE" ]; then
    echo ""
    echo "설정 파일 업데이트 중..."
    # 절대 경로를 상대 경로로 변경
    sed -i "s|/home/[^/]*/dev/sam3d_gui|$PROJECT_ROOT|g" "$CONFIG_FILE"
    echo "✅ 설정 파일 업데이트 완료"
fi

# ==========================================
# 완료 메시지
# ==========================================
echo ""
echo "============================================="
echo "🎉 환경 설정 완료!"
echo "============================================="
echo ""
echo "✅ 설치 완료:"
echo "  - Python 3.10"
echo "  - PyTorch 2.0.0 + CUDA 11.8"
echo "  - NumPy 1.x"
echo "  - Kaolin 0.17.0"
echo "  - pytorch3d 0.7.7"
echo "  - gsplat"
echo "  - SAM 3D dependencies (Lightning 제외)"
echo "  - SAM2 checkpoint"
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
