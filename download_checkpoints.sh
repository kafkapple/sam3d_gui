#!/bin/bash
# SAM 3D GUI - 통합 체크포인트 다운로드 스크립트
#
# 체크포인트 구조:
#   checkpoints/
#   ├── sam2/     # SAM2 체크포인트
#   │   └── sam2_hiera_large.pt
#   └── sam3d/    # SAM3D 체크포인트 (HuggingFace clone)
#       ├── pipeline.yaml
#       ├── slat_*.ckpt
#       └── ss_*.ckpt

set -e

echo "=========================================="
echo "SAM 3D GUI - 체크포인트 다운로드"
echo "=========================================="
echo ""

# 프로젝트 루트 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints"

echo "프로젝트 루트: $PROJECT_ROOT"
echo "체크포인트 디렉토리: $CHECKPOINT_DIR"
echo ""

# ==========================================
# 1. SAM2 체크포인트 다운로드
# ==========================================
echo "[1/2] SAM2 체크포인트"
echo "----------------------------------------"

SAM2_DIR="$CHECKPOINT_DIR/sam2"
mkdir -p "$SAM2_DIR"

if [ -f "$SAM2_DIR/sam2_hiera_large.pt" ]; then
    echo "✅ SAM2 체크포인트 이미 존재"
    ls -lh "$SAM2_DIR/sam2_hiera_large.pt"
else
    echo "📥 SAM2 체크포인트 다운로드 중..."
    cd "$SAM2_DIR"
    wget -q --show-progress https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
    echo "✅ SAM2 다운로드 완료"
    cd "$PROJECT_ROOT"
fi
echo ""

# ==========================================
# 2. SAM3D 체크포인트 다운로드
# ==========================================
echo "[2/2] SAM3D 체크포인트"
echo "----------------------------------------"

SAM3D_DIR="$CHECKPOINT_DIR/sam3d"

# .env 파일에서 HuggingFace 토큰 로드
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "✓ .env 파일 로드 중..."
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | grep -v '^$' | xargs)
fi

# HuggingFace 토큰 확인
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  HF_TOKEN이 설정되지 않았습니다."
    echo ""
    echo "HuggingFace 토큰이 필요합니다:"
    echo "  1. https://huggingface.co/settings/tokens 에서 토큰 생성"
    echo "  2. .env 파일에 HF_TOKEN=your_actual_token 추가"
    echo ""

    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        echo "📋 .env 파일 생성..."
        echo "# HuggingFace 토큰 (SAM3D 다운로드용)" > "$PROJECT_ROOT/.env"
        echo "HF_TOKEN=your_token_here" >> "$PROJECT_ROOT/.env"
        echo ""
        echo "⚠️  .env 파일이 생성되었습니다. HF_TOKEN을 설정하세요:"
        echo "   nano $PROJECT_ROOT/.env"
    fi

    echo "토큰 없이 계속 진행하시겠습니까? (인증 실패 가능) (y/n)"
    read -p "> " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
elif [ "$HF_TOKEN" = "your_token_here" ]; then
    echo "❌ HF_TOKEN이 기본값입니다. 실제 토큰으로 교체하세요:"
    echo "   nano $PROJECT_ROOT/.env"
    exit 1
else
    echo "✓ HuggingFace 토큰 감지됨 (${HF_TOKEN:0:8}...)"
fi

# Git LFS 확인
if ! command -v git-lfs &> /dev/null; then
    echo "⚠️  Git LFS가 설치되어 있지 않습니다."
    echo ""
    echo "설치 방법:"
    echo "  Ubuntu/Debian: sudo apt-get install git-lfs"
    echo "  Conda: conda install -c conda-forge git-lfs"
    echo ""
    read -p "Git LFS를 설치하시겠습니까? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo apt-get update && sudo apt-get install -y git-lfs
        git lfs install
    else
        echo "Git LFS 없이 계속 진행합니다 (대용량 파일 다운로드 실패 가능)"
    fi
else
    echo "✓ Git LFS 감지됨"
    git lfs install --skip-smudge 2>/dev/null || true
fi

# SAM3D 체크포인트 다운로드
if [ -d "$SAM3D_DIR" ] && [ "$(ls -A $SAM3D_DIR/*.ckpt 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "✅ SAM3D 체크포인트 이미 존재"
    ls -lh "$SAM3D_DIR"/*.ckpt 2>/dev/null | head -5
else
    echo "📥 SAM3D 체크포인트 다운로드 중..."
    echo "   (HuggingFace에서 ~2GB 다운로드, 시간 소요)"
    echo ""

    # 임시 디렉토리에 클론
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"

    if [ -n "$HF_TOKEN" ]; then
        git clone --depth 1 https://oauth2:${HF_TOKEN}@huggingface.co/facebook/sam-3d-objects .
    else
        git clone --depth 1 https://huggingface.co/facebook/sam-3d-objects .
    fi

    # LFS 파일 pull
    git lfs pull

    # 체크포인트만 복사 (소스코드 제외)
    mkdir -p "$SAM3D_DIR"
    cp -v *.ckpt *.pt *.yaml *.safetensors "$SAM3D_DIR/" 2>/dev/null || true

    # 정리
    cd "$PROJECT_ROOT"
    rm -rf "$TEMP_DIR"

    echo "✅ SAM3D 다운로드 완료"
fi
echo ""

# ==========================================
# 완료 메시지
# ==========================================
echo "=========================================="
echo "✅ 체크포인트 다운로드 완료!"
echo "=========================================="
echo ""
echo "체크포인트 위치:"
echo "  SAM2:  $SAM2_DIR"
echo "  SAM3D: $SAM3D_DIR"
echo ""
echo "파일 목록:"
echo "--- SAM2 ---"
ls -lh "$SAM2_DIR" 2>/dev/null || echo "  (없음)"
echo ""
echo "--- SAM3D ---"
ls -lh "$SAM3D_DIR"/*.ckpt 2>/dev/null | head -5 || echo "  (없음)"
echo ""
echo "이제 ./run.sh 로 웹 인터페이스를 실행할 수 있습니다."
