#!/bin/bash
# SAM 3D 체크포인트 다운로드 스크립트

echo "=========================================="
echo "SAM 3D Objects - 체크포인트 다운로드"
echo "=========================================="
echo ""

# .env 파일에서 HuggingFace 토큰 로드
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
    echo "✓ .env 파일 로드 중..."
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
else
    echo "⚠️  .env 파일을 찾을 수 없습니다."
    echo "   .env.example을 복사하여 .env를 생성하고 HF_TOKEN을 설정하세요."
fi

# HuggingFace 토큰 확인
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  HF_TOKEN이 설정되지 않았습니다."
    echo ""
    echo "HuggingFace 토큰이 필요합니다:"
    echo "  1. https://huggingface.co/settings/tokens 에서 토큰 생성"
    echo "  2. .env 파일에 HF_TOKEN=your_token 추가"
    echo ""
    echo "계속 진행하시겠습니까? (인증 실패 가능) (y/n)"
    read -p "> " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ HuggingFace 토큰 감지됨"
fi

# 체크포인트 디렉토리 생성
CHECKPOINT_DIR="$HOME/dev/sam-3d-objects/checkpoints/hf"
mkdir -p "$CHECKPOINT_DIR"

echo "다운로드 위치: $CHECKPOINT_DIR"
echo ""

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
    git lfs install
fi

echo ""
echo "HuggingFace에서 SAM 3D 체크포인트 다운로드 중..."
echo ""

cd "$HOME/dev/sam-3d-objects"

# HuggingFace 레포지토리 클론 (토큰 인증 사용)
if [ ! -d "checkpoints/hf/.git" ]; then
    echo "📥 체크포인트 다운로드 시작..."

    if [ -n "$HF_TOKEN" ]; then
        # 토큰을 사용하여 인증
        echo "   HuggingFace 토큰으로 인증 중..."
        git clone https://oauth2:${HF_TOKEN}@huggingface.co/facebook/sam-3d-objects checkpoints/hf
    else
        # 토큰 없이 시도 (실패 가능)
        git clone https://huggingface.co/facebook/sam-3d-objects checkpoints/hf
    fi
else
    echo "📥 체크포인트 업데이트 중..."
    cd checkpoints/hf

    if [ -n "$HF_TOKEN" ]; then
        # 토큰을 사용하여 인증
        git pull https://oauth2:${HF_TOKEN}@huggingface.co/facebook/sam-3d-objects
    else
        git pull
    fi
    cd ../..
fi

echo ""
echo "=========================================="
echo "다운로드 완료!"
echo "=========================================="
echo ""
echo "체크포인트 위치: $CHECKPOINT_DIR"
echo ""
echo "다운로드된 파일:"
ls -lh "$CHECKPOINT_DIR"

echo ""
echo "이제 web GUI에서 'Generate 3D Mesh'를 사용할 수 있습니다."
