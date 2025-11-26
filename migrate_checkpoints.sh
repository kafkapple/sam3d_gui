#!/bin/bash
# SAM 3D GUI - 체크포인트 마이그레이션 스크립트
#
# 기존 여러 위치에 흩어진 체크포인트를 통합 구조로 이동합니다:
#   checkpoints/
#   ├── sam2/     # SAM2 체크포인트
#   └── sam3d/    # SAM3D 체크포인트

set -e

echo "=========================================="
echo "SAM 3D GUI - 체크포인트 마이그레이션"
echo "=========================================="
echo ""

# 프로젝트 루트 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints"

echo "프로젝트 루트: $PROJECT_ROOT"
echo "통합 체크포인트 디렉토리: $CHECKPOINT_DIR"
echo ""

# 디렉토리 생성
mkdir -p "$CHECKPOINT_DIR/sam2"
mkdir -p "$CHECKPOINT_DIR/sam3d"

# ==========================================
# 1. SAM2 체크포인트 마이그레이션
# ==========================================
echo "[1/2] SAM2 체크포인트 마이그레이션"
echo "----------------------------------------"

SAM2_SOURCES=(
    "$PROJECT_ROOT/external/sam2/checkpoints"
    "$HOME/dev/segment-anything-2/checkpoints"
    "$HOME/segment-anything-2/checkpoints"
)

SAM2_DEST="$CHECKPOINT_DIR/sam2"
SAM2_MIGRATED=false

for src in "${SAM2_SOURCES[@]}"; do
    if [ -d "$src" ] && [ "$(ls -A "$src"/*.pt 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "발견: $src"
        for f in "$src"/*.pt; do
            if [ -f "$f" ]; then
                fname=$(basename "$f")
                if [ ! -f "$SAM2_DEST/$fname" ]; then
                    echo "  이동: $fname"
                    cp "$f" "$SAM2_DEST/"
                    SAM2_MIGRATED=true
                else
                    echo "  스킵 (이미 존재): $fname"
                fi
            fi
        done
    fi
done

if [ "$SAM2_MIGRATED" = true ]; then
    echo "✅ SAM2 마이그레이션 완료"
else
    echo "ℹ️  마이그레이션할 SAM2 체크포인트 없음"
fi
echo ""

# ==========================================
# 2. SAM3D 체크포인트 마이그레이션
# ==========================================
echo "[2/2] SAM3D 체크포인트 마이그레이션"
echo "----------------------------------------"

SAM3D_SOURCES=(
    "$PROJECT_ROOT/external/sam-3d-objects/checkpoints/hf"
    "$PROJECT_ROOT/external/sam-3d-objects/checkpoints/external/sam-3d-objects/checkpoints/hf"
    "$PROJECT_ROOT/external/sam-3d-objects/checkpoints/external/sam-3d-objects/checkpoints/hf/checkpoints"
    "$HOME/sam-3d-objects/checkpoints/hf"
    "$HOME/dev/sam-3d-objects/checkpoints/hf"
)

SAM3D_DEST="$CHECKPOINT_DIR/sam3d"
SAM3D_MIGRATED=false

for src in "${SAM3D_SOURCES[@]}"; do
    if [ -d "$src" ] && [ "$(ls -A "$src"/*.ckpt 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "발견: $src"
        for f in "$src"/*.ckpt "$src"/*.pt "$src"/*.yaml "$src"/*.safetensors; do
            if [ -f "$f" ]; then
                fname=$(basename "$f")
                if [ ! -f "$SAM3D_DEST/$fname" ]; then
                    echo "  이동: $fname"
                    cp "$f" "$SAM3D_DEST/"
                    SAM3D_MIGRATED=true
                else
                    echo "  스킵 (이미 존재): $fname"
                fi
            fi
        done
    fi
done

if [ "$SAM3D_MIGRATED" = true ]; then
    echo "✅ SAM3D 마이그레이션 완료"
else
    echo "ℹ️  마이그레이션할 SAM3D 체크포인트 없음"
fi
echo ""

# ==========================================
# 3. 중복 디렉토리 정리 제안
# ==========================================
echo "=========================================="
echo "정리 제안"
echo "=========================================="
echo ""
echo "마이그레이션 후 다음 디렉토리를 삭제할 수 있습니다:"
echo ""

# 중복된 레거시 디렉토리 확인
CLEANUP_CANDIDATES=(
    "$PROJECT_ROOT/external/sam2/checkpoints"
    "$PROJECT_ROOT/external/sam-3d-objects/checkpoints"
)

for dir in "${CLEANUP_CANDIDATES[@]}"; do
    if [ -d "$dir" ]; then
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "  - $dir ($size)"
    fi
done

echo ""
echo "⚠️  삭제 전 백업을 권장합니다."
echo ""
echo "삭제 명령어:"
echo "  rm -rf $PROJECT_ROOT/external/sam2/checkpoints"
echo "  rm -rf $PROJECT_ROOT/external/sam-3d-objects/checkpoints"
echo ""

# ==========================================
# 완료 메시지
# ==========================================
echo "=========================================="
echo "✅ 마이그레이션 완료!"
echo "=========================================="
echo ""
echo "통합 체크포인트 위치:"
echo "  SAM2:  $SAM2_DEST"
echo "  SAM3D: $SAM3D_DEST"
echo ""
echo "파일 목록:"
echo "--- SAM2 ---"
ls -lh "$SAM2_DEST"/*.pt 2>/dev/null || echo "  (없음)"
echo ""
echo "--- SAM3D ---"
ls -lh "$SAM3D_DEST"/*.ckpt 2>/dev/null | head -5 || echo "  (없음)"
