#!/bin/bash
# SAM2 환경 진단 스크립트
# GPU05 서버에서 "Contour (fallback)" 문제 진단용

echo "============================================="
echo "SAM2 환경 진단"
echo "============================================="
echo ""

# 1. NumPy 버전 확인
echo "[1/5] NumPy 버전 확인..."
NUMPY_VERSION=$(conda run -n sam3d_gui python -c "import numpy; print(numpy.__version__)" 2>&1)
echo "NumPy: $NUMPY_VERSION"

NUMPY_MAJOR=$(echo $NUMPY_VERSION | cut -d. -f1 2>/dev/null)
if [[ "$NUMPY_MAJOR" -ge 2 ]]; then
    echo "❌ NumPy 2.x 감지! (PyTorch 2.0.0과 호환 안됨)"
    echo "   해결: conda run -n sam3d_gui pip install 'numpy<2' --force-reinstall"
else
    echo "✅ NumPy 1.x 확인됨"
fi
echo ""

# 2. PyTorch 버전 확인
echo "[2/5] PyTorch 버전 확인..."
TORCH_VERSION=$(conda run -n sam3d_gui python -c "import torch; print(torch.__version__)" 2>&1)
echo "PyTorch: $TORCH_VERSION"

if [[ "$TORCH_VERSION" != "2.0.0+cu118" ]]; then
    echo "⚠️  예상 버전: 2.0.0+cu118"
fi
echo ""

# 3. SAM2 패키지 설치 확인
echo "[3/5] SAM2 패키지 설치 확인..."
SAM2_INSTALLED=$(conda run -n sam3d_gui pip list | grep SAM-2)
if [[ -z "$SAM2_INSTALLED" ]]; then
    echo "❌ SAM2 패키지 미설치"
    echo "   해결: conda run -n sam3d_gui pip install --no-deps git+https://github.com/facebookresearch/segment-anything-2.git"
else
    echo "✅ SAM2 패키지 설치됨: $SAM2_INSTALLED"
fi
echo ""

# 4. SAM2 import 테스트
echo "[4/5] SAM2 import 테스트..."
SAM2_IMPORT=$(conda run -n sam3d_gui python -c "from sam2.sam2_image_predictor import SAM2ImagePredictor; print('SUCCESS')" 2>&1)

if [[ "$SAM2_IMPORT" == *"SUCCESS"* ]]; then
    echo "✅ SAM2 import 성공"
elif [[ "$SAM2_IMPORT" == *"NumPy"* ]]; then
    echo "❌ NumPy 호환성 오류 감지"
    echo "   에러: $SAM2_IMPORT"
    echo "   해결: conda run -n sam3d_gui pip install 'numpy<2' --force-reinstall"
else
    echo "❌ SAM2 import 실패"
    echo "   에러: $SAM2_IMPORT"
fi
echo ""

# 5. SAM2 체크포인트 확인
echo "[5/5] SAM2 체크포인트 확인..."
# 프로젝트 루트 자동 감지 (dev 폴더 있거나 없거나)
if [ -d "$HOME/dev/sam3d_gui" ]; then
    PROJECT_ROOT="$HOME/dev/sam3d_gui"
elif [ -d "$HOME/sam3d_gui" ]; then
    PROJECT_ROOT="$HOME/sam3d_gui"
else
    PROJECT_ROOT="$(pwd)"
fi

SAM2_CHECKPOINT="$PROJECT_ROOT/external/sam2/checkpoints/sam2_hiera_large.pt"
if [ -f "$SAM2_CHECKPOINT" ]; then
    SIZE=$(du -h "$SAM2_CHECKPOINT" | cut -f1)
    echo "✅ SAM2 체크포인트 발견: $SIZE"
    echo "   위치: $SAM2_CHECKPOINT"
else
    echo "⚠️  SAM2 체크포인트 없음"
    echo "   예상 위치: $SAM2_CHECKPOINT"
    echo "   해결: ./setup_system_cuda.sh 또는 ./setup_conda_cuda.sh 실행"
fi
echo ""

# 종합 진단
echo "============================================="
echo "진단 요약"
echo "============================================="

ISSUES=0

if [[ "$NUMPY_MAJOR" -ge 2 ]]; then
    echo "❌ NumPy 2.x → NumPy 1.x로 다운그레이드 필요"
    ((ISSUES++))
fi

if [[ -z "$SAM2_INSTALLED" ]]; then
    echo "❌ SAM2 패키지 미설치 → 설치 필요"
    ((ISSUES++))
fi

if [[ "$SAM2_IMPORT" != *"SUCCESS"* ]]; then
    echo "❌ SAM2 import 실패 → NumPy 버전 또는 패키지 설치 확인"
    ((ISSUES++))
fi

if [[ $ISSUES -eq 0 ]]; then
    echo "✅ 모든 검사 통과! SAM2가 정상 작동해야 합니다."
    echo ""
    echo "만약 여전히 'Contour (fallback)'이 표시되면:"
    echo "  1. 웹 서버 재시작: Ctrl+C 후 ./run.sh"
    echo "  2. 캐시 초기화: 브라우저 새로고침 (Ctrl+Shift+R)"
else
    echo ""
    echo "⚠️  $ISSUES 개 문제 발견. 위 해결 방법을 따라주세요."
    echo ""
    echo "빠른 수정 명령어:"
    echo "  conda run -n sam3d_gui pip install 'numpy<2' --force-reinstall"
    echo "  conda run -n sam3d_gui pip install --no-deps git+https://github.com/facebookresearch/segment-anything-2.git"
fi

echo ""
echo "============================================="
