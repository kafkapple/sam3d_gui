#!/bin/bash
# SAM 3D GUI - 웹 인터페이스 실행 (상대 경로 기반)

# 프로젝트 루트 경로 설정
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT"

echo "==========================================="
echo "SAM 3D GUI - 웹 인터페이스"
echo "==========================================="
echo "프로젝트 루트: $PROJECT_ROOT"
echo ""

# Conda 환경 확인
if ! conda env list | grep -q "^sam3d_gui "; then
    echo "❌ Conda 환경 'sam3d_gui'가 없습니다."
    echo ""
    echo "먼저 환경을 설정하세요:"
    echo "  ./setup.sh"
    echo ""
    exit 1
fi

# 환경 변수 설정 (SAM3D 초기화 스킵)
export LIDRA_SKIP_INIT=1

echo "Conda 환경 활성화: sam3d_gui"
echo "웹 인터페이스 시작..."
echo ""
echo "📱 접속 주소:"
echo "  - 로컬:  http://localhost:7860"
echo "  - 네트워크: http://$(hostname -I | awk '{print $1}'):7860"
echo ""
echo "🎬 기능:"
echo "  Tab 1: 🚀 Quick Mode - 자동 세그멘테이션 & 모션 감지"
echo "  Tab 2: 🎨 Interactive Mode - Point annotation & Propagation"
echo "  Tab 3: 📦 Batch Processing - 대량 비디오 처리"
echo "  Tab 4: 🎯 Data Augmentation - RGB + Mask 증강"
echo ""
echo "종료: Ctrl+C"
echo ""

# 웹 앱 실행 (상대 경로)
conda run -n sam3d_gui python "$PROJECT_ROOT/src/web_app.py"

echo ""
echo "서버 종료됨."
