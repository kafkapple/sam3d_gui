#!/bin/bash
# SAM 3D GUI - 웹 인터페이스 실행 (상대 경로 기반)
export LIDRA_SKIP_INIT=1

# 프로젝트 루트 경로 설정
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT"

# Ctrl+C 시그널 처리 - 자식 프로세스도 함께 종료
cleanup() {
    echo ""
    echo "종료 중... 프로세스 정리"
    # 현재 스크립트의 자식 프로세스 그룹 종료
    pkill -P $$ 2>/dev/null
    # web_app.py 프로세스 직접 종료
    pkill -f "python.*web_app.py" 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

# 기존 프로세스 확인 및 정리
check_existing_process() {
    local existing_pid=$(pgrep -f "python.*web_app.py" 2>/dev/null)
    if [[ -n "$existing_pid" ]]; then
        echo "⚠️  기존 sam3d_gui 프로세스 발견 (PID: $existing_pid)"
        read -p "기존 프로세스를 종료하시겠습니까? (y/N): " answer
        if [[ "$answer" =~ ^[Yy]$ ]]; then
            kill $existing_pid 2>/dev/null
            sleep 1
            # 강제 종료 필요시
            if ps -p $existing_pid > /dev/null 2>&1; then
                kill -9 $existing_pid 2>/dev/null
            fi
            echo "✅ 기존 프로세스 종료됨"
        else
            echo "기존 프로세스가 실행 중입니다. 새 인스턴스를 시작하지 않습니다."
            exit 1
        fi
    fi
}

# 디버그 모드 확인
DEBUG_MODE=0
if [[ "$1" == "--debug" ]] || [[ "$1" == "-d" ]]; then
    DEBUG_MODE=1
    export SAM3D_DEBUG=1
    export PYTHONUNBUFFERED=1
fi

echo "==========================================="
echo "SAM 3D GUI - 웹 인터페이스"
echo "==========================================="
echo "프로젝트 루트: $PROJECT_ROOT"
if [[ "$DEBUG_MODE" == "1" ]]; then
    echo "🔧 디버그 모드: ON"
fi
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

# 기존 프로세스 확인
check_existing_process

# 환경 변수 설정 (SAM3D 초기화 스킵)
export LIDRA_SKIP_INIT=1

# CUDA 환경 설정 (nvdiffrast JIT 컴파일용)
# 시스템에 여러 CUDA 버전이 있을 경우, PyTorch와 맞는 버전 사용
if [[ -d "/usr/local/cuda-11.8" ]]; then
    export CUDA_HOME=/usr/local/cuda-11.8
    export PATH=/usr/local/cuda-11.8/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
    echo "✓ CUDA 11.8 환경 설정됨 (Texture Baking 지원)"
fi

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
echo "💡 디버그 모드: ./run.sh --debug"
echo ""

# 웹 앱 실행 (상대 경로)
# conda run은 부모 shell의 환경변수(CUDA_HOME 등)를 상속하지 않음
# 따라서 직접 conda activate 후 실행
eval "$(conda shell.bash hook)"
conda activate sam3d_gui

if [[ "$DEBUG_MODE" == "1" ]]; then
    # 디버그 모드: 출력 버퍼링 없이 실행
    python -u "$PROJECT_ROOT/src/web_app.py"
else
    python "$PROJECT_ROOT/src/web_app.py"
fi

echo ""
echo "서버 종료됨."
