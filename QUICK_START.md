# SAM 3D GUI - Quick Start Guide

## ⚠️ 첫 실행 전 필수 설정

### HuggingFace 인증 설정

SAM 3D 모델 다운로드를 위해 **HuggingFace 토큰이 필요**합니다.

```bash
# 1. 모델 액세스 요청
# https://huggingface.co/facebook/sam-3d-objects 에서 "Request access" 클릭

# 2. 토큰 생성
# https://huggingface.co/settings/tokens 에서 Read 권한 토큰 생성

# 3. .env 파일에 토큰 추가
cd /home/joon/dev/sam3d_gui
echo 'HF_TOKEN="hf_your_token_here"' > .env
```

**자세한 설정 방법**: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md#huggingface-인증-설정-중요)

---

## 🚀 서버 실행 방법

### 기본 실행
```bash
cd /home/joon/dev/sam3d_gui
./run.sh
```

### 포트 변경하여 실행
```bash
# 7861 포트로 실행
GRADIO_SERVER_PORT=7861 ./run.sh

# 또는 export로 설정
export GRADIO_SERVER_PORT=7862
./run.sh
```

### 백그라운드 실행
```bash
nohup ./run.sh > /tmp/sam_gui.log 2>&1 &

# 로그 확인
tail -f /tmp/sam_gui.log
```

---

## 🛑 서버 종료 방법

### 모든 서버 종료
```bash
pkill -f "web_app.py"
```

### 특정 프로세스 종료
```bash
# 프로세스 확인
ps aux | grep "[w]eb_app.py"

# PID로 종료
kill <PID>
```

---

## 🌐 접속 방법

### 로컬 접속
```
http://localhost:7860
```

### 네트워크 접속
```
http://192.168.45.10:7860
```

### 포트 변경 시
```
http://localhost:<GRADIO_SERVER_PORT>
```

---

## ⚙️ CUDA/GPU 설정

### 현재 설정 (config/model_config.yaml)
```yaml
sam2:
  device: "auto"  # GPU 자동 감지
  # RTX 3060, A6000 지원
```

### GPU 상태 확인
```bash
# CUDA 사용 가능 확인
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# GPU 이름 확인
python -c "import torch; print(torch.cuda.get_device_name(0))"

# nvidia-smi로 확인
nvidia-smi
```

### 강제로 CPU 사용
`config/model_config.yaml` 수정:
```yaml
sam2:
  device: "cpu"  # auto → cpu
```

### 강제로 CUDA 사용
```yaml
sam2:
  device: "cuda"  # auto → cuda
```

---

## 🎨 Interactive Mode (기본 탭)

현재 기본 탭은 **🎨 Interactive Mode**입니다.

### Quick Mode로 변경하려면
`src/web_app.py:948` 수정:
```python
# 현재
with gr.Tabs():

# Quick Mode를 기본으로
with gr.Tabs(selected=0):
```

---

## 📝 로그 확인

### 서버 시작 로그
```bash
# 실행 직후
tail -30 /tmp/sam_gui.log

# 실시간 모니터링
tail -f /tmp/sam_gui.log
```

### GPU 사용 확인
실행 시 다음과 같은 메시지 확인:
```
✓ CUDA detected: NVIDIA GeForce RTX 3060
✓ SAM 2 loaded: SAM 2.1 Hiera Large on cuda
```

CPU 사용 시:
```
Warning: CUDA not available, using CPU
✓ SAM 2 loaded: SAM 2.1 Hiera Large on cpu
```

---

## 🐛 문제 해결

### 오류 1: "Cannot find empty port"
**증상:**
```
OSError: Cannot find empty port in range: 7860-7860
```

**해결:**
```bash
# 방법 1: 기존 서버 종료
pkill -f "web_app.py"
./run.sh

# 방법 2: 다른 포트 사용
GRADIO_SERVER_PORT=7861 ./run.sh
```

---

### 오류 2: "CUDA not available"
**증상:**
```
Warning: CUDA not available, using CPU
```

**해결:**
```bash
# CUDA 확인
nvidia-smi

# PyTorch CUDA 확인
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# conda 환경 재설치 (필요시)
conda activate sam3d_gui
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

---

### 오류 3: SAM 2 로딩 실패
**증상:**
```
Warning: SAM 2 checkpoint not found at ...
```

**해결:**
```bash
# 체크포인트 경로 확인
ls -lh ~/dev/segment-anything-2/checkpoints/sam2.1_hiera_large.pt

# config 확인
cat config/model_config.yaml | grep checkpoint
```

---

## 🔧 고급 설정

### 동시 처리 스레드 수 변경
`src/web_app.py:1269` 수정:
```python
max_threads=40  # 기본값, 필요시 증가
```

### 디버그 모드 끄기
`src/web_app.py:1268` 수정:
```python
debug=False  # True → False
```

### 외부 접속 차단
`src/web_app.py:1265` 수정:
```python
server_name="127.0.0.1",  # "0.0.0.0" → "127.0.0.1"
```

---

## 📊 시스템 요구사항

### 최소 요구사항
- **GPU**: RTX 3060 이상 (12GB VRAM)
- **RAM**: 16GB 이상
- **디스크**: 20GB 이상 (체크포인트 + 데이터)

### 권장 사양
- **GPU**: A6000 (48GB VRAM)
- **RAM**: 32GB 이상
- **디스크**: 100GB 이상

### 지원 GPU
- NVIDIA RTX 3060 (12GB) - ✅ 테스트 완료
- NVIDIA A6000 (48GB) - ✅ 지원 (자동 감지)
- 기타 CUDA 호환 GPU - ✅ 지원

---

## 📚 추가 문서

- [UPDATES_LOG.md](UPDATES_LOG.md) - 전체 업데이트 내역
- [SESSION_MANAGEMENT.md](docs/SESSION_MANAGEMENT.md) - 세션 저장/로드 가이드
- [README_CHECKPOINTS.md](README_CHECKPOINTS.md) - 체크포인트 관리
- [config/model_config.yaml](config/model_config.yaml) - 모델 설정

---

**작성일**: 2025-11-24
**버전**: 1.0
**상태**: ✅ GPU 자동 감지 지원
