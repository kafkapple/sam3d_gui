# 환경 진단 가이드 - 새 서버 setup 전 필수 체크

## 📋 빠른 시작

새로운 서버에서 `setup.sh` 실행 전:

```bash
cd /path/to/sam3d_gui
./check_environment.sh > environment_report.txt 2>&1
cat environment_report.txt
```

---

## 🔍 수동 체크 명령어 모음

### 1. 운영체제 정보

```bash
# OS 버전
cat /etc/os-release

# 커널 버전
uname -r

# 아키텍처
uname -m
```

**필요한 정보**: Ubuntu/CentOS 버전, 커널 버전

---

### 2. GPU 정보

```bash
# GPU 확인
nvidia-smi

# GPU 상세 정보
nvidia-smi --query-gpu=gpu_name,memory.total,driver_version,compute_cap --format=csv

# CUDA Compute Capability (중요!)
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

**필요한 정보**:
- GPU 모델 (예: RTX 3060, A6000)
- VRAM 크기 (예: 12GB, 48GB)
- Driver 버전 (예: 525.147.05)
- **Compute Capability** (예: 8.0, 8.6) ← 가장 중요!

**Compute Capability 매핑**:
```
7.5 = RTX 20xx, T4
8.0 = A100, A6000
8.6 = RTX 30xx (3060, 3070, 3080, 3090)
8.9 = RTX 40xx (4080, 4090)
```

---

### 3. CUDA 정보

```bash
# nvcc 확인
nvcc --version

# nvcc 위치
which nvcc

# CUDA 설치 위치 확인
ls -d /usr/local/cuda*

# CUDA 환경변수
echo $CUDA_HOME
echo $CUDA_PATH
echo $LD_LIBRARY_PATH
```

**필요한 정보**:
- CUDA 버전 (11.7, 11.8, 12.0 등)
- nvcc 경로
- CUDA_HOME 설정 여부

**추천**: CUDA 11.8 (Kaolin 0.17.0 요구사항)

---

### 4. Python & Conda

```bash
# Conda 확인
conda --version
which conda
conda info

# Python 확인
python --version
python3 --version

# 기존 환경 확인
conda env list
```

**필요한 정보**:
- Conda 버전
- Python 버전 (3.10 필요)
- 기존 sam3d_gui 환경 존재 여부

---

### 5. 컴파일러

```bash
# GCC 버전
gcc --version

# G++ 버전
g++ --version

# Make
make --version
```

**필요한 정보**:
- GCC 버전 (7.x 이상 권장)
- 컴파일러 설치 여부

**없으면**:
```bash
sudo apt install build-essential
```

---

### 6. FFmpeg 라이브러리 (PyAV 설치용)

```bash
# FFmpeg 확인
ffmpeg -version

# 라이브러리 확인
ldconfig -p | grep libavcodec
ldconfig -p | grep libavformat
ldconfig -p | grep libavutil
ldconfig -p | grep libswscale
ldconfig -p | grep libswresample

# pkg-config로 버전 확인
pkg-config --modversion libavcodec
pkg-config --modversion libavformat
```

**필요한 정보**:
- FFmpeg 버전
- libavcodec, libavformat, libavutil 존재 여부

**❌ 없으면 (PyAV 설치 실패 원인)**:
```bash
# 방법 1: 시스템 패키지
sudo apt update
sudo apt install ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev

# 방법 2: Conda (더 안전)
conda install -c conda-forge av
```

---

### 7. Git & Git LFS

```bash
# Git 확인
git --version

# Git LFS 확인
git-lfs --version
```

**필요한 정보**:
- Git LFS 설치 여부 (SAM 3D 체크포인트 다운로드용)

**없으면**:
```bash
# Conda (sudo 불필요)
conda install -c conda-forge git-lfs
git lfs install

# 또는 시스템 패키지
sudo apt install git-lfs
```

---

### 8. 디스크 공간

```bash
# 전체 디스크
df -h

# 홈 디렉토리
df -h /home

# 현재 디렉토리 크기
du -sh .
```

**필요한 정보**:
- 최소 20GB 여유 공간 (체크포인트 + 환경)
- 권장 100GB 이상

---

### 9. 네트워크

```bash
# GitHub 접속
ping -c 3 github.com

# HuggingFace 접속
ping -c 3 huggingface.co

# PyPI 접속
ping -c 3 pypi.org

# 프록시 확인
echo $http_proxy
echo $https_proxy
```

**필요한 정보**:
- 외부 네트워크 접속 가능 여부
- 프록시 설정

---

## 🐛 현재 오류 분석: PyAV 설치 실패

### 오류 메시지
```
av/filter/loudnorm_impl.c:86:43: error: 'AVCodecParameters' has no member named 'ch_layout'
error: command '/usr/bin/gcc' failed with exit code 1
```

### 원인
**FFmpeg 버전 불일치**:
- PyAV는 FFmpeg 5.0+ 필요
- 시스템에 구버전 FFmpeg 설치되어 있음
- `ch_layout`는 FFmpeg 5.0+에서 도입된 새 API

### 해결 방법

#### 방법 1: Conda로 av 설치 (가장 안전) ✅

```bash
# setup.sh 수정 전에 먼저 테스트
conda activate sam3d_gui

# Conda로 av 설치 (FFmpeg 포함)
conda install -c conda-forge av

# 확인
python -c "import av; print(av.__version__)"
```

**setup.sh 수정**:
```bash
# 기존 (line 105-122)
conda run -n sam3d_gui pip install \
    spconv-cu118==2.3.8 \
    xatlas roma einops-exts \
    av decord open3d trimesh \
    ...

# 수정 후
# av는 conda로 설치
conda run -n sam3d_gui conda install -c conda-forge av -y

# 나머지는 pip
conda run -n sam3d_gui pip install \
    spconv-cu118==2.3.8 \
    xatlas roma einops-exts \
    decord open3d trimesh \
    ...
```

#### 방법 2: 시스템 FFmpeg 업그레이드

```bash
# FFmpeg 버전 확인
ffmpeg -version

# Ubuntu 22.04+
sudo apt update
sudo apt install ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev

# Ubuntu 20.04 (PPA 필요)
sudo add-apt-repository ppa:savoury1/ffmpeg5
sudo apt update
sudo apt install ffmpeg libavcodec-dev libavformat-dev libavutil-dev

# 확인
pkg-config --modversion libavcodec  # 59.x 이상이어야 함
```

#### 방법 3: av 제외하고 설치

PyAV가 필수가 아니라면:

```bash
# av 없이 설치
conda run -n sam3d_gui pip install \
    spconv-cu118==2.3.8 \
    xatlas roma einops-exts \
    decord open3d trimesh \
    ...
```

---

## 📊 환경별 체크리스트

### 최소 요구사항

| 항목 | 최소 사양 | 권장 사양 |
|------|----------|----------|
| GPU | RTX 3060 12GB | A6000 48GB |
| VRAM | 12GB | 16GB+ |
| RAM | 16GB | 32GB+ |
| 디스크 | 20GB | 100GB+ |
| CUDA | 11.8 | 11.8 |
| Python | 3.10 | 3.10 |
| GCC | 7.x+ | 9.x+ |
| FFmpeg | 4.4+ | 5.0+ |

### GPU별 TORCH_CUDA_ARCH_LIST 설정

```bash
# RTX 3060/3070/3080/3090
export TORCH_CUDA_ARCH_LIST="8.6"

# A6000/A100
export TORCH_CUDA_ARCH_LIST="8.0"

# 두 GPU 모두 지원 (기본값)
export TORCH_CUDA_ARCH_LIST="8.0;8.6"

# RTX 20xx/T4
export TORCH_CUDA_ARCH_LIST="7.5"

# RTX 40xx
export TORCH_CUDA_ARCH_LIST="8.9"
```

---

## 🚨 일반적인 문제 해결

### 문제 1: CUDA not found
```bash
# nvcc 경로 추가
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 문제 2: GCC not found
```bash
sudo apt update
sudo apt install build-essential
```

### 문제 3: Out of Memory (컴파일 중)
```bash
# 스왑 메모리 추가
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 문제 4: Git LFS 없음
```bash
# Conda 설치 (sudo 불필요)
conda install -c conda-forge git-lfs
git lfs install
```

### 문제 5: Kaolin 컴파일 실패
```bash
# CUDA architecture 명시
export TORCH_CUDA_ARCH_LIST="8.6"  # GPU에 맞게 조정
export FORCE_CUDA=1

# 재시도
conda run -n sam3d_gui pip install --no-build-isolation \
    git+https://github.com/NVIDIAGameWorks/kaolin.git@v0.17.0
```

---

## 📝 환경 보고서 생성

다른 사람에게 공유하거나 문제 해결 시:

```bash
cd /path/to/sam3d_gui

# 전체 진단 실행
./check_environment.sh > environment_report_$(hostname)_$(date +%Y%m%d).txt 2>&1

# 압축하여 전송
tar -czf environment_report.tar.gz environment_report_*.txt

# 또는 직접 확인
cat environment_report_*.txt
```

보고서 포함 정보:
- OS, CPU, RAM
- GPU, CUDA, nvcc
- Python, Conda
- GCC, FFmpeg
- Git, Git LFS
- 디스크 공간
- 네트워크 연결
- 기존 설치 확인
- 추천 설정

---

## 🎯 체크리스트 요약

setup.sh 실행 전 필수 확인:

- [ ] GPU 확인: `nvidia-smi`
- [ ] Compute Capability 확인: `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`
- [ ] CUDA 확인: `nvcc --version`
- [ ] Conda 확인: `conda --version`
- [ ] GCC 확인: `gcc --version`
- [ ] FFmpeg 확인: `ffmpeg -version` 또는 `ldconfig -p | grep libavcodec`
- [ ] Git LFS 확인: `git-lfs --version`
- [ ] 디스크 공간 확인: `df -h /home` (20GB+ 여유)
- [ ] 네트워크 확인: `ping -c 1 github.com`

모두 ✅면 setup.sh 실행 가능!

---

**작성일**: 2025-11-25
**버전**: 1.0
