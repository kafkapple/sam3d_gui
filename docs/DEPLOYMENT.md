# SAM 3D GUI - Deployment Guide

## 📋 목차

1. [체크포인트 관리](#체크포인트-관리)
2. [Git LFS 요구사항](#git-lfs-요구사항)
3. [서버 환경별 배포 방법](#서버-환경별-배포-방법)
4. [SAM 3D 체크포인트 다운로드](#sam-3d-체크포인트-다운로드)
5. [Git Repository 관리](#git-repository-관리)

---

## 체크포인트 관리

### 📋 개요

SAM 3D GUI는 두 가지 SAM 모델을 사용합니다:
- **SAM 2**: Interactive point annotation (foreground/background 세그멘테이션)
- **SAM 3D**: 3D mesh 생성

모든 체크포인트 경로는 `config/model_config.yaml`에서 중앙 관리됩니다.

### 🗂️ 체크포인트 위치

**현재 설정 (config/model_config.yaml)**

```yaml
sam2:
  checkpoint: ~/dev/segment-anything-2/checkpoints/sam2.1_hiera_large.pt
  config: "configs/sam2.1/sam2.1_hiera_l.yaml"

sam3d:
  checkpoint_dir: ~/dev/sam3d_gui/external/sam-3d-objects/checkpoints/hf
  checkpoint_dir_alt: ~/dev/sam-3d-objects/checkpoints/hf
```

### ✅ 체크포인트 상태

#### SAM 2 (Interactive Segmentation)
- **위치**: `/home/joon/dev/segment-anything-2/checkpoints/sam2.1_hiera_large.pt`
- **상태**: ✅ 다운로드 완료
- **용량**: ~2.4GB
- **용도**: Point annotation, foreground/background 분리

#### SAM 3D (3D Reconstruction)
- **위치**: `/home/joon/dev/sam-3d-objects/checkpoints/hf/`
- **상태**: ❌ **다운로드 필요**
- **용량**: 약 5-10GB
- **용도**: 3D mesh 생성 (PLY 파일)

### 🔧 경로 변경 방법

모든 경로는 `config/model_config.yaml`에서 관리됩니다:

```yaml
sam2:
  checkpoint: "${oc.env:HOME}/dev/segment-anything-2/checkpoints/sam2.1_hiera_large.pt"
  # 경로 변경 시 이 줄만 수정

sam3d:
  checkpoint_dir: "${oc.env:HOME}/dev/sam3d_gui/external/sam-3d-objects/checkpoints/hf"
  # 경로 변경 시 이 줄만 수정
```

**환경 변수 활용**: `${oc.env:HOME}`은 자동으로 `/home/joon`으로 확장됩니다.

### 📂 권장 디렉토리 구조

```
/home/joon/dev/
├── sam3d_gui/                      # 이 프로젝트
│   ├── config/
│   │   └── model_config.yaml       # 체크포인트 경로 설정
│   ├── external/
│   │   └── sam-3d-objects/         # Git submodule (옵션)
│   │       └── checkpoints/hf/
│   └── src/
│       ├── web_app.py
│       └── config_loader.py
│
├── segment-anything-2/              # SAM 2 레포지토리
│   └── checkpoints/
│       └── sam2.1_hiera_large.pt    # ✅ 존재
│
└── sam-3d-objects/                  # SAM 3D standalone (대체 경로)
    └── checkpoints/hf/
        ├── pipeline.yaml            # ❌ 다운로드 필요
        └── ... (model files)
```

---

## Git LFS 요구사항

### 왜 Git LFS가 필요한가?

SAM 3D Objects 모델 체크포인트는 **5-10GB의 대용량 파일**입니다. 일반 Git은 이런 대용량 파일을 효율적으로 처리할 수 없어 Git LFS (Large File Storage)가 필요합니다.

### Sudo 권한 필요 이유

**`src/web_app.py`의 자동 다운로드 기능은 다음 명령어를 실행합니다:**

```python
# download_sam3d_checkpoint() 함수 내부 (lines 562-619)
subprocess.run(["sudo", "apt-get", "update"], check=True)
subprocess.run(["sudo", "apt-get", "install", "-y", "git-lfs"], check=True)
```

**왜 sudo가 필요한가?**

1. **시스템 패키지 설치**: `apt-get install`은 시스템 디렉토리(`/usr/bin`, `/usr/lib`)에 파일을 설치합니다.
2. **권한 보호**: Linux는 일반 사용자가 시스템 디렉토리에 쓰는 것을 차단합니다.
3. **보안**: 시스템 전체에 영향을 주는 작업은 관리자 권한이 필요합니다.

**Sudo 없이 실행하면?**

```bash
Permission denied: Cannot write to /usr/bin
E: Could not open lock file - open (13: Permission denied)
```

---

## 서버 환경별 배포 방법

### 환경 1: Sudo 권한이 있는 개발 서버

**GUI 자동 다운로드 기능이 작동합니다.**

#### 설정 방법:

```bash
# 1. GUI 실행
cd /home/joon/dev/sam3d_gui
./run.sh

# 2. GUI에서 "Generate 3D Mesh" 버튼 클릭
# 3. 체크포인트가 없으면 자동으로 다운로드됩니다.
```

#### 동작 과정:

1. Git LFS 확인
2. 없으면 `sudo apt-get install git-lfs` 실행
3. HuggingFace에서 자동 다운로드
4. 완료 후 3D mesh 생성 진행

**장점**: 사용자가 아무것도 하지 않아도 자동으로 설치됩니다.

---

### 환경 2: Sudo 권한이 없는 운영 서버

**GUI 자동 다운로드는 실패합니다. 사전 다운로드가 필요합니다.**

#### 해결 방법 A: `download_sam3d.sh` 사용 (권장)

**사전 다운로드 스크립트를 사용하여 미리 다운로드합니다.**

```bash
# 1. 서버에 접속 (sudo 권한 없음)
cd /home/joon/dev/sam3d_gui

# 2. 다운로드 스크립트 실행
./download_sam3d.sh
```

**스크립트 동작:**

1. Git LFS 확인
2. 없으면 설치 방법 안내 및 선택
   - `sudo apt-get install` (가능하면)
   - `conda install -c conda-forge git-lfs` (sudo 없이)
3. HuggingFace에서 다운로드
4. `~/dev/sam-3d-objects/checkpoints/hf/` 위치에 저장

**장점**:
- Sudo 없어도 Conda로 설치 가능
- GUI 실행 전에 완료
- 네트워크 타임아웃 걱정 없음

---

#### 해결 방법 B: Conda로 Git LFS 설치

**Sudo 없이 사용자 공간에 Git LFS 설치:**

```bash
# 1. Conda 환경에서 Git LFS 설치
conda activate sam3d_gui
conda install -c conda-forge git-lfs

# 2. Git LFS 초기화
git lfs install

# 3. 다운로드 스크립트 실행
cd /home/joon/dev/sam3d_gui
./download_sam3d.sh
```

**장점**:
- Sudo 권한 불필요
- 사용자 환경에만 설치
- 안전하고 독립적

---

#### 해결 방법 C: 수동 다운로드

**다른 컴퓨터에서 다운받아 서버로 복사:**

```bash
# 로컬 컴퓨터 (Git LFS 설치된 환경)
cd ~/Downloads
git clone https://huggingface.co/facebook/sam-3d-objects
tar -czf sam3d_checkpoints.tar.gz sam-3d-objects/

# 서버로 복사
scp sam3d_checkpoints.tar.gz user@server:/home/user/

# 서버에서
cd /home/user
tar -xzf sam3d_checkpoints.tar.gz
mv sam-3d-objects ~/dev/sam-3d-objects/checkpoints/hf
```

**장점**:
- Git LFS 설치 불필요
- 네트워크 제약 없음

---

### 환경 3: Docker 컨테이너

**Dockerfile에 Git LFS 포함:**

```dockerfile
FROM python:3.10

# Git LFS 설치 (sudo 불필요, root로 실행)
RUN apt-get update && \
    apt-get install -y git-lfs && \
    git lfs install

# 애플리케이션 설치
COPY . /app
WORKDIR /app

# 체크포인트 다운로드
RUN ./download_sam3d.sh

# GUI 실행
CMD ["./run.sh"]
```

**장점**:
- 이미지 빌드 시 한 번만 다운로드
- 런타임에 다운로드 불필요

---

## HuggingFace 인증 설정 (중요!)

### ⚠️ 모델 액세스 권한 요청

SAM 3D Objects 모델은 **gated model**로, 사용 전 액세스 권한을 받아야 합니다.

**필수 단계:**

1. **HuggingFace 계정 생성**
   - https://huggingface.co/join

2. **모델 페이지에서 액세스 요청**
   - https://huggingface.co/facebook/sam-3d-objects
   - "Request access to this model" 버튼 클릭
   - Meta 팀의 승인 대기 (일반적으로 즉시~24시간)

3. **HuggingFace Token 생성**
   - https://huggingface.co/settings/tokens
   - "New token" 클릭
   - 이름: `sam3d_gui_token` (예시)
   - Type: **Read** (읽기 권한만 필요)
   - 생성된 토큰 복사: `hf_xxxxxxxxxxxxxxxxxxxxx`

4. **`.env` 파일에 토큰 추가**

```bash
cd /home/joon/dev/sam3d_gui

# .env 파일 생성 (없으면)
cp .env.example .env

# 토큰 설정
echo 'HF_TOKEN="hf_your_token_here"' > .env
```

**`.env` 파일 예시:**
```bash
# HuggingFace Authentication
HF_TOKEN=

# Optional
GRADIO_SERVER_PORT=7860
```

### 인증 확인

**토큰이 제대로 설정되었는지 확인:**

```bash
cd /home/joon/dev/sam3d_gui

# 스크립트 실행 (토큰 자동 로드)
./download_sam3d.sh

# 출력 확인:
# ✓ .env 파일 로드 중...
# ✓ HuggingFace 토큰 감지됨
```

**만약 403 에러가 발생하면:**
```
fatal: unable to access '...': The requested URL returned error: 403
```

→ **원인**: 아직 모델 액세스가 승인되지 않음
→ **해결**: https://huggingface.co/facebook/sam-3d-objects 에서 승인 상태 확인

---

## SAM 3D 체크포인트 다운로드

### 다운로드 위치 (Config 관리)

**`config/model_config.yaml`에 정의된 경로:**

```yaml
sam3d:
  name: "SAM 3D Objects"
  checkpoint_dir: "${oc.env:HOME}/dev/sam3d_gui/external/sam-3d-objects/checkpoints/hf"
  checkpoint_dir_alt: "${oc.env:HOME}/dev/sam-3d-objects/checkpoints/hf"
```

**실제 경로 (환경 변수 확장 후):**

```
기본 경로: /home/joon/dev/sam3d_gui/external/sam-3d-objects/checkpoints/hf
대체 경로: /home/joon/dev/sam-3d-objects/checkpoints/hf
```

### 다운로드 방법 비교

| 방법 | Sudo 필요 | 시간 | 자동화 | 권장 환경 |
|------|----------|------|--------|-----------|
| **GUI 자동 다운로드** | ✅ 필요 | 10-20분 | ✅ 완전 자동 | 개발 서버 |
| **download_sam3d.sh** | ❌ 불필요 (Conda 사용 시) | 10-20분 | ⚠️ 수동 실행 | 운영 서버 |
| **Conda + 스크립트** | ❌ 불필요 | 10-20분 | ⚠️ 수동 실행 | 제한된 서버 |
| **수동 복사** | ❌ 불필요 | 30분+ | ❌ 수동 | 네트워크 제약 |

### `download_sam3d.sh` 사용법

**기본 사용:**

```bash
cd /home/joon/dev/sam3d_gui
./download_sam3d.sh
```

**출력 예시:**

```
==========================================
SAM 3D Objects - 체크포인트 다운로드
==========================================

다운로드 위치: /home/joon/dev/sam-3d-objects/checkpoints/hf

⚠️  Git LFS가 설치되어 있지 않습니다.

설치 방법:
  Ubuntu/Debian: sudo apt-get install git-lfs
  Conda: conda install -c conda-forge git-lfs

Git LFS를 설치하시겠습니까? (y/n) y

[설치 진행...]

✓ Git LFS 감지됨

HuggingFace에서 SAM 3D 체크포인트 다운로드 중...

📥 체크포인트 다운로드 시작...
Cloning into 'checkpoints/hf'...
[다운로드 진행...]

==========================================
다운로드 완료!
==========================================

체크포인트 위치: /home/joon/dev/sam-3d-objects/checkpoints/hf

다운로드된 파일:
total 8.5G
-rw-r--r-- 1 user user 5.2G pipeline.yaml
[...]

이제 web GUI에서 'Generate 3D Mesh'를 사용할 수 있습니다.
```

---

## Git Repository 관리

### 디렉토리 구조

```
sam3d_gui/
├── .gitignore                  # ✅ 준비 완료
├── config/
│   └── model_config.yaml       # 체크포인트 경로 설정
├── src/
│   └── web_app.py              # 메인 애플리케이션
├── external/                   # ❌ Git에 포함 안됨 (대용량)
│   └── sam-3d-objects/
│       └── checkpoints/hf/
├── outputs/                    # ❌ Git에 포함 안됨 (결과 파일)
│   ├── sessions/
│   └── *.ply, *.obj
├── checkpoints/                # ❌ Git에 포함 안됨 (SAM 2)
│   └── sam2.1_hiera_large.pt
├── logs/                       # ❌ Git에 포함 안됨
│   └── *.log
├── download_sam3d.sh           # ✅ Git에 포함 (스크립트)
├── run.sh                      # ✅ Git에 포함
├── QUICK_START.md              # ✅ Git에 포함
└── docs/
    ├── SESSION_MANAGEMENT.md   # ✅ Git에 포함
    └── DEPLOYMENT.md           # ✅ Git에 포함 (이 문서)
```

### `.gitignore` 주요 패턴

```gitignore
# Model checkpoints (large files)
checkpoints/
*.pth
*.pt
*.ckpt
*.safetensors

# SAM 3D checkpoints (HuggingFace)
external/sam-3d-objects/
**/sam-3d-objects/checkpoints/

# Output files
outputs/
*.ply
*.obj
*.mp4
*.avi
*.mov
*.mkv

# Logs
*.log
logs/
nohup.out
sam_gui*.log
/tmp/

# Python
__pycache__/
*.pyc
venv/
```

### Git 저장소 초기화

**처음 설정:**

```bash
cd /home/joon/dev/sam3d_gui

# Git 저장소 초기화
git init

# .gitignore 확인
cat .gitignore

# 파일 추가
git add .
git status  # 제외된 파일 확인

# 첫 커밋
git commit -m "Initial commit: SAM 3D GUI with auto-download"

# 원격 저장소 연결 (선택사항)
git remote add origin https://github.com/your-username/sam3d_gui.git
git push -u origin main
```

### 체크포인트 파일 제외 확인

```bash
# Git에 포함되지 않는 파일 확인
git status --ignored

# 예상 출력:
# Ignored files:
#   external/sam-3d-objects/
#   outputs/
#   checkpoints/
#   *.log
```

---

## 배포 체크리스트

### 새 서버 배포 시

- [ ] Conda 환경 설치 (`sam3d_gui`)
- [ ] Git LFS 설치 (sudo 또는 conda)
- [ ] `download_sam3d.sh` 실행
- [ ] 체크포인트 다운로드 완료 확인
- [ ] `config/model_config.yaml` 경로 확인
- [ ] `./run.sh` 실행
- [ ] GUI 접속 (http://localhost:7860)
- [ ] "Generate 3D Mesh" 테스트

### Git 저장소 클론 후

```bash
# 1. 저장소 클론
git clone https://github.com/your-username/sam3d_gui.git
cd sam3d_gui

# 2. Conda 환경 생성
conda env create -f environment.yml
conda activate sam3d_gui

# 3. Git LFS 설치 (conda)
conda install -c conda-forge git-lfs
git lfs install

# 4. 체크포인트 다운로드
./download_sam3d.sh

# 5. GUI 실행
./run.sh
```

---

## 문제 해결

### 문제 1: "Permission denied" 에러

**증상:**

```
E: Could not open lock file - open (13: Permission denied)
```

**원인:** Sudo 권한 없이 `apt-get install` 시도

**해결:**

```bash
# Conda로 설치
conda install -c conda-forge git-lfs
```

---

### 문제 2: Git LFS 파일이 placeholder

**증상:**

```
version https://git-lfs.github.com/spec/v1
oid sha256:abc123...
size 5368709120
```

**원인:** Git LFS가 설치되지 않은 상태로 clone

**해결:**

```bash
# Git LFS 설치 후
git lfs install
git lfs pull
```

---

### 문제 3: 체크포인트 경로 불일치

**증상:**

```
SAM 3D config not found at: /path/to/checkpoint
```

**원인:** `config/model_config.yaml` 경로가 실제 다운로드 위치와 다름

**해결:**

```bash
# 1. 실제 다운로드된 위치 확인
ls -l ~/dev/sam-3d-objects/checkpoints/hf/pipeline.yaml

# 2. config 수정
vim config/model_config.yaml

# checkpoint_dir 경로 수정
sam3d:
  checkpoint_dir: "${oc.env:HOME}/dev/sam-3d-objects/checkpoints/hf"
```

---

## 참고 문서

- [QUICK_START.md](../QUICK_START.md) - 빠른 시작 가이드
- [SESSION_MANAGEMENT.md](SESSION_MANAGEMENT.md) - 세션 저장/로드
- [config/model_config.yaml](../config/model_config.yaml) - 체크포인트 경로 설정

---

**작성일**: 2025-11-24  
**버전**: 1.0  
**상태**: ✅ Git LFS 요구사항 문서화 완료
