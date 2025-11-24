# SAM 3D GUI - 체크포인트 관리 가이드

## 📋 개요

SAM 3D GUI는 두 가지 SAM 모델을 사용합니다:
- **SAM 2**: Interactive point annotation (foreground/background 세그멘테이션)
- **SAM 3D**: 3D mesh 생성

모든 체크포인트 경로는 `config/model_config.yaml`에서 중앙 관리됩니다.

---

## 🗂️ 체크포인트 위치

### 현재 설정 (config/model_config.yaml)

```yaml
sam2:
  checkpoint: ~/dev/segment-anything-2/checkpoints/sam2.1_hiera_large.pt
  config: "configs/sam2.1/sam2.1_hiera_l.yaml"

sam3d:
  checkpoint_dir: ~/dev/sam3d_gui/external/sam-3d-objects/checkpoints/hf
  checkpoint_dir_alt: ~/dev/sam-3d-objects/checkpoints/hf
```

---

## ✅ 현재 상태

### SAM 2 (Interactive Segmentation)
- **위치**: `/home/joon/dev/segment-anything-2/checkpoints/sam2.1_hiera_large.pt`
- **상태**: ✅ 다운로드 완료
- **용량**: ~약 2GB
- **용도**: Point annotation, foreground/background 분리

### SAM 3D (3D Reconstruction)
- **위치**: `/home/joon/dev/sam-3d-objects/checkpoints/hf/`
- **상태**: ❌ **다운로드 필요**
- **용량**: 약 5-10GB (예상)
- **용도**: 3D mesh 생성 (PLY 파일)

---

## 📥 SAM 3D 체크포인트 다운로드

### 방법 1: HuggingFace에서 다운로드

```bash
cd /home/joon/dev/sam-3d-objects

# Git LFS 설치 (필요시)
sudo apt-get install git-lfs
git lfs install

# HuggingFace에서 체크포인트 다운로드
git clone https://huggingface.co/facebook/sam-3d-objects checkpoints/hf

# 또는 Python 스크립트 사용 (있다면)
python download_checkpoints.py
```

### 방법 2: 수동 다운로드

1. HuggingFace 방문: https://huggingface.co/facebook/sam-3d-objects
2. 필요한 파일 다운로드:
   - `pipeline.yaml` (필수)
   - 모델 weights (`.pth`, `.pt` 파일들)
3. 다음 경로에 저장:
   ```
   /home/joon/dev/sam-3d-objects/checkpoints/hf/
   ├── pipeline.yaml
   ├── model.pth
   └── ...
   ```

---

## 🔧 경로 변경 방법

### Config 파일 수정

모든 경로는 `config/model_config.yaml`에서 관리됩니다:

```yaml
sam2:
  checkpoint: "${oc.env:HOME}/dev/segment-anything-2/checkpoints/sam2.1_hiera_large.pt"
  # 경로 변경 시 이 줄만 수정

sam3d:
  checkpoint_dir: "${oc.env:HOME}/dev/sam3d_gui/external/sam-3d-objects/checkpoints/hf"
  # 경로 변경 시 이 줄만 수정
```

### 환경 변수 활용

`${oc.env:HOME}`은 자동으로 `/home/joon`으로 확장됩니다.

---

## 📂 권장 디렉토리 구조

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

## 🚀 사용 방법

### 1. Interactive Mode (SAM 2)

**현재 상태**: ✅ 사용 가능

1. 웹 GUI 실행: `./run.sh`
2. Interactive Mode 탭 선택
3. 비디오 로드
4. Foreground/Background points 클릭
5. "Segment Current Frame" 클릭 → **SAM 2가 자동 실행됨**

### 2. 3D Mesh 생성 (SAM 3D)

**현재 상태**: ❌ **체크포인트 다운로드 필요**

1. SAM 3D 체크포인트 다운로드 (위 참조)
2. "Generate 3D Mesh" 클릭
3. PLY 파일 자동 생성 & 다운로드

---

## 🔍 문제 해결

### 오류: "SAM 3D config not found"

```
3D 재구성 실패: SAM 3D config not found at .../pipeline.yaml
```

**해결**:
1. SAM 3D 체크포인트 다운로드 (위 참조)
2. `pipeline.yaml` 파일이 올바른 경로에 있는지 확인
3. `config/model_config.yaml`의 경로가 정확한지 확인

### SAM 2 로딩 실패

**증상**: Segment Current Frame 클릭 시 "fallback" 모드 사용

**해결**:
1. SAM 2 체크포인트 확인:
   ```bash
   ls -lh ~/dev/segment-anything-2/checkpoints/sam2.1_hiera_large.pt
   ```
2. CUDA 사용 가능 여부 확인:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
3. 로그 확인:
   ```bash
   tail -100 /tmp/sam_gui_final.log
   ```

---

## 📊 체크포인트 크기

| 모델 | 파일 크기 | 상태 |
|------|----------|------|
| SAM 2.1 Hiera Large | ~2.4 GB | ✅ |
| SAM 2.1 Hiera Base+ | ~약 900 MB | 옵션 |
| SAM 3D Objects | ~5-10 GB | ❌ 다운로드 필요 |

---

## 📝 다음 단계

1. **SAM 3D 체크포인트 다운로드** (필수)
2. Web GUI 테스트:
   - Interactive Mode로 마우스 세그멘테이션
   - 3D mesh 생성 테스트
3. 결과 확인:
   - 세그멘테이션 품질
   - 3D mesh 품질 (MeshLab으로 확인)

---

**작성일**: 2025-11-24
**상태**: SAM 2 ✅ / SAM 3D ❌ (체크포인트 다운로드 대기)
