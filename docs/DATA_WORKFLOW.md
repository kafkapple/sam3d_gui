# SAM 3D GUI - Data Workflow Guide

## Overview

SAM 3D GUI는 Fauna 데이터셋 호환 구조로 annotation 및 augmentation을 수행합니다.

**핵심 원칙**: 모든 데이터는 Fauna 형식으로 저장되어 즉시 학습에 사용 가능

---

## 1. Data Structure

### Fauna 호환 구조

```
session_directory/
├── session_metadata.json
├── Camera1_0_frame0000/
│   ├── rgb.png
│   ├── mask.png
│   ├── box.txt          (optional)
│   └── metadata.json    (optional)
├── Camera1_0_frame0001/
│   ├── rgb.png
│   └── mask.png
└── Camera2_0_frame0000/
    ├── rgb.png
    └── mask.png
```

**특징**:
- ✅ 각 프레임이 독립 디렉토리
- ✅ RGB와 mask가 같은 위치 (Fauna 요구사항)
- ✅ 디렉토리명에 카메라/비디오/프레임 정보 포함
- ✅ 즉시 Fauna 학습 사용 가능

### 파일명 규칙 (Naming Convention)

```
{camera_name}_{video_name}_frame{idx:04d}/
  ├── rgb.png
  └── mask.png

예시:
- Camera1_0_frame0000/      # Camera1, video 0.mp4, frame 0
- Camera1_12000_frame0050/  # Camera1, video 12000.mp4, frame 50
- Camera2_0_frame0099/      # Camera2, video 0.mp4, frame 99
```

---

## 2. Annotation Workflow

### 2.1 Interactive Annotation (개별 비디오)

**사용 시나리오**: 소수 비디오를 정밀하게 annotation

```
입력: video.mp4
↓
[Interactive Tab]
- SAM2로 마스크 annotation
- 프레임별 저장
↓
출력: outputs/sessions/TIMESTAMP/
  ├── session_metadata.json
  ├── frame_0000/
  │   ├── rgb.png
  │   └── mask.png
  └── frame_0001/
      ├── rgb.png
      └── mask.png
```

### 2.2 Batch Annotation (다중 비디오)

**사용 시나리오**: 대량 비디오를 자동으로 annotation

```
입력: /data/markerless_mouse/
  ├── mouse_1/
  │   ├── Camera1/
  │   │   ├── 0.mp4
  │   │   ├── 3000.mp4
  │   │   └── ...
  │   └── Camera2/
  │       └── ...
  └── mouse_2/
      └── ...

↓
[Batch Processing Tab]
- Reference annotation 1회
- SAM2가 모든 비디오/프레임에 자동 적용
↓
출력: outputs/sessions/mouse_batch_TIMESTAMP/
  ├── session_metadata.json
  ├── Camera1_0_frame0000/
  │   ├── rgb.png
  │   └── mask.png
  ├── Camera1_0_frame0001/
  │   ├── rgb.png
  │   └── mask.png
  └── Camera2_3000_frame0050/
      ├── rgb.png
      └── mask.png

총 프레임 수: 72 videos × 100 frames = 7,200 frames
```

**Session Metadata 예시**:
```json
{
  "session_id": "mouse_batch_20251125_185700",
  "session_type": "batch",
  "total_videos": 72,
  "total_frames": 7200,
  "fauna_compatible": true,
  "naming_convention": "{camera}_{video}_frame{idx}",
  "videos": [
    {
      "camera": "Camera1",
      "video_name": "0.mp4",
      "num_frames": 100,
      "filename_pattern": "Camera1_0_frame*.png"
    }
  ]
}
```

---

## 3. Data Augmentation Workflow

### 3.1 Augmentation 설정

**Data Augmentation Tab**에서 설정:

1. **Session 로드**:
   - Scan Sessions → Select Session → Load

2. **Augmentation 파라미터**:
   - **Crop-Based Scale**: 마스크 영역만 크롭하여 확대/축소
     - Scale Range: 0.5x - 2.0x
     - Horizontal/Vertical Offset: 위치 이동
   - **Geometric**: Rotation, Flip
   - **Photometric**: Brightness, Noise, Contrast

3. **Multiplier**: 프레임당 증강 버전 수 (예: 5개)

### 3.2 Augmentation 출력 구조

```
입력: outputs/sessions/mouse_batch_TIMESTAMP/
  └── Camera1_0_frame0000/
      ├── rgb.png
      └── mask.png

↓
[Apply Augmentation]
- Multiplier: 5
- Crop-based scale, rotation, etc.
↓
출력: outputs/augmented/TIMESTAMP/
  ├── session_metadata.json
  ├── Camera1_0_frame0000_aug00/  # 첫 번째 증강
  │   ├── rgb.png
  │   └── mask.png
  ├── Camera1_0_frame0000_aug01/  # 두 번째 증강
  │   ├── rgb.png
  │   └── mask.png
  ├── Camera1_0_frame0000_aug02/
  │   ├── rgb.png
  │   └── mask.png
  └── ...

총 프레임 수: 7,200 × 5 = 36,000 frames
```

**파일명 규칙**:
```
{original_name}_aug{idx:02d}/
  ├── rgb.png
  └── mask.png

예시:
- Camera1_0_frame0000_aug00/  # 원본 Camera1_0_frame0000의 증강 버전 1
- Camera1_0_frame0000_aug01/  # 원본 Camera1_0_frame0000의 증강 버전 2
```

**Augmentation Metadata 예시**:
```json
{
  "session_id": "augmented_20251126_154500",
  "source_session": "outputs/sessions/mouse_batch_20251125_185700",
  "augmentation_params": {
    "multiplier": 5,
    "crop_based_scale": true,
    "scale_range": [0.5, 2.0],
    "offset_x_max": 0.2,
    "offset_y_max": 0.2,
    "rotation_range": [-30, 30],
    "brightness_range": [0.7, 1.3]
  },
  "total_original_frames": 7200,
  "total_augmented_frames": 36000,
  "fauna_compatible": true,
  "timestamp": "2025-11-26T15:45:00"
}
```

---

## 4. Quality Analysis

### 4.1 품질 분석 실행

**Generate Quality Report** 버튼 클릭:

```
입력: outputs/augmented/TIMESTAMP/
↓
[Feature Extraction]
- Simple: Histogram-based features
- ResNet: Deep learning features
↓
[Clustering]
- K-means or DBSCAN
- Diversity metrics
↓
출력: outputs/augmented/TIMESTAMP/
  ├── clustering_results.json
  └── quality_report.html
```

### 4.2 Quality Report 내용

**HTML Report 포함 사항**:
- 📊 Diversity Metrics
  - Silhouette Score (클러스터 품질)
  - Davies-Bouldin Score (클러스터 분리도)
  - Cluster Size Distribution
- 🗺️ Feature Space Visualization (t-SNE/UMAP)
- 🖼️ Representative Images (클러스터별 대표 이미지)
- 📝 Quality Indicators (자동 평가 및 권장사항)

**평가 기준**:
- Silhouette > 0.4: Good diversity
- Balanced cluster sizes: 균등한 variation
- Low Davies-Bouldin: 명확한 클러스터 분리

---

## 5. Usage Scenarios

### Scenario 1: Fauna 학습에 원본 데이터만 사용

```yaml
# config/train_fauna_mouse.yaml
train_data_dir: outputs/sessions/mouse_batch_20251125_185700/
```

**장점**: 원본 데이터의 품질 유지

**사용 케이스**:
- 충분한 데이터가 있는 경우
- 데이터 품질이 우선인 경우

### Scenario 2: Fauna 학습에 증강 데이터만 사용

```yaml
# config/train_fauna_mouse.yaml
train_data_dir: outputs/augmented/20251126_154500/
```

**장점**: 다양한 variation 학습

**사용 케이스**:
- 원본 데이터가 부족한 경우
- Generalization 중요한 경우
- Augmentation 효과 검증

### Scenario 3: Fauna 학습에 원본 + 증강 혼합

```yaml
# config/train_fauna_mouse.yaml
train_data_dir:
  - outputs/sessions/mouse_batch_20251125_185700/     # 7,200 frames
  - outputs/augmented/20251126_154500/                 # 36,000 frames
```

**장점**: 최대 데이터 활용

**사용 케이스**:
- 최고 성능 추구
- 데이터 다양성과 품질 모두 필요
- Production 모델

**총 데이터**: 7,200 + 36,000 = 43,200 frames

### Scenario 4: 증강 체이닝 (Augmentation Chaining)

```
원본 데이터
  ↓
[Augmentation 1]
- Crop-based scale
- Multiplier: 5
  ↓
outputs/augmented/TIMESTAMP_1/  (36,000 frames)
  ↓
[Quality Analysis]
- Review quality report
- Adjust parameters
  ↓
[Augmentation 2]
- Different parameters
- Multiplier: 2
  ↓
outputs/augmented/TIMESTAMP_2/  (72,000 frames)
```

**사용 케이스**:
- 파라미터 최적화
- 다단계 augmentation
- 극대량 데이터 생성

---

## 6. Best Practices

### 6.1 Annotation

✅ **DO**:
- Batch annotation 사용 (대량 비디오)
- Reference annotation은 명확하게
- 주기적으로 결과 확인

❌ **DON'T**:
- 모든 비디오를 interactive로 처리
- Reference annotation을 대충 설정

### 6.2 Augmentation

✅ **DO**:
- Preview 먼저 확인
- Multiplier는 5-10 권장
- Crop-based scale 활성화 (가장 효과적)
- Quality Report로 품질 검증

❌ **DON'T**:
- 과도한 augmentation (artifact 발생)
- Quality Report 건너뛰기
- 원본보다 너무 많은 증강 (10x 이상)

### 6.3 Training Data Selection

**원본만 (Scenario 1)**:
- ✅ 데이터 > 10K frames
- ✅ 품질 > 다양성

**증강만 (Scenario 2)**:
- ✅ 데이터 < 5K frames
- ✅ 다양성 > 품질
- ✅ Augmentation quality report 좋음

**혼합 (Scenario 3)**:
- ✅ 최고 성능 필요
- ✅ 충분한 컴퓨팅 자원
- ✅ Production 모델

---

## 7. File Organization

### 7.1 권장 디렉토리 구조

```
project_root/
├── outputs/
│   ├── sessions/                    # Annotation 결과
│   │   ├── mouse_batch_TIMESTAMP/   # Batch annotation
│   │   └── interactive_TIMESTAMP/   # Interactive annotation
│   └── augmented/                   # Augmentation 결과
│       ├── TIMESTAMP_1/
│       │   ├── quality_report.html
│       │   └── Camera1_0_frame0000_aug00/
│       └── TIMESTAMP_2/
└── data/                            # 원본 비디오
    └── markerless_mouse/
        ├── mouse_1/
        └── mouse_2/
```

### 7.2 용량 관리

**예상 용량**:
- 원본 프레임: ~1MB × 7,200 = ~7GB
- 증강 프레임 (5x): ~1MB × 36,000 = ~36GB
- **총 용량**: ~43GB (원본 + 증강)

**절약 방법**:
- Quality Report 확인 후 불필요한 augmentation 삭제
- 학습 완료 후 원본만 보관
- Compression 사용 (JPEG quality 85)

---

## 8. Troubleshooting

### 문제: Augmentation 결과가 Load Session에 안 보임

**원인**: 세션 구조 불일치

**해결**:
```bash
# Scan Sessions 다시 실행
# Session Directory 경로 확인: outputs/augmented
```

### 문제: Fauna 학습 시 데이터 로드 실패

**원인**: 디렉토리 구조 불일치

**확인사항**:
```bash
# 각 프레임 디렉토리에 rgb.png, mask.png 있는지 확인
ls outputs/sessions/TIMESTAMP/Camera1_0_frame0000/
# 출력: rgb.png  mask.png
```

### 문제: Quality Report에서 클러스터 품질 낮음

**의미**: Augmentation diversity 부족

**해결**:
- Scale range 확대 (0.3 - 3.0)
- Offset 증가 (0.3 - 0.5)
- Rotation range 확대
- 다른 augmentation 조합 시도

---

## 9. Command Line Usage

### Batch Session 변환 (Legacy)

기존 batch 세션을 augmentation 호환 형식으로 변환:

```bash
python3 convert_batch_session_for_augmentation.py \
  outputs/sessions/mouse_batch_20251125_185700
```

### Quality Analysis (Standalone)

```bash
python3 -c "
from feature_clustering import analyze_augmentation_quality
from html_report_generator import generate_html_report
from pathlib import Path

results = analyze_augmentation_quality(
    image_paths=list(Path('outputs/augmented/TIMESTAMP').rglob('*/rgb.png')),
    output_dir=Path('outputs/augmented/TIMESTAMP'),
    feature_type='simple',
    cluster_method='kmeans',
    n_clusters=5
)

generate_html_report(
    results=results,
    output_path=Path('outputs/augmented/TIMESTAMP/quality_report.html')
)
"
```

---

## 10. Summary

### 전체 워크플로우

```
1. Annotation
   ↓
   outputs/sessions/TIMESTAMP/
   (Fauna 호환 구조)
   ↓
2. Augmentation (optional)
   ↓
   outputs/augmented/TIMESTAMP/
   (Fauna 호환 구조)
   ↓
3. Quality Analysis
   ↓
   quality_report.html
   (품질 검증)
   ↓
4. Fauna Training
   ↓
   config.yaml: train_data_dir 설정
   (Scenario 1/2/3 선택)
```

### 핵심 장점

1. **통일된 구조**: 모든 단계에서 Fauna 호환
2. **즉시 사용**: 변환 없이 바로 학습
3. **추적 가능**: 파일명으로 원본 추적
4. **유연성**: 3가지 사용 시나리오 지원
5. **품질 보증**: 자동 품질 분석

---

## References

- [Fauna Dataset Format](https://github.com/3DAnimals/3DAnimals)
- [SAM 2 Documentation](https://github.com/facebookresearch/sam2)
- [Data Augmentation Best Practices](./AUGMENTATION.md)
