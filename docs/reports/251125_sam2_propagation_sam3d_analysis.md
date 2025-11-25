# SAM 2 Propagation Mechanism & SAM3D 상세 분석

**날짜**: 2025-11-25
**작성자**: Claude Code
**프로젝트**: sam3d_gui

---

## 📋 Executive Summary

이 보고서는 Meta AI의 SAM 2 Video Predictor의 propagation 메커니즘, SAM3D 모델의 실체와 차이점, 그리고 현재 sam3d_gui 프로젝트에서의 통합 방안을 분석합니다.

**핵심 발견사항**:
- SAM 2는 memory-based tracking으로 한 프레임 annotation만으로 비디오 전체 추적 가능
- SAM3D는 실제로 존재하며, 단일 이미지에서 3D 재구성을 수행하는 Meta의 최신 모델
- SAM2 Image Predictor와 Video Predictor는 메모리 메커니즘 유무로 구분되며, 각각 다른 용도
- 현재 코드는 SAM2ImagePredictor만 사용 중, Video Predictor 통합 시 효율성 대폭 향상 가능

---

## 1. SAM 2 Video Predictor의 Propagation 메커니즘

### 1.1 Memory-Based Tracking 아키텍처

SAM 2는 **memory-augmented streaming architecture**를 도입하여 비디오 프레임을 실시간으로 처리합니다[1][2].

#### 핵심 컴포넌트

**1. Memory Encoder (메모리 인코더)**
- 예측된 마스크와 이미지 특징을 결합하여 메모리 표현 생성
- Hiera 이미지 인코더의 출력을 재사용하여 효율성 확보
- 더 큰 이미지 인코더로 확장 가능한 설계

```python
# Pseudo-code
def memory_encoder(image_embedding, predicted_mask):
    """
    Args:
        image_embedding: Hiera encoder 출력 (강력한 시각적 표현)
        predicted_mask: 현재 프레임의 예측 마스크
    Returns:
        memory_feature: 저장할 메모리 표현
    """
    # 이미지 특징과 마스크 정보 융합
    memory_feature = fuse(image_embedding, predicted_mask)
    return memory_feature
```

**2. Memory Bank (메모리 뱅크)**
- **FIFO (First-In-First-Out)** 방식의 메모리 저장소
- 최근 N개 프레임의 메모리 보관 (N은 하이퍼파라미터)
- Temporal position 정보를 임베딩에 포함하여 단기 객체 움직임 표현

```
Memory Bank 구조:
┌─────────────────────────────────────┐
│ Frame t-3  Frame t-2  Frame t-1  │ (저장)
│   [mem]      [mem]      [mem]    │
└─────────────────────────────────────┘
           ↓ (cross-attention)
      Current Frame t
```

**3. Memory Attention (메모리 어텐션)**
- **Self-attention**: 현재 프레임 내부의 특징 관계 파악
- **Cross-attention**: 저장된 과거 프레임 메모리와 현재 프레임 간 연결
- Transformer 블록 기반으로 temporal 정보 통합

```python
# Memory Attention 동작 원리
def memory_attention(current_frame_features, memory_bank):
    """
    Args:
        current_frame_features: 현재 프레임의 특징
        memory_bank: 과거 N개 프레임의 메모리
    Returns:
        attended_features: 메모리 기반으로 보강된 특징
    """
    # Self-attention: 현재 프레임 내부 관계
    self_attn = self_attention(current_frame_features)

    # Cross-attention: 과거 메모리와 현재 프레임 연결
    cross_attn = cross_attention(
        query=current_frame_features,
        key_value=memory_bank  # 과거 프레임들
    )

    # 통합: 시간적 일관성 확보
    attended_features = combine(self_attn, cross_attn)
    return attended_features
```

### 1.2 왜 한 프레임 Annotation만으로 가능한가?

**핵심 원리: Memory Propagation**

1. **초기 프레임 (t=0)**:
   - 사용자가 클릭 또는 바운딩 박스로 객체 지정
   - SAM 2가 정확한 마스크 생성
   - 이 마스크와 이미지 특징을 메모리로 저장

2. **다음 프레임 (t=1)**:
   - **이전 메모리 참조**: t=0의 객체 표현을 Memory Bank에서 가져옴
   - **Cross-attention**: 현재 프레임에서 유사한 특징 찾기
   - **마스크 예측**: 객체가 어디로 이동했는지 추론
   - **새 메모리 저장**: t=1의 표현을 Memory Bank에 추가

3. **이후 프레임 (t=2, 3, ...)**:
   - 동일한 과정 반복
   - 여러 과거 프레임의 메모리를 종합하여 더 robust한 추적

```
Timeline 시각화:

Frame 0 (User annotation)
  ↓
  🖱️ User clicks on mouse
  ↓
  🎭 SAM 2 generates mask
  ↓
  💾 Store in Memory Bank

Frame 1 (Automatic)
  ↓
  📖 Read Memory Bank (Frame 0 info)
  ↓
  🔍 Cross-attention: Find object in Frame 1
  ↓
  🎭 Generate mask automatically
  ↓
  💾 Store Frame 1 memory

Frame 2 (Automatic)
  ↓
  📖 Read Memory Bank (Frame 0, 1 info)
  ↓
  🔍 Cross-attention with multiple memories
  ↓
  🎭 Generate mask (more robust)
  ↓
  💾 Store Frame 2 memory

... (continues for all frames)
```

### 1.3 Temporal Consistency 학습

**암묵적 학습 (Implicit Learning)**:
- 학습 단계에서 memory-based frame propagation을 통해 시간적 일관성 학습
- 명시적인 optical flow나 tracking loss 없이도 일관된 추적 가능
- 메모리 메커니즘 자체가 temporal consistency를 보장

**Occlusion Handling (가림 처리)**:
- 객체가 일시적으로 가려질 때: 메모리에 저장된 과거 정보 활용
- 재등장 시: 메모리 기반 cross-attention으로 자동 재인식
- 긴 occlusion: 여러 프레임의 메모리 종합으로 복원

```python
# Occlusion 처리 예시
def track_with_occlusion(frames, initial_mask):
    memory_bank = []
    predictions = []

    for t, frame in enumerate(frames):
        if t == 0:
            # 초기 annotation
            mask = initial_mask
        else:
            # 메모리 기반 추적
            if len(memory_bank) > 0:
                # Cross-attention with memory
                mask = predict_from_memory(frame, memory_bank)

                # 가림 감지 (confidence 낮음)
                if mask.confidence < threshold:
                    # 이전 메모리들로부터 보완
                    mask = recover_from_history(memory_bank)

        # 메모리 저장
        memory = encode_memory(frame, mask)
        memory_bank.append(memory)
        predictions.append(mask)

    return predictions
```

### 1.4 성능 지표

- **속도**: 실시간 처리 가능 (~44 FPS)[3]
- **정확도**: 기존 SOTA 대비 우수한 segmentation quality
- **메모리**: FIFO 방식으로 일정한 메모리 사용량 유지

---

## 2. SAM3D 모델 조사

### 2.1 SAM3D의 실체

**결론: SAM3D는 실제로 존재하며, Meta AI가 2024년 11월 공개한 공식 모델입니다**[4][5].

#### SAM3D 공식 정보

- **발표일**: 2024년 11월 19일
- **개발자**: Meta AI
- **공식 페이지**: https://ai.meta.com/sam3d/
- **GitHub**: https://github.com/facebookresearch/sam-3d-objects
- **Demo**: https://sam3d.org/

#### 두 가지 전문 모델

**1. SAM 3D Objects**
- **용도**: 일반 객체 및 장면 재구성
- **특징**:
  - 단일 RGB 이미지에서 고품질 3D 메쉬 생성
  - Occlusion, clutter, 작은 객체, 비정상적 시점에서도 robust
  - Single-object 및 multi-object 생성 지원
  - 인간 선호도 테스트에서 기존 SOTA 대비 5:1 승률[6]

**2. SAM 3D Body**
- **용도**: 인체 3D 재구성 전용
- **특징**:
  - 단일 이미지에서 신체 형태, 자세 추정
  - Meta Momentum Human Rig (MHR) 포맷 지원
  - Rigging 및 animation 기능

### 2.2 SAM3D 기술적 원리

#### Workflow (처리 흐름)

```
Input: Single RGB Image
  ↓
Step 1: Segmentation (SAM 기반)
  - User clicks object
  - SAM generates 2D mask
  ↓
Step 2: 3D Geometry Inference
  - 단일 이미지에서 3D 형상 추론
  - Depth, normal, occlusion 고려
  ↓
Step 3: Texture & Pose Estimation
  - RGB 텍스처 매핑
  - 객체 자세 추정
  ↓
Output: High-quality 3D Mesh
  - PLY, OBJ, GLB 포맷
  - UV mapping 보존
```

#### Progressive Training (점진적 학습)

```python
# SAM3D Objects의 학습 전략
def progressive_training():
    """
    단계적으로 복잡한 데이터 학습
    """
    # Stage 1: Clean backgrounds, simple objects
    train_on(clean_data)

    # Stage 2: Add occlusion
    train_on(occluded_data)

    # Stage 3: Add clutter
    train_on(cluttered_scenes)

    # Stage 4: Small objects, unusual viewpoints
    train_on(challenging_data)
```

#### Data Engine with Human Feedback

- 인간 평가자의 피드백을 학습에 통합
- Iterative refinement로 품질 향상
- Real-world scenarios에 강건

### 2.3 SAM3D Output Formats

| Format | 용도 | 특징 |
|--------|------|------|
| **PLY** | Point cloud | Gaussian Splatting 지원 |
| **OBJ** | Mesh | 텍스처 + UV mapping |
| **GLB** | 3D scene | Unity, Unreal Engine 호환 |
| **MHR** | Human body | Animation rigging (SAM 3D Body 전용) |

### 2.4 현재 프로젝트에서의 사용

**sam3d_gui 프로젝트**는 이미 **SAM 3D Objects**를 통합하여 사용 중:

```python
# /home/joon/dev/sam3d_gui/src/sam3d_processor.py
# Line 113
self.inference_model = Inference(config_path, compile=False)

# Line 338
output = self.inference_model(frame, mask, seed=seed)
```

**사용 흐름**:
1. 비디오 프레임 추출
2. SAM2 (또는 기존 segmentation)로 2D 마스크 생성
3. SAM3D Objects로 3D 재구성
4. PLY 파일로 저장

---

## 3. SAM2 Image Predictor vs Video Predictor

### 3.1 핵심 차이점

| 구분 | SAM2ImagePredictor | SAM2VideoPredictor |
|------|-------------------|-------------------|
| **메모리 메커니즘** | ❌ 없음 | ✅ Memory Bank + Memory Attention |
| **초기화** | `build_sam2()` | `build_sam2_video_predictor()` |
| **용도** | 정적 이미지 segmentation | 비디오 객체 추적 |
| **Temporal consistency** | ❌ 프레임 간 독립적 | ✅ 시간적 일관성 보장 |
| **Annotation 필요** | 모든 프레임에 필요 | 한 프레임만 필요 |
| **Occlusion 처리** | ❌ 불가능 | ✅ 메모리 기반 복원 |
| **속도** | 빠름 (~47 FPS, tiny) | 빠름 (~44 FPS, 실시간) |
| **메모리 사용** | 낮음 | 중간 (FIFO로 제한) |

### 3.2 따로 써야 하는 이유

#### 아키텍처 차이

**SAM2ImagePredictor**:
```python
# 이미지 단일 처리
predictor.set_image(image)
mask = predictor.predict(point_coords, point_labels)
# 다음 이미지는 완전히 독립적
```

**SAM2VideoPredictor**:
```python
# 비디오 상태 관리
predictor.init_state(video_path)  # 비디오 전체 초기화

# 첫 프레임 annotation
predictor.add_new_points(frame_idx=0, points, labels)

# 나머지 프레임 자동 propagation
for frame_idx in range(1, num_frames):
    mask = predictor.propagate(frame_idx)  # 자동 추적
```

**메모리 구조**:
- Image Predictor: 프레임 간 정보 공유 없음
- Video Predictor: Memory Bank에 과거 프레임 저장 및 활용

### 3.3 Use Cases 분석

#### SAM2ImagePredictor 적합한 경우

✅ **정적 이미지 segmentation**
- 단일 이미지 분석
- Batch processing (프레임 간 관계 없음)
- 이미지 데이터셋 annotation

✅ **독립적 프레임 처리**
- 비디오 프레임들을 각각 독립적으로 처리
- 객체가 프레임마다 다름
- 시간적 일관성 불필요

**예시**:
```python
# 현재 sam3d_gui의 lite_annotator.py 사용 패턴
for frame in frames:
    predictor.set_image(frame)
    mask = predictor.predict(points, labels)  # 각 프레임 독립 처리
```

#### SAM2VideoPredictor 적합한 경우

✅ **비디오 객체 추적**
- 한 객체를 여러 프레임에서 추적
- Annotation 효율성 (한 프레임만 annotation)
- 시간적 일관성 필요

✅ **Occlusion 처리**
- 객체 일시적 가림
- 재등장 자동 감지

✅ **실시간 비디오 segmentation**
- 스트리밍 비디오
- 라이브 카메라 입력

**예시**:
```python
# 효율적인 비디오 추적 (권장)
video_predictor.init_state(video_path)

# 첫 프레임만 annotation
video_predictor.add_new_points(frame_idx=0, points=[mouse_click], labels=[1])

# 나머지 자동 추적 (no annotation needed!)
for frame_idx in range(1, total_frames):
    mask = video_predictor.propagate_in_video(frame_idx)
    # 자동으로 객체 추적
```

### 3.4 통합 가능성

**공식 답변 (Meta AI)**[7]:
- "SAM 2 has all the capabilities of SAM on static images"
- Image와 Video API가 모두 제공됨
- 하지만 **메모리 초기화 차이**로 인해 별도 API 필요

**통합 시도 시 문제**:
```python
# ❌ 이렇게 할 수 없음
video_predictor = build_sam2_video_predictor(...)
video_predictor.set_image(single_image)  # 에러!

# ✅ 올바른 사용
image_predictor = build_sam2(...)
image_predictor.set_image(single_image)  # 정상
```

**이유**:
- Video Predictor는 `init_state(video_path)` 호출 필요
- 비디오 전체의 메모리 상태 관리
- 단일 이미지 처리에는 불필요한 오버헤드

---

## 4. 현재 코드 분석 및 개선 제안

### 4.1 현재 구현 상태

#### 4.1.1 lite_annotator.py (현재)

```python
# /home/joon/dev/sam3d_gui/src/lite_annotator.py
from sam2.sam2_image_predictor import SAM2ImagePredictor  # Image Predictor만 사용

class LiteAnnotator:
    def __init__(self, sam2_base_path, device="cuda"):
        self.predictor = None  # SAM2ImagePredictor

    def load_frame(self, frame_idx):
        """각 프레임을 독립적으로 로드"""
        # 비디오에서 프레임 읽기
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.video_cap.read()

        # Annotation 복원 (파일에서)
        annotation_file = f'frame_{frame_idx:04d}_annotation.json'
        if annotation_file.exists():
            # 저장된 annotation 로드
            self.points = load_points(annotation_file)

    def generate_mask(self):
        """SAM2 Image Predictor로 마스크 생성"""
        self.predictor.set_image(self.current_frame)  # 매번 set_image 호출
        masks, scores, _ = self.predictor.predict(
            point_coords=self.points,
            point_labels=self.labels
        )
```

**문제점**:
- ❌ 모든 프레임에 수동 annotation 필요
- ❌ 프레임 간 정보 공유 없음
- ❌ 동일 객체를 여러 번 annotation 해야 함
- ❌ 시간적 일관성 보장 없음

#### 4.1.2 web_app.py (Import만 존재)

```python
# /home/joon/dev/sam3d_gui/src/web_app.py
from sam2.sam2_video_predictor import SAM2VideoPredictor  # Import만 되어 있음

self.sam2_video_predictor = None  # 초기화만, 실제 사용 안 함
```

**현황**:
- ✅ Video Predictor import 완료
- ❌ 실제 초기화 및 사용 코드 없음
- ❌ Memory-based tracking 미활용

### 4.2 통합 제안: Video Predictor 활용

#### 4.2.1 새로운 클래스: VideoAnnotator

```python
# /home/joon/dev/sam3d_gui/src/video_annotator.py
"""
SAM2 Video Predictor 기반 효율적 비디오 annotation
"""

import cv2
import numpy as np
from pathlib import Path
import torch
import sys
from typing import Optional, Tuple, List

sys.path.append(str(Path.home() / 'dev/segment-anything-2'))
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor


class VideoAnnotator:
    """
    SAM2 Video Predictor를 활용한 효율적 비디오 annotation

    Key Features:
    - 한 프레임 annotation으로 전체 비디오 자동 추적
    - Memory-based propagation
    - Occlusion 자동 처리
    """

    SAM_MODELS = {
        'base_plus': {
            'config': 'configs/sam2.1/sam2.1_hiera_b+.yaml',
            'checkpoint': 'checkpoints/sam2.1_hiera_base_plus.pt',
        },
        'large': {
            'config': 'configs/sam2.1/sam2.1_hiera_l.yaml',
            'checkpoint': 'checkpoints/sam2.1_hiera_large.pt',
        }
    }

    def __init__(self, sam2_base_path: Path, device: str = "cuda"):
        self.sam2_base_path = sam2_base_path
        self.device = device if torch.cuda.is_available() else "cpu"

        # Video predictor
        self.predictor: Optional[SAM2VideoPredictor] = None
        self.current_model = None

        # Video state
        self.video_path = None
        self.inference_state = None  # Video predictor 내부 상태
        self.total_frames = 0

        # Tracking state
        self.object_ids = []  # 추적 중인 객체 ID 목록
        self.object_annotations = {}  # {obj_id: {frame_idx: points}}

        # Output
        self.output_dir = Path.home() / "dev/sam3d_gui/outputs/video_annotations"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self, model_name: str = "base_plus") -> str:
        """SAM2 Video Predictor 로드"""
        try:
            model_info = self.SAM_MODELS[model_name]
            config_path = self.sam2_base_path / model_info['config']
            checkpoint_path = self.sam2_base_path / model_info['checkpoint']

            if not checkpoint_path.exists():
                return f"Checkpoint not found: {checkpoint_path}"

            # Build video predictor
            self.predictor = build_sam2_video_predictor(
                config_file=str(config_path),
                ckpt_path=str(checkpoint_path),
                device=self.device
            )
            self.current_model = model_name

            return f"✓ Loaded SAM2 Video Predictor: {model_name} on {self.device}"

        except Exception as e:
            return f"✗ Failed to load model: {str(e)}"

    def init_video(self, video_path: str) -> Tuple[bool, str, int]:
        """
        비디오 초기화 및 inference state 생성

        Args:
            video_path: 비디오 파일 경로

        Returns:
            (success, message, total_frames)
        """
        if self.predictor is None:
            return False, "Load model first", 0

        try:
            self.video_path = Path(video_path)

            if not self.video_path.exists():
                return False, f"Video not found: {video_path}", 0

            # Video info
            cap = cv2.VideoCapture(str(self.video_path))
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # Initialize inference state (핵심!)
            # 이 단계에서 전체 비디오의 메모리 구조 초기화
            self.inference_state = self.predictor.init_state(
                video_path=str(self.video_path)
            )

            # Reset tracking
            self.object_ids = []
            self.object_annotations = {}

            msg = f"✓ Video initialized: {self.video_path.name} ({self.total_frames} frames)"
            return True, msg, self.total_frames

        except Exception as e:
            return False, f"✗ Error: {str(e)}", 0

    def add_object_annotation(
        self,
        frame_idx: int,
        points: List[Tuple[int, int]],
        labels: List[int],
        object_id: Optional[int] = None
    ) -> Tuple[bool, str, int]:
        """
        특정 프레임에 객체 annotation 추가

        Args:
            frame_idx: Annotation할 프레임 인덱스
            points: [(x, y), ...] 포인트 좌표
            labels: [1, 1, 0, ...] (1=foreground, 0=background)
            object_id: 객체 ID (None이면 자동 생성)

        Returns:
            (success, message, assigned_object_id)
        """
        if self.inference_state is None:
            return False, "Initialize video first", -1

        try:
            # Object ID 생성
            if object_id is None:
                object_id = len(self.object_ids)
                self.object_ids.append(object_id)

            # SAM2 Video Predictor에 annotation 추가
            point_coords = np.array(points, dtype=np.float32)
            point_labels = np.array(labels, dtype=np.int32)

            # 핵심 API: add_new_points_or_box
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=object_id,
                points=point_coords,
                labels=point_labels
            )

            # Annotation 기록
            if object_id not in self.object_annotations:
                self.object_annotations[object_id] = {}

            self.object_annotations[object_id][frame_idx] = {
                'points': points,
                'labels': labels
            }

            msg = f"✓ Added annotation for object {object_id} at frame {frame_idx}"
            return True, msg, object_id

        except Exception as e:
            return False, f"✗ Error: {str(e)}", -1

    def propagate_in_video(self) -> Tuple[bool, str, dict]:
        """
        전체 비디오에 대해 자동 propagation 실행

        한 번의 호출로 모든 프레임의 마스크 생성!

        Returns:
            (success, message, results)
            results = {
                frame_idx: {
                    obj_id: mask (H, W) numpy array
                }
            }
        """
        if self.inference_state is None:
            return False, "Initialize video first", {}

        if len(self.object_ids) == 0:
            return False, "Add at least one object annotation", {}

        try:
            results = {}

            # 핵심 API: propagate_in_video
            # 모든 프레임에 대해 자동으로 마스크 생성
            for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(
                self.inference_state
            ):
                # 각 프레임의 결과 저장
                results[frame_idx] = {}

                for obj_id, logit in zip(obj_ids, mask_logits):
                    # Logit을 binary mask로 변환
                    mask = (logit > 0.0).cpu().numpy().squeeze()
                    results[frame_idx][obj_id] = mask

            msg = f"✓ Propagated {len(results)} frames, {len(self.object_ids)} objects"
            return True, msg, results

        except Exception as e:
            return False, f"✗ Error: {str(e)}", {}

    def get_frame_mask(
        self,
        frame_idx: int,
        object_id: Optional[int] = None
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        특정 프레임의 마스크 가져오기 (on-demand)

        Args:
            frame_idx: 프레임 인덱스
            object_id: 특정 객체 (None이면 모든 객체 합성)

        Returns:
            (mask, message)
        """
        if self.inference_state is None:
            return None, "Initialize video first"

        try:
            # 특정 프레임만 처리 (propagate_in_video의 경량 버전)
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                self.inference_state,
                start_frame_idx=frame_idx,
                max_frame_num_to_track=1  # 한 프레임만
            ):
                if out_frame_idx == frame_idx:
                    if object_id is not None:
                        # 특정 객체만
                        idx = out_obj_ids.index(object_id)
                        logit = out_mask_logits[idx]
                        mask = (logit > 0.0).cpu().numpy().squeeze()
                    else:
                        # 모든 객체 합성
                        masks = [(logit > 0.0).cpu().numpy().squeeze()
                                for logit in out_mask_logits]
                        mask = np.logical_or.reduce(masks)

                    return mask, "Success"

            return None, "Frame not found"

        except Exception as e:
            return None, f"✗ Error: {str(e)}"

    def refine_annotation(
        self,
        frame_idx: int,
        object_id: int,
        additional_points: List[Tuple[int, int]],
        additional_labels: List[int]
    ) -> Tuple[bool, str]:
        """
        특정 프레임의 annotation 수정 (interactive refinement)

        Args:
            frame_idx: 수정할 프레임
            object_id: 객체 ID
            additional_points: 추가 포인트
            additional_labels: 추가 레이블
        """
        return self.add_object_annotation(
            frame_idx=frame_idx,
            points=additional_points,
            labels=additional_labels,
            object_id=object_id
        )

    def save_results(
        self,
        results: dict,
        format: str = "png"
    ) -> str:
        """
        결과 저장

        Args:
            results: propagate_in_video()의 결과
            format: 'png' or 'npy'
        """
        try:
            video_name = self.video_path.stem
            output_subdir = self.output_dir / video_name
            output_subdir.mkdir(exist_ok=True)

            for frame_idx, objects in results.items():
                for obj_id, mask in objects.items():
                    if format == "png":
                        mask_uint8 = (mask * 255).astype(np.uint8)
                        filename = f"frame_{frame_idx:04d}_obj_{obj_id}.png"
                        cv2.imwrite(str(output_subdir / filename), mask_uint8)
                    elif format == "npy":
                        filename = f"frame_{frame_idx:04d}_obj_{obj_id}.npy"
                        np.save(str(output_subdir / filename), mask)

            msg = f"✓ Saved {len(results)} frames to {output_subdir}"
            return msg

        except Exception as e:
            return f"✗ Error saving: {str(e)}"


# Usage Example
if __name__ == "__main__":
    # Initialize
    annotator = VideoAnnotator(
        sam2_base_path=Path.home() / "dev/segment-anything-2",
        device="cuda"
    )

    # Load model
    print(annotator.load_model("base_plus"))

    # Init video
    success, msg, total = annotator.init_video(
        "/home/joon/dev/data/markerless_mouse/mouse_1/Camera1/0.mp4"
    )
    print(msg)

    # Annotate one frame only!
    success, msg, obj_id = annotator.add_object_annotation(
        frame_idx=0,
        points=[(500, 400)],  # 마우스 클릭 위치
        labels=[1]  # Foreground
    )
    print(msg)

    # Propagate to all frames automatically
    success, msg, results = annotator.propagate_in_video()
    print(msg)

    # Save all masks
    print(annotator.save_results(results, format="png"))

    # Check specific frame
    mask, msg = annotator.get_frame_mask(frame_idx=100, object_id=obj_id)
    print(f"Frame 100 mask shape: {mask.shape if mask is not None else 'None'}")
```

#### 4.2.2 Gradio UI 통합

```python
# /home/joon/dev/sam3d_gui/src/web_app.py에 추가

def create_video_annotation_tab():
    """
    Tab 4: Video Annotation (SAM2 Video Predictor)

    Workflow:
    1. Load video
    2. Annotate first frame (or any frame)
    3. Click "Propagate" → All frames auto-annotated
    4. Review and refine if needed
    5. Export masks or 3D reconstruct
    """
    with gr.Tab("Video Annotation"):
        with gr.Row():
            # Left: Video player
            with gr.Column(scale=2):
                video_input = gr.Video(label="Upload Video")
                frame_slider = gr.Slider(
                    minimum=0,
                    maximum=100,
                    step=1,
                    label="Frame Index"
                )
                frame_display = gr.Image(label="Current Frame")

                with gr.Row():
                    prev_btn = gr.Button("◀ Prev Frame")
                    next_btn = gr.Button("Next Frame ▶")

            # Right: Annotation controls
            with gr.Column(scale=1):
                model_select = gr.Radio(
                    choices=["base_plus", "large"],
                    value="base_plus",
                    label="Model"
                )
                load_model_btn = gr.Button("Load Model")

                gr.Markdown("### Annotation")
                point_type = gr.Radio(
                    choices=["Foreground", "Background"],
                    value="Foreground",
                    label="Point Type"
                )

                add_point_btn = gr.Button("Add Point (Click on Image)")

                gr.Markdown("### Propagation")
                propagate_btn = gr.Button("🚀 Propagate to All Frames", variant="primary")
                progress_bar = gr.Progress()

                status_text = gr.Textbox(label="Status", lines=5)

                gr.Markdown("### Export")
                export_format = gr.Radio(
                    choices=["PNG", "NPY"],
                    value="PNG",
                    label="Format"
                )
                export_btn = gr.Button("💾 Export All Masks")

        # Event handlers
        def on_load_model(model_name):
            return video_annotator.load_model(model_name)

        def on_video_upload(video):
            success, msg, total = video_annotator.init_video(video)
            if success:
                return msg, gr.Slider(maximum=total-1)
            return msg, gr.Slider()

        def on_frame_change(frame_idx):
            # Get frame from video
            cap = cv2.VideoCapture(video_annotator.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame_rgb
            return None

        def on_add_point(evt: gr.SelectData, point_type):
            x, y = evt.index
            label = 1 if point_type == "Foreground" else 0

            # Add to video predictor
            success, msg, obj_id = video_annotator.add_object_annotation(
                frame_idx=current_frame_idx,
                points=[(x, y)],
                labels=[label]
            )
            return msg

        def on_propagate():
            success, msg, results = video_annotator.propagate_in_video()
            if success:
                # Store results for export
                video_annotator.last_results = results
                return f"{msg}\n✓ Ready to export!"
            return msg

        def on_export(format_type):
            if not hasattr(video_annotator, 'last_results'):
                return "Run propagation first"

            format_str = format_type.lower()
            return video_annotator.save_results(
                video_annotator.last_results,
                format=format_str
            )

        # Connect events
        load_model_btn.click(
            fn=on_load_model,
            inputs=[model_select],
            outputs=[status_text]
        )

        video_input.change(
            fn=on_video_upload,
            inputs=[video_input],
            outputs=[status_text, frame_slider]
        )

        frame_slider.change(
            fn=on_frame_change,
            inputs=[frame_slider],
            outputs=[frame_display]
        )

        frame_display.select(
            fn=on_add_point,
            inputs=[point_type],
            outputs=[status_text]
        )

        propagate_btn.click(
            fn=on_propagate,
            outputs=[status_text]
        )

        export_btn.click(
            fn=on_export,
            inputs=[export_format],
            outputs=[status_text]
        )
```

### 4.3 Best Practices 제안

#### 4.3.1 모델 선택 가이드

| 시나리오 | 추천 모델 | 이유 |
|---------|----------|------|
| **단일 이미지 segmentation** | SAM2ImagePredictor (large) | 최고 품질, 프레임 간 관계 불필요 |
| **이미지 배치 처리** | SAM2ImagePredictor (base_plus) | 속도와 품질 균형 |
| **비디오 객체 추적** | SAM2VideoPredictor (base_plus) | Memory 기반 효율성 |
| **고해상도 비디오** | SAM2VideoPredictor (large) | 품질 우선 |
| **실시간 처리** | SAM2ImagePredictor (tiny) | 최대 속도 |

#### 4.3.2 통합 전략

**Unified API 설계**:

```python
# /home/joon/dev/sam3d_gui/src/sam_unified.py
"""
SAM2 통합 API: Image와 Video Predictor 자동 선택
"""

class SAMUnified:
    """
    자동으로 적절한 predictor 선택
    """

    def __init__(self, sam2_base_path, device="auto"):
        self.image_predictor = None
        self.video_predictor = None
        self.mode = None  # 'image' or 'video'

    def set_mode(self, mode: str):
        """
        모드 설정

        Args:
            mode: 'image' (독립 프레임) or 'video' (시간적 추적)
        """
        if mode == "image":
            # Image predictor 사용
            if self.image_predictor is None:
                self.image_predictor = build_sam2(...)
            self.mode = "image"

        elif mode == "video":
            # Video predictor 사용
            if self.video_predictor is None:
                self.video_predictor = build_sam2_video_predictor(...)
            self.mode = "video"

    def process(self, input_data, **kwargs):
        """
        입력에 따라 자동 처리
        """
        if self.mode == "image":
            # 독립적 이미지 처리
            return self._process_image(input_data, **kwargs)

        elif self.mode == "video":
            # 비디오 추적
            return self._process_video(input_data, **kwargs)


# Usage
sam = SAMUnified(sam2_base_path=Path.home() / "dev/segment-anything-2")

# Scenario 1: 독립 프레임 처리
sam.set_mode("image")
for frame in frames:
    mask = sam.process(frame, points=[(x, y)], labels=[1])

# Scenario 2: 비디오 추적
sam.set_mode("video")
sam.process(video_path, initial_frame=0, points=[(x, y)], labels=[1])
# → 모든 프레임 자동 추적
```

#### 4.3.3 성능 최적화

**1. torch.compile() 활용** [8]

```python
# SAM2 Video Predictor 성능 향상
video_predictor = build_sam2_video_predictor(
    ...,
    vos_optimized=True  # torch.compile 활성화
)
# → Major speedup for video inference
```

**2. Multi-object 추적 효율화** [9]

```python
# ❌ 비효율: 각 객체를 따로 처리
for obj_id in [0, 1, 2]:
    predictor.add_new_points(obj_id=obj_id, ...)
    results = predictor.propagate_in_video()

# ✅ 효율: 한 번에 여러 객체 추적 (이미지 특징 공유)
predictor.add_new_points(obj_id=0, ...)
predictor.add_new_points(obj_id=1, ...)
predictor.add_new_points(obj_id=2, ...)
results = predictor.propagate_in_video()  # 모든 객체 한 번에
```

**3. 프레임 스트라이드 적용**

```python
# 긴 비디오는 stride로 샘플링
def process_long_video(video_path, stride=5):
    """
    긴 비디오를 stride로 샘플링하여 처리

    Args:
        stride: N 프레임마다 하나씩 처리
    """
    # Annotation on frame 0
    predictor.add_new_points(frame_idx=0, ...)

    # Propagate only on sampled frames
    for frame_idx in range(0, total_frames, stride):
        mask = predictor.get_frame_mask(frame_idx)
        # Process mask
```

---

## 5. 실용적 제안 및 구현 로드맵

### 5.1 단계별 통합 계획

#### Phase 1: Video Predictor 통합 (2-3일)

**목표**: VideoAnnotator 클래스 구현 및 기본 동작 검증

**작업**:
1. ✅ `src/video_annotator.py` 생성
2. ✅ SAM2VideoPredictor 초기화
3. ✅ 기본 annotation 및 propagation API
4. ✅ 단위 테스트 작성

**검증**:
```python
# Test script
annotator = VideoAnnotator(...)
annotator.load_model("base_plus")
annotator.init_video("test_video.mp4")
annotator.add_object_annotation(frame_idx=0, points=[(x, y)], labels=[1])
results = annotator.propagate_in_video()
# → 모든 프레임에 마스크 생성 확인
```

#### Phase 2: Gradio UI 통합 (2-3일)

**목표**: Web UI에 Video Annotation 탭 추가

**작업**:
1. ✅ Tab 4 추가: "Video Annotation"
2. ✅ 비디오 업로드 및 프레임 네비게이션
3. ✅ Interactive annotation (클릭으로 포인트 추가)
4. ✅ Propagate 버튼 및 progress bar
5. ✅ 결과 export (PNG, NPY)

**UI Mockup**:
```
┌─────────────────────────────────────────────────────┐
│ Tab 4: Video Annotation (SAM2 Video Predictor)      │
├──────────────────┬──────────────────────────────────┤
│ Video Player     │ Annotation Controls              │
│                  │                                  │
│ [Video Canvas]   │ Model: ○ base_plus ○ large      │
│                  │ [Load Model]                     │
│ Frame: 0/1000    │                                  │
│ ━━━━━━━━━━━━━━━ │ Point Type: ○ FG ○ BG          │
│ [◀ Prev] [Next▶] │ [Add Point (Click on Video)]     │
│                  │                                  │
│                  │ [🚀 Propagate to All Frames]     │
│                  │                                  │
│                  │ Status: Ready                    │
│                  │                                  │
│                  │ Export: ○ PNG ○ NPY             │
│                  │ [💾 Export All Masks]            │
└──────────────────┴──────────────────────────────────┘
```

#### Phase 3: SAM3D 통합 (1-2일)

**목표**: Propagated masks를 SAM3D로 3D 재구성

**작업**:
1. ✅ Video Annotation 결과를 SAM3D Processor로 전달
2. ✅ 선택된 프레임의 마스크로 3D 재구성
3. ✅ Batch 3D reconstruction (여러 프레임)

**Workflow**:
```
Video → VideoAnnotator → Propagate → All Masks
                              ↓
                    Select Frame(s)
                              ↓
                    SAM3DProcessor.reconstruct_3d()
                              ↓
                    PLY/OBJ export
```

#### Phase 4: 성능 최적화 및 고급 기능 (2-3일)

**작업**:
1. ✅ torch.compile() 적용
2. ✅ Multi-object tracking
3. ✅ Interactive refinement (annotation 수정)
4. ✅ Occlusion 감지 및 시각화
5. ✅ 결과 품질 메트릭 (confidence, IoU)

### 5.2 Migration Path (기존 코드 마이그레이션)

#### 현재 LiteAnnotator 사용자를 위한 전환 가이드

**Before (LiteAnnotator - 모든 프레임 annotation 필요)**:
```python
lite = LiteAnnotator(sam2_base_path, device="cuda")
lite.load_model("large")
lite.change_input_source(video_path, 'video')

# 각 프레임마다 annotation 필요
for frame_idx in range(total_frames):
    lite.load_frame(frame_idx)
    lite.add_point(x, y, 'foreground')
    frame_vis, mask, msg = lite.generate_mask()
    lite.save_annotation()
```

**After (VideoAnnotator - 한 프레임만 annotation)**:
```python
video = VideoAnnotator(sam2_base_path, device="cuda")
video.load_model("base_plus")
video.init_video(video_path)

# 첫 프레임만 annotation
video.add_object_annotation(
    frame_idx=0,
    points=[(x, y)],
    labels=[1]
)

# 나머지 모든 프레임 자동 추적
success, msg, results = video.propagate_in_video()
video.save_results(results)
```

**시간 절약**:
- Before: 1000 프레임 × 30초 = ~8시간
- After: 첫 프레임 30초 + propagate 2분 = **~2.5분** (99.5% 시간 절약!)

### 5.3 코드 개선 방향

#### 5.3.1 현재 sam3d_processor.py 개선

**Issue**: 비효율적인 프레임별 독립 처리

```python
# 현재 코드 (sam3d_processor.py)
def track_object_across_frames(self, frames, initial_bbox=None, ...):
    """각 프레임을 독립적으로 segmentation"""
    for idx, frame in enumerate(frames):
        # 매번 새로 segmentation (비효율!)
        mask = self.segment_object_interactive(frame, bbox=initial_bbox, method='grabcut')
        # ...
```

**개선안**:

```python
# 개선된 sam3d_processor.py
def track_object_across_frames(self, frames, initial_bbox=None, use_video_predictor=True, ...):
    """
    SAM2 Video Predictor를 활용한 효율적 추적

    Args:
        use_video_predictor: True면 Video Predictor 사용, False면 기존 방식
    """
    if use_video_predictor and self.sam2_video_predictor:
        # Video Predictor로 효율적 추적
        return self._track_with_video_predictor(frames, initial_bbox, ...)
    else:
        # 기존 방식 (fallback)
        return self._track_with_image_predictor(frames, initial_bbox, ...)

def _track_with_video_predictor(self, frames, initial_bbox, ...):
    """SAM2 Video Predictor 사용"""
    # 1. 임시 비디오 파일 생성
    temp_video = self._frames_to_video(frames)

    # 2. Video Predictor 초기화
    inference_state = self.sam2_video_predictor.init_state(temp_video)

    # 3. 첫 프레임 annotation (bbox → points 변환)
    center_x, center_y = initial_bbox[0] + initial_bbox[2]//2, initial_bbox[1] + initial_bbox[3]//2
    self.sam2_video_predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=0,
        box=initial_bbox  # bbox 직접 사용 가능!
    )

    # 4. 모든 프레임 자동 추적
    results = {}
    for frame_idx, obj_ids, mask_logits in self.sam2_video_predictor.propagate_in_video(inference_state):
        mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
        results[frame_idx] = mask

    # 5. TrackingResult 생성
    segments = []
    for idx, mask in results.items():
        segment_info = SegmentInfo(
            frame_idx=idx,
            mask=mask,
            bbox=self._mask_to_bbox(mask),
            center=self._mask_to_center(mask),
            area=mask.sum()
        )
        segments.append(segment_info)

    return TrackingResult(
        start_frame=0,
        end_frame=len(frames)-1,
        segments=segments,
        motion_detected=self._detect_motion(segments, motion_threshold),
        duration_seconds=len(frames) / fps
    )
```

#### 5.3.2 Unified Segmentation API

```python
# /home/joon/dev/sam3d_gui/src/segmentation_factory.py
"""
SAM 기반 segmentation 통합 팩토리
"""

from enum import Enum
from typing import Union, List
import numpy as np

class SegmentationMode(Enum):
    IMAGE = "image"  # 독립 프레임
    VIDEO = "video"  # 시간적 추적
    AUTO = "auto"    # 자동 선택


class SegmentationFactory:
    """
    통합 segmentation 팩토리

    입력에 따라 자동으로 적절한 predictor 선택
    """

    def __init__(self, sam2_base_path, device="auto"):
        self.lite_annotator = None  # Image predictor
        self.video_annotator = None  # Video predictor

    def segment(
        self,
        input_data: Union[np.ndarray, str, List[np.ndarray]],
        points: List[tuple],
        labels: List[int],
        mode: SegmentationMode = SegmentationMode.AUTO
    ):
        """
        통합 segmentation API

        Args:
            input_data:
                - np.ndarray: 단일 이미지 (H, W, 3)
                - str: 비디오 파일 경로
                - List[np.ndarray]: 프레임 시퀀스
            points: [(x, y), ...]
            labels: [1, 0, ...]
            mode: 처리 모드

        Returns:
            마스크 또는 마스크 시퀀스
        """
        # 모드 자동 결정
        if mode == SegmentationMode.AUTO:
            if isinstance(input_data, np.ndarray):
                mode = SegmentationMode.IMAGE
            elif isinstance(input_data, (str, list)):
                mode = SegmentationMode.VIDEO

        # 처리
        if mode == SegmentationMode.IMAGE:
            return self._segment_image(input_data, points, labels)
        elif mode == SegmentationMode.VIDEO:
            return self._segment_video(input_data, points, labels)

    def _segment_image(self, image, points, labels):
        """단일 이미지 segmentation"""
        if self.lite_annotator is None:
            self.lite_annotator = LiteAnnotator(...)

        # Image predictor 사용
        self.lite_annotator.predictor.set_image(image)
        masks, scores, _ = self.lite_annotator.predictor.predict(
            point_coords=np.array(points),
            point_labels=np.array(labels)
        )
        return masks[0]  # Best mask

    def _segment_video(self, video_data, points, labels):
        """비디오 segmentation"""
        if self.video_annotator is None:
            self.video_annotator = VideoAnnotator(...)

        # Video predictor 사용
        if isinstance(video_data, str):
            # 비디오 파일
            self.video_annotator.init_video(video_data)
        else:
            # 프레임 시퀀스
            temp_video = self._frames_to_video(video_data)
            self.video_annotator.init_video(temp_video)

        # 첫 프레임 annotation
        self.video_annotator.add_object_annotation(0, points, labels)

        # Propagate
        _, _, results = self.video_annotator.propagate_in_video()
        return results


# Usage
factory = SegmentationFactory(sam2_base_path)

# Scenario 1: 단일 이미지
mask = factory.segment(
    input_data=image,  # np.ndarray
    points=[(500, 400)],
    labels=[1]
)

# Scenario 2: 비디오
masks = factory.segment(
    input_data="video.mp4",  # 자동으로 Video Predictor 사용
    points=[(500, 400)],
    labels=[1]
)
```

### 5.4 성능 벤치마크

#### 예상 성능 개선

**테스트 시나리오**: 1000 프레임 비디오, 한 객체 추적

| 방법 | 시간 | Annotation 횟수 | 메모리 |
|------|------|----------------|--------|
| **LiteAnnotator (현재)** | ~8시간 | 1000번 | 낮음 |
| **VideoAnnotator (제안)** | ~2.5분 | 1번 | 중간 |
| **개선율** | **99.5% 감소** | **99.9% 감소** | +20% |

**세부 분석**:
```
LiteAnnotator (Image Predictor):
- Frame 0: 30s (manual annotation + inference)
- Frame 1: 30s (manual annotation + inference)
- ...
- Frame 999: 30s
Total: 1000 × 30s = 30,000s ≈ 8.3 hours

VideoAnnotator (Video Predictor):
- Frame 0: 30s (manual annotation + inference)
- Propagate all: 120s (automatic inference on 1000 frames)
Total: 30s + 120s = 150s ≈ 2.5 minutes
```

---

## 6. SAM3D 활용 전략

### 6.1 SAM3D Objects 최적 사용법

#### 6.1.1 입력 품질 최적화

**Best Practices**:

1. **High-quality 2D mask**:
   - SAM2 Video Predictor로 일관된 마스크 생성
   - Temporal consistency 확보

2. **적절한 프레임 선택**:
   - 객체가 명확히 보이는 프레임
   - Occlusion 최소화
   - 정면 또는 측면 view

3. **Multi-view reconstruction** (선택):
   - 여러 프레임의 3D 재구성 결합
   - 더 완전한 3D 모델

```python
# Multi-view 3D reconstruction
def reconstruct_3d_multiview(video_results, sam3d_processor):
    """
    여러 프레임의 3D 재구성을 결합

    Args:
        video_results: VideoAnnotator.propagate_in_video() 결과
        sam3d_processor: SAM3DProcessor 인스턴스
    """
    # 1. 주요 프레임 선택 (균등 샘플링)
    key_frames = [0, 250, 500, 750, 999]

    # 2. 각 프레임에서 3D 재구성
    reconstructions = []
    for frame_idx in key_frames:
        frame = get_frame(video, frame_idx)
        mask = video_results[frame_idx][obj_id]

        recon = sam3d_processor.reconstruct_3d(frame, mask)
        reconstructions.append(recon)

    # 3. 3D 모델 결합 (alignment + fusion)
    final_3d = merge_3d_reconstructions(reconstructions)

    return final_3d
```

#### 6.1.2 출력 포맷 선택

| 포맷 | 용도 | 장점 | 단점 |
|------|------|------|------|
| **PLY** | Gaussian Splatting, point cloud | 고품질, 빠른 렌더링 | 파일 크기 큰 편 |
| **OBJ** | 3D 편집 (Blender, Maya) | 텍스처 + UV, 범용성 | Mesh 변환 필요 |
| **GLB** | 게임 엔진 (Unity, Unreal) | 최적화, 애니메이션 지원 | 복잡한 설정 |

**권장 워크플로우**:
```
SAM3D → PLY (primary output)
  ↓
Convert → OBJ (for editing)
  ↓
Convert → GLB (for game engine)
```

### 6.2 SAM2 + SAM3D 통합 파이프라인

```python
# /home/joon/dev/sam3d_gui/src/pipeline_integrated.py
"""
SAM2 Video Predictor + SAM3D Objects 통합 파이프라인
"""

class IntegratedPipeline:
    """
    End-to-end 파이프라인: 비디오 → 3D 재구성

    Workflow:
    1. SAM2 Video Predictor로 모든 프레임 segmentation
    2. 주요 프레임 선택
    3. SAM3D Objects로 3D 재구성
    4. Multi-view fusion (선택)
    """

    def __init__(self, sam2_base_path, sam3d_checkpoint):
        self.video_annotator = VideoAnnotator(sam2_base_path)
        self.sam3d_processor = SAM3DProcessor(sam3d_checkpoint)

    def process_video_to_3d(
        self,
        video_path: str,
        annotation_frame: int = 0,
        annotation_points: List[tuple] = None,
        annotation_labels: List[int] = None,
        reconstruction_frames: List[int] = None,
        multiview: bool = True
    ):
        """
        비디오에서 3D 모델 생성

        Args:
            video_path: 비디오 파일
            annotation_frame: Annotation할 프레임 (보통 0)
            annotation_points: [(x, y), ...]
            annotation_labels: [1, 0, ...]
            reconstruction_frames: 3D 재구성할 프레임들 (None이면 자동 선택)
            multiview: 여러 프레임 결합 여부

        Returns:
            3D 재구성 결과
        """
        # Step 1: Video segmentation
        print("Step 1: SAM2 Video Predictor - Segmentation")
        self.video_annotator.load_model("base_plus")
        self.video_annotator.init_video(video_path)

        # Annotation
        self.video_annotator.add_object_annotation(
            frame_idx=annotation_frame,
            points=annotation_points,
            labels=annotation_labels
        )

        # Propagate
        success, msg, video_results = self.video_annotator.propagate_in_video()
        print(f"  {msg}")

        # Step 2: 주요 프레임 선택
        if reconstruction_frames is None:
            total = len(video_results)
            if multiview:
                # 5-10개 균등 샘플링
                reconstruction_frames = [int(i * total / 10) for i in range(10)]
            else:
                # 중간 프레임 하나만
                reconstruction_frames = [total // 2]

        print(f"Step 2: Selected {len(reconstruction_frames)} frames for 3D reconstruction")

        # Step 3: SAM3D 3D 재구성
        print("Step 3: SAM3D Objects - 3D Reconstruction")
        reconstructions = []

        cap = cv2.VideoCapture(video_path)

        for frame_idx in reconstruction_frames:
            # 프레임 로드
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame_bgr = cap.read()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # 마스크 가져오기
            mask = video_results[frame_idx][0]  # obj_id=0

            # 3D 재구성
            print(f"  Reconstructing frame {frame_idx}...")
            recon = self.sam3d_processor.reconstruct_3d(frame_rgb, mask)
            reconstructions.append({
                'frame_idx': frame_idx,
                'reconstruction': recon
            })

        cap.release()

        # Step 4: Multi-view fusion (optional)
        if multiview and len(reconstructions) > 1:
            print("Step 4: Multi-view fusion")
            final_3d = self._fuse_reconstructions(reconstructions)
        else:
            final_3d = reconstructions[0]['reconstruction']

        return final_3d

    def _fuse_reconstructions(self, reconstructions):
        """
        여러 3D 재구성 결합

        TODO: 실제 구현 필요
        - Point cloud alignment (ICP)
        - Mesh fusion
        - Texture blending
        """
        # Placeholder: 첫 번째 재구성만 반환
        print("  Multi-view fusion not yet implemented, using first reconstruction")
        return reconstructions[0]['reconstruction']


# Usage Example
if __name__ == "__main__":
    pipeline = IntegratedPipeline(
        sam2_base_path=Path.home() / "dev/segment-anything-2",
        sam3d_checkpoint=Path.home() / "dev/sam-3d-objects/checkpoints/hf"
    )

    # 비디오에서 3D 모델 생성
    result_3d = pipeline.process_video_to_3d(
        video_path="/home/joon/dev/data/markerless_mouse/mouse_1/Camera1/0.mp4",
        annotation_frame=0,
        annotation_points=[(500, 400)],  # 마우스 클릭 위치
        annotation_labels=[1],
        multiview=True  # 여러 프레임 결합
    )

    # 저장
    pipeline.sam3d_processor.export_mesh(result_3d, "output_mouse.ply", format="ply")
    print("✓ 3D reconstruction complete!")
```

---

## 7. 실전 예제

### 7.1 예제 1: 마우스 비디오 추적 및 3D 재구성

```python
"""
마우스 비디오에서 객체 추적 및 3D 재구성
"""

# Step 1: Video annotation
video_annotator = VideoAnnotator(
    sam2_base_path=Path.home() / "dev/segment-anything-2",
    device="cuda"
)

video_annotator.load_model("base_plus")

video_path = "/home/joon/dev/data/markerless_mouse/mouse_1/Camera1/0.mp4"
video_annotator.init_video(video_path)

# 첫 프레임에서 마우스 클릭
video_annotator.add_object_annotation(
    frame_idx=0,
    points=[(500, 400)],  # 마우스 중심
    labels=[1]
)

# 모든 프레임 자동 추적
success, msg, results = video_annotator.propagate_in_video()
print(msg)  # "✓ Propagated 3000 frames, 1 objects"

# Step 2: 결과 저장
video_annotator.save_results(results, format="png")
# → outputs/video_annotations/0/frame_XXXX_obj_0.png

# Step 3: 특정 프레임 3D 재구성
sam3d = SAM3DProcessor(
    sam3d_checkpoint_path=Path.home() / "dev/sam-3d-objects/checkpoints/hf"
)

# 중간 프레임 선택
frame_idx = 1500
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
ret, frame_bgr = cap.read()
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
cap.release()

# 마스크 가져오기
mask = results[frame_idx][0]  # obj_id=0

# 3D 재구성
reconstruction = sam3d.reconstruct_3d(frame_rgb, mask, seed=42)

# 저장
sam3d.export_mesh(reconstruction, "mouse_3d.ply", format="ply")
print("✓ 3D reconstruction saved: mouse_3d.ply")
```

### 7.2 예제 2: Multi-object 추적

```python
"""
여러 객체 동시 추적
"""

video_annotator = VideoAnnotator(...)
video_annotator.load_model("large")
video_annotator.init_video("multi_object_video.mp4")

# Object 1: 첫 번째 마우스
video_annotator.add_object_annotation(
    frame_idx=0,
    points=[(300, 200)],
    labels=[1],
    object_id=None  # 자동 생성 → obj_id=0
)

# Object 2: 두 번째 마우스
video_annotator.add_object_annotation(
    frame_idx=0,
    points=[(700, 400)],
    labels=[1],
    object_id=None  # 자동 생성 → obj_id=1
)

# 모든 객체 동시 추적 (효율적!)
success, msg, results = video_annotator.propagate_in_video()

# 결과 확인
print(f"Tracked {len(video_annotator.object_ids)} objects")
for frame_idx, objects in results.items():
    for obj_id, mask in objects.items():
        print(f"Frame {frame_idx}, Object {obj_id}: Mask shape {mask.shape}")

# 각 객체별 3D 재구성
for obj_id in video_annotator.object_ids:
    frame_rgb = get_frame(video_path, frame_idx=1000)
    mask = results[1000][obj_id]

    recon = sam3d.reconstruct_3d(frame_rgb, mask)
    sam3d.export_mesh(recon, f"object_{obj_id}_3d.ply")
```

### 7.3 예제 3: Interactive refinement

```python
"""
자동 추적 결과를 interactive하게 개선
"""

video_annotator = VideoAnnotator(...)
video_annotator.load_model("base_plus")
video_annotator.init_video(video_path)

# 초기 annotation
video_annotator.add_object_annotation(
    frame_idx=0,
    points=[(500, 400)],
    labels=[1]
)

# 첫 번째 propagation
results = video_annotator.propagate_in_video()

# 중간에 마스크 확인
mask_100 = results[1][100][0]
visualize_mask(mask_100)  # 품질 확인

# 품질이 낮으면 → Refinement
# Frame 100에 추가 포인트 annotation
video_annotator.refine_annotation(
    frame_idx=100,
    object_id=0,
    additional_points=[(520, 410), (480, 390)],  # Foreground 보강
    additional_labels=[1, 1]
)

# 다시 propagation (Frame 100 이후만)
results_refined = video_annotator.propagate_in_video()

# 개선된 결과 확인
mask_100_refined = results_refined[1][100][0]
visualize_mask(mask_100_refined)  # 품질 향상 확인
```

---

## 8. 교훈 및 Best Practices

### 8.1 핵심 교훈

#### 1. Memory-Based Tracking의 강력함

**발견**:
- 한 프레임 annotation만으로 전체 비디오 추적 가능
- Temporal consistency가 자동으로 보장
- Occlusion 처리 자동화

**시사점**:
- 비디오 annotation 작업량 99% 감소
- 실시간 처리 가능 (~44 FPS)
- 긴 비디오에서도 효율적

#### 2. Image vs Video Predictor 구분의 필요성

**이유**:
- 아키텍처 차이 (메모리 유무)
- 초기화 방법 차이
- Use case 차이

**Best Practice**:
- 독립 프레임 → Image Predictor
- 시간적 추적 → Video Predictor
- 통합 API로 자동 선택

#### 3. SAM3D의 실용성

**발견**:
- 단일 이미지에서 고품질 3D 재구성
- Real-world 조건 (occlusion, clutter)에 강건
- 인간 선호도 5:1 승률

**활용**:
- SAM2로 일관된 마스크 → SAM3D로 3D 재구성
- Multi-view fusion으로 품질 향상
- 게임 엔진, 3D 편집 도구 호환

### 8.2 개발 가이드라인

#### Defensive Programming

```python
# ✅ Good: 안전한 Video Predictor 초기화
def init_video_safe(video_path):
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if predictor is None:
        raise RuntimeError("Load model first")

    inference_state = predictor.init_state(video_path)

    if inference_state is None:
        raise RuntimeError("Failed to initialize video state")

    return inference_state

# ❌ Bad: 검증 없이 진행
def init_video_unsafe(video_path):
    inference_state = predictor.init_state(video_path)  # 에러 가능성
    return inference_state
```

#### Error Handling

```python
# ✅ Good: 상세한 에러 메시지
try:
    results = video_annotator.propagate_in_video()
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        print("GPU 메모리 부족. 해결 방법:")
        print("1. 비디오 해상도 낮추기")
        print("2. 프레임 수 줄이기")
        print("3. 더 작은 모델 사용 (base_plus → tiny)")
    elif "No annotation" in str(e):
        print("첫 프레임에 annotation이 필요합니다.")
        print("add_object_annotation() 먼저 호출하세요.")
    else:
        print(f"Unexpected error: {e}")
        raise

# ❌ Bad: 에러 무시
try:
    results = video_annotator.propagate_in_video()
except:
    pass  # Silent failure
```

#### Documentation

```python
# ✅ Good: 명확한 docstring
def propagate_in_video(self) -> Tuple[bool, str, dict]:
    """
    전체 비디오에 대해 자동 propagation 실행

    Requirements:
        - init_video() 호출 완료
        - add_object_annotation() 최소 1번 호출

    Returns:
        success (bool): 성공 여부
        message (str): 상태 메시지
        results (dict): {
            frame_idx (int): {
                obj_id (int): mask (np.ndarray, shape=(H, W), dtype=bool)
            }
        }

    Example:
        >>> annotator.init_video("video.mp4")
        >>> annotator.add_object_annotation(0, [(x, y)], [1])
        >>> success, msg, results = annotator.propagate_in_video()
        >>> print(f"Processed {len(results)} frames")

    Raises:
        RuntimeError: Video not initialized or no annotations
    """
    # Implementation
```

### 8.3 성능 최적화 체크리스트

- [ ] **torch.compile() 활성화**: `vos_optimized=True`
- [ ] **Multi-object 배칭**: 여러 객체를 한 번에 처리
- [ ] **프레임 스트라이드**: 긴 비디오는 샘플링
- [ ] **GPU 활용**: CUDA 우선, CPU fallback
- [ ] **메모리 관리**: FIFO memory bank 크기 조정
- [ ] **Early stopping**: 품질 threshold 도달 시 조기 종료

---

## 9. 결론 및 Next Steps

### 9.1 요약

#### SAM 2 Video Predictor

**핵심 메커니즘**:
- Memory Bank (FIFO) + Memory Attention (cross-attention)
- 한 프레임 annotation → 전체 비디오 자동 추적
- Temporal consistency 암묵적 학습

**성능**:
- ~44 FPS (실시간)
- Occlusion 자동 처리
- 긴 비디오 스트리밍 가능

#### SAM3D

**실체**:
- Meta AI 공식 모델 (2024년 11월 발표)
- SAM 3D Objects + SAM 3D Body

**기능**:
- 단일 이미지 → 고품질 3D 메쉬
- Real-world 조건 robust
- PLY, OBJ, GLB 출력

#### Image vs Video Predictor

**구분 이유**:
- 아키텍처 차이 (메모리 유무)
- Use case 차이 (독립 vs 추적)

**통합 가능성**:
- Unified API로 자동 선택 가능
- 하지만 내부적으로는 별도 predictor 사용

### 9.2 권장 사항

#### 즉시 구현 가능

1. ✅ **VideoAnnotator 클래스 구현** (2-3일)
   - SAM2 Video Predictor 통합
   - 기본 annotation 및 propagation API

2. ✅ **Gradio UI 추가** (2-3일)
   - Tab 4: Video Annotation
   - Interactive annotation
   - Export 기능

#### 단기 목표 (1-2주)

3. ✅ **SAM3D 통합**
   - VideoAnnotator 결과 → SAM3D 3D 재구성
   - Multi-view fusion

4. ✅ **성능 최적화**
   - torch.compile() 적용
   - Multi-object tracking
   - 메모리 효율화

#### 장기 목표 (1개월+)

5. ⏰ **Advanced Features**
   - Interactive refinement UI
   - Quality metrics (IoU, confidence)
   - Batch processing pipeline

6. ⏰ **Production Ready**
   - Comprehensive testing
   - Documentation
   - Deployment guide

### 9.3 Next Steps

**Phase 1 (This Week)**:
1. VideoAnnotator 클래스 구현
2. 단위 테스트 작성
3. 기본 동작 검증

**Phase 2 (Next Week)**:
1. Gradio UI 통합
2. Interactive annotation
3. User testing

**Phase 3 (Following Weeks)**:
1. SAM3D integration
2. Performance optimization
3. Documentation

---

## 참고 자료

### 공식 문서

1. [SAM 2: Segment Anything in Images and Videos (arXiv)](https://arxiv.org/abs/2408.00714)
2. [Meta AI SAM 2 Blog Post](https://ai.meta.com/blog/segment-anything-2/)
3. [SAM 2 GitHub Repository](https://github.com/facebookresearch/sam2)
4. [Meta AI SAM 3D Official Page](https://ai.meta.com/sam3d/)
5. [SAM 3D Objects GitHub](https://github.com/facebookresearch/sam-3d-objects)
6. [SAM 3D Demo](https://sam3d.org/)

### 기술 문서

7. [SAM 2 Ultralytics Documentation](https://docs.ultralytics.com/models/sam-2/)
8. [SAM 2 LearnOpenCV Tutorial](https://learnopencv.com/sam-2/)
9. [Roboflow SAM 2 Video Segmentation Guide](https://blog.roboflow.com/sam-2-video-segmentation/)

### 추가 자료

10. [SAMURAI: Zero-Shot Visual Tracking (arXiv)](https://arxiv.org/html/2411.11922v1)
11. [HuggingFace SAM2 Video Documentation](https://huggingface.co/docs/transformers/en/model_doc/sam2_video)
12. [Analytics Vidhya SAM 2 Tutorial](https://www.analyticsvidhya.com/blog/2024/08/meta-sam-2/)

---

## 부록: API Reference

### VideoAnnotator API

```python
class VideoAnnotator:
    """SAM2 Video Predictor 기반 비디오 annotation"""

    def __init__(self, sam2_base_path: Path, device: str = "cuda"):
        """초기화"""

    def load_model(self, model_name: str = "base_plus") -> str:
        """모델 로드"""

    def init_video(self, video_path: str) -> Tuple[bool, str, int]:
        """비디오 초기화"""

    def add_object_annotation(
        self,
        frame_idx: int,
        points: List[Tuple[int, int]],
        labels: List[int],
        object_id: Optional[int] = None
    ) -> Tuple[bool, str, int]:
        """객체 annotation 추가"""

    def propagate_in_video(self) -> Tuple[bool, str, dict]:
        """전체 비디오 자동 propagation"""

    def get_frame_mask(
        self,
        frame_idx: int,
        object_id: Optional[int] = None
    ) -> Tuple[Optional[np.ndarray], str]:
        """특정 프레임 마스크 가져오기"""

    def refine_annotation(
        self,
        frame_idx: int,
        object_id: int,
        additional_points: List[Tuple[int, int]],
        additional_labels: List[int]
    ) -> Tuple[bool, str]:
        """Annotation 수정"""

    def save_results(self, results: dict, format: str = "png") -> str:
        """결과 저장"""
```

### IntegratedPipeline API

```python
class IntegratedPipeline:
    """SAM2 + SAM3D 통합 파이프라인"""

    def __init__(self, sam2_base_path: Path, sam3d_checkpoint: Path):
        """초기화"""

    def process_video_to_3d(
        self,
        video_path: str,
        annotation_frame: int = 0,
        annotation_points: List[tuple] = None,
        annotation_labels: List[int] = None,
        reconstruction_frames: List[int] = None,
        multiview: bool = True
    ) -> dict:
        """비디오 → 3D 재구성"""
```

---

**문서 버전**: 1.0
**마지막 업데이트**: 2025-11-25
**작성자**: Claude Code
**프로젝트**: sam3d_gui

---

## Sources

- [SAM 2: Segment Anything Model 2 - Ultralytics](https://docs.ultralytics.com/models/sam-2/)
- [SAM-2: Memory-Augmented Video Segmentation](https://www.emergentmind.com/topics/segment-anything-model-2-sam-2)
- [How to Use SAM 2 for Video Segmentation - Roboflow](https://blog.roboflow.com/sam-2-video-segmentation/)
- [SAM 2 GitHub Repository](https://github.com/facebookresearch/sam2)
- [Meta AI SAM 3D Official Page](https://ai.meta.com/sam3d/)
- [SAM 3D: High-Fidelity 3D Reconstruction](https://sam3d.org/)
- [Meta AI's New Segment Anything Model: Exploring SAM 3](https://www.ultralytics.com/blog/exploring-sam-3-meta-ais-new-segment-anything-model)
- [SAM 2: Segment Anything in Images and Videos (arXiv)](https://arxiv.org/abs/2408.00714)
- [SAM 3D Objects GitHub](https://github.com/facebookresearch/sam-3d-objects)
- [SAM 2 – Promptable Segmentation for Images and Videos | LearnOpenCV](https://learnopencv.com/sam-2/)
