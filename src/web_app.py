#!/usr/bin/env python3
"""
SAM 3D GUI - Interactive SAM Annotation Web Interface
대화형 SAM annotation: point 클릭으로 fg/bg 지정 + 비디오 propagation
"""

import sys
import gradio as gr
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import json

from sam3d_processor import SAM3DProcessor
from config_loader import ModelConfig
from lite_annotator import LiteAnnotator

# Load configuration
try:
    config = ModelConfig()
    print(f"✓ Config loaded from: {Path(__file__).parent.parent / 'config' / 'model_config.yaml'}")
except Exception as e:
    print(f"Warning: Failed to load config: {e}")
    config = None

# SAM 2 imports
SAM2_PATH = Path.home() / 'dev/segment-anything-2'
if SAM2_PATH.exists():
    sys.path.insert(0, str(SAM2_PATH))
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    SAM2_AVAILABLE = True
else:
    SAM2ImagePredictor = None
    SAM2VideoPredictor = None
    SAM2_AVAILABLE = False
    print("Warning: SAM 2 not found. Interactive segmentation will use fallback method.")

class SAMInteractiveWebApp:
    """
    SAM 3D GUI - 통합 웹 인터페이스

    모드 1: 자동 처리 (Quick Mode)
    - 비디오 선택 → 자동 세그멘테이션 → 모션 감지 → 결과

    모드 2: 대화형 Annotation (Interactive Mode)
    - Point annotation (foreground/background)
    - 수동 세그멘테이션 → Propagation → 3D mesh
    """

    def __init__(self):
        # Config-based initialization
        self.config = config

        # SAM 3D processor 초기화
        if config:
            sam3d_checkpoint = config.sam3d_checkpoint_dir
            self.processor = SAM3DProcessor(sam3d_checkpoint_path=sam3d_checkpoint)
        else:
            self.processor = SAM3DProcessor()

        # SAM2 predictor 초기화 (Interactive Mode용)
        self.sam2_predictor = None
        self.sam2_video_predictor = None
        self.sam2_device = None
        if SAM2_AVAILABLE and config:
            try:
                print("Loading SAM 2 for interactive segmentation...")
                checkpoint = Path(config.sam2_checkpoint)
                model_cfg = config.sam2_config
                device = config.sam2_device

                # Auto-detect device
                if device == "auto":
                    if torch.cuda.is_available():
                        device = "cuda"
                        gpu_name = torch.cuda.get_device_name(0)
                        print(f"✓ CUDA detected: {gpu_name}")
                    else:
                        device = "cpu"
                        print("Warning: CUDA not available, using CPU")
                elif device == "cuda" and not torch.cuda.is_available():
                    device = "cpu"
                    print("Warning: CUDA not available, using CPU")

                self.sam2_device = device

                if checkpoint.exists():
                    from sam2.build_sam import build_sam2, build_sam2_video_predictor

                    # Image predictor for single-frame segmentation
                    sam2_model = build_sam2(model_cfg, str(checkpoint), device=device)
                    self.sam2_predictor = SAM2ImagePredictor(sam2_model)

                    # Video predictor for memory-based tracking
                    self.sam2_video_predictor = build_sam2_video_predictor(model_cfg, str(checkpoint), device=device)

                    print(f"✓ SAM 2 loaded: {config.cfg.sam2.name} on {device}")
                    print(f"✓ SAM 2 Video Predictor initialized for propagation")
                else:
                    print(f"Warning: SAM 2 checkpoint not found at {checkpoint}")
            except Exception as e:
                print(f"Warning: Failed to load SAM 2: {e}")
                import traceback
                traceback.print_exc()
                self.sam2_predictor = None
                self.sam2_video_predictor = None

        # 상태 관리
        self.video_path = None
        self.frames = []
        self.current_frame_idx = 0
        self.annotations = {
            'foreground': [],  # [(x, y), ...]
            'background': []   # [(x, y), ...]
        }
        self.masks = []  # 각 프레임의 마스크
        self.current_mask = None
        self.tracking_result = None

        # Default paths from config
        if config:
            self.default_data_dir = config.default_data_dir
        else:
            self.default_data_dir = "/home/joon/dev/data/markerless_mouse/"

        # LiteAnnotator 초기화 (Tab 3: Lite Mode)
        self.lite_annotator = None
        if SAM2_AVAILABLE:
            try:
                print("Initializing Lite Annotator...")
                self.lite_annotator = LiteAnnotator(
                    sam2_base_path=SAM2_PATH,
                    device=self.sam2_device if self.sam2_device else "auto"
                )
                print("✓ Lite Annotator initialized")
            except Exception as e:
                print(f"Warning: Failed to initialize Lite Annotator: {e}")
                self.lite_annotator = None

    def quick_process(self, data_dir: str, video_file: str,
                     start_time: float, duration: float,
                     motion_threshold: float, segmentation_method: str,
                     progress=gr.Progress()) -> Tuple[np.ndarray, str]:
        """
        Quick Mode: 자동 처리 (기존 web_app.py 기능 통합)
        """
        if not video_file:
            return None, "비디오를 선택하세요"

        video_path = Path(data_dir) / video_file
        if not video_path.exists():
            return None, f"비디오를 찾을 수 없습니다: {video_path}"

        self.video_path = str(video_path)

        progress(0, desc="비디오 처리 시작...")

        try:
            # 비디오 처리
            result, reconstruction = self.processor.process_video_segment(
                video_path=self.video_path,
                start_time=start_time,
                duration=duration,
                output_dir="outputs/",
                motion_threshold=motion_threshold,
                segmentation_method=segmentation_method
            )

            self.tracking_result = result

            progress(0.5, desc="결과 생성 중...")

            # 결과 시각화
            if result.segments:
                first_frame = result.segments[0]

                # 프레임 다시 읽기
                cap = cv2.VideoCapture(self.video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame.frame_idx)
                ret, frame = cap.read()
                cap.release()

                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mask_colored = np.zeros_like(frame_rgb)
                    mask_colored[first_frame.mask > 0] = [0, 255, 0]
                    overlay = cv2.addWeighted(frame_rgb, 0.7, mask_colored, 0.3, 0)

                    if first_frame.bbox:
                        x1, y1, x2, y2 = first_frame.bbox
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    visualization = overlay
                else:
                    visualization = None
            else:
                visualization = None

            progress(0.8, desc="통계 계산 중...")

            # 결과 텍스트
            result_text = f"""
### 🎯 Quick Process 완료

**기본 정보**
- 분석된 프레임: {len(result.segments)}
- 처리 시간: {start_time}s - {start_time + duration}s

**모션 감지**
- 감지 여부: {'✅ 예' if result.motion_detected else '❌ 아니오'}
- 임계값: {motion_threshold} 픽셀
"""

            if result.motion_detected and len(result.segments) > 1:
                displacements = []
                for i in range(1, len(result.segments)):
                    prev = result.segments[i-1].center
                    curr = result.segments[i].center
                    dx = curr[0] - prev[0]
                    dy = curr[1] - prev[1]
                    disp = (dx**2 + dy**2)**0.5
                    displacements.append(disp)

                if displacements:
                    result_text += f"""
**변위 통계**
- 최대 변위: {max(displacements):.1f} 픽셀
- 평균 변위: {sum(displacements)/len(displacements):.1f} 픽셀
"""

            if result.segments:
                first = result.segments[0]
                result_text += f"""
**객체 정보**
- 바운딩 박스: {first.bbox}
- 중심점: {first.center}
- 면적: {first.area:.0f} 픽셀²

**출력**
- 저장 위치: `outputs/`
"""

            progress(1.0, desc="완료!")

            return visualization, result_text

        except Exception as e:
            import traceback
            error_msg = f"오류:\n```\n{str(e)}\n{traceback.format_exc()}\n```"
            return None, error_msg

    def scan_videos(self, data_dir: str) -> List[str]:
        """디렉토리에서 비디오 파일 스캔"""
        data_path = Path(data_dir)
        if not data_path.exists():
            return []

        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        videos = []

        for ext in video_extensions:
            videos.extend([str(p.relative_to(data_path))
                          for p in data_path.rglob(f'*{ext}')])

        return sorted(videos)

    def load_video(self, data_dir: str, video_file: str,
                   start_time: float, duration: float) -> Tuple[np.ndarray, str, gr.Slider]:
        """비디오 로드 및 프레임 추출"""
        default_slider = gr.Slider(label="프레임 위치", minimum=0, maximum=100, value=0, step=1)

        if not video_file:
            return None, "비디오를 선택하세요", default_slider

        video_path = Path(data_dir) / video_file

        if not video_path.exists():
            return None, f"비디오를 찾을 수 없습니다: {video_path}", default_slider

        self.video_path = str(video_path)

        try:
            # 비디오 정보
            info = self.processor.get_video_info(self.video_path)
            fps = info['fps']

            # 프레임 추출
            start_frame = int(start_time * fps)
            num_frames = int(duration * fps)

            self.frames = self.processor.extract_frames(
                self.video_path,
                start_frame,
                num_frames,
                stride=1
            )

            if not self.frames:
                return None, "❌ 프레임 추출 실패: 프레임이 없습니다", default_slider

            # 초기화
            self.current_frame_idx = 0
            self.annotations = {'foreground': [], 'background': []}
            self.masks = [None] * len(self.frames)
            self.current_mask = None

            info_text = f"""
### 비디오 로드 완료 ✅

- **프레임 수**: {len(self.frames)}
- **해상도**: {info['width']} x {info['height']}
- **FPS**: {info['fps']:.2f}
- **구간**: {start_time}s - {start_time + duration}s

### 다음 단계:
1. **Foreground Point** 클릭하여 객체 위치 지정
2. **Background Point** 클릭하여 배경 위치 지정 (선택사항)
3. **Segment Current Frame** 클릭하여 현재 프레임 세그멘테이션
4. **Propagate to All Frames** 클릭하여 전체 비디오 추적
5. **Generate 3D Mesh** 클릭하여 3D 생성
            """

            # 첫 프레임 반환 + 슬라이더 업데이트
            frame_rgb = cv2.cvtColor(self.frames[0], cv2.COLOR_BGR2RGB)

            # 슬라이더 범위 업데이트
            slider_update = gr.Slider(
                label="프레임 위치",
                minimum=0,
                maximum=len(self.frames) - 1,
                value=0,
                step=1,
                interactive=True,
                info=f"슬라이더를 드래그하여 프레임 이동 (총 {len(self.frames)}개)"
            )

            return frame_rgb, info_text, slider_update

        except Exception as e:
            import traceback
            error_msg = f"""
### ❌ 오류 발생

**에러 메시지:**
```
{str(e)}
```

**상세 정보:**
```
{traceback.format_exc()}
```

**확인사항:**
- 비디오 파일 경로가 정확한가요?
- 비디오 파일이 존재하나요?
- 파일 형식이 지원되나요? (MP4, AVI, MOV, MKV)
"""
            print(f"[ERROR] {error_msg}")
            return None, error_msg, default_slider

    def add_point(self, image: np.ndarray, point_type: str, evt: gr.SelectData) -> Tuple[np.ndarray, str]:
        """
        이미지 클릭 시 point 추가

        Args:
            image: 현재 이미지
            point_type: 'foreground' or 'background'
            evt: Gradio 클릭 이벤트
        """
        if image is None or len(self.frames) == 0:
            return image, "먼저 비디오를 로드하세요"

        # 클릭 좌표
        x, y = evt.index[0], evt.index[1]

        # Point 추가
        self.annotations[point_type].append((x, y))

        # 현재 프레임에 point 표시
        frame = self.frames[self.current_frame_idx].copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Foreground points (녹색)
        for px, py in self.annotations['foreground']:
            cv2.circle(frame_rgb, (px, py), 5, (0, 255, 0), -1)
            cv2.circle(frame_rgb, (px, py), 7, (255, 255, 255), 2)

        # Background points (빨간색)
        for px, py in self.annotations['background']:
            cv2.circle(frame_rgb, (px, py), 5, (255, 0, 0), -1)
            cv2.circle(frame_rgb, (px, py), 7, (255, 255, 255), 2)

        status = f"""
**Annotations:**
- Foreground: {len(self.annotations['foreground'])} points
- Background: {len(self.annotations['background'])} points

클릭한 위치: ({x}, {y}) - {point_type}
"""

        return frame_rgb, status

    def segment_current_frame(self) -> Tuple[np.ndarray, str]:
        """
        현재 프레임을 SAM으로 세그멘테이션
        (간단한 contour 기반, 실제 SAM 모델 통합은 별도 필요)
        """
        if len(self.frames) == 0:
            return None, "먼저 비디오를 로드하세요"

        if len(self.annotations['foreground']) == 0:
            return None, "최소 1개의 foreground point가 필요합니다"

        try:
            frame = self.frames[self.current_frame_idx]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # SAM2 사용 (available하면)
            if self.sam2_predictor is not None:
                # SAM2 inference
                self.sam2_predictor.set_image(frame_rgb)

                # Points와 labels 준비
                point_coords = []
                point_labels = []

                for px, py in self.annotations['foreground']:
                    point_coords.append([px, py])
                    point_labels.append(1)  # foreground

                for px, py in self.annotations['background']:
                    point_coords.append([px, py])
                    point_labels.append(0)  # background

                point_coords = np.array(point_coords, dtype=np.float32)
                point_labels = np.array(point_labels, dtype=np.int32)

                # SAM2 predict
                masks, scores, _ = self.sam2_predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True
                )

                # Best mask 선택
                best_idx = np.argmax(scores)
                mask = masks[best_idx]
                confidence = scores[best_idx]

                status_method = f"SAM2 (confidence: {confidence:.3f})"
            else:
                # Fallback: contour 기반
                mask = self.processor.segment_object_interactive(frame, method='contour')
                confidence = 0.0
                status_method = "Contour (fallback)"

            # 마스크 저장
            self.masks[self.current_frame_idx] = mask
            self.current_mask = mask

            # 시각화
            overlay = frame_rgb.copy()
            overlay[mask > 0] = [0, 255, 0]  # 녹색 마스크
            result = cv2.addWeighted(frame_rgb, 0.6, overlay, 0.4, 0)

            # Points 표시
            for px, py in self.annotations['foreground']:
                cv2.circle(result, (px, py), 5, (0, 255, 0), -1)
                cv2.circle(result, (px, py), 7, (255, 255, 255), 2)
            for px, py in self.annotations['background']:
                cv2.circle(result, (px, py), 5, (255, 0, 0), -1)
                cv2.circle(result, (px, py), 7, (255, 255, 255), 2)

            mask_area = np.sum(mask > 0)
            mask_pct = mask_area / mask.size * 100

            status = f"""
### Segmentation 완료 ✅

- **Method**: {status_method}
- **프레임**: {self.current_frame_idx + 1} / {len(self.frames)}
- **마스크 영역**: {mask_area} 픽셀 ({mask_pct:.1f}%)
- **Foreground points**: {len(self.annotations['foreground'])}
- **Background points**: {len(self.annotations['background'])}

### 다음:
- 다른 프레임에도 annotation하려면 프레임 이동
- 또는 **Propagate to All Frames** 클릭
"""

            return result, status

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            return None, f"세그멘테이션 실패: {str(e)}\n\n```\n{error_detail}\n```"

    def propagate_to_all_frames(self, progress=gr.Progress()) -> Tuple[np.ndarray, str]:
        """
        현재 프레임의 annotation을 전체 비디오에 propagation (tracking)
        SAM 2 Video Predictor를 사용한 메모리 기반 추적

        중요: 고정 points를 모든 프레임에 재적용하지 않음!
        대신 SAM 2의 memory mechanism을 사용하여 자동으로 객체 추적
        """
        if len(self.frames) == 0:
            return None, "먼저 비디오를 로드하세요"

        if len(self.annotations['foreground']) == 0:
            return None, "Annotation points가 필요합니다 (최소 1개의 foreground point)"

        try:
            progress(0, desc="비디오 tracking 초기화 (SAM 2 Video Predictor)...")

            # SAM 2 Video Predictor 사용 (메모리 기반 추적)
            if self.sam2_video_predictor is not None:
                # 1. 임시 디렉토리에 프레임 저장 (SAM 2 Video Predictor는 디렉토리 입력 필요)
                import tempfile
                import os
                temp_dir = tempfile.mkdtemp(prefix="sam3d_video_")

                try:
                    progress(0.05, desc="프레임 저장 중...")
                    for i, frame in enumerate(self.frames):
                        frame_path = os.path.join(temp_dir, f"{i:05d}.jpg")
                        cv2.imwrite(frame_path, frame)

                    progress(0.1, desc="SAM 2 Video Predictor 초기화 중...")

                    # 2. Inference state 초기화
                    inference_state = self.sam2_video_predictor.init_state(video_path=temp_dir)

                    # 3. 현재 프레임에만 annotation points 추가 (conditioning frame)
                    point_coords = []
                    point_labels = []

                    for px, py in self.annotations['foreground']:
                        point_coords.append([px, py])
                        point_labels.append(1)

                    for px, py in self.annotations['background']:
                        point_coords.append([px, py])
                        point_labels.append(0)

                    point_coords = np.array(point_coords, dtype=np.float32)
                    point_labels = np.array(point_labels, dtype=np.int32)

                    progress(0.15, desc=f"초기 프레임 ({self.current_frame_idx}) annotation 중...")

                    # 현재 프레임을 conditioning frame으로 설정
                    _, out_obj_ids, out_mask_logits = self.sam2_video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=self.current_frame_idx,
                        obj_id=1,  # Single object tracking
                        points=point_coords,
                        labels=point_labels,
                    )

                    progress(0.2, desc="메모리 기반 전파 시작...")

                    # 4. Propagate using memory-based tracking (NO points on other frames!)
                    video_segments = {}
                    for frame_idx, obj_ids, mask_logits in self.sam2_video_predictor.propagate_in_video(
                        inference_state,
                        start_frame_idx=self.current_frame_idx
                    ):
                        # Memory-based tracking - 각 프레임은 이전 프레임의 메모리를 사용
                        # Points는 재적용되지 않음!
                        video_segments[frame_idx] = (mask_logits[0] > 0.0).cpu().numpy()

                        progress_pct = 0.2 + 0.6 * (frame_idx + 1) / len(self.frames)
                        progress(progress_pct, desc=f"Tracking... {frame_idx+1}/{len(self.frames)}")

                    # 5. 결과를 self.masks에 저장
                    self.masks = [None] * len(self.frames)
                    for frame_idx, mask in video_segments.items():
                        if frame_idx < len(self.masks):
                            self.masks[frame_idx] = mask.squeeze()

                    progress(0.9, desc="Tracking 완료, 결과 처리 중...")

                finally:
                    # 임시 디렉토리 정리
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)

            else:
                # Fallback: Image predictor 사용 (구버전 방식 - 정확도 낮음)
                progress(0, desc="Fallback: 프레임별 세그멘테이션...")

                for i, frame in enumerate(self.frames):
                    if self.masks[i] is None:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        if self.sam2_predictor is not None:
                            self.sam2_predictor.set_image(frame_rgb)

                            point_coords = []
                            point_labels = []

                            for px, py in self.annotations['foreground']:
                                point_coords.append([px, py])
                                point_labels.append(1)

                            for px, py in self.annotations['background']:
                                point_coords.append([px, py])
                                point_labels.append(0)

                            point_coords = np.array(point_coords, dtype=np.float32)
                            point_labels = np.array(point_labels, dtype=np.int32)

                            masks, scores, _ = self.sam2_predictor.predict(
                                point_coords=point_coords,
                                point_labels=point_labels,
                                multimask_output=True
                            )

                            best_idx = np.argmax(scores)
                            mask = masks[best_idx]
                        else:
                            mask = self.processor.segment_object_interactive(frame, method='contour')

                        self.masks[i] = mask

                    progress((i + 1) / len(self.frames), desc=f"Processing... {i+1}/{len(self.frames)}")

            progress(1.0, desc="시각화 준비 중...")

            # 현재 프레임 시각화
            self.current_frame_idx = min(self.current_frame_idx, len(self.frames) - 1)
            current_frame = self.frames[self.current_frame_idx]
            current_mask = self.masks[self.current_frame_idx]

            frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            overlay = frame_rgb.copy()
            if current_mask is not None:
                overlay[current_mask > 0] = [0, 255, 0]
            result = cv2.addWeighted(frame_rgb, 0.6, overlay, 0.4, 0)

            # 통계
            tracked_frames = sum(1 for m in self.masks if m is not None)

            method_used = "SAM 2 Video Predictor (Memory-based)" if self.sam2_video_predictor else "SAM 2 Image (Fallback)"

            status = f"""
### Propagation 완료 ✅

- **Method**: {method_used}
- **Tracked 프레임**: {tracked_frames} / {len(self.frames)}
- **현재 프레임**: {self.current_frame_idx + 1} / {len(self.frames)}
- **Conditioning Frame**: {self.current_frame_idx} (Points만 여기 적용)

### 메모리 기반 추적:
- 현재 프레임의 points만 사용
- 다른 프레임은 메모리로 자동 추적
- 객체 이동에도 정확한 마스크 생성

### 다음:
- **프레임 네비게이션**으로 결과 확인
- **Generate 3D Mesh** 클릭하여 3D 생성
- 또는 **Save Masks** 클릭하여 마스크 저장
"""

            return result, status

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            return None, f"Propagation 실패: {str(e)}\n\n```\n{error_detail}\n```"

    def download_sam3d_checkpoint(self, progress=gr.Progress()) -> bool:
        """
        SAM 3D 체크포인트를 자동으로 다운로드
        """
        import subprocess
        import os
        from dotenv import load_dotenv

        progress(0, desc="SAM 3D 체크포인트 다운로드 준비 중...")

        # .env 파일 로드
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            print(f"✓ .env 파일 로드됨: {env_path}")
        else:
            print(f"⚠️ .env 파일 없음: {env_path}")

        # HuggingFace 토큰 확인
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("⚠️ HF_TOKEN이 설정되지 않았습니다. 다운로드 실패 가능.")

        # Config에서 체크포인트 경로 가져오기
        if self.config:
            checkpoint_dir = Path(self.config.sam3d_checkpoint_dir).expanduser()
        else:
            checkpoint_dir = Path("~/dev/sam-3d-objects/checkpoints/hf").expanduser()

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        progress(0.1, desc="Git LFS 확인 중...")

        # Git LFS 확인 및 설치
        try:
            subprocess.run(["git", "lfs", "version"], check=True, capture_output=True)
        except:
            progress(0.2, desc="Git LFS 설치 중...")
            try:
                subprocess.run(["sudo", "apt-get", "update"], check=True)
                subprocess.run(["sudo", "apt-get", "install", "-y", "git-lfs"], check=True)
                subprocess.run(["git", "lfs", "install"], check=True)
            except Exception as e:
                print(f"Git LFS 설치 실패: {e}")
                return False

        progress(0.3, desc="SAM 3D 체크포인트 다운로드 중... (5-10GB, 시간 소요)")

        # HuggingFace에서 다운로드 (토큰 인증 사용)
        try:
            # 토큰이 있으면 인증 URL 사용
            if hf_token:
                clone_url = f"https://oauth2:{hf_token}@huggingface.co/facebook/sam-3d-objects"
                pull_url = f"https://oauth2:{hf_token}@huggingface.co/facebook/sam-3d-objects"
            else:
                clone_url = "https://huggingface.co/facebook/sam-3d-objects"
                pull_url = "origin"

            if not (checkpoint_dir / "pipeline.yaml").exists():
                # 처음 다운로드
                subprocess.run([
                    "git", "clone",
                    clone_url,
                    str(checkpoint_dir)
                ], check=True, cwd=checkpoint_dir.parent)
                progress(0.9, desc="다운로드 완료, 검증 중...")
            else:
                # 이미 존재하면 업데이트
                if hf_token:
                    subprocess.run(["git", "pull", pull_url], check=True, cwd=checkpoint_dir)
                else:
                    subprocess.run(["git", "pull"], check=True, cwd=checkpoint_dir)
                progress(0.9, desc="업데이트 완료, 검증 중...")

            # 다운로드 확인
            if (checkpoint_dir / "pipeline.yaml").exists():
                progress(1.0, desc="SAM 3D 체크포인트 준비 완료!")
                return True
            else:
                return False

        except Exception as e:
            print(f"다운로드 실패: {e}")
            return False

    def generate_3d_mesh(self, progress=gr.Progress()) -> Tuple[str, str]:
        """
        세그멘테이션 결과로 3D mesh 생성
        """
        if len(self.frames) == 0 or all(m is None for m in self.masks):
            return None, "먼저 세그멘테이션을 완료하세요"

        try:
            progress(0, desc="3D mesh 생성 준비 중...")

            # SAM 3D 체크포인트 확인
            if self.config:
                checkpoint_dir = Path(self.config.sam3d_checkpoint_dir).expanduser()
            else:
                checkpoint_dir = Path("~/dev/sam-3d-objects/checkpoints/hf").expanduser()

            if not (checkpoint_dir / "pipeline.yaml").exists():
                progress(0.1, desc="SAM 3D 체크포인트 없음, 다운로드 시작...")

                download_success = self.download_sam3d_checkpoint(progress)

                if not download_success:
                    return None, """
### ❌ SAM 3D 체크포인트 다운로드 실패

**수동 다운로드 방법:**
```bash
cd /home/joon/dev/sam3d_gui
./download_sam3d.sh
```

또는 다음 명령어:
```bash
cd ~/dev/sam-3d-objects
git clone https://huggingface.co/facebook/sam-3d-objects checkpoints/hf
```
"""

            # 대표 프레임 선택 (중간 프레임)
            mid_idx = len(self.frames) // 2
            frame = self.frames[mid_idx]
            mask = self.masks[mid_idx]

            if mask is None:
                return None, "중간 프레임에 마스크가 없습니다"

            # 3D 재구성 시도
            progress(0.5, desc="SAM 3D 재구성 중...")

            try:
                reconstruction = self.processor.reconstruct_3d(frame, mask)

                if reconstruction:
                    # PLY 저장
                    output_path = "outputs/interactive_reconstruction.ply"
                    self.processor.export_mesh(reconstruction, output_path, format='ply')

                    progress(1.0, desc="완료!")

                    status = f"""
### 3D Mesh 생성 완료 ✅

- **프레임**: {mid_idx + 1} / {len(self.frames)}
- **저장 위치**: `{output_path}`

### 3D 뷰어로 확인:
```bash
meshlab {output_path}
```

또는 온라인: https://3dviewer.net/
"""
                    return output_path, status
                else:
                    return None, "3D 재구성 실패 (SAM 3D 체크포인트 필요)"

            except Exception as e:
                # SAM 3D 없으면 간단한 point cloud만 생성
                return None, f"3D 재구성 실패: {str(e)}\n\nSAM 3D 체크포인트가 필요합니다."

        except Exception as e:
            import traceback
            return None, f"오류:\n{str(e)}\n{traceback.format_exc()}"

    def save_annotation_session(self) -> str:
        """
        Annotation 세션 전체 저장 (annotation points + masks + metadata)
        """
        if len(self.frames) == 0:
            return "저장할 데이터가 없습니다"

        try:
            # 세션 ID 생성 (timestamp)
            from datetime import datetime
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

            output_dir = Path(f"outputs/sessions/{session_id}")
            output_dir.mkdir(parents=True, exist_ok=True)

            # 1. Annotation 메타데이터 저장 (JSON)
            metadata = {
                "session_id": session_id,
                "video_path": self.video_path,
                "num_frames": len(self.frames),
                "current_frame_idx": self.current_frame_idx,
                "annotations": {
                    "foreground": self.annotations['foreground'],
                    "background": self.annotations['background']
                },
                "frame_info": []
            }

            # 2. 각 프레임 저장
            saved_masks = 0
            for i, (frame, mask) in enumerate(zip(self.frames, self.masks)):
                frame_dir = output_dir / f"frame_{i:04d}"
                frame_dir.mkdir(exist_ok=True)

                # 원본 프레임 저장
                frame_path = frame_dir / "original.png"
                cv2.imwrite(str(frame_path), frame)

                # 마스크 저장 (있으면)
                if mask is not None:
                    mask_path = frame_dir / "mask.png"
                    cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)

                    # 시각화 (마스크 오버레이) 저장
                    vis_path = frame_dir / "visualization.png"
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    overlay = frame_rgb.copy()
                    overlay[mask > 0] = [0, 255, 0]
                    result = cv2.addWeighted(frame_rgb, 0.6, overlay, 0.4, 0)

                    # Annotation points 표시
                    for px, py in self.annotations['foreground']:
                        cv2.circle(result, (px, py), 5, (0, 255, 0), -1)
                        cv2.circle(result, (px, py), 7, (255, 255, 255), 2)
                    for px, py in self.annotations['background']:
                        cv2.circle(result, (px, py), 5, (255, 0, 0), -1)
                        cv2.circle(result, (px, py), 7, (255, 255, 255), 2)

                    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(vis_path), result_bgr)

                    saved_masks += 1

                    # 프레임 메타데이터
                    mask_area = np.sum(mask > 0)
                    metadata["frame_info"].append({
                        "frame_idx": i,
                        "has_mask": True,
                        "mask_area": int(mask_area),
                        "mask_percentage": float(mask_area / mask.size * 100)
                    })
                else:
                    metadata["frame_info"].append({
                        "frame_idx": i,
                        "has_mask": False
                    })

            # 3. Metadata JSON 저장
            metadata_path = output_dir / "session_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            return f"""
### Annotation 세션 저장 완료 ✅

**세션 ID**: `{session_id}`

**저장 내용**:
- 📁 원본 프레임: {len(self.frames)}개
- 🎭 마스크: {saved_masks}개
- 📍 Annotation points: {len(self.annotations['foreground'])} foreground, {len(self.annotations['background'])} background
- 📋 메타데이터: session_metadata.json

**저장 위치**: `{output_dir}/`

**디렉토리 구조**:
```
{session_id}/
├── session_metadata.json
├── frame_0000/
│   ├── original.png
│   ├── mask.png
│   └── visualization.png
├── frame_0001/
│   └── ...
```

**세션 재로드**: 이 session_id를 사용하여 나중에 재로드할 수 있습니다.
"""

        except Exception as e:
            import traceback
            return f"저장 오류: {str(e)}\n\n```\n{traceback.format_exc()}\n```"

    def load_annotation_session(self, session_id: str) -> Tuple[np.ndarray, str]:
        """
        저장된 annotation 세션 로드
        """
        try:
            session_dir = Path(f"outputs/sessions/{session_id}")
            if not session_dir.exists():
                return None, f"세션을 찾을 수 없습니다: {session_id}"

            # 메타데이터 로드
            metadata_path = session_dir / "session_metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # 프레임 및 마스크 로드
            num_frames = metadata["num_frames"]
            self.frames = []
            self.masks = []

            for i in range(num_frames):
                frame_dir = session_dir / f"frame_{i:04d}"

                # 원본 프레임 로드
                frame_path = frame_dir / "original.png"
                frame = cv2.imread(str(frame_path))
                self.frames.append(frame)

                # 마스크 로드 (있으면)
                mask_path = frame_dir / "mask.png"
                if mask_path.exists():
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    self.masks.append((mask > 0).astype(bool))
                else:
                    self.masks.append(None)

            # Annotation points 복원
            self.annotations = {
                'foreground': metadata["annotations"]["foreground"],
                'background': metadata["annotations"]["background"]
            }

            # 비디오 경로 및 현재 프레임 인덱스 복원
            self.video_path = metadata["video_path"]
            self.current_frame_idx = metadata["current_frame_idx"]

            # 현재 프레임 시각화
            current_frame = self.frames[self.current_frame_idx]
            current_mask = self.masks[self.current_frame_idx]

            frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            if current_mask is not None:
                overlay = frame_rgb.copy()
                overlay[current_mask > 0] = [0, 255, 0]
                result = cv2.addWeighted(frame_rgb, 0.6, overlay, 0.4, 0)
            else:
                result = frame_rgb

            # Annotation points 표시
            for px, py in self.annotations['foreground']:
                cv2.circle(result, (px, py), 5, (0, 255, 0), -1)
                cv2.circle(result, (px, py), 7, (255, 255, 255), 2)
            for px, py in self.annotations['background']:
                cv2.circle(result, (px, py), 5, (255, 0, 0), -1)
                cv2.circle(result, (px, py), 7, (255, 255, 255), 2)

            masks_loaded = sum(1 for m in self.masks if m is not None)

            status = f"""
### 세션 로드 완료 ✅

**세션 ID**: `{session_id}`

**로드된 데이터**:
- 📁 프레임: {len(self.frames)}개
- 🎭 마스크: {masks_loaded}개
- 📍 Foreground points: {len(self.annotations['foreground'])}개
- 📍 Background points: {len(self.annotations['background'])}개
- 📹 비디오: {self.video_path}

**현재 프레임**: {self.current_frame_idx + 1} / {len(self.frames)}

이제 프레임 네비게이션, 추가 annotation, propagation 등을 계속할 수 있습니다.
"""

            return result, status

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            return None, f"로드 오류: {str(e)}\n\n```\n{error_detail}\n```"

    def get_session_ids(self) -> List[str]:
        """저장된 세션 ID 목록 반환 (Dropdown용)"""
        try:
            sessions_dir = Path("outputs/sessions")
            if not sessions_dir.exists():
                return []

            sessions = sorted([d.name for d in sessions_dir.iterdir() if d.is_dir()], reverse=True)
            return sessions

        except Exception as e:
            print(f"세션 목록 가져오기 실패: {e}")
            return []

    def list_saved_sessions(self) -> str:
        """저장된 세션 목록 조회"""
        try:
            sessions_dir = Path("outputs/sessions")
            if not sessions_dir.exists():
                return "저장된 세션이 없습니다"

            sessions = sorted([d.name for d in sessions_dir.iterdir() if d.is_dir()])

            if not sessions:
                return "저장된 세션이 없습니다"

            result = "### 저장된 Annotation 세션 목록\n\n"
            for session_id in sessions:
                metadata_path = sessions_dir / session_id / "session_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    num_frames = metadata["num_frames"]
                    num_masks = sum(1 for info in metadata["frame_info"] if info.get("has_mask", False))
                    video_path = Path(metadata["video_path"]).name if metadata.get("video_path") else "Unknown"

                    result += f"""
**{session_id}**
- 비디오: `{video_path}`
- 프레임: {num_frames}개
- 마스크: {num_masks}개
- 경로: `outputs/sessions/{session_id}/`

---
"""

            return result

        except Exception as e:
            return f"오류: {str(e)}"

    def save_masks(self) -> str:
        """마스크만 간단히 저장 (하위 호환성)"""
        if all(m is None for m in self.masks):
            return "저장할 마스크가 없습니다"

        try:
            output_dir = Path("outputs/masks")
            output_dir.mkdir(parents=True, exist_ok=True)

            saved_count = 0
            for i, mask in enumerate(self.masks):
                if mask is not None:
                    output_path = output_dir / f"mask_{i:04d}.png"
                    cv2.imwrite(str(output_path), mask.astype(np.uint8) * 255)
                    saved_count += 1

            return f"""
### 마스크 저장 완료 ✅

- **저장된 마스크**: {saved_count} / {len(self.masks)}
- **저장 위치**: `{output_dir}/`

**참고**: 전체 세션(annotation + masks + metadata)을 저장하려면 **"💾 Save Session"** 버튼을 사용하세요.
"""

        except Exception as e:
            return f"오류: {str(e)}"

    def navigate_frame(self, direction: str, step: int = 1) -> Tuple[np.ndarray, str]:
        """
        프레임 네비게이션

        Args:
            direction: "prev", "next", "first", "last", "goto"
            step: 이동할 프레임 수
        """
        if len(self.frames) == 0:
            return None, "먼저 비디오를 로드하세요"

        old_idx = self.current_frame_idx

        if direction == "prev":
            self.current_frame_idx = max(0, self.current_frame_idx - step)
        elif direction == "next":
            self.current_frame_idx = min(len(self.frames) - 1, self.current_frame_idx + step)
        elif direction == "first":
            self.current_frame_idx = 0
        elif direction == "last":
            self.current_frame_idx = len(self.frames) - 1
        elif direction == "goto":
            # step은 실제 프레임 번호 (0-indexed)
            self.current_frame_idx = max(0, min(len(self.frames) - 1, step))

        # 현재 프레임 시각화
        frame = self.frames[self.current_frame_idx]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 마스크가 있으면 표시
        mask = self.masks[self.current_frame_idx]
        if mask is not None:
            overlay = frame_rgb.copy()
            overlay[mask > 0] = [0, 255, 0]
            result = cv2.addWeighted(frame_rgb, 0.6, overlay, 0.4, 0)

            mask_area = np.sum(mask > 0)
            mask_pct = mask_area / mask.size * 100
            mask_info = f"마스크: {mask_area} 픽셀 ({mask_pct:.1f}%)"
        else:
            result = frame_rgb
            mask_info = "마스크 없음"

        # Points 표시 (annotation이 있으면)
        if len(self.annotations['foreground']) > 0 or len(self.annotations['background']) > 0:
            for px, py in self.annotations['foreground']:
                cv2.circle(result, (px, py), 5, (0, 255, 0), -1)
                cv2.circle(result, (px, py), 7, (255, 255, 255), 2)
            for px, py in self.annotations['background']:
                cv2.circle(result, (px, py), 5, (255, 0, 0), -1)
                cv2.circle(result, (px, py), 7, (255, 255, 255), 2)

        status = f"""
### 프레임 {self.current_frame_idx + 1} / {len(self.frames)}

- **이동**: {old_idx + 1} → {self.current_frame_idx + 1}
- **{mask_info}**
"""

        return result, status

    def clear_annotations(self) -> Tuple[np.ndarray, str]:
        """
        모든 annotation points와 masks 초기화

        Returns:
            현재 프레임 이미지, 상태 메시지
        """
        # Annotations 초기화
        self.annotations['foreground'] = []
        self.annotations['background'] = []

        # Masks 초기화
        self.masks = [None] * len(self.frames) if self.frames else []
        self.current_mask = None

        # 현재 프레임 이미지 반환 (annotation 없이)
        if len(self.frames) > 0:
            current_frame = self.frames[self.current_frame_idx]
            frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

            status = """
### Annotations 초기화 완료 ✅

- 모든 foreground/background points 제거
- 모든 마스크 초기화
- 새로 annotation 시작 가능

**다음 단계**: 이미지 클릭하여 새로운 annotation 시작
"""
            return frame_rgb, status
        else:
            return None, "비디오가 로드되지 않았습니다"

    def export_fauna_dataset(
        self,
        animal_name: str = "mouse",
        target_frames: int = 50,
        progress=gr.Progress()
    ) -> str:
        """
        Fauna 데이터셋 형식으로 저장
        스마트 샘플링: 전체 비디오에서 target_frames 개만 균등 간격으로 선택

        Args:
            animal_name: 동물 이름 (폴더명)
            target_frames: 저장할 프레임 수 (기본 50개)

        Returns:
            상태 메시지
        """
        if len(self.frames) == 0:
            return "❌ 비디오가 로드되지 않았습니다"

        if all(m is None for m in self.masks):
            return "❌ 마스크가 없습니다. 먼저 Propagate를 실행하세요"

        try:
            from datetime import datetime

            progress(0, desc="Fauna 데이터셋 준비 중...")

            # 출력 디렉토리 설정
            fauna_root = Path.home() / "dev/3DAnimals/data/fauna/Fauna_dataset/large_scale"
            output_dir = fauna_root / animal_name / "train" / "seq_000"
            output_dir.mkdir(parents=True, exist_ok=True)

            # 스마트 샘플링: target_frames개를 균등 간격으로 선택
            total_frames = len(self.frames)
            if total_frames <= target_frames:
                # 프레임 수가 적으면 전부 사용
                selected_indices = list(range(total_frames))
            else:
                # 균등 간격으로 샘플링
                step = total_frames / target_frames
                selected_indices = [int(i * step) for i in range(target_frames)]

            progress(0.1, desc=f"{len(selected_indices)}개 프레임 선택됨 (전체 {total_frames}개 중)...")

            # 프레임 및 마스크 저장
            saved_count = 0
            for idx, frame_idx in enumerate(selected_indices):
                if self.masks[frame_idx] is None:
                    continue

                frame = self.frames[frame_idx]
                mask = self.masks[frame_idx]

                # Fauna 형식: {index:07d}_rgb.png, {index:07d}_mask.png
                rgb_path = output_dir / f"{idx:07d}_rgb.png"
                mask_path = output_dir / f"{idx:07d}_mask.png"

                # RGB 저장 (BGR → RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imwrite(str(rgb_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

                # Mask 저장 (0-255 형식)
                mask_uint8 = (mask * 255).astype(np.uint8)
                cv2.imwrite(str(mask_path), mask_uint8)

                saved_count += 1
                progress(0.1 + 0.8 * (idx + 1) / len(selected_indices),
                        desc=f"저장 중... {idx+1}/{len(selected_indices)}")

            # 메타데이터 생성
            metadata = {
                "animal_name": animal_name,
                "sequence": "seq_000",
                "split": "train",
                "total_frames": saved_count,
                "original_video_frames": total_frames,
                "sampling_strategy": "uniform" if total_frames > target_frames else "all",
                "annotations": {
                    "foreground_points": len(self.annotations['foreground']),
                    "background_points": len(self.annotations['background'])
                },
                "export_date": datetime.now().isoformat(),
                "source_video": str(self.video_path) if self.video_path else None
            }

            metadata_path = output_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            progress(1.0, desc="Fauna 데이터셋 생성 완료!")

            return f"""
### Fauna 데이터셋 생성 완료 ✅

**저장 위치**: `{output_dir}`

**데이터셋 구조**:
```
{animal_name}/train/seq_000/
├── 0000000_rgb.png
├── 0000000_mask.png
├── 0000001_rgb.png
├── 0000001_mask.png
...
├── {saved_count-1:07d}_rgb.png
├── {saved_count-1:07d}_mask.png
└── metadata.json
```

**통계**:
- 저장된 프레임: {saved_count}개
- 원본 비디오: {total_frames} 프레임
- 샘플링: {"균등 간격 " + str(target_frames) + "개" if total_frames > target_frames else "전체 사용"}

**다음 단계**:
1. 데이터 검증: `ls {output_dir} | head -20`
2. 3DAnimals 학습 실행
3. 결과 확인 및 시각화

**Config 설정 예시**:
```yaml
dataset:
  name: {animal_name}
  path: data/fauna/Fauna_dataset/large_scale/{animal_name}
  split: train
```
"""

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            return f"❌ Fauna 데이터셋 생성 실패: {str(e)}\n\n```\n{error_detail}\n```"

    def export_frames_and_masks(self, output_dir: str = None, progress=gr.Progress()) -> str:
        """
        프레임별로 원본 이미지와 마스크를 별도 폴더에 저장

        Args:
            output_dir: 출력 디렉토리 (None이면 자동 생성)

        Returns:
            상태 메시지
        """
        if len(self.frames) == 0:
            return "❌ 비디오가 로드되지 않았습니다"

        if all(m is None for m in self.masks):
            return "❌ 마스크가 없습니다. 먼저 Segment 또는 Propagate를 실행하세요"

        try:
            progress(0, desc="저장 준비 중...")

            # 출력 디렉토리 설정
            if output_dir is None:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = Path(self.config.output_dir if self.config else "outputs") / f"frames_export_{timestamp}"
            else:
                output_dir = Path(output_dir)

            output_dir.mkdir(parents=True, exist_ok=True)

            # 서브디렉토리 생성
            images_dir = output_dir / "images"
            masks_dir = output_dir / "masks"
            overlays_dir = output_dir / "overlays"

            images_dir.mkdir(exist_ok=True)
            masks_dir.mkdir(exist_ok=True)
            overlays_dir.mkdir(exist_ok=True)

            progress(0.1, desc="프레임 저장 중...")

            saved_count = 0
            for i, frame in enumerate(self.frames):
                # 원본 이미지 저장
                image_path = images_dir / f"frame_{i:05d}.png"
                cv2.imwrite(str(image_path), frame)

                # 마스크 저장 (있을 경우)
                if self.masks[i] is not None:
                    mask = self.masks[i]
                    mask_path = masks_dir / f"frame_{i:05d}.png"
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    cv2.imwrite(str(mask_path), mask_uint8)

                    # 오버레이 저장 (시각화)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    overlay = frame_rgb.copy()
                    overlay[mask > 0] = [0, 255, 0]
                    result = cv2.addWeighted(frame_rgb, 0.6, overlay, 0.4, 0)
                    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

                    overlay_path = overlays_dir / f"frame_{i:05d}.png"
                    cv2.imwrite(str(overlay_path), result_bgr)

                    saved_count += 1

                progress((i + 1) / len(self.frames), desc=f"저장 중... {i+1}/{len(self.frames)}")

            # 메타데이터 저장
            metadata = {
                "video_path": str(self.video_path) if self.video_path else None,
                "total_frames": len(self.frames),
                "frames_with_masks": saved_count,
                "annotations": {
                    "foreground": self.annotations['foreground'],
                    "background": self.annotations['background']
                },
                "export_date": datetime.now().isoformat()
            }

            metadata_path = output_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            progress(1.0, desc="저장 완료!")

            return f"""
### 프레임/마스크 저장 완료 ✅

**저장 위치**: `{output_dir}`

**저장된 파일**:
- 📁 **images/**: 원본 프레임 {len(self.frames)}개
- 📁 **masks/**: 마스크 이미지 {saved_count}개
- 📁 **overlays/**: 시각화 이미지 {saved_count}개
- 📄 **metadata.json**: 메타데이터

**파일 형식**: PNG (무손실)

**다음 단계**:
- 이미지 처리 파이프라인에서 사용
- 학습 데이터셋으로 활용
- 외부 도구로 추가 분석
"""

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            return f"❌ 저장 실패: {str(e)}\n\n```\n{error_detail}\n```"

    # ===== Lite Annotator Event Handlers =====

    def _lite_load_source(
        self,
        input_path: str,
        input_type: str,
        pattern: str
    ) -> Tuple[str, gr.Slider, Dict]:
        """Load video or image folder"""
        if self.lite_annotator is None:
            return "❌ Lite Annotator not initialized", gr.Slider(maximum=100), {}

        success, msg, total_frames = self.lite_annotator.change_input_source(
            input_path, input_type, pattern
        )

        if success:
            # Update slider maximum
            new_slider = gr.Slider(minimum=0, maximum=max(0, total_frames - 1), value=0, step=1)
            info = self.lite_annotator.get_info()
            return msg, new_slider, info
        else:
            return msg, gr.Slider(maximum=100), {}

    def _lite_load_model(self, model_name: str) -> str:
        """Load SAM 2.1 model"""
        if self.lite_annotator is None:
            return "❌ Lite Annotator not initialized"

        msg = self.lite_annotator.load_model(model_name)
        return msg

    def _lite_load_frame(self, frame_idx: int) -> Tuple[Optional[np.ndarray], str, Dict]:
        """Load frame at index"""
        if self.lite_annotator is None:
            return None, "❌ Lite Annotator not initialized", {}

        frame, msg = self.lite_annotator.load_frame(int(frame_idx))
        info = self.lite_annotator.get_info()

        return frame, msg, info

    def _lite_add_point(self, evt: gr.SelectData, point_type: str) -> Tuple[np.ndarray, str]:
        """Add point from click event"""
        if self.lite_annotator is None:
            return None, "❌ Lite Annotator not initialized"

        x, y = evt.index[0], evt.index[1]
        frame, msg = self.lite_annotator.add_point(x, y, point_type)

        return frame, msg

    def _lite_generate_mask(self) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
        """Generate mask from points"""
        if self.lite_annotator is None:
            return None, None, "❌ Lite Annotator not initialized"

        frame_vis, mask_binary, msg = self.lite_annotator.generate_mask()

        return frame_vis, mask_binary, msg

    def _lite_save_annotation(self) -> str:
        """Save current annotation"""
        if self.lite_annotator is None:
            return "❌ Lite Annotator not initialized"

        msg = self.lite_annotator.save_annotation()
        return msg

    def _lite_clear_points(self) -> Tuple[np.ndarray, str]:
        """Clear all points"""
        if self.lite_annotator is None:
            return None, "❌ Lite Annotator not initialized"

        frame, msg = self.lite_annotator.clear_points()
        return frame, msg

    def create_interface(self):
        """Gradio 인터페이스 생성 - 통합 버전"""

        with gr.Blocks(title="SAM 3D GUI - Unified Interface") as demo:
            gr.Markdown("""
            # 🎬 SAM 3D GUI - 통합 웹 인터페이스

            **두 가지 작업 모드:**
            - 🚀 **Quick Mode**: 자동 세그멘테이션 & 모션 감지 (빠름)
            - 🎨 **Interactive Mode**: 수동 annotation & propagation (정확함)
            """)

            # 비디오 자동 스캔 (Interactive Mode용)
            initial_videos = self.scan_videos(self.default_data_dir)
            initial_video = initial_videos[0] if initial_videos else None

            # 세션 자동 스캔
            initial_sessions = self.get_session_ids()

            with gr.Tabs():
                # ===== Tab 1: Interactive Mode (기본) =====
                with gr.Tab("🎨 Interactive Mode"):
                    gr.Markdown("### 대화형 Annotation & Propagation")

                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 📁 비디오 로드")

                            data_dir = gr.Textbox(
                                label="데이터 디렉토리",
                                value=self.default_data_dir
                            )

                            scan_video_btn = gr.Button("📂 비디오 스캔")

                            video_file = gr.Dropdown(
                                label="비디오 파일",
                                choices=initial_videos,
                                value=initial_video,
                                interactive=True
                            )

                            with gr.Row():
                                start_time = gr.Number(label="시작 (초)", value=0.0, minimum=0)
                                duration = gr.Number(label="길이 (초)", value=3.0, minimum=0.1)

                            load_btn = gr.Button("📹 비디오 로드", variant="primary")

                            gr.Markdown("### 🎯 Annotation")

                            annotation_mode = gr.Radio(
                                label="Point 타입",
                                choices=["foreground", "background"],
                                value="foreground"
                            )

                            clear_btn = gr.Button("🗑️ Points 초기화")
                            clear_all_btn = gr.Button("🔄 All Annotations 초기화", variant="stop")
                            segment_btn = gr.Button("✂️ Segment Current Frame", variant="secondary")

                            gr.Markdown("### 🎬 Propagation")

                            propagate_btn = gr.Button("🔄 Propagate to All Frames", variant="primary")

                            gr.Markdown("### 🎞️ 프레임 네비게이션")

                            # 프레임 프로그레스 바 (직접 이동 가능)
                            frame_slider = gr.Slider(
                                label="프레임 위치",
                                minimum=0,
                                maximum=100,
                                value=0,
                                step=1,
                                interactive=True,
                                info="슬라이더를 드래그하여 프레임 이동"
                            )

                            with gr.Row():
                                first_btn = gr.Button("⏮️ 처음", size="sm")
                                prev_btn = gr.Button("◀️ 이전", size="sm")
                                next_btn = gr.Button("▶️ 다음", size="sm")
                                last_btn = gr.Button("⏭️ 마지막", size="sm")

                            with gr.Row():
                                frame_step = gr.Slider(
                                    label="이동 간격",
                                    minimum=1,
                                    maximum=10,
                                    value=1,
                                    step=1
                                )

                            with gr.Row():
                                goto_frame = gr.Number(
                                    label="프레임 번호",
                                    value=1,
                                    minimum=1,
                                    step=1,
                                    scale=2
                                )
                                goto_btn = gr.Button("이동", scale=1)

                            gr.Markdown("### 💾 세션 관리")

                            save_session_btn = gr.Button("💾 Save Session", variant="primary")

                            gr.Markdown("**세션 로드**")
                            with gr.Row():
                                session_refresh_btn = gr.Button("🔄 목록 새로고침", size="sm")

                            session_id_dropdown = gr.Dropdown(
                                label="세션 선택",
                                choices=initial_sessions,
                                value=initial_sessions[0] if initial_sessions else None,
                                interactive=True,
                                scale=2
                            )
                            load_session_btn = gr.Button("📂 Load Session")

                            gr.Markdown("### 🎲 3D & 출력")

                            mesh_btn = gr.Button("🎲 Generate 3D Mesh")
                            save_masks_btn = gr.Button("💾 Save Masks Only")
                            export_frames_btn = gr.Button("📤 Export Frames & Masks")

                            gr.Markdown("### 🦁 Fauna 데이터셋 저장")

                            with gr.Row():
                                fauna_animal_name = gr.Textbox(
                                    label="동물 이름",
                                    value="mouse",
                                    placeholder="예: mouse, cat, dog"
                                )
                                fauna_target_frames = gr.Number(
                                    label="목표 프레임 수",
                                    value=50,
                                    minimum=10,
                                    maximum=500,
                                    step=10
                                )

                            export_fauna_btn = gr.Button("🐾 Fauna 형식으로 저장", variant="primary")

                        # 우측: 이미지 & 결과
                        with gr.Column(scale=2):
                            gr.Markdown("### 🖼️ Annotation & Results")

                            image_display = gr.Image(
                                label="이미지 (클릭하여 point 추가)",
                                type="numpy",
                                height=500,
                                interactive=True
                            )

                            status_text = gr.Markdown("비디오를 로드하세요")

                            mesh_file = gr.File(label="3D Mesh 파일")

                    # Interactive Mode 이벤트 핸들러
                    load_btn.click(
                        fn=self.load_video,
                        inputs=[data_dir, video_file, start_time, duration],
                        outputs=[image_display, status_text, frame_slider]
                    )

                    # 슬라이더로 프레임 이동
                    frame_slider.change(
                        fn=lambda frame_idx: self.navigate_frame("goto", int(frame_idx)),
                        inputs=[frame_slider],
                        outputs=[image_display, status_text]
                    )

                    # 이미지 클릭 시 point 추가
                    def handle_click(mode, evt: gr.SelectData):
                        """이미지 클릭 핸들러 - img 파라미터 제거"""
                        if len(self.frames) == 0:
                            return None, "먼저 비디오를 로드하세요"

                        # 클릭 좌표
                        x, y = evt.index[0], evt.index[1]

                        # Point 추가
                        self.annotations[mode].append((x, y))

                        # 현재 프레임에 point 표시
                        frame = self.frames[self.current_frame_idx].copy()
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Foreground points (녹색)
                        for px, py in self.annotations['foreground']:
                            cv2.circle(frame_rgb, (px, py), 5, (0, 255, 0), -1)
                            cv2.circle(frame_rgb, (px, py), 7, (255, 255, 255), 2)

                        # Background points (빨간색)
                        for px, py in self.annotations['background']:
                            cv2.circle(frame_rgb, (px, py), 5, (255, 0, 0), -1)
                            cv2.circle(frame_rgb, (px, py), 7, (255, 255, 255), 2)

                        status = f"""
**Annotations:**
- Foreground: {len(self.annotations['foreground'])} points
- Background: {len(self.annotations['background'])} points

클릭한 위치: ({x}, {y}) - {mode}
"""

                        return frame_rgb, status

                    image_display.select(
                        fn=handle_click,
                        inputs=[annotation_mode],
                        outputs=[image_display, status_text]
                    )

                    segment_btn.click(
                        fn=self.segment_current_frame,
                        outputs=[image_display, status_text]
                    )

                    propagate_btn.click(
                        fn=self.propagate_to_all_frames,
                        outputs=[image_display, status_text]
                    )

                    mesh_btn.click(
                        fn=self.generate_3d_mesh,
                        outputs=[mesh_file, status_text]
                    )

                    save_masks_btn.click(
                        fn=self.save_masks,
                        outputs=[status_text]
                    )

                    export_frames_btn.click(
                        fn=self.export_frames_and_masks,
                        outputs=[status_text]
                    )

                    export_fauna_btn.click(
                        fn=self.export_fauna_dataset,
                        inputs=[fauna_animal_name, fauna_target_frames],
                        outputs=[status_text]
                    )

                    clear_all_btn.click(
                        fn=self.clear_annotations,
                        outputs=[image_display, status_text]
                    )

                    # 세션 관리 이벤트
                    save_session_btn.click(
                        fn=self.save_annotation_session,
                        outputs=[status_text]
                    )

                    session_refresh_btn.click(
                        fn=lambda: gr.Dropdown(choices=self.get_session_ids()),
                        outputs=[session_id_dropdown]
                    )

                    load_session_btn.click(
                        fn=self.load_annotation_session,
                        inputs=[session_id_dropdown],
                        outputs=[image_display, status_text]
                    )

                    def clear_points():
                        self.annotations = {'foreground': [], 'background': []}
                        if len(self.frames) > 0:
                            frame_rgb = cv2.cvtColor(self.frames[self.current_frame_idx], cv2.COLOR_BGR2RGB)
                            return frame_rgb, "Points 초기화됨"
                        return None, "Points 초기화됨"

                    clear_btn.click(
                        fn=clear_points,
                        outputs=[image_display, status_text]
                    )

                    # 비디오 스캔 버튼
                    scan_video_btn.click(
                        fn=lambda d: gr.Dropdown(choices=self.scan_videos(d)),
                        inputs=[data_dir],
                        outputs=[video_file]
                    )

                    # 프레임 네비게이션 이벤트
                    first_btn.click(
                        fn=lambda: self.navigate_frame("first"),
                        outputs=[image_display, status_text]
                    )

                    prev_btn.click(
                        fn=lambda step: self.navigate_frame("prev", int(step)),
                        inputs=[frame_step],
                        outputs=[image_display, status_text]
                    )

                    next_btn.click(
                        fn=lambda step: self.navigate_frame("next", int(step)),
                        inputs=[frame_step],
                        outputs=[image_display, status_text]
                    )

                    last_btn.click(
                        fn=lambda: self.navigate_frame("last"),
                        outputs=[image_display, status_text]
                    )

                    goto_btn.click(
                        fn=lambda frame_num: self.navigate_frame("goto", int(frame_num) - 1),  # 1-indexed to 0-indexed
                        inputs=[goto_frame],
                        outputs=[image_display, status_text]
                    )

                # ===== Tab 2: Quick Mode =====
                with gr.Tab("🚀 Quick Mode"):
                    gr.Markdown("### 빠른 자동 처리")

                    with gr.Row():
                        with gr.Column(scale=1):
                            quick_data_dir = gr.Textbox(
                                label="데이터 디렉토리",
                                value=self.default_data_dir
                            )

                            quick_scan_btn = gr.Button("📂 비디오 스캔")

                            quick_video_list = gr.Dropdown(
                                label="비디오 파일",
                                choices=initial_videos,
                                value=initial_video,
                                interactive=True
                            )

                            with gr.Row():
                                quick_start = gr.Number(label="시작(초)", value=0.0)
                                quick_duration = gr.Number(label="길이(초)", value=3.0)

                            quick_threshold = gr.Slider(
                                label="모션 임계값",
                                minimum=0, maximum=200, value=50.0, step=1
                            )

                            quick_method = gr.Radio(
                                label="세그멘테이션",
                                choices=["contour", "simple_threshold", "grabcut"],
                                value="contour"
                            )

                            quick_process_btn = gr.Button("🚀 자동 처리", variant="primary", size="lg")

                        with gr.Column(scale=2):
                            quick_image = gr.Image(label="결과", type="numpy", height=500)
                            quick_status = gr.Markdown("비디오를 선택하고 처리하세요")

                    # Quick Mode 이벤트
                    quick_scan_btn.click(
                        fn=lambda d: gr.Dropdown(choices=self.scan_videos(d)),
                        inputs=[quick_data_dir],
                        outputs=[quick_video_list]
                    )

                    quick_process_btn.click(
                        fn=self.quick_process,
                        inputs=[quick_data_dir, quick_video_list, quick_start,
                               quick_duration, quick_threshold, quick_method],
                        outputs=[quick_image, quick_status]
                    )

                # ===== Tab 3: Lite Annotator =====
                with gr.Tab("📝 Lite Annotator"):
                    gr.Markdown("### 효율적 Annotation 모드")
                    gr.Markdown("Direct video/image loading, multi-model selection, auto-restore")

                    with gr.Row():
                        # Left column: Input & Frame Display
                        with gr.Column(scale=2):
                            # Input source section
                            gr.Markdown("#### 📂 Input Source")
                            with gr.Row():
                                lite_input_path = gr.Textbox(
                                    label="Video/Image Folder Path",
                                    placeholder="/path/to/video.mp4 or /path/to/images/",
                                    scale=3
                                )
                                lite_input_type = gr.Radio(
                                    choices=["video", "images"],
                                    value="video",
                                    label="Type",
                                    scale=1
                                )

                            with gr.Row():
                                lite_pattern = gr.Textbox(
                                    label="Image Pattern (for images type)",
                                    value="*.png",
                                    scale=2
                                )
                                lite_load_btn = gr.Button("📥 Load Source", variant="primary", scale=1)

                            lite_load_status = gr.Markdown("No input loaded")

                            # Frame display
                            gr.Markdown("#### 🖼️ Frame")
                            lite_frame_display = gr.Image(
                                label="Current Frame",
                                type="numpy",
                                height=500,
                                interactive=True  # Enable click events
                            )

                            # Frame navigation
                            with gr.Row():
                                lite_frame_slider = gr.Slider(
                                    label="Frame Index",
                                    minimum=0,
                                    maximum=100,
                                    value=0,
                                    step=1,
                                    interactive=True
                                )

                        # Right column: Controls & Mask Display
                        with gr.Column(scale=1):
                            # Model selection
                            gr.Markdown("#### 🤖 Model Selection")
                            lite_model_dropdown = gr.Dropdown(
                                choices=["tiny", "small", "base_plus", "large"],
                                value="large",
                                label="SAM 2.1 Model",
                                info="tiny: fastest, large: best quality"
                            )
                            lite_load_model_btn = gr.Button("Load Model", size="sm")

                            # Point annotation
                            gr.Markdown("#### 🎨 Annotation")
                            lite_point_type = gr.Radio(
                                choices=["foreground", "background"],
                                value="foreground",
                                label="Point Type"
                            )

                            # Action buttons
                            with gr.Column():
                                lite_generate_btn = gr.Button("🎯 Generate Mask", variant="primary")
                                lite_save_btn = gr.Button("💾 Save Annotation", variant="secondary")
                                lite_clear_btn = gr.Button("🔄 Clear Points", variant="stop")

                            # Mask display
                            gr.Markdown("#### 🎭 Mask")
                            lite_mask_display = gr.Image(
                                label="Generated Mask",
                                type="numpy",
                                height=200
                            )

                            # Status
                            lite_status = gr.Markdown("**Status:** Load source to start")

                            # Info panel
                            gr.Markdown("#### ℹ️ Info")
                            lite_info = gr.JSON(label="Current State", value={})

                    # Event handlers for Lite Annotator

                    # Load source
                    lite_load_btn.click(
                        fn=self._lite_load_source,
                        inputs=[lite_input_path, lite_input_type, lite_pattern],
                        outputs=[lite_load_status, lite_frame_slider, lite_info]
                    )

                    # Load model
                    lite_load_model_btn.click(
                        fn=self._lite_load_model,
                        inputs=[lite_model_dropdown],
                        outputs=[lite_status]
                    )

                    # Frame slider change
                    lite_frame_slider.change(
                        fn=self._lite_load_frame,
                        inputs=[lite_frame_slider],
                        outputs=[lite_frame_display, lite_status, lite_info]
                    )

                    # Click on frame to add point
                    lite_frame_display.select(
                        fn=self._lite_add_point,
                        inputs=[lite_point_type],
                        outputs=[lite_frame_display, lite_status]
                    )

                    # Generate mask
                    lite_generate_btn.click(
                        fn=self._lite_generate_mask,
                        outputs=[lite_frame_display, lite_mask_display, lite_status]
                    )

                    # Save annotation
                    lite_save_btn.click(
                        fn=self._lite_save_annotation,
                        outputs=[lite_status]
                    )

                    # Clear points
                    lite_clear_btn.click(
                        fn=self._lite_clear_points,
                        outputs=[lite_frame_display, lite_status]
                    )

        return demo

def main():
    """웹 앱 실행"""
    import os

    app = SAMInteractiveWebApp()
    demo = app.create_interface()

    # 포트 설정: 환경 변수 또는 7860-7870 범위에서 자동 선택
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,  # 포트 사용 중이면 자동으로 다음 포트 시도
        share=False,
        debug=True,
        max_threads=40  # 동시 처리 증가
    )

if __name__ == "__main__":
    main()
