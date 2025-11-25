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
import os

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Set environment variable to skip SAM3D init (which requires missing module)
os.environ['LIDRA_SKIP_INIT'] = '1'

from sam3d_processor import SAM3DProcessor
from config_loader import ModelConfig
from lite_annotator import LiteAnnotator
from augmentation import DataAugmentor, generate_augmentation_configs

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
            self.default_output_dir = config.output_dir
        else:
            self.default_data_dir = "/home/joon/dev/data/markerless_mouse/"
            self.default_output_dir = "/home/joon/dev/sam3d_gui/outputs/"

        # Data Augmentor 초기화
        self.augmentor = DataAugmentor()
        self.augmentation_preview = None

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

    def unload_sam2_models(self):
        """
        Unload SAM2 models to free GPU memory before SAM 3D inference

        This is critical for RTX 3060 12GB where:
        - SAM2 Large uses ~2-3GB
        - SAM3D uses ~8-10GB
        - Total 11-13GB > 12GB available

        By unloading SAM2 before SAM3D, we free ~3GB for SAM3D inference.
        """
        import gc

        if self.sam2_predictor is not None or self.sam2_video_predictor is not None:
            print("\n🧹 SAM2 모델 언로드 시작 (메모리 확보)...")

            # Print memory before cleanup
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                print(f"   현재 GPU 메모리: {allocated:.2f} GB")

            # Delete SAM2 image predictor
            if self.sam2_predictor is not None:
                del self.sam2_predictor
                self.sam2_predictor = None
                print("   ✓ SAM2 Image Predictor 해제")

            # Delete SAM2 video predictor
            if self.sam2_video_predictor is not None:
                del self.sam2_video_predictor
                self.sam2_video_predictor = None
                print("   ✓ SAM2 Video Predictor 해제")

            # Force garbage collection
            gc.collect()

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("   ✓ CUDA 캐시 정리 완료")

                # Print memory after cleanup
                allocated_after = torch.cuda.memory_allocated(0) / 1024**3
                freed = allocated - allocated_after
                print(f"   ✓ GPU 메모리 해제: {freed:.2f} GB")
                print(f"   현재 GPU 메모리: {allocated_after:.2f} GB")

            print("✅ SAM2 모델 언로드 완료\n")
        else:
            print("ℹ️  SAM2 모델이 이미 언로드되어 있습니다.")

    def reload_sam2_models(self):
        """
        Reload SAM2 models after SAM 3D inference completes

        This allows users to continue using interactive segmentation after 3D reconstruction.
        """
        if not SAM2_AVAILABLE or not self.config:
            print("⚠️  SAM2를 다시 로드할 수 없습니다 (SAM2 unavailable or no config)")
            return

        if self.sam2_predictor is not None and self.sam2_video_predictor is not None:
            print("ℹ️  SAM2 모델이 이미 로드되어 있습니다.")
            return

        print("\n🔄 SAM2 모델 재로드 중...")

        try:
            checkpoint = Path(self.config.sam2_checkpoint)
            model_cfg = self.config.sam2_config
            device = self.sam2_device

            if checkpoint.exists():
                from sam2.build_sam import build_sam2, build_sam2_video_predictor

                # Rebuild models
                sam2_model = build_sam2(model_cfg, str(checkpoint), device=device)
                self.sam2_predictor = SAM2ImagePredictor(sam2_model)
                self.sam2_video_predictor = build_sam2_video_predictor(model_cfg, str(checkpoint), device=device)

                print(f"✅ SAM2 모델 재로드 완료 (device: {device})\n")
            else:
                print(f"❌ SAM2 checkpoint를 찾을 수 없습니다: {checkpoint}")
        except Exception as e:
            print(f"❌ SAM2 재로드 실패: {e}")
            import traceback
            traceback.print_exc()

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

    def calculate_stride_from_target(self, target_frames: int) -> int:
        """
        목표 프레임 수를 기반으로 stride 자동 계산

        Args:
            target_frames: 처리할 목표 프레임 수

        Returns:
            계산된 stride 값
        """
        if len(self.frames) == 0:
            return 1

        stride = max(1, len(self.frames) // target_frames)
        return stride

    def scan_batch_videos(self, data_dir: str, pattern: str = "*.mp4") -> Tuple[List[str], str, gr.CheckboxGroup]:
        """
        폴더 내 모든 비디오 스캔 및 메타데이터 수집 (recursive)
        폴더별로 그룹화하여 표시

        Args:
            data_dir: 비디오가 있는 디렉토리
            pattern: 비디오 파일 패턴 (예: *.mp4, *.avi)

        Returns:
            (비디오 경로 리스트, 상태 메시지, CheckboxGroup 업데이트)
        """
        try:
            data_path = Path(data_dir)
            if not data_path.exists():
                empty_checkbox = gr.CheckboxGroup(choices=[], value=[])
                return [], f"❌ 디렉토리를 찾을 수 없습니다: {data_dir}", empty_checkbox

            # 비디오 파일 찾기 (recursive)
            video_files = sorted(data_path.rglob(pattern))

            if not video_files:
                empty_checkbox = gr.CheckboxGroup(choices=[], value=[])
                return [], f"❌ 비디오 파일을 찾을 수 없습니다: {pattern} (recursive 탐색)", empty_checkbox

            # 메타데이터 수집
            total_frames = 0
            total_duration = 0
            video_info = []

            for video_path in video_files:
                try:
                    info = self.processor.get_video_info(str(video_path))
                    total_frames += info['frame_count']
                    total_duration += info['duration']
                    video_info.append({
                        'path': str(video_path),
                        'name': video_path.name,
                        'frames': info['frame_count'],
                        'duration': info['duration'],
                        'fps': info['fps'],
                        'resolution': f"{info['width']}x{info['height']}"
                    })
                except Exception as e:
                    print(f"⚠️ {video_path.name} 정보 읽기 실패: {e}")
                    continue

            # 평균 FPS 계산
            avg_fps = total_frames / total_duration if total_duration > 0 else 0

            # 상태 메시지 생성
            status = f"""
### 📂 Batch 비디오 스캔 완료 ✅

- **비디오 파일 수**: {len(video_info)}
- **총 프레임 수**: {total_frames:,}
- **총 길이**: {total_duration:.1f}초 ({total_duration/60:.1f}분)
- **평균 FPS**: {avg_fps:.1f}

<details>
<summary><b>📋 비디오 목록 ({len(video_info)}개) - 클릭하여 펼치기/접기</b></summary>

"""
            for idx, info in enumerate(video_info, 1):
                # 전체 상대 경로 표시
                rel_path = str(Path(info['path']).relative_to(data_path))
                status += f"\n{idx}. **{rel_path}**"
                status += f"\n   - 프레임: {info['frames']}, 길이: {info['duration']:.1f}초, FPS: {info['fps']:.1f}, 해상도: {info['resolution']}\n"

            status += "\n</details>"

            # 비디오 경로 리스트 반환
            video_paths = [str(v) for v in video_files]
            self.batch_videos = video_paths
            self.batch_video_info = video_info

            # CheckboxGroup 업데이트 - 전체 상대 경로로 표시 (고유하게)
            video_relative_paths = [str(Path(info['path']).relative_to(data_path)) for info in video_info]

            # 전체 계층적 경로를 레이블로 사용
            video_labels = video_relative_paths  # 전체 경로 사용

            updated_checkbox = gr.CheckboxGroup(
                choices=video_labels,
                value=video_labels,  # 기본적으로 모두 선택
                label="🎬 처리할 비디오 선택 (계층적 경로)",
                info="선택된 비디오만 처리됩니다"
            )

            # 레이블과 실제 경로 매핑 저장
            self.batch_video_label_map = dict(zip(video_labels, video_paths))

            return video_paths, status, updated_checkbox

        except Exception as e:
            import traceback
            empty_checkbox = gr.CheckboxGroup(choices=[], value=[])
            return [], f"❌ 스캔 실패:\n{str(e)}\n{traceback.format_exc()}", empty_checkbox

    def batch_load_reference_frame(self, selected_videos: List[str]) -> Tuple[np.ndarray, str]:
        """
        Batch 모드에서 선택된 비디오 중 첫 번째 비디오의 첫 프레임을 reference로 로드

        Args:
            selected_videos: 선택된 비디오 레이블 리스트

        Returns:
            (reference_frame, status_message)
        """
        if not hasattr(self, 'batch_videos') or not self.batch_videos:
            return None, "❌ 먼저 비디오를 스캔하세요"

        if not selected_videos or len(selected_videos) == 0:
            return None, "❌ 최소 1개의 비디오를 선택하세요"

        try:
            # 선택된 첫 번째 비디오의 실제 경로 찾기
            first_selected_label = selected_videos[0]

            if hasattr(self, 'batch_video_label_map') and first_selected_label in self.batch_video_label_map:
                first_video_path = self.batch_video_label_map[first_selected_label]
            else:
                # Fallback: 전체 리스트의 첫 번째
                first_video_path = self.batch_videos[0]

            # 첫 프레임 추출
            frames = self.processor.extract_frames(
                first_video_path,
                start_frame=0,
                num_frames=1,
                stride=1
            )

            if not frames:
                return None, "❌ Reference 프레임 추출 실패"

            # 프레임 저장 (annotation용)
            self.frames = frames
            self.current_frame_idx = 0
            self.annotations = {'foreground': [], 'background': []}
            self.masks = [None] * len(frames)

            # RGB 변환 (Gradio는 RGB 사용)
            frame_rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)

            status = f"""
### ✅ Reference 프레임 로드 완료

- **선택된 비디오**: {len(selected_videos)}개 중 첫 번째
- **Reference 비디오**: {first_selected_label}
- **파일명**: {Path(first_video_path).name}
- **해상도**: {frame_rgb.shape[1]} x {frame_rgb.shape[0]}

이제 이미지를 클릭하여 annotation을 추가하세요.
"""

            return frame_rgb, status

        except Exception as e:
            import traceback
            return None, f"❌ Reference 프레임 로드 실패:\n{str(e)}\n{traceback.format_exc()}"

    def batch_propagate_videos(
        self,
        target_frames: int = 100,
        selected_videos: List[str] = None,
        progress=gr.Progress()
    ) -> Tuple[str, str]:
        """
        여러 비디오를 순차적으로 처리 (Batch Processing)

        각 비디오마다:
        1. 로드 (목표 프레임 수에 맞춰 stride 자동 계산)
        2. 현재 annotation으로 propagate
        3. 결과 임시 저장
        4. 메모리 해제

        Args:
            target_frames: 각 비디오에서 추출할 목표 프레임 수
            selected_videos: 처리할 비디오 이름 리스트 (None이면 전체)
            progress: Gradio progress bar

        Returns:
            (상태 메시지, 완료 메시지)
        """
        if not hasattr(self, 'batch_videos') or not self.batch_videos:
            return "먼저 비디오를 스캔하세요", "❌ 비디오 없음"

        if len(self.annotations['foreground']) == 0:
            return "Annotation이 필요합니다 (최소 1개의 foreground point)", "❌ Annotation 없음"

        try:
            import tempfile
            import shutil
            import torch

            # 임시 결과 저장 디렉토리
            batch_temp_dir = Path(tempfile.mkdtemp(prefix="sam3d_batch_"))

            # Reference annotation 저장
            reference_annotations = {
                'foreground': self.annotations['foreground'].copy(),
                'background': self.annotations['background'].copy()
            }

            # 선택된 비디오 필터링 (레이블 → 경로 매핑 사용)
            if selected_videos and len(selected_videos) > 0:
                # 선택된 레이블을 실제 경로로 변환
                videos_to_process = []
                if hasattr(self, 'batch_video_label_map'):
                    for label in selected_videos:
                        if label in self.batch_video_label_map:
                            videos_to_process.append(self.batch_video_label_map[label])
                else:
                    # 레이블 맵이 없으면 이름으로 매칭 (하위 호환성)
                    for video_path in self.batch_videos:
                        video_name = Path(video_path).name
                        if video_name in selected_videos:
                            videos_to_process.append(video_path)
            else:
                # 선택이 없으면 모든 비디오 처리
                videos_to_process = self.batch_videos

            if not videos_to_process:
                return "처리할 비디오를 선택하세요", "❌ 선택된 비디오 없음"

            total_videos = len(videos_to_process)
            total_processed_frames = 0
            video_results = []

            progress(0, desc=f"Batch 처리 시작: {total_videos}개 비디오...")

            for video_idx, video_path in enumerate(videos_to_process):
                video_name = Path(video_path).name
                progress(video_idx / total_videos, desc=f"처리 중: {video_name} ({video_idx+1}/{total_videos})")

                print(f"\n{'='*80}")
                print(f"📹 비디오 {video_idx+1}/{total_videos}: {video_name}")
                print(f"{'='*80}")

                # 1. 비디오 로드 (stride 간격)
                # stride를 찾을 때는 batch_video_info에서 전체 프레임 찾아야 함
                # 현재 video_idx는 videos_to_process의 인덱스이므로, 원본 video_info를 찾아야 함
                matching_info = None
                for info in self.batch_video_info:
                    if info['path'] == video_path:
                        matching_info = info
                        break

                if matching_info is None:
                    print(f"⚠️ {video_name}: 비디오 정보를 찾을 수 없음, 건너뜀")
                    continue

                num_frames = matching_info['frames']

                # stride 계산: target_frames에 맞춰 자동 조정
                # 목표: target_frames 프레임을 추출하도록 stride 계산
                # stride = num_frames // target_frames (최소 1)
                # 실제 추출되는 프레임 수: ceil(num_frames / stride)
                calculated_stride = max(1, num_frames // target_frames)
                actual_num_frames_to_extract = (num_frames + calculated_stride - 1) // calculated_stride

                frame_indices = list(range(0, num_frames, calculated_stride))

                print(f"✓ 프레임 추출 계획:")
                print(f"  - 총 비디오 프레임: {num_frames}")
                print(f"  - 목표 프레임 수: {target_frames}")
                print(f"  - 계산된 stride: {calculated_stride}")
                print(f"  - 실제 추출 프레임 수: {actual_num_frames_to_extract}")
                print(f"  - 공식: ceil({num_frames} / {calculated_stride}) = {actual_num_frames_to_extract}")

                # Extract frames
                frames = self.processor.extract_frames(video_path, 0, num_frames, stride=calculated_stride)
                if not frames:
                    print(f"⚠️ {video_name}: 프레임 추출 실패, 건너뜀")
                    continue

                # 2. Propagate (SAM 2 Video Predictor)
                print(f"✓ Propagation 시작...")

                # 임시 디렉토리에 프레임 저장
                video_temp_dir = tempfile.mkdtemp(prefix=f"sam3d_video_{video_idx}_")

                try:
                    for idx, frame in enumerate(frames):
                        frame_path = Path(video_temp_dir) / f"{idx:05d}.jpg"
                        cv2.imwrite(str(frame_path), frame)

                    # SAM 2 inference
                    if self.sam2_video_predictor is not None:
                        inference_state = self.sam2_video_predictor.init_state(video_path=video_temp_dir)

                        # Reference annotations 적용 (첫 프레임)
                        point_coords = []
                        point_labels = []

                        for px, py in reference_annotations['foreground']:
                            point_coords.append([px, py])
                            point_labels.append(1)

                        for px, py in reference_annotations['background']:
                            point_coords.append([px, py])
                            point_labels.append(0)

                        point_coords = np.array(point_coords, dtype=np.float32)
                        point_labels = np.array(point_labels, dtype=np.int32)

                        # Add points to first frame
                        self.sam2_video_predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=0,
                            obj_id=1,
                            points=point_coords,
                            labels=point_labels,
                        )

                        # Propagate
                        video_segments = {}
                        for frame_idx, obj_ids, mask_logits in self.sam2_video_predictor.propagate_in_video(
                            inference_state,
                            start_frame_idx=0
                        ):
                            video_segments[frame_idx] = (mask_logits[0] > 0.0).cpu().numpy()

                        # 3. 결과 저장 (비디오별 디렉토리)
                        video_result_dir = batch_temp_dir / f"video_{video_idx:03d}"
                        video_result_dir.mkdir(exist_ok=True)

                        for frame_idx, mask in video_segments.items():
                            frame_dir = video_result_dir / f"frame_{frame_idx:04d}"
                            frame_dir.mkdir(exist_ok=True)

                            # Save frame and mask
                            cv2.imwrite(str(frame_dir / "original.png"), frames[frame_idx])

                            mask_uint8 = mask.squeeze().astype(np.uint8) * 255
                            cv2.imwrite(str(frame_dir / "mask.png"), mask_uint8)

                        print(f"✓ {len(video_segments)} 프레임 저장 완료")
                        total_processed_frames += len(video_segments)

                        video_results.append({
                            'video_idx': video_idx,
                            'video_name': video_name,
                            'video_path': video_path,
                            'frames': len(video_segments),
                            'result_dir': str(video_result_dir)
                        })

                finally:
                    # 임시 디렉토리 정리
                    shutil.rmtree(video_temp_dir, ignore_errors=True)

                    # 적극적인 메모리 해제
                    # SAM 2 inference_state 정리
                    if 'inference_state' in locals():
                        del inference_state
                    if 'video_segments' in locals():
                        del video_segments

                    # 프레임 메모리 해제
                    del frames

                    # CUDA 메모리 정리
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                    # Python garbage collection 강제 실행
                    import gc
                    gc.collect()

                print(f"✓ {video_name} 완료 (메모리 해제됨)")

            # 결과 저장
            self.batch_results = {
                'temp_dir': str(batch_temp_dir),
                'videos': video_results,
                'total_frames': total_processed_frames,
                'target_frames': target_frames,
                'reference_annotations': reference_annotations
            }

            progress(1.0, desc="Batch 처리 완료!")

            status = f"""
### 🎉 Batch Propagation 완료 ✅

- **처리된 비디오**: {len(video_results)} / {total_videos}
- **총 프레임 수**: {total_processed_frames}
- **목표 프레임 수**: {target_frames} (각 비디오당)
- **임시 저장 위치**: {batch_temp_dir}

### 다음 단계:
- **Export to Fauna** 클릭하여 통합 데이터셋 생성
"""

            return status, "✅ 완료"

        except Exception as e:
            import traceback
            error_msg = f"❌ Batch 처리 실패:\n{str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return error_msg, "❌ 실패"

    def save_batch_session(self, session_name: str = "") -> Tuple[str, str]:
        """
        Batch 처리 결과를 세션으로 저장 (Fauna 형식이 아닌 개별 비디오별 저장)

        Args:
            session_name: 세션 이름

        Returns:
            (저장 경로, 상태 메시지)
        """
        if not hasattr(self, 'batch_results') or not self.batch_results:
            return "", "❌ 먼저 Batch Propagation을 실행하세요"

        try:
            from datetime import datetime
            import shutil
            import json

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if session_name and session_name.strip():
                session_id = f"{session_name.strip()}_{timestamp}"
            else:
                session_id = f"batch_{timestamp}"

            output_dir = Path(f"outputs/sessions/{session_id}")
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'='*80}")
            print(f"💾 Batch 세션 저장: {output_dir}")
            print(f"{'='*80}")

            batch_results = self.batch_results

            # 메타데이터
            metadata = {
                'session_id': session_id,
                'session_type': 'batch',
                'timestamp': timestamp,
                'total_videos': len(batch_results['videos']),
                'total_frames': batch_results['total_frames'],
                'target_frames': batch_results['target_frames'],
                'reference_annotations': batch_results['reference_annotations'],
                'videos': []
            }

            # 각 비디오 결과를 개별 폴더에 저장
            for video_result in batch_results['videos']:
                video_name = video_result['video_name']
                video_result_dir = Path(video_result['result_dir'])
                video_idx = video_result['video_idx']

                # 비디오별 저장 디렉토리
                video_save_dir = output_dir / f"video_{video_idx:03d}_{Path(video_name).stem}"
                video_save_dir.mkdir(exist_ok=True)

                print(f"\n📹 저장 중: {video_name}")

                # 프레임 복사
                if video_result_dir.exists():
                    for frame_dir in video_result_dir.iterdir():
                        if frame_dir.is_dir() and frame_dir.name.startswith('frame_'):
                            dst = video_save_dir / frame_dir.name
                            shutil.copytree(frame_dir, dst, dirs_exist_ok=True)

                # 비디오 메타데이터
                video_meta = {
                    'video_idx': video_idx,
                    'video_name': video_name,
                    'video_path': video_result['video_path'],
                    'num_frames': video_result['frames'],
                    'saved_dir': str(video_save_dir.relative_to(output_dir))
                }
                metadata['videos'].append(video_meta)

                print(f"  ✓ {video_result['frames']} 프레임 저장 완료")

            # 메타데이터 저장
            metadata_path = output_dir / "session_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            # 임시 디렉토리 정리
            if 'temp_dir' in batch_results:
                temp_dir = Path(batch_results['temp_dir'])
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)

            print(f"\n✅ Batch 세션 저장 완료!")
            print(f"   경로: {output_dir}")

            status = f"""
### 💾 Batch 세션 저장 완료 ✅

- **세션 ID**: `{session_id}`
- **저장 경로**: `{output_dir}`
- **비디오 수**: {len(batch_results['videos'])}
- **총 프레임 수**: {batch_results['total_frames']}
- **목표 프레임 수**: {batch_results['target_frames']} (각 비디오당)

### 저장된 비디오:
"""
            for video_meta in metadata['videos']:
                status += f"\n- **{video_meta['video_name']}**: {video_meta['num_frames']} 프레임 (→ `{video_meta['saved_dir']}`)"

            status += f"""

### 세션 구조:
```
{session_id}/
├── video_000_{Path(metadata['videos'][0]['video_name']).stem}/
│   ├── frame_0000/
│   │   ├── original.png
│   │   └── mask.png
│   └── ...
├── video_001_.../
└── session_metadata.json
```

### 다음 단계:
- 저장된 세션은 나중에 로드 가능
- 또는 **Export to Fauna**로 통합 데이터셋 생성
"""

            return str(output_dir), status

        except Exception as e:
            import traceback
            error_msg = f"❌ Batch 세션 저장 실패:\n{str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return "", error_msg

    def load_batch_session(self, session_path: str) -> Tuple[str, str]:
        """
        저장된 Batch 세션 로드

        Args:
            session_path: 세션 디렉토리 경로 또는 session_metadata.json 경로

        Returns:
            (상태 메시지, 성공 여부)
        """
        try:
            import json

            session_path = Path(session_path)

            # session_metadata.json 경로 찾기
            if session_path.is_file() and session_path.name == "session_metadata.json":
                metadata_path = session_path
                session_dir = session_path.parent
            elif session_path.is_dir():
                metadata_path = session_path / "session_metadata.json"
                session_dir = session_path
            else:
                return "❌ 유효하지 않은 세션 경로입니다", ""

            if not metadata_path.exists():
                return f"❌ 세션 메타데이터를 찾을 수 없습니다: {metadata_path}", ""

            # 메타데이터 로드
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # 세션 타입 확인
            if metadata.get('session_type') != 'batch':
                return f"❌ Batch 세션이 아닙니다. (타입: {metadata.get('session_type')})", ""

            print(f"\n{'='*80}")
            print(f"📂 Batch 세션 로드: {session_dir}")
            print(f"{'='*80}")

            # batch_results 복원
            video_results = []
            for video_meta in metadata['videos']:
                video_result_dir = session_dir / video_meta['saved_dir']

                if not video_result_dir.exists():
                    print(f"  ⚠️  경고: {video_result_dir} 없음")
                    continue

                # 프레임 개수 확인
                num_frames = len(list(video_result_dir.glob("frame_*")))

                video_results.append({
                    'video_idx': video_meta['video_idx'],
                    'video_name': video_meta['video_name'],
                    'video_path': video_meta['video_path'],
                    'frames': num_frames,
                    'result_dir': str(video_result_dir)
                })

                print(f"  ✓ {video_meta['video_name']}: {num_frames} 프레임")

            # batch_results 설정
            self.batch_results = {
                'temp_dir': '',  # 로드된 세션은 임시 디렉토리 없음
                'videos': video_results,
                'total_frames': metadata['total_frames'],
                'target_frames': metadata['target_frames'],
                'reference_annotations': metadata['reference_annotations']
            }

            print(f"\n✅ Batch 세션 로드 완료!")

            status = f"""
### 📂 Batch 세션 로드 완료 ✅

- **세션 ID**: `{metadata['session_id']}`
- **로드 경로**: `{session_dir}`
- **비디오 수**: {len(video_results)}
- **총 프레임 수**: {metadata['total_frames']}
- **목표 프레임 수**: {metadata['target_frames']} (각 비디오당)

### 로드된 비디오:
"""
            for video_result in video_results:
                status += f"\n- **{video_result['video_name']}**: {video_result['frames']} 프레임"

            status += """

### 다음 단계:
- **Export to Fauna** 클릭하여 통합 데이터셋 생성
- 또는 추가 편집 수행
"""

            return status, "✅ 로드 완료"

        except Exception as e:
            import traceback
            error_msg = f"❌ Batch 세션 로드 실패:\n{str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return error_msg, ""

    def export_batch_to_fauna(self, output_name: str = "fauna_dataset", file_structure: str = "video_folders") -> Tuple[str, str]:
        """
        Batch 처리 결과를 Fauna 데이터셋 형식으로 export

        Args:
            output_name: 출력 데이터셋 이름
            file_structure: 파일 구조 ("video_folders" 또는 "flat")

        Returns:
            (Fauna 데이터셋 경로, 상태 메시지)
        """
        if not hasattr(self, 'batch_results') or not self.batch_results:
            return "", "❌ 먼저 Batch Propagation을 실행하세요"

        try:
            from datetime import datetime
            import shutil
            import json

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"outputs/fauna_datasets/{output_name}_{timestamp}")
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'='*80}")
            print(f"📦 Fauna 데이터셋 생성: {output_dir}")
            print(f"📁 파일 구조: {file_structure}")
            print(f"{'='*80}")

            batch_results = self.batch_results
            total_frames_exported = 0
            video_segments_info = []

            # 각 비디오 결과를 export
            for video_idx, video_result in enumerate(batch_results['videos']):
                video_name = video_result['video_name']
                video_result_dir = Path(video_result['result_dir'])
                num_frames = video_result['frames']

                print(f"\n📹 {video_name}: {num_frames} 프레임 export 중...")

                # 비디오 이름에서 안전한 prefix 생성 (경로 구분자 제거)
                video_prefix = f"video{video_idx:03d}"

                video_segment = {
                    'video_name': video_name,
                    'video_path': video_result['video_path'],
                    'video_idx': video_result['video_idx'],
                    'video_prefix': video_prefix,
                    'num_frames': num_frames
                }
                video_segments_info.append(video_segment)

                # 비디오별 폴더 생성 (video_folders 모드인 경우)
                if file_structure == "video_folders":
                    video_output_dir = output_dir / video_prefix
                    video_output_dir.mkdir(exist_ok=True)

                # 프레임 복사
                for local_frame_idx in range(num_frames):
                    src_frame_dir = video_result_dir / f"frame_{local_frame_idx:04d}"

                    if not src_frame_dir.exists():
                        print(f"  ⚠️  경고: {src_frame_dir} 없음, 건너뜀")
                        continue

                    # 원본 파일 읽기
                    src_rgb = src_frame_dir / "original.png"
                    src_mask = src_frame_dir / "mask.png"

                    if not src_rgb.exists() or not src_mask.exists():
                        print(f"  ⚠️  경고: frame_{local_frame_idx:04d} 파일 누락")
                        continue

                    # 목적지 경로 결정
                    if file_structure == "video_folders":
                        # video001/frame_0000_rgb.png
                        dst_rgb = video_output_dir / f"frame_{local_frame_idx:04d}_rgb.png"
                        dst_mask = video_output_dir / f"frame_{local_frame_idx:04d}_mask.png"
                    else:  # flat
                        # video001_frame_0000_rgb.png
                        dst_rgb = output_dir / f"{video_prefix}_frame_{local_frame_idx:04d}_rgb.png"
                        dst_mask = output_dir / f"{video_prefix}_frame_{local_frame_idx:04d}_mask.png"

                    # 파일 복사
                    shutil.copy2(src_rgb, dst_rgb)
                    shutil.copy2(src_mask, dst_mask)

                total_frames_exported += num_frames
                print(f"  ✓ {num_frames} 프레임 복사 완료")

            # 메타데이터 생성
            metadata = {
                'dataset_name': output_name,
                'timestamp': timestamp,
                'file_structure': file_structure,
                'total_frames': total_frames_exported,
                'num_videos': len(batch_results['videos']),
                'target_frames': batch_results['target_frames'],
                'reference_annotations': batch_results['reference_annotations'],
                'video_segments': video_segments_info
            }

            # 메타데이터 저장
            metadata_path = output_dir / "dataset_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"\n✅ Fauna 데이터셋 생성 완료!")
            print(f"   경로: {output_dir}")
            print(f"   총 프레임: {total_frames_exported}")

            # 임시 디렉토리 정리
            if 'temp_dir' in batch_results:
                temp_dir = Path(batch_results['temp_dir'])
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    print(f"✓ 임시 디렉토리 정리 완료")

            # 상태 메시지 생성
            structure_example = ""
            if file_structure == "video_folders":
                structure_example = f"""
```
{output_dir.name}/
├── video000/
│   ├── frame_0000_rgb.png
│   ├── frame_0000_mask.png
│   ├── frame_0001_rgb.png
│   └── frame_0001_mask.png
├── video001/
│   └── ...
└── dataset_metadata.json
```
"""
            else:  # flat
                structure_example = f"""
```
{output_dir.name}/
├── video000_frame_0000_rgb.png
├── video000_frame_0000_mask.png
├── video000_frame_0001_rgb.png
├── video000_frame_0001_mask.png
├── video001_frame_0000_rgb.png
└── dataset_metadata.json
```
"""

            status = f"""
### 🎉 Fauna 데이터셋 생성 완료 ✅

- **출력 경로**: `{output_dir}`
- **파일 구조**: {file_structure}
- **총 프레임 수**: {total_frames_exported}
- **비디오 수**: {len(batch_results['videos'])}
- **목표 프레임 수**: {batch_results['target_frames']} (각 비디오당)

### 비디오 정보:
"""
            for seg in video_segments_info:
                status += f"\n- **{seg['video_name']}**: {seg['num_frames']} 프레임 (prefix: {seg['video_prefix']})"

            status += f"\n\n### 데이터셋 구조:{structure_example}"

            return str(output_dir), status

        except Exception as e:
            import traceback
            error_msg = f"❌ Fauna Export 실패: {str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            return "", error_msg

    def get_video_duration(self, data_dir: str, video_file: str) -> float:
        """
        비디오 파일의 전체 길이(초) 반환

        Args:
            data_dir: 데이터 디렉토리 경로
            video_file: 비디오 파일명

        Returns:
            비디오 길이(초), 실패 시 3.0 (기본값)
        """
        if not video_file:
            return 3.0

        video_path = Path(data_dir) / video_file
        if not video_path.exists():
            return 3.0

        try:
            info = self.processor.get_video_info(str(video_path))
            duration = info['frame_count'] / info['fps']
            print(f"✓ 비디오 길이: {duration:.2f}초 ({info['frame_count']} 프레임, {info['fps']:.2f} fps)")
            return round(duration, 2)
        except Exception as e:
            print(f"비디오 정보 읽기 실패: {e}")
            return 3.0

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
            total_duration = info['frame_count'] / fps

            # duration이 비디오 길이를 초과하면 전체 길이 사용
            if duration <= 0 or duration > total_duration:
                duration = total_duration
                print(f"✓ Duration을 비디오 전체 길이로 설정: {duration:.2f}초")

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

    def propagate_to_all_frames(self, stride: int = 1, progress=gr.Progress()) -> Tuple[np.ndarray, str]:
        """
        현재 프레임의 annotation을 전체 비디오에 propagation (tracking)
        SAM 2 Video Predictor를 사용한 메모리 기반 추적

        Args:
            stride: 프레임 처리 간격 (1=모든 프레임, 10=10프레임마다 처리)

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
                    # 메모리 보호: 최대 500 프레임으로 제한 (약 6GB 메모리)
                    MAX_FRAMES = 500
                    effective_stride = stride
                    frame_indices = list(range(0, len(self.frames), stride))

                    if len(frame_indices) > MAX_FRAMES:
                        # stride 자동 조정
                        effective_stride = max(stride, len(self.frames) // MAX_FRAMES)
                        frame_indices = list(range(0, len(self.frames), effective_stride))
                        print(f"⚠️ 메모리 보호: stride {stride} → {effective_stride} 자동 조정 ({len(frame_indices)} 프레임)")

                    progress(0.05, desc=f"프레임 저장 중 (stride={effective_stride}, 총 {len(frame_indices)} 프레임)...")

                    # stride 간격으로만 프레임 저장
                    for idx, i in enumerate(frame_indices):
                        frame_path = os.path.join(temp_dir, f"{idx:05d}.jpg")
                        cv2.imwrite(frame_path, self.frames[i])

                    # 원본 인덱스 매핑 저장 (나중에 결과를 원본 인덱스로 복원)
                    self.stride_frame_mapping = {idx: i for idx, i in enumerate(frame_indices)}
                    self.effective_stride = effective_stride  # status 메시지를 위해 저장

                    print(f"✓ {len(frame_indices)} 프레임 저장 완료 (원본 {len(self.frames)}개 중)")

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

                    # 현재 프레임 인덱스를 stride 기반 인덱스로 변환
                    # 예: 원본 프레임 20, stride=10 -> stride 인덱스 2
                    stride_frame_idx = self.current_frame_idx // stride
                    if self.current_frame_idx not in frame_indices:
                        # 현재 프레임이 stride에 포함되지 않으면 가장 가까운 프레임 사용
                        stride_frame_idx = min(range(len(frame_indices)),
                                              key=lambda i: abs(frame_indices[i] - self.current_frame_idx))

                    progress(0.15, desc=f"초기 프레임 ({self.current_frame_idx} -> stride idx {stride_frame_idx}) annotation 중...")

                    # 현재 프레임을 conditioning frame으로 설정
                    _, out_obj_ids, out_mask_logits = self.sam2_video_predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=stride_frame_idx,
                        obj_id=1,  # Single object tracking
                        points=point_coords,
                        labels=point_labels,
                    )

                    progress(0.2, desc="메모리 기반 전파 시작...")

                    # 4. Propagate using memory-based tracking (NO points on other frames!)
                    video_segments = {}
                    for stride_idx, obj_ids, mask_logits in self.sam2_video_predictor.propagate_in_video(
                        inference_state,
                        start_frame_idx=stride_frame_idx
                    ):
                        # Memory-based tracking - 각 프레임은 이전 프레임의 메모리를 사용
                        # Points는 재적용되지 않음!
                        video_segments[stride_idx] = (mask_logits[0] > 0.0).cpu().numpy()

                        progress_pct = 0.2 + 0.6 * (stride_idx + 1) / len(frame_indices)
                        progress(progress_pct, desc=f"Tracking... {stride_idx+1}/{len(frame_indices)} (stride {stride})")

                    # 5. 결과를 self.masks에 저장 (stride 간격의 프레임만)
                    self.masks = [None] * len(self.frames)
                    for stride_idx, mask in video_segments.items():
                        original_idx = self.stride_frame_mapping.get(stride_idx)
                        if original_idx is not None and original_idx < len(self.masks):
                            self.masks[original_idx] = mask.squeeze()

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

            # 실제 사용된 stride 계산
            used_stride = getattr(self, 'effective_stride', stride)

            status = f"""
### Propagation 완료 ✅

- **Method**: {method_used}
- **Stride**: {used_stride} (처리된 프레임: {tracked_frames}, 전체: {len(self.frames)})
- **효율성**: {100 * tracked_frames / len(self.frames):.1f}% 프레임만 처리
- **현재 프레임**: {self.current_frame_idx + 1} / {len(self.frames)}
- **Conditioning Frame**: {self.current_frame_idx} (Points만 여기 적용)

### 메모리 기반 추적 (Stride 적용):
- 현재 프레임의 points만 사용
- {used_stride} 간격으로 프레임 처리 (예: 3000 프레임 → {tracked_frames} 프레임)
- 객체 이동에도 정확한 마스크 생성
- 메모리 보호: 최대 500 프레임으로 자동 제한

### 다음:
- **프레임 네비게이션**으로 결과 확인 (stride 간격만 마스크 존재)
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
        print("\n" + "="*80)
        print("🔹 generate_3d_mesh() 시작")
        print("="*80)

        if len(self.frames) == 0 or all(m is None for m in self.masks):
            print("❌ 프레임 또는 마스크 없음")
            return None, "먼저 세그멘테이션을 완료하세요"

        try:
            progress(0, desc="3D mesh 생성 준비 중...")

            # SAM 3D 체크포인트 확인
            if self.config:
                checkpoint_dir = Path(self.config.sam3d_checkpoint_dir).expanduser()
                print(f"✓ Config에서 checkpoint 경로 로드: {checkpoint_dir}")
            else:
                checkpoint_dir = Path("~/dev/sam-3d-objects/checkpoints/hf/checkpoints").expanduser()
                print(f"✓ 기본 checkpoint 경로 사용: {checkpoint_dir}")

            print(f"✓ Checkpoint 존재 확인 중: {checkpoint_dir}")
            print(f"   pipeline.yaml 존재: {(checkpoint_dir / 'pipeline.yaml').exists()}")

            if not (checkpoint_dir / "pipeline.yaml").exists():
                print("❌ pipeline.yaml 파일이 없음")
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

            print(f"\n✓ 대표 프레임 선택: {mid_idx + 1}/{len(self.frames)}")
            print(f"   Frame shape: {frame.shape}")
            print(f"   Mask shape: {mask.shape if mask is not None else 'None'}")
            print(f"   Mask type: {type(mask)}")

            if mask is None:
                print("❌ 중간 프레임에 마스크 없음")
                return None, "중간 프레임에 마스크가 없습니다"

            # 3D 재구성 시도
            print("\n✓ 3D 재구성 시작...")
            progress(0.5, desc="SAM 3D 재구성 중...")

            # Unload SAM2 models to free GPU memory for SAM 3D
            # Critical for RTX 3060 12GB: SAM2 (3GB) + SAM3D (10GB) = 13GB > 12GB
            self.unload_sam2_models()

            try:
                reconstruction = self.processor.reconstruct_3d(frame, mask)
                print(f"✓ Reconstruction 완료: {type(reconstruction)}")

                if reconstruction:
                    # PLY 저장
                    project_root = Path(__file__).parent.parent
                    output_dir = project_root / "outputs" / "3d_meshes"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / "reconstruction.ply"

                    print(f"\n✓ Mesh 저장 중: {output_path}")
                    self.processor.export_mesh(reconstruction, str(output_path), format='ply')
                    print(f"✓ Mesh 저장 완료")

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
                    print("✅ generate_3d_mesh() 완료")

                    # Reload SAM2 models for continued use
                    self.reload_sam2_models()

                    return str(output_path), status
                else:
                    print("❌ Reconstruction이 None")

                    # Reload SAM2 models even on failure
                    self.reload_sam2_models()

                    return None, "3D 재구성 실패 (SAM 3D 체크포인트 필요)"

            except Exception as e:
                # SAM 3D 없으면 간단한 point cloud만 생성
                print(f"❌ 3D 재구성 실패: {e}")
                import traceback
                traceback.print_exc()

                # Reload SAM2 models even on failure
                self.reload_sam2_models()

                return None, f"3D 재구성 실패: {str(e)}\n\nSAM 3D 체크포인트가 필요합니다."

        except Exception as e:
            import traceback
            return None, f"오류:\n{str(e)}\n{traceback.format_exc()}"

    def save_annotation_session(self, session_name: str = "") -> str:
        """
        Annotation 세션 전체 저장 (annotation points + masks + metadata)

        Args:
            session_name: 세션 이름 (비어있으면 timestamp 사용)
        """
        print("\n" + "="*80)
        print("🔹 save_annotation_session() 시작")
        print("="*80)

        if len(self.frames) == 0:
            print("❌ 저장 실패: 프레임 없음")
            return "저장할 데이터가 없습니다"

        print(f"✓ 프레임 수: {len(self.frames)}")
        print(f"✓ 마스크 수: {len(self.masks)}")
        print(f"✓ Foreground points: {len(self.annotations['foreground'])}")
        print(f"✓ Background points: {len(self.annotations['background'])}")

        try:
            # 세션 ID 생성
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if session_name and session_name.strip():
                # 사용자 지정 이름 사용 (timestamp 추가)
                session_id = f"{session_name.strip()}_{timestamp}"
            else:
                # timestamp만 사용
                session_id = timestamp

            print(f"✓ 세션 ID 생성: {session_id}")

            output_dir = Path(f"outputs/sessions/{session_id}")
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ 출력 디렉토리 생성: {output_dir}")

            # 1. Annotation 메타데이터 저장 (JSON)
            print("\n🔹 Step 1: 메타데이터 구성 중...")

            # Stride 계산 (stride_frame_mapping이 있으면 추출, 없으면 1로 가정)
            effective_stride = 1
            if hasattr(self, 'stride_frame_mapping') and self.stride_frame_mapping:
                frame_indices = sorted(self.stride_frame_mapping.values())
                if len(frame_indices) > 1:
                    effective_stride = frame_indices[1] - frame_indices[0]

            num_frames_saved = sum(1 for m in self.masks if m is not None)

            metadata = {
                "session_id": session_id,
                "video_path": self.video_path,
                "num_frames": num_frames_saved,  # Load 함수가 찾는 키
                "num_frames_total": len(self.frames),
                "num_frames_saved": num_frames_saved,
                "stride": effective_stride,
                "current_frame_idx": self.current_frame_idx,
                "annotations": {
                    "foreground": self.annotations['foreground'],
                    "background": self.annotations['background']
                },
                "frame_info": []
            }
            print(f"✓ 메타데이터 구성 완료 (stride={effective_stride})")

            # 2. 각 프레임 저장 (stride 간격의 프레임만)
            print("\n🔹 Step 2: 프레임별 저장 시작 (마스크가 있는 프레임만)...")
            saved_masks = 0
            saved_frame_idx = 0
            for i, (frame, mask) in enumerate(zip(self.frames, self.masks)):
                # 마스크가 없는 프레임은 건너뛰기 (stride로 생략된 프레임)
                if mask is None:
                    continue

                if saved_frame_idx % 10 == 0:  # 10프레임마다 진행상황 출력
                    print(f"  진행: {saved_frame_idx} 프레임 저장됨 (원본 인덱스: {i}/{len(self.frames)})...")

                frame_dir = output_dir / f"frame_{saved_frame_idx:04d}"
                frame_dir.mkdir(exist_ok=True)

                # 원본 프레임 저장
                try:
                    frame_path = frame_dir / "original.png"
                    success = cv2.imwrite(str(frame_path), frame)
                    if not success:
                        print(f"  ⚠️ 프레임 {i} 저장 실패: {frame_path}")
                except Exception as e:
                    print(f"  ❌ 프레임 {i} 저장 오류: {str(e)}")
                    raise

                # 마스크 저장 (이미 mask is not None 체크로 들어옴)
                try:
                    mask_path = frame_dir / "mask.png"
                    mask_uint8 = mask.astype(np.uint8) * 255
                    success = cv2.imwrite(str(mask_path), mask_uint8)
                    if not success:
                        print(f"  ⚠️ 마스크 {i} 저장 실패: {mask_path}")

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
                    success = cv2.imwrite(str(vis_path), result_bgr)
                    if not success:
                        print(f"  ⚠️ 시각화 {i} 저장 실패: {vis_path}")

                    saved_masks += 1

                    # 프레임 메타데이터 (원본 인덱스와 저장 인덱스 모두 기록)
                    mask_area = np.sum(mask > 0)
                    metadata["frame_info"].append({
                        "saved_frame_idx": saved_frame_idx,  # Fauna 형식 인덱스
                        "original_frame_idx": i,  # 원본 비디오 인덱스
                        "has_mask": True,
                        "mask_area": int(mask_area),
                        "mask_percentage": float(mask_area / mask.size * 100)
                    })

                    saved_frame_idx += 1

                except Exception as e:
                    print(f"  ❌ 마스크 {i} 처리 오류: {str(e)}")
                    raise

            # 마스크가 없는 프레임에 대한 메타데이터는 더 이상 추가하지 않음
            # (stride로 생략된 프레임)

            print(f"✓ 프레임별 저장 완료: 원본 {len(self.frames)}개 중 마스크가 있는 {saved_masks}개만 저장")

            # 3. Metadata JSON 저장
            print("\n🔹 Step 3: 메타데이터 JSON 저장 중...")
            metadata_path = output_dir / "session_metadata.json"
            try:
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                print(f"✓ 메타데이터 저장 완료: {metadata_path}")
            except Exception as e:
                print(f"❌ 메타데이터 저장 오류: {str(e)}")
                raise

            print("\n" + "="*80)
            print("✅ save_annotation_session() 완료!")
            print("="*80 + "\n")

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
            error_detail = traceback.format_exc()
            print("\n" + "="*80)
            print("❌ save_annotation_session() 실패!")
            print("="*80)
            print(f"오류 타입: {type(e).__name__}")
            print(f"오류 메시지: {str(e)}")
            print("\n전체 스택 트레이스:")
            print(error_detail)
            print("="*80 + "\n")
            return f"저장 오류: {str(e)}\n\n```\n{error_detail}\n```"

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

    def _find_next_sequence(self, fauna_root: Path, animal_name: str) -> str:
        """
        다음 사용 가능한 시퀀스 번호 찾기

        Args:
            fauna_root: Fauna 데이터셋 루트 경로
            animal_name: 동물 이름

        Returns:
            "seq_XXX" 형식의 시퀀스 이름
        """
        train_dir = fauna_root / animal_name / "train"

        # train 디렉토리가 없으면 seq_000부터 시작
        if not train_dir.exists():
            return "seq_000"

        # 기존 seq_XXX 디렉토리 찾기
        existing_sequences = [
            d.name for d in train_dir.iterdir()
            if d.is_dir() and d.name.startswith("seq_")
        ]

        # 기존 시퀀스가 없으면 seq_000
        if not existing_sequences:
            return "seq_000"

        # 가장 큰 시퀀스 번호 찾기
        try:
            max_seq_num = max([int(s.split("_")[1]) for s in existing_sequences])
            next_seq_num = max_seq_num + 1
            return f"seq_{next_seq_num:03d}"
        except (IndexError, ValueError):
            # 파싱 실패 시 안전하게 seq_000 반환
            return "seq_000"

    def export_fauna_dataset(
        self,
        animal_name: str = "mouse",
        target_frames: int = 50,
        progress=gr.Progress()
    ) -> str:
        """
        Fauna 데이터셋 형식으로 저장
        스마트 샘플링: 전체 비디오에서 target_frames 개만 균등 간격으로 선택
        자동 시퀀스 번호 할당: 기존 시퀀스를 덮어쓰지 않음

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

            # 출력 디렉토리 설정 - outputs 하위에 체계적으로 저장
            project_root = Path(__file__).parent.parent
            fauna_root = project_root / "outputs" / "fauna_datasets"
            sequence_name = self._find_next_sequence(fauna_root, animal_name)
            output_dir = fauna_root / animal_name / "train" / sequence_name
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n🔹 Fauna 데이터셋 저장:")
            print(f"   Animal: {animal_name}")
            print(f"   Sequence: {sequence_name}")
            print(f"   Path: {output_dir}")

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
                "sequence": sequence_name,
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
**시퀀스**: `{sequence_name}` (자동 할당 - 기존 데이터 보존)

**데이터셋 구조**:
```
{animal_name}/train/{sequence_name}/
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

                            with gr.Row():
                                target_frames = gr.Number(
                                    label="목표 프레임 수",
                                    value=300,
                                    minimum=10,
                                    maximum=1000,
                                    step=10,
                                    info="처리할 총 프레임 수 (Stride 자동 계산)"
                                )
                                auto_stride = gr.Number(
                                    label="자동 Stride",
                                    value=10,
                                    interactive=False,
                                    info="목표 프레임 수 기반 자동 계산"
                                )

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
                                    label="이동 간격 (Stride)",
                                    minimum=1,
                                    maximum=100,
                                    value=1,
                                    step=1,
                                    info="Propagate 시에도 이 간격으로 처리됩니다"
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

                            gr.Markdown("### 💾 세션 저장")

                            session_name_input = gr.Textbox(
                                label="세션 이름 (선택사항)",
                                placeholder="예: mouse_experiment_1",
                                info="비어있으면 timestamp만 사용"
                            )

                            save_session_btn = gr.Button("💾 Save Session", variant="secondary")

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
                    # 비디오 파일 선택 시 duration 자동 업데이트
                    video_file.change(
                        fn=self.get_video_duration,
                        inputs=[data_dir, video_file],
                        outputs=[duration]
                    )

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

                    # 목표 프레임 수 변경 시 auto_stride 자동 계산
                    target_frames.change(
                        fn=self.calculate_stride_from_target,
                        inputs=[target_frames],
                        outputs=[auto_stride]
                    )

                    propagate_btn.click(
                        fn=self.propagate_to_all_frames,
                        inputs=[auto_stride],  # frame_step 대신 auto_stride 사용
                        outputs=[image_display, status_text]
                    )

                    save_session_btn.click(
                        fn=self.save_annotation_session,
                        inputs=[session_name_input],
                        outputs=[status_text]
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

                # ===== Tab 2: Batch Mode =====
                with gr.Tab("📦 Batch Mode"):
                    gr.Markdown("### 여러 비디오 일괄 처리")

                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 📂 비디오 스캔")

                            batch_data_dir = gr.Textbox(
                                label="비디오 폴더",
                                value=self.default_data_dir
                            )

                            batch_pattern = gr.Textbox(
                                label="파일 패턴",
                                value="*.mp4",
                                info="예: *.mp4, *.avi, video_*.mp4"
                            )

                            batch_scan_btn = gr.Button("📂 비디오 스캔", variant="primary")

                            batch_info = gr.Markdown("비디오를 스캔하세요")

                            # 비디오 선택 UI (Accordion으로 감싸기)
                            with gr.Accordion("🎬 처리할 비디오 선택", open=True):
                                # 전체 선택/해제 버튼 (상단)
                                with gr.Row():
                                    batch_select_all_btn = gr.Button("✅ 전체 선택", size="sm")
                                    batch_deselect_all_btn = gr.Button("❌ 전체 해제", size="sm")

                                batch_video_selection = gr.CheckboxGroup(
                                    label="비디오 목록",
                                    choices=[],
                                    value=[],
                                    interactive=True,
                                    info="선택된 비디오만 처리됩니다"
                                )

                            gr.Markdown("### 🎯 Reference Annotation")

                            gr.Markdown("""
첫 번째 비디오의 대표 프레임에 annotation을 추가하세요.
모든 비디오에 동일한 annotation이 적용됩니다.
                            """)

                            # Interactive Mode에서 사용하는 annotation UI 재사용
                            batch_load_ref_btn = gr.Button("📹 Reference 프레임 로드")

                            batch_annotation_mode = gr.Radio(
                                label="Point 타입",
                                choices=["foreground", "background"],
                                value="foreground"
                            )

                            with gr.Row():
                                batch_segment_btn = gr.Button("🎯 Segment (미리보기)", variant="secondary", size="sm")
                                batch_clear_btn = gr.Button("🗑️ Points 초기화", size="sm")

                            gr.Markdown("### ⚙️ Batch 설정")

                            with gr.Row():
                                batch_target_frames = gr.Number(
                                    label="비디오당 목표 프레임 수",
                                    value=100,
                                    minimum=10,
                                    maximum=500,
                                    step=10,
                                    info="각 비디오에서 추출할 목표 프레임 수 (실제 stride 자동 계산)"
                                )

                            batch_propagate_btn = gr.Button("🔄 Batch Propagate", variant="primary", size="lg")

                            gr.Markdown("### 💾 세션 관리")

                            # 세션 로드
                            with gr.Accordion("📂 세션 불러오기", open=False):
                                batch_load_session_path = gr.Textbox(
                                    label="세션 경로",
                                    placeholder="예: outputs/sessions/mouse_batch_20251125_123456",
                                    info="세션 폴더 경로 또는 session_metadata.json 경로"
                                )
                                batch_load_session_btn = gr.Button("📂 세션 로드", variant="secondary")

                            # 세션 저장 및 Export
                            batch_session_name = gr.Textbox(
                                label="세션/데이터셋 이름",
                                value="mouse_batch",
                                placeholder="예: mouse_experiment_batch",
                                info="세션 저장 및 Fauna export 시 사용"
                            )

                            batch_file_structure = gr.Radio(
                                choices=[
                                    ("📁 비디오별 폴더 (video001/frame_0000_rgb.png)", "video_folders"),
                                    ("📄 완전 평면 (video001_frame_0000_rgb.png)", "flat")
                                ],
                                value="video_folders",
                                label="📁 파일 구조 (Export 시)",
                                info="비디오별로 폴더 구조 vs 모든 파일 평면 구조"
                            )

                            with gr.Row():
                                batch_save_session_btn = gr.Button("💾 Save Session", variant="secondary")
                                batch_export_btn = gr.Button("📦 Export to Fauna", variant="secondary")

                        with gr.Column(scale=2):
                            batch_image_display = gr.Image(
                                label="Reference Frame",
                                type="numpy"
                            )

                            batch_status_text = gr.Markdown("### 상태: 대기 중")

                            batch_output_path = gr.Textbox(
                                label="출력 경로",
                                interactive=False
                            )

                    # Event handlers
                    batch_scan_btn.click(
                        fn=self.scan_batch_videos,
                        inputs=[batch_data_dir, batch_pattern],
                        outputs=[gr.State(), batch_info, batch_video_selection]
                    )

                    # 전체 선택/해제 버튼
                    def select_all_videos():
                        if hasattr(self, 'batch_video_label_map'):
                            all_labels = list(self.batch_video_label_map.keys())
                            return gr.CheckboxGroup(value=all_labels)
                        return gr.CheckboxGroup(value=[])

                    def deselect_all_videos():
                        return gr.CheckboxGroup(value=[])

                    batch_select_all_btn.click(
                        fn=select_all_videos,
                        outputs=[batch_video_selection]
                    )

                    batch_deselect_all_btn.click(
                        fn=deselect_all_videos,
                        outputs=[batch_video_selection]
                    )

                    # Reference frame 로드
                    batch_load_ref_btn.click(
                        fn=self.batch_load_reference_frame,
                        inputs=[batch_video_selection],
                        outputs=[batch_image_display, batch_status_text]
                    )

                    # Batch 모드 point annotation 클릭 이벤트
                    batch_image_display.select(
                        fn=self.add_point,
                        inputs=[batch_image_display, batch_annotation_mode],
                        outputs=[batch_image_display, batch_status_text]
                    )

                    # Batch segment (미리보기)
                    batch_segment_btn.click(
                        fn=self.segment_current_frame,
                        outputs=[batch_image_display, batch_status_text]
                    )

                    # Batch clear points
                    def batch_clear_points():
                        self.annotations = {'foreground': [], 'background': []}
                        if len(self.frames) > 0:
                            frame_rgb = cv2.cvtColor(self.frames[self.current_frame_idx], cv2.COLOR_BGR2RGB)
                            return frame_rgb, "Points 초기화됨"
                        return None, "Points 초기화됨"

                    batch_clear_btn.click(
                        fn=batch_clear_points,
                        outputs=[batch_image_display, batch_status_text]
                    )

                    batch_propagate_btn.click(
                        fn=self.batch_propagate_videos,
                        inputs=[batch_target_frames, batch_video_selection],
                        outputs=[batch_status_text, gr.State()]
                    )

                    # 세션 로드
                    batch_load_session_btn.click(
                        fn=self.load_batch_session,
                        inputs=[batch_load_session_path],
                        outputs=[batch_status_text, batch_output_path]
                    )

                    # 세션 저장
                    batch_save_session_btn.click(
                        fn=self.save_batch_session,
                        inputs=[batch_session_name],
                        outputs=[batch_output_path, batch_status_text]
                    )

                    # Fauna export
                    batch_export_btn.click(
                        fn=self.export_batch_to_fauna,
                        inputs=[batch_session_name, batch_file_structure],
                        outputs=[batch_output_path, batch_status_text]
                    )

                # ===== Tab 3: Quick Mode =====
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

                # ===== Tab 4: Data Augmentation =====
                with gr.Tab("🎲 Data Augmentation"):
                    gr.Markdown("### 데이터 증강")
                    gr.Markdown("RGB 이미지와 마스크를 함께 증강합니다. 기하학적 변환은 동일하게, 색상 변환은 RGB만 적용됩니다.")

                    with gr.Row():
                        # Left column: Input & Controls
                        with gr.Column(scale=1):
                            gr.Markdown("#### 📂 Input Source")

                            # Session selection
                            aug_session_dir = gr.Textbox(
                                label="Session Directory",
                                value=str(Path(self.default_output_dir) / "sessions"),
                                placeholder="Path to saved annotation sessions"
                            )

                            aug_scan_btn = gr.Button("📂 Scan Sessions", size="sm")

                            aug_session_list = gr.Dropdown(
                                label="Select Session",
                                choices=[],
                                interactive=True
                            )

                            aug_load_session_btn = gr.Button("📥 Load Session", variant="primary")

                            aug_session_info = gr.Markdown("No session loaded")

                            # Augmentation parameters
                            gr.Markdown("#### ⚙️ Augmentation Parameters")

                            with gr.Accordion("🔄 Geometric Transforms (RGB + Mask)", open=True):
                                aug_scale_enable = gr.Checkbox(label="Enable Scale", value=True)
                                with gr.Row():
                                    aug_scale_min = gr.Slider(
                                        label="Scale Min",
                                        minimum=0.3, maximum=1.0, value=0.5, step=0.05
                                    )
                                    aug_scale_max = gr.Slider(
                                        label="Scale Max",
                                        minimum=1.0, maximum=3.0, value=2.0, step=0.1
                                    )

                                aug_fill_color = gr.Dropdown(
                                    label="Fill Color (for shrinking)",
                                    choices=["white", "black", "nearest"],
                                    value="white"
                                )

                                aug_rotation_enable = gr.Checkbox(label="Enable Rotation", value=True)
                                with gr.Row():
                                    aug_rotation_min = gr.Slider(
                                        label="Rotation Min (deg)",
                                        minimum=-180, maximum=0, value=-30, step=5
                                    )
                                    aug_rotation_max = gr.Slider(
                                        label="Rotation Max (deg)",
                                        minimum=0, maximum=180, value=30, step=5
                                    )

                                aug_flip_enable = gr.Checkbox(label="Enable Random Flip", value=True)

                            with gr.Accordion("🎨 Photometric Transforms (RGB only)", open=False):
                                aug_noise_enable = gr.Checkbox(label="Enable Gaussian Noise", value=True)
                                aug_noise_std = gr.Slider(
                                    label="Noise Std",
                                    minimum=0, maximum=30, value=10, step=1
                                )

                                aug_brightness_enable = gr.Checkbox(label="Enable Brightness", value=True)
                                with gr.Row():
                                    aug_brightness_min = gr.Slider(
                                        label="Brightness Min",
                                        minimum=0.5, maximum=1.0, value=0.7, step=0.05
                                    )
                                    aug_brightness_max = gr.Slider(
                                        label="Brightness Max",
                                        minimum=1.0, maximum=1.5, value=1.3, step=0.05
                                    )

                                aug_contrast_enable = gr.Checkbox(label="Enable Contrast", value=False)
                                aug_color_jitter_enable = gr.Checkbox(label="Enable Color Jitter", value=False)
                                aug_blur_enable = gr.Checkbox(label="Enable Gaussian Blur", value=False)

                            # Preview settings
                            gr.Markdown("#### 👀 Preview Settings")
                            with gr.Row():
                                aug_preview_rows = gr.Slider(
                                    label="Grid Rows",
                                    minimum=1, maximum=5, value=3, step=1
                                )
                                aug_preview_cols = gr.Slider(
                                    label="Grid Cols",
                                    minimum=1, maximum=5, value=3, step=1
                                )

                            aug_preview_btn = gr.Button("🔍 Generate Preview", variant="secondary", size="lg")

                            # Batch augmentation settings
                            gr.Markdown("#### 🚀 Batch Augmentation")
                            aug_multiplier = gr.Number(
                                label="Augmentation Multiplier",
                                value=5,
                                minimum=1,
                                maximum=20,
                                step=1,
                                info="Number of augmented versions per sample"
                            )

                            aug_output_dir = gr.Textbox(
                                label="Output Directory",
                                value=str(Path(self.default_output_dir) / "augmented"),
                                placeholder="Output path for augmented data"
                            )

                            aug_apply_btn = gr.Button("✨ Apply Augmentation", variant="primary", size="lg")

                        # Right column: Preview & Results
                        with gr.Column(scale=2):
                            gr.Markdown("#### 🖼️ Preview Grid")
                            aug_preview_display = gr.Image(
                                label="Augmentation Preview",
                                type="numpy",
                                height=600
                            )

                            aug_status = gr.Markdown("Load a session to start")

                            aug_progress = gr.Markdown("")

                    # Event handlers for Data Augmentation

                    # Scan sessions
                    def scan_aug_sessions(session_dir):
                        """Scan for available annotation sessions"""
                        try:
                            session_path = Path(session_dir)
                            if not session_path.exists():
                                return gr.Dropdown(choices=[]), "❌ Session directory not found"

                            # Find all session files (both session.json and session_metadata.json)
                            sessions = []

                            # Search for interactive sessions (session.json)
                            for session_file in session_path.rglob("session.json"):
                                # Verify it's an interactive session by reading the file
                                try:
                                    with open(session_file, 'r') as f:
                                        metadata = json.load(f)
                                        if metadata.get('session_type') != 'batch':
                                            session_name = session_file.parent.name
                                            sessions.append(str(session_file.parent))
                                except:
                                    # If can't read, assume it's valid
                                    session_name = session_file.parent.name
                                    sessions.append(str(session_file.parent))

                            # Also search for batch sessions (session_metadata.json) - they can also be augmented
                            for session_file in session_path.rglob("session_metadata.json"):
                                try:
                                    with open(session_file, 'r') as f:
                                        metadata = json.load(f)
                                        if metadata.get('session_type') == 'batch':
                                            session_name = session_file.parent.name
                                            sessions.append(str(session_file.parent))
                                except:
                                    pass

                            if not sessions:
                                return gr.Dropdown(choices=[]), "⚠️ No sessions found"

                            return gr.Dropdown(choices=sessions), f"✅ Found {len(sessions)} sessions"
                        except Exception as e:
                            return gr.Dropdown(choices=[]), f"❌ Error: {str(e)}"

                    aug_scan_btn.click(
                        fn=scan_aug_sessions,
                        inputs=[aug_session_dir],
                        outputs=[aug_session_list, aug_session_info]
                    )

                    # Load session
                    def load_aug_session(session_path):
                        """Load annotation session for augmentation"""
                        try:
                            if not session_path:
                                return None, "⚠️ Please select a session"

                            session_path = Path(session_path)
                            session_file = session_path / "session.json"

                            if not session_file.exists():
                                return None, f"❌ Session file not found: {session_file}"

                            # Load session metadata
                            with open(session_file, 'r') as f:
                                metadata = json.load(f)

                            # Store for augmentation
                            self.aug_session_path = session_path
                            self.aug_metadata = metadata

                            info = f"""
✅ Session loaded successfully

**Session ID:** {metadata.get('session_id', 'N/A')}
**Frames:** {metadata.get('num_frames', 0)} frames
**Created:** {metadata.get('created_at', 'N/A')}
"""
                            return None, info
                        except Exception as e:
                            return None, f"❌ Error loading session: {str(e)}"

                    aug_load_session_btn.click(
                        fn=load_aug_session,
                        inputs=[aug_session_list],
                        outputs=[aug_preview_display, aug_session_info]
                    )

                    # Generate preview
                    def generate_aug_preview(
                        rows, cols,
                        scale_enable, scale_min, scale_max, fill_color,
                        rotation_enable, rotation_min, rotation_max,
                        flip_enable,
                        noise_enable, noise_std,
                        brightness_enable, brightness_min, brightness_max,
                        contrast_enable, color_jitter_enable, blur_enable
                    ):
                        """Generate augmentation preview grid"""
                        try:
                            if not hasattr(self, 'aug_session_path'):
                                return None, "❌ Please load a session first"

                            # Load first frame and mask as example
                            rgb_files = sorted((self.aug_session_path / "rgb").glob("*.png"))
                            mask_files = sorted((self.aug_session_path / "masks").glob("*.png"))

                            if not rgb_files or not mask_files:
                                return None, "❌ No RGB or mask files found in session"

                            # Load first frame
                            rgb = cv2.imread(str(rgb_files[0]))
                            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

                            mask = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE)
                            mask = mask > 127  # Convert to boolean

                            # Build base config
                            base_config = {
                                'scale': scale_enable,
                                'rotation': rotation_enable,
                                'flip': flip_enable,
                                'noise': noise_enable,
                                'brightness': brightness_enable,
                                'contrast': contrast_enable,
                                'color_jitter': color_jitter_enable,
                                'blur': blur_enable,
                                'fill_color': fill_color
                            }

                            # Generate random configs
                            num_variations = int(rows * cols)
                            configs = []

                            import random
                            for _ in range(num_variations):
                                config = {'fill_color': fill_color}

                                if scale_enable:
                                    config['scale'] = random.uniform(scale_min, scale_max)

                                if rotation_enable:
                                    config['rotation'] = random.uniform(rotation_min, rotation_max)

                                if flip_enable and random.random() > 0.5:
                                    config['flip'] = random.choice(['horizontal', 'vertical'])

                                if noise_enable:
                                    config['noise'] = random.uniform(5, noise_std)

                                if brightness_enable:
                                    config['brightness'] = random.uniform(brightness_min, brightness_max)

                                if contrast_enable:
                                    config['contrast'] = random.uniform(0.8, 1.2)

                                if color_jitter_enable:
                                    config['color_jitter'] = True

                                if blur_enable:
                                    config['blur'] = random.choice([3, 5, 7])

                                configs.append(config)

                            # Generate preview grid
                            grid = self.augmentor.generate_preview_grid(
                                rgb, mask, configs, grid_size=(int(rows), int(cols))
                            )

                            return grid, f"✅ Preview generated with {num_variations} variations"

                        except Exception as e:
                            import traceback
                            return None, f"❌ Error: {str(e)}\n{traceback.format_exc()}"

                    aug_preview_btn.click(
                        fn=generate_aug_preview,
                        inputs=[
                            aug_preview_rows, aug_preview_cols,
                            aug_scale_enable, aug_scale_min, aug_scale_max, aug_fill_color,
                            aug_rotation_enable, aug_rotation_min, aug_rotation_max,
                            aug_flip_enable,
                            aug_noise_enable, aug_noise_std,
                            aug_brightness_enable, aug_brightness_min, aug_brightness_max,
                            aug_contrast_enable, aug_color_jitter_enable, aug_blur_enable
                        ],
                        outputs=[aug_preview_display, aug_status]
                    )

                    # Apply batch augmentation
                    def apply_batch_augmentation(
                        multiplier, output_dir,
                        scale_enable, scale_min, scale_max, fill_color,
                        rotation_enable, rotation_min, rotation_max,
                        flip_enable,
                        noise_enable, noise_std,
                        brightness_enable, brightness_min, brightness_max,
                        contrast_enable, color_jitter_enable, blur_enable
                    ):
                        """Apply augmentation to all frames in session"""
                        try:
                            if not hasattr(self, 'aug_session_path'):
                                return "❌ Please load a session first", ""

                            output_path = Path(output_dir)
                            output_path.mkdir(parents=True, exist_ok=True)

                            # Create subdirectories
                            rgb_out = output_path / "rgb"
                            mask_out = output_path / "masks"
                            rgb_out.mkdir(exist_ok=True)
                            mask_out.mkdir(exist_ok=True)

                            # Load all frames
                            rgb_files = sorted((self.aug_session_path / "rgb").glob("*.png"))
                            mask_files = sorted((self.aug_session_path / "masks").glob("*.png"))

                            total_frames = len(rgb_files)
                            total_outputs = total_frames * int(multiplier)

                            progress_msg = f"🚀 Processing {total_frames} frames × {int(multiplier)} = {total_outputs} outputs..."

                            import random
                            processed = 0

                            for idx, (rgb_file, mask_file) in enumerate(zip(rgb_files, mask_files)):
                                # Load frame
                                rgb = cv2.imread(str(rgb_file))
                                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

                                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                                mask = mask > 127

                                # Generate augmentations
                                for aug_idx in range(int(multiplier)):
                                    # Random config
                                    config = {'fill_color': fill_color}

                                    if scale_enable:
                                        config['scale'] = random.uniform(scale_min, scale_max)

                                    if rotation_enable:
                                        config['rotation'] = random.uniform(rotation_min, rotation_max)

                                    if flip_enable and random.random() > 0.5:
                                        config['flip'] = random.choice(['horizontal', 'vertical'])

                                    if noise_enable:
                                        config['noise'] = random.uniform(5, noise_std)

                                    if brightness_enable:
                                        config['brightness'] = random.uniform(brightness_min, brightness_max)

                                    if contrast_enable:
                                        config['contrast'] = random.uniform(0.8, 1.2)

                                    if color_jitter_enable:
                                        config['color_jitter'] = True

                                    if blur_enable:
                                        config['blur'] = random.choice([3, 5, 7])

                                    # Apply augmentation
                                    aug_rgb, aug_mask, applied = self.augmentor.augment(rgb, mask, config)

                                    # Save with consistent naming: frame{idx:04d}_aug{aug_idx:02d}.png
                                    output_name = f"frame{idx:04d}_aug{aug_idx:02d}.png"

                                    # Save RGB
                                    rgb_bgr = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR)
                                    cv2.imwrite(str(rgb_out / output_name), rgb_bgr)

                                    # Save mask
                                    mask_img = (aug_mask * 255).astype(np.uint8)
                                    cv2.imwrite(str(mask_out / output_name), mask_img)

                                    processed += 1

                                # Update progress
                                if (idx + 1) % 10 == 0:
                                    progress_msg = f"⏳ Processed {idx + 1}/{total_frames} frames ({processed}/{total_outputs} outputs)"

                            # Save metadata
                            metadata = {
                                'source_session': str(self.aug_session_path),
                                'original_frames': total_frames,
                                'multiplier': int(multiplier),
                                'total_augmented': processed,
                                'augmentation_config': {
                                    'scale': {'enabled': scale_enable, 'min': scale_min, 'max': scale_max} if scale_enable else None,
                                    'rotation': {'enabled': rotation_enable, 'min': rotation_min, 'max': rotation_max} if rotation_enable else None,
                                    'flip': flip_enable,
                                    'noise': {'enabled': noise_enable, 'std': noise_std} if noise_enable else None,
                                    'brightness': {'enabled': brightness_enable, 'min': brightness_min, 'max': brightness_max} if brightness_enable else None,
                                    'fill_color': fill_color
                                },
                                'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }

                            with open(output_path / "augmentation_metadata.json", 'w') as f:
                                json.dump(metadata, f, indent=2)

                            final_msg = f"""
✅ Augmentation complete!

**Original frames:** {total_frames}
**Multiplier:** {int(multiplier)}×
**Total generated:** {processed} augmented samples

**Output location:**
`{output_path}`
"""
                            return final_msg, f"Saved to: {output_path}"

                        except Exception as e:
                            import traceback
                            return f"❌ Error: {str(e)}\n{traceback.format_exc()}", ""

                    aug_apply_btn.click(
                        fn=apply_batch_augmentation,
                        inputs=[
                            aug_multiplier, aug_output_dir,
                            aug_scale_enable, aug_scale_min, aug_scale_max, aug_fill_color,
                            aug_rotation_enable, aug_rotation_min, aug_rotation_max,
                            aug_flip_enable,
                            aug_noise_enable, aug_noise_std,
                            aug_brightness_enable, aug_brightness_min, aug_brightness_max,
                            aug_contrast_enable, aug_color_jitter_enable, aug_blur_enable
                        ],
                        outputs=[aug_status, aug_progress]
                    )

        return demo

def main():
    """웹 앱 실행"""
    import os
    import socket

    app = SAMInteractiveWebApp()
    demo = app.create_interface()

    # 포트 설정: 환경 변수 또는 7860-7900 범위에서 자동 선택
    start_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))

    # 사용 가능한 포트 찾기
    def find_free_port(start, end=None):
        """Find a free port in the range [start, end]"""
        if end is None:
            end = start + 40  # 7860-7900

        for port in range(start, end + 1):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    return port
            except OSError:
                continue
        return None

    port = find_free_port(start_port)
    if port is None:
        print(f"❌ Cannot find free port in range {start_port}-{start_port + 40}")
        print("💡 Kill existing processes: pkill -f web_app.py")
        return

    print(f"✓ Using port: {port}")

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        debug=True,
        max_threads=40  # 동시 처리 증가
    )

if __name__ == "__main__":
    main()
