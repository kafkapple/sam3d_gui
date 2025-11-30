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
import logging

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Set environment variable to skip SAM3D init (which requires missing module)
os.environ['LIDRA_SKIP_INIT'] = '1'

# ==========================================
# 로깅 설정
# ==========================================
def setup_logging():
    """디버그 모드에 따른 로깅 설정"""
    debug_mode = os.environ.get('SAM3D_DEBUG', '0') == '1'

    # 로그 레벨 설정
    log_level = logging.DEBUG if debug_mode else logging.INFO

    # 포맷 설정
    if debug_mode:
        log_format = '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s'
    else:
        log_format = '%(asctime)s [%(levelname)s] %(message)s'

    # 기본 로깅 설정
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # 외부 라이브러리 로그 레벨 조정 (너무 verbose 방지)
    if not debug_mode:
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)

    logger = logging.getLogger('sam3d_gui')
    logger.setLevel(log_level)

    if debug_mode:
        logger.info("🔧 디버그 모드 활성화")
        logger.debug(f"Python: {sys.version}")
        logger.debug(f"PyTorch: {torch.__version__}")
        logger.debug(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.debug(f"CUDA device: {torch.cuda.get_device_name(0)}")
            logger.debug(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    return logger

# 로거 초기화
logger = setup_logging()

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
# Try to import SAM2 from installed package (via pip install)
try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    SAM2_AVAILABLE = True
    print("✓ SAM2 package found (installed via pip)")
except ImportError:
    # Fallback: Try legacy path-based import
    SAM2_PATH = Path.home() / 'dev/segment-anything-2'
    if SAM2_PATH.exists():
        sys.path.insert(0, str(SAM2_PATH))
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from sam2.sam2_video_predictor import SAM2VideoPredictor
            SAM2_AVAILABLE = True
            print(f"✓ SAM2 found at legacy path: {SAM2_PATH}")
        except ImportError:
            SAM2ImagePredictor = None
            SAM2VideoPredictor = None
            SAM2_AVAILABLE = False
            print("Warning: SAM 2 not found. Interactive segmentation will use fallback method.")
    else:
        SAM2ImagePredictor = None
        SAM2VideoPredictor = None
        SAM2_AVAILABLE = False
        print("Warning: SAM 2 not found. Interactive segmentation will use fallback method.")

class SAMInteractiveWebApp:
    """
    SAM 3D GUI - 통합 웹 인터페이스

    모드 1: 대화형 Annotation (Interactive Mode)
    - Point annotation (foreground/background)
    - 수동 세그멘테이션 → Propagation → 결과

    모드 2: 일괄 처리 (Batch Mode)
    - 다중 비디오 일괄 처리, 세션 관리

    모드 3: Lite Annotator
    - 효율적 단일 프레임 annotation
    """

    # SAM2 체크포인트 기본 경로
    SAM2_CHECKPOINT_PATH = Path(__file__).parent.parent / "checkpoints" / "sam2" / "sam2_hiera_large.pt"
    SAM2_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"

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

        # 현재 로드된 세션 경로 (덮어쓰기용)
        self.current_session_path = None

        # Default paths from config
        if config:
            self.default_data_dir = config.default_data_dir
            self.default_output_dir = config.output_dir
        else:
            # Fallback: data one level above project root, output inside
            project_root = Path(__file__).parent.parent
            self.default_data_dir = str(project_root.parent / "data" / "markerless_mouse")
            self.default_output_dir = str(project_root / "outputs")

        # Data Augmentor 초기화
        self.augmentor = DataAugmentor()
        self.augmentation_preview = None

        # LiteAnnotator 초기화 (Tab 3: Lite Mode)
        self.lite_annotator = None
        if SAM2_AVAILABLE:
            try:
                print("Initializing Lite Annotator...")
                # Try to find SAM2 base path
                sam2_base_path = None

                # Option 1: Legacy path
                legacy_path = Path.home() / 'dev/segment-anything-2'
                if legacy_path.exists():
                    sam2_base_path = legacy_path

                # Option 2: Use None (LiteAnnotator will use installed package)
                self.lite_annotator = LiteAnnotator(
                    sam2_base_path=sam2_base_path,
                    device=self.sam2_device if self.sam2_device else "auto"
                )
                print("✓ Lite Annotator initialized")
            except Exception as e:
                print(f"Warning: Failed to initialize Lite Annotator: {e}")
                self.lite_annotator = None

    def check_sam2_available(self) -> Tuple[bool, str]:
        """
        SAM2 모델 사용 가능 여부 확인

        Returns:
            (available, status_message)
        """
        if not SAM2_AVAILABLE:
            return False, "SAM2 패키지가 설치되지 않았습니다. `pip install sam2` 실행 필요"

        if self.sam2_predictor is None or self.sam2_video_predictor is None:
            checkpoint = self.SAM2_CHECKPOINT_PATH
            if not checkpoint.exists():
                return False, f"SAM2 체크포인트가 없습니다: {checkpoint}"
            return False, "SAM2 모델이 로드되지 않았습니다"

        return True, f"SAM2 모델 사용 가능 ({self.sam2_device})"

    def download_sam2_checkpoint(self, progress_callback=None) -> Tuple[bool, str]:
        """
        SAM2 체크포인트 다운로드

        Args:
            progress_callback: 진행률 콜백 함수 (0.0 ~ 1.0)

        Returns:
            (success, message)
        """
        import urllib.request
        import ssl

        checkpoint_path = self.SAM2_CHECKPOINT_PATH
        checkpoint_dir = checkpoint_path.parent

        # 이미 존재하면 스킵
        if checkpoint_path.exists():
            return True, f"체크포인트가 이미 존재합니다: {checkpoint_path}"

        try:
            # 디렉토리 생성
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            print(f"📥 SAM2 체크포인트 다운로드 시작...")
            print(f"   URL: {self.SAM2_DOWNLOAD_URL}")
            print(f"   저장 위치: {checkpoint_path}")

            # SSL context 설정
            ssl_context = ssl.create_default_context()

            # 진행률 표시를 위한 다운로드
            def reporthook(block_num, block_size, total_size):
                if total_size > 0:
                    downloaded = block_num * block_size
                    percent = min(downloaded / total_size, 1.0)
                    if progress_callback:
                        progress_callback(percent)
                    # 10% 단위로 출력
                    if int(percent * 10) > int((downloaded - block_size) / total_size * 10):
                        print(f"   다운로드 진행: {percent*100:.0f}%")

            urllib.request.urlretrieve(
                self.SAM2_DOWNLOAD_URL,
                str(checkpoint_path),
                reporthook=reporthook
            )

            # 파일 크기 확인
            file_size = checkpoint_path.stat().st_size / (1024 * 1024)
            print(f"✅ SAM2 다운로드 완료: {file_size:.1f} MB")

            return True, f"SAM2 체크포인트 다운로드 완료 ({file_size:.1f} MB)"

        except Exception as e:
            # 실패 시 부분 다운로드 파일 삭제
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            return False, f"다운로드 실패: {str(e)}"

    def load_sam2_models(self) -> Tuple[bool, str]:
        """
        SAM2 모델 로드 (체크포인트가 있어야 함)

        Returns:
            (success, message)
        """
        if not SAM2_AVAILABLE:
            return False, "SAM2 패키지가 설치되지 않았습니다"

        checkpoint = self.SAM2_CHECKPOINT_PATH

        # config에서 경로 가져오기 (있으면)
        if self.config:
            config_checkpoint = Path(self.config.sam2_checkpoint)
            if config_checkpoint.exists():
                checkpoint = config_checkpoint

        if not checkpoint.exists():
            return False, f"SAM2 체크포인트를 찾을 수 없습니다: {checkpoint}"

        try:
            from sam2.build_sam import build_sam2, build_sam2_video_predictor

            # Device 설정
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.sam2_device = device

            model_cfg = self.config.sam2_config if self.config else "sam2_hiera_l.yaml"

            print(f"🔄 SAM2 모델 로딩 중... (device: {device})")

            # Image predictor
            sam2_model = build_sam2(model_cfg, str(checkpoint), device=device)
            self.sam2_predictor = SAM2ImagePredictor(sam2_model)

            # Video predictor
            self.sam2_video_predictor = build_sam2_video_predictor(model_cfg, str(checkpoint), device=device)

            # Lite Annotator에 공용 predictor 전달
            if self.lite_annotator is not None:
                self.lite_annotator.set_predictor(self.sam2_predictor, "shared-large")
                print(f"  └─ Lite Annotator에 공용 predictor 전달됨")

            print(f"✅ SAM2 모델 로드 완료")
            return True, f"SAM2 모델 로드 완료 (device: {device})"

        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"SAM2 로드 실패: {str(e)}"

    def _get_sam2_status_markdown(self) -> str:
        """SAM2 상태를 Markdown 형식으로 반환 (모델 정보 포함)"""
        # 모델 정보
        model_name = "Hiera Large"
        model_size = "~897MB"

        if self.sam2_predictor is not None and self.sam2_video_predictor is not None:
            return f"✅ **SAM2** ({model_name}) - {self.sam2_device}"
        elif not SAM2_AVAILABLE:
            return f"❌ **SAM2**: 패키지 미설치 (`pip install sam2`)"
        else:
            checkpoint = self.SAM2_CHECKPOINT_PATH
            if self.config:
                config_checkpoint = Path(self.config.sam2_checkpoint)
                if config_checkpoint.exists():
                    checkpoint = config_checkpoint

            if not checkpoint.exists():
                return f"⚠️ **SAM2** ({model_name}, {model_size}) - 다운로드 필요"
            else:
                return f"⚠️ **SAM2** ({model_name}) - 버튼 클릭하여 로드"

    def ensure_sam2_ready(self, progress_callback=None) -> Tuple[bool, str]:
        """
        SAM2 모델이 준비되었는지 확인하고, 없으면 다운로드 후 로드

        Args:
            progress_callback: 진행률 콜백

        Returns:
            (success, message)
        """
        # 이미 로드되어 있으면 OK
        if self.sam2_predictor is not None and self.sam2_video_predictor is not None:
            return True, "SAM2 모델 사용 준비됨"

        if not SAM2_AVAILABLE:
            return False, "❌ SAM2 패키지가 설치되지 않았습니다.\n\n`pip install sam2` 명령으로 설치하세요."

        # 체크포인트 확인
        checkpoint = self.SAM2_CHECKPOINT_PATH
        if self.config:
            config_checkpoint = Path(self.config.sam2_checkpoint)
            if config_checkpoint.exists():
                checkpoint = config_checkpoint

        # 체크포인트 없으면 다운로드
        if not checkpoint.exists():
            print("📥 SAM2 체크포인트가 없습니다. 자동 다운로드를 시작합니다...")
            success, msg = self.download_sam2_checkpoint(progress_callback)
            if not success:
                return False, f"❌ SAM2 다운로드 실패: {msg}"

        # 모델 로드
        success, msg = self.load_sam2_models()
        return success, msg

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

    def _extract_subject_id(self, video_path: str) -> Optional[str]:
        """
        비디오 경로에서 subject ID (예: mouse_1, mouse_2) 추출

        Args:
            video_path: 비디오 파일 경로

        Returns:
            subject ID 문자열 또는 None
        """
        import re
        # mouse_1, mouse_2 등의 패턴 찾기
        match = re.search(r'mouse_(\d+)', video_path, re.IGNORECASE)
        if match:
            return f"mouse_{match.group(1)}"

        # subject_1, subject_2 패턴도 지원
        match = re.search(r'subject_(\d+)', video_path, re.IGNORECASE)
        if match:
            return f"subject_{match.group(1)}"

        return None

    def _extract_camera_id(self, video_path: str) -> Optional[str]:
        """
        비디오 경로에서 camera ID (예: Camera1, cam2) 추출

        Args:
            video_path: 비디오 파일 경로

        Returns:
            camera ID 문자열 또는 None
        """
        import re
        # Camera1, Camera2 패턴
        match = re.search(r'Camera(\d+)', video_path, re.IGNORECASE)
        if match:
            return f"cam{match.group(1)}"

        # cam1, cam2 패턴
        match = re.search(r'cam(\d+)', video_path, re.IGNORECASE)
        if match:
            return f"cam{match.group(1)}"

        # view1, view2 패턴
        match = re.search(r'view(\d+)', video_path, re.IGNORECASE)
        if match:
            return f"view{match.group(1)}"

        return None

    def _generate_unique_video_id(self, video_path: str) -> str:
        """
        비디오 경로에서 고유한 ID 생성 (mouse + camera + 파일명)

        예: /media/.../mouse_1/Camera1/0.mp4 -> "m1_cam1_0"

        Args:
            video_path: 비디오 파일 경로

        Returns:
            고유한 비디오 ID 문자열
        """
        import re
        parts = []

        # Subject ID 추출
        subject_match = re.search(r'mouse_(\d+)', video_path, re.IGNORECASE)
        if subject_match:
            parts.append(f"m{subject_match.group(1)}")
        else:
            subject_match = re.search(r'subject_(\d+)', video_path, re.IGNORECASE)
            if subject_match:
                parts.append(f"s{subject_match.group(1)}")

        # Camera ID 추출
        camera_match = re.search(r'Camera(\d+)', video_path, re.IGNORECASE)
        if camera_match:
            parts.append(f"cam{camera_match.group(1)}")
        else:
            camera_match = re.search(r'cam(\d+)', video_path, re.IGNORECASE)
            if camera_match:
                parts.append(f"cam{camera_match.group(1)}")

        # 파일명 (확장자 제외)
        filename = Path(video_path).stem
        parts.append(filename)

        if parts:
            return "_".join(parts)
        else:
            # 추출 실패 시 전체 경로 기반 해시
            return Path(video_path).stem

    def _format_video_label_with_subject(self, video_path: str, video_name: str, base_path: Path = None) -> str:
        """
        비디오 레이블 생성 (unique_id 형식: m1_cam1_frame)

        Args:
            video_path: 전체 비디오 경로
            video_name: 비디오 파일명
            base_path: 기준 경로 (상대 경로 계산용)

        Returns:
            포맷된 레이블 문자열 (예: m1_cam1_0)
        """
        unique_id = self._generate_unique_video_id(video_path)

        # unique_id가 video_name과 다르면 unique_id 사용, 같으면 상대경로 사용
        if unique_id != video_name:
            return unique_id

        if base_path:
            rel_path = str(Path(video_path).relative_to(base_path))
            return rel_path

        return video_name

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

            # CheckboxGroup 업데이트 - subject ID 포함하여 표시
            video_labels = [
                self._format_video_label_with_subject(info['path'], info['name'], data_path)
                for info in video_info
            ]

            updated_checkbox = gr.CheckboxGroup(
                choices=video_labels,
                value=video_labels,  # 기본적으로 모두 선택
                label="🎬 처리할 비디오 선택 (Subject ID + 경로)",
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
                        # frames는 RGB이므로 BGR로 변환하여 저장
                        cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

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

                            # Save frame and mask (RGB→BGR 변환)
                            cv2.imwrite(str(frame_dir / "original.png"), cv2.cvtColor(frames[frame_idx], cv2.COLOR_RGB2BGR))

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

            # per_video_annotations 저장 (있으면)
            if hasattr(self, 'per_video_annotations') and self.per_video_annotations:
                metadata['per_video_annotations'] = self.per_video_annotations
            elif 'per_video_annotations' in batch_results:
                metadata['per_video_annotations'] = batch_results['per_video_annotations']

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

                # result_dir 업데이트 (Export Fauna에서 사용)
                video_result['result_dir'] = str(video_save_dir)

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

    def generate_batch_visualization(
        self,
        session_path: str = None,
        output_format: str = "images",
        progress=None
    ) -> Tuple[str, str]:
        """
        Batch 결과의 마스크 시각화 생성

        Args:
            session_path: 세션 경로 (None이면 현재 batch_results 사용)
            output_format: "images" (개별 이미지) 또는 "video" (비디오)
            progress: Gradio progress

        Returns:
            (출력 경로, 상태 메시지)
        """
        try:
            import tempfile

            # 데이터 소스 결정
            if session_path:
                session_dir = Path(session_path)
                if not session_dir.exists():
                    return "", "❌ 세션 경로를 찾을 수 없습니다"

                # 메타데이터 로드
                metadata_path = session_dir / "session_metadata.json"
                if not metadata_path.exists():
                    return "", "❌ session_metadata.json을 찾을 수 없습니다"

                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                video_dirs = [session_dir / v['saved_dir'] for v in metadata.get('videos', [])]
            elif hasattr(self, 'batch_results') and self.batch_results:
                # 임시 결과 사용
                temp_dir = Path(self.batch_results['temp_dir'])
                video_dirs = [Path(v['result_dir']) for v in self.batch_results['videos']]
            else:
                return "", "❌ 시각화할 데이터가 없습니다. 먼저 Batch 처리를 실행하거나 세션을 로드하세요."

            # 출력 디렉토리 생성
            vis_output_dir = Path(self.default_output_dir) / "visualizations" / f"vis_{Path(tempfile.mktemp()).name[-8:]}"
            vis_output_dir.mkdir(parents=True, exist_ok=True)

            total_frames = 0
            processed_frames = 0

            # 전체 프레임 수 계산
            for video_dir in video_dirs:
                if video_dir.exists():
                    frame_dirs = [d for d in video_dir.iterdir() if d.is_dir() and d.name.startswith('frame_')]
                    total_frames += len(frame_dirs)

            if progress:
                progress(0, desc="🎨 시각화 생성 중...")

            # 각 비디오 처리
            for video_idx, video_dir in enumerate(video_dirs):
                if not video_dir.exists():
                    continue

                video_name = video_dir.name
                video_vis_dir = vis_output_dir / video_name
                video_vis_dir.mkdir(exist_ok=True)

                frame_dirs = sorted([d for d in video_dir.iterdir() if d.is_dir() and d.name.startswith('frame_')])

                for frame_dir in frame_dirs:
                    original_path = frame_dir / "original.png"
                    mask_path = frame_dir / "mask.png"

                    if not original_path.exists() or not mask_path.exists():
                        continue

                    # 이미지 로드
                    original = cv2.imread(str(original_path))
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

                    if original is None or mask is None:
                        continue

                    # 마스크 오버레이 생성 (녹색, 40% 투명도)
                    overlay = original.copy()
                    mask_bool = mask > 127
                    overlay[mask_bool] = overlay[mask_bool] * 0.6 + np.array([0, 255, 0]) * 0.4

                    # 마스크 윤곽선 추가 (빨간색)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

                    # 저장
                    vis_path = video_vis_dir / f"{frame_dir.name}_vis.png"
                    cv2.imwrite(str(vis_path), overlay.astype(np.uint8))

                    processed_frames += 1
                    if progress and total_frames > 0:
                        progress(processed_frames / total_frames, desc=f"🎨 {video_name}: {frame_dir.name}")

                # 비디오 생성 (선택적)
                if output_format == "video":
                    vis_images = sorted(video_vis_dir.glob("*_vis.png"))
                    if vis_images:
                        first_img = cv2.imread(str(vis_images[0]))
                        h, w = first_img.shape[:2]

                        video_path = vis_output_dir / f"{video_name}_visualization.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(str(video_path), fourcc, 10, (w, h))

                        for img_path in vis_images:
                            img = cv2.imread(str(img_path))
                            out.write(img)

                        out.release()

            if progress:
                progress(1.0, desc="✅ 시각화 완료!")

            status = f"""
### 🎨 시각화 생성 완료 ✅

- **출력 경로**: `{vis_output_dir}`
- **처리된 프레임**: {processed_frames}개
- **비디오 수**: {len(video_dirs)}개
- **형식**: {output_format}

각 비디오 폴더에서 `*_vis.png` 파일을 확인하세요.
녹색 영역이 마스크, 빨간 윤곽선이 경계입니다.
"""

            return str(vis_output_dir), status

        except Exception as e:
            import traceback
            return "", f"❌ 시각화 실패: {str(e)}\n{traceback.format_exc()}"

    def get_batch_frame_list(self) -> List[Dict]:
        """
        Batch 결과의 전체 프레임 목록 반환 (슬라이더용)

        Returns:
            프레임 정보 리스트 [{video_idx, video_name, frame_idx, frame_dir}, ...]
        """
        frame_list = []

        if not hasattr(self, 'batch_results') or not self.batch_results:
            return frame_list

        for video_result in self.batch_results['videos']:
            video_dir = Path(video_result['result_dir'])
            video_name = video_result['video_name']
            video_idx = video_result['video_idx']

            if not video_dir.exists():
                continue

            frame_dirs = sorted([d for d in video_dir.iterdir() if d.is_dir() and d.name.startswith('frame_')])

            for frame_dir in frame_dirs:
                frame_idx = int(frame_dir.name.split('_')[1])
                frame_list.append({
                    'video_idx': video_idx,
                    'video_name': video_name,
                    'frame_idx': frame_idx,
                    'frame_dir': str(frame_dir)
                })

        return frame_list

    def get_visualization_frame(self, global_idx: int) -> Tuple[np.ndarray, str]:
        """
        특정 인덱스의 시각화 프레임 반환 (슬라이더용)

        Args:
            global_idx: 전체 프레임 리스트에서의 인덱스

        Returns:
            (시각화 이미지, 상태 텍스트)
        """
        frame_list = self.get_batch_frame_list()

        if not frame_list:
            return None, "결과가 없습니다. 먼저 Batch Propagate를 실행하세요."

        if global_idx < 0 or global_idx >= len(frame_list):
            return None, f"유효하지 않은 인덱스: {global_idx}"

        frame_info = frame_list[global_idx]
        frame_dir = Path(frame_info['frame_dir'])

        original_path = frame_dir / "original.png"
        mask_path = frame_dir / "mask.png"

        if not original_path.exists() or not mask_path.exists():
            return None, f"프레임 파일을 찾을 수 없습니다: {frame_dir}"

        # 이미지 로드
        original = cv2.imread(str(original_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if original is None or mask is None:
            return None, "이미지 로드 실패"

        # BGR → RGB 변환
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        # 마스크 오버레이 생성 (녹색, 40% 투명도)
        overlay = original.copy().astype(np.float32)
        mask_bool = mask > 127
        overlay[mask_bool] = overlay[mask_bool] * 0.6 + np.array([0, 255, 0]) * 0.4

        # 마스크 윤곽선 추가 (빨간색)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)

        status = f"📹 **{frame_info['video_name']}** | 🖼️ Frame {frame_info['frame_idx']} | ({global_idx + 1}/{len(frame_list)})"

        return overlay.astype(np.uint8), status

    # ========== Per-Video Annotation Support ==========

    def _draw_points_on_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        프레임에 현재 annotation points 표시

        Args:
            frame: RGB 프레임 이미지

        Returns:
            points가 표시된 이미지
        """
        frame_with_points = frame.copy()

        # Foreground points (녹색)
        for px, py in self.annotations['foreground']:
            cv2.circle(frame_with_points, (px, py), 5, (0, 255, 0), -1)
            cv2.circle(frame_with_points, (px, py), 7, (255, 255, 255), 2)

        # Background points (빨간색)
        for px, py in self.annotations['background']:
            cv2.circle(frame_with_points, (px, py), 5, (255, 0, 0), -1)
            cv2.circle(frame_with_points, (px, py), 7, (255, 255, 255), 2)

        return frame_with_points

    def init_per_video_annotations(self):
        """비디오별 annotation 저장소 초기화"""
        if not hasattr(self, 'per_video_annotations'):
            self.per_video_annotations = {}

    def save_current_annotation_for_video(self, video_label: str) -> str:
        """
        현재 annotation을 특정 비디오용으로 저장

        Args:
            video_label: 비디오 레이블 (UI에서 선택한 것)

        Returns:
            상태 메시지
        """
        self.init_per_video_annotations()

        if len(self.annotations['foreground']) == 0:
            return f"❌ Annotation이 없습니다. 먼저 foreground point를 추가하세요."

        # 레이블 → 경로 변환
        if hasattr(self, 'batch_video_label_map') and video_label in self.batch_video_label_map:
            video_path = self.batch_video_label_map[video_label]
        else:
            video_path = video_label

        self.per_video_annotations[video_path] = {
            'foreground': self.annotations['foreground'].copy(),
            'background': self.annotations['background'].copy(),
            'video_label': video_label
        }

        fg_count = len(self.annotations['foreground'])
        bg_count = len(self.annotations['background'])

        return f"✅ **{video_label}** annotation 저장됨 (FG: {fg_count}, BG: {bg_count})"

    def load_video_for_annotation(self, video_label: str) -> Tuple[np.ndarray, str]:
        """
        특정 비디오의 첫 프레임을 로드하고 기존 annotation 복원

        Args:
            video_label: 비디오 레이블

        Returns:
            (프레임 이미지, 상태 메시지)
        """
        self.init_per_video_annotations()

        if not hasattr(self, 'batch_video_label_map'):
            return None, "❌ 먼저 비디오를 스캔하세요."

        if video_label not in self.batch_video_label_map:
            return None, f"❌ 비디오를 찾을 수 없습니다: {video_label}"

        video_path = self.batch_video_label_map[video_label]

        # 첫 프레임 추출
        frames = self.processor.extract_frames(video_path, 0, 1, stride=1)
        if not frames:
            return None, f"❌ 프레임 추출 실패: {video_label}"

        # 현재 프레임 설정
        self.frames = frames
        self.current_frame_idx = 0

        # 기존 annotation 복원 (있으면)
        if video_path in self.per_video_annotations:
            saved = self.per_video_annotations[video_path]
            self.annotations = {
                'foreground': saved['foreground'].copy(),
                'background': saved['background'].copy()
            }
            status = f"📹 **{video_label}** 로드 완료 (기존 annotation 복원됨)"
        else:
            # 새 비디오면 annotation 초기화
            self.annotations = {'foreground': [], 'background': []}
            status = f"📹 **{video_label}** 로드 완료 (새 annotation)"

        # 현재 annotation 표시
        frame_with_points = self._draw_points_on_frame(frames[0])

        return frame_with_points, status

    def get_per_video_annotation_status(self) -> str:
        """비디오별 annotation 상태 반환"""
        self.init_per_video_annotations()

        if not self.per_video_annotations:
            return "### 📋 비디오별 Annotation: 없음"

        lines = ["### 📋 비디오별 Annotation 현황\n"]
        for video_path, anno in self.per_video_annotations.items():
            label = anno.get('video_label', Path(video_path).name)
            fg = len(anno['foreground'])
            bg = len(anno['background'])
            lines.append(f"- **{label}**: FG {fg}개, BG {bg}개")

        return "\n".join(lines)

    def save_per_video_annotations_to_file(self, filename: str = "") -> Tuple[str, str]:
        """
        비디오별 annotation을 JSON 파일로 저장 (propagation 전에도 사용 가능)

        Args:
            filename: 파일 이름 (비어있으면 자동 생성)

        Returns:
            (저장 경로, 상태 메시지)
        """
        self.init_per_video_annotations()

        if not self.per_video_annotations:
            return "", "❌ 저장할 비디오별 annotation이 없습니다."

        try:
            from datetime import datetime
            import json

            # 저장 경로 설정
            annotations_dir = Path(self.default_output_dir) / "annotations"
            annotations_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if filename and filename.strip():
                save_filename = f"{filename.strip()}_{timestamp}.json"
            else:
                save_filename = f"per_video_annotations_{timestamp}.json"

            save_path = annotations_dir / save_filename

            # 저장 데이터 구성
            save_data = {
                'timestamp': timestamp,
                'num_videos': len(self.per_video_annotations),
                'per_video_annotations': self.per_video_annotations
            }

            with open(save_path, 'w') as f:
                json.dump(save_data, f, indent=2)

            status = f"""
### 💾 비디오별 Annotation 저장 완료 ✅

- **파일**: `{save_path}`
- **비디오 수**: {len(self.per_video_annotations)}개

나중에 **Annotation 로드** 버튼으로 불러올 수 있습니다.
"""
            return str(save_path), status

        except Exception as e:
            import traceback
            return "", f"❌ 저장 실패: {str(e)}\n{traceback.format_exc()}"

    def load_per_video_annotations_from_file(self, filepath: str) -> Tuple[str, str]:
        """
        저장된 비디오별 annotation JSON 파일 로드

        Args:
            filepath: JSON 파일 경로

        Returns:
            (상태 텍스트, annotation 상태)
        """
        try:
            import json

            filepath = Path(filepath)
            if not filepath.exists():
                return "❌ 파일을 찾을 수 없습니다.", self.get_per_video_annotation_status()

            with open(filepath, 'r') as f:
                data = json.load(f)

            if 'per_video_annotations' not in data:
                return "❌ 유효하지 않은 annotation 파일입니다.", self.get_per_video_annotation_status()

            self.per_video_annotations = data['per_video_annotations']

            status = f"""
### 📂 비디오별 Annotation 로드 완료 ✅

- **파일**: `{filepath}`
- **비디오 수**: {len(self.per_video_annotations)}개 복원됨

이제 **비디오별 Batch Propagate**를 실행할 수 있습니다.
"""
            return status, self.get_per_video_annotation_status()

        except Exception as e:
            import traceback
            return f"❌ 로드 실패: {str(e)}", self.get_per_video_annotation_status()

    def scan_annotation_files(self) -> List[str]:
        """저장된 annotation 파일 목록 스캔"""
        annotations_dir = Path(self.default_output_dir) / "annotations"
        if not annotations_dir.exists():
            return []

        files = sorted(annotations_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        return [str(f) for f in files]

    # ========== Preview Video Generation ==========

    def get_batch_video_list(self) -> List[Dict]:
        """
        Batch 결과의 비디오 목록 반환

        Returns:
            비디오 정보 리스트 [{video_idx, video_name, video_path, result_dir, num_frames, subject_id}, ...]
        """
        if not hasattr(self, 'batch_results') or not self.batch_results:
            return []

        video_list = []
        for video_result in self.batch_results['videos']:
            video_dir = Path(video_result['result_dir'])
            if video_dir.exists():
                num_frames = len(list(video_dir.glob("frame_*")))
                video_path = video_result.get('video_path', '')
                subject_id = self._extract_subject_id(video_path)
                unique_id = self._generate_unique_video_id(video_path)
                video_list.append({
                    'video_idx': video_result['video_idx'],
                    'video_name': video_result['video_name'],
                    'video_path': video_path,
                    'result_dir': str(video_dir),
                    'num_frames': num_frames,
                    'subject_id': subject_id,
                    'unique_id': unique_id
                })
        return video_list

    def get_video_frame_for_preview(
        self,
        video_idx: int,
        frame_idx: int,
        display_mode: str = "overlay"
    ) -> Tuple[np.ndarray, str]:
        """
        특정 비디오의 특정 프레임 반환 (프리뷰용)

        Args:
            video_idx: 비디오 인덱스
            frame_idx: 프레임 인덱스
            display_mode: "mask", "overlay", "side_by_side"

        Returns:
            (이미지, 상태 텍스트)
        """
        video_list = self.get_batch_video_list()

        if not video_list:
            return None, "결과가 없습니다."

        # video_idx로 비디오 찾기
        video_info = None
        for v in video_list:
            if v['video_idx'] == video_idx:
                video_info = v
                break

        if video_info is None:
            return None, f"비디오 {video_idx}를 찾을 수 없습니다."

        video_dir = Path(video_info['result_dir'])
        frame_dirs = sorted([d for d in video_dir.iterdir() if d.is_dir() and d.name.startswith('frame_')])

        if frame_idx < 0 or frame_idx >= len(frame_dirs):
            return None, f"유효하지 않은 프레임 인덱스: {frame_idx}"

        frame_dir = frame_dirs[frame_idx]
        original_path = frame_dir / "original.png"
        mask_path = frame_dir / "mask.png"

        if not original_path.exists() or not mask_path.exists():
            return None, f"프레임 파일 없음: {frame_dir}"

        # 이미지 로드
        original = cv2.imread(str(original_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if original is None or mask is None:
            return None, "이미지 로드 실패"

        # BGR → RGB
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        # 디스플레이 모드에 따라 출력
        if display_mode == "mask":
            # Binary mask (3채널로 변환)
            result = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        elif display_mode == "overlay":
            # 마스크 오버레이 (녹색, 40% 투명도)
            result = original.copy().astype(np.float32)
            mask_bool = mask > 127
            result[mask_bool] = result[mask_bool] * 0.6 + np.array([0, 255, 0]) * 0.4
            # 윤곽선 추가 (빨간색)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, (255, 0, 0), 2)
            result = result.astype(np.uint8)

        elif display_mode == "side_by_side":
            # 원본 | 마스크 | 오버레이 (3개 나란히, 저해상도)
            h, w = original.shape[:2]
            scale = min(1.0, 400 / w)  # 최대 너비 400px
            new_w, new_h = int(w * scale), int(h * scale)

            orig_small = cv2.resize(original, (new_w, new_h))
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            mask_small = cv2.resize(mask_rgb, (new_w, new_h))

            overlay = original.copy().astype(np.float32)
            mask_bool = mask > 127
            overlay[mask_bool] = overlay[mask_bool] * 0.6 + np.array([0, 255, 0]) * 0.4
            overlay_small = cv2.resize(overlay.astype(np.uint8), (new_w, new_h))

            result = np.hstack([orig_small, mask_small, overlay_small])

        else:
            result = original

        status = f"📹 **{video_info['video_name']}** | Frame {frame_idx + 1}/{len(frame_dirs)}"
        return result, status

    def generate_preview_video(
        self,
        video_idx: int,
        display_mode: str = "overlay",
        fps: int = 15,
        scale: float = 0.5,
        progress=None
    ) -> Tuple[str, str]:
        """
        특정 비디오의 프리뷰 영상 생성 (저해상도, 빠른 확인용)

        Args:
            video_idx: 비디오 인덱스
            display_mode: "mask", "overlay", "side_by_side"
            fps: 프레임 레이트
            scale: 해상도 스케일 (0.25 ~ 1.0)

        Returns:
            (비디오 경로, 상태 메시지)
        """
        video_list = self.get_batch_video_list()

        if not video_list:
            return "", "결과가 없습니다. 먼저 Batch Propagate를 실행하세요."

        # video_idx로 비디오 찾기
        video_info = None
        for v in video_list:
            if v['video_idx'] == video_idx:
                video_info = v
                break

        if video_info is None:
            return "", f"비디오 {video_idx}를 찾을 수 없습니다."

        try:
            video_dir = Path(video_info['result_dir'])
            frame_dirs = sorted([d for d in video_dir.iterdir() if d.is_dir() and d.name.startswith('frame_')])

            if not frame_dirs:
                return "", "프레임이 없습니다."

            # 출력 경로
            preview_dir = Path(self.default_output_dir) / "previews"
            preview_dir.mkdir(parents=True, exist_ok=True)

            # unique_id 사용 (m1_cam1_0 형식)으로 72개 비디오 모두 구분 가능
            unique_id = video_info.get('unique_id')
            if not unique_id:
                # fallback: video_path에서 unique_id 생성
                video_path = video_info.get('video_path', '')
                unique_id = self._generate_unique_video_id(video_path) if video_path else Path(video_info['video_name']).stem
            output_path = preview_dir / f"{unique_id}_{display_mode}_preview.mp4"

            # 첫 프레임으로 크기 결정
            first_frame, _ = self.get_video_frame_for_preview(video_idx, 0, display_mode)
            if first_frame is None:
                return "", "첫 프레임 로드 실패"

            h, w = first_frame.shape[:2]
            new_w, new_h = int(w * scale), int(h * scale)
            # 짝수로 맞추기 (코덱 요구사항)
            new_w = new_w if new_w % 2 == 0 else new_w + 1
            new_h = new_h if new_h % 2 == 0 else new_h + 1

            # VideoWriter 설정
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (new_w, new_h))

            if progress:
                progress(0, desc=f"🎬 프리뷰 생성 중: {unique_id}")

            for i, frame_dir in enumerate(frame_dirs):
                frame, _ = self.get_video_frame_for_preview(video_idx, i, display_mode)
                if frame is not None:
                    # 리사이즈 및 BGR 변환
                    frame_resized = cv2.resize(frame, (new_w, new_h))
                    frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)

                if progress:
                    progress((i + 1) / len(frame_dirs), desc=f"🎬 {unique_id}: {i+1}/{len(frame_dirs)}")

            out.release()

            if progress:
                progress(1.0, desc="✅ 프리뷰 생성 완료")

            status = f"""
### 🎬 프리뷰 영상 생성 완료 ✅

- **비디오**: {unique_id} ({video_info['video_name']})
- **모드**: {display_mode}
- **프레임 수**: {len(frame_dirs)}
- **FPS**: {fps}
- **해상도**: {new_w}x{new_h} (원본의 {int(scale*100)}%)
- **파일**: `{output_path}`
"""
            return str(output_path), status

        except Exception as e:
            import traceback
            return "", f"❌ 프리뷰 생성 실패: {str(e)}\n{traceback.format_exc()}"

    def batch_propagate_with_per_video_annotations(
        self,
        target_frames: int = 100,
        selected_videos: List[str] = None,
        progress=gr.Progress()
    ) -> Tuple[str, str]:
        """
        비디오별 개별 annotation을 사용한 Batch Propagation

        각 비디오마다 해당 비디오에 저장된 annotation 사용.
        annotation이 없는 비디오는 기본 reference annotation 사용.
        """
        self.init_per_video_annotations()

        if not hasattr(self, 'batch_videos') or not self.batch_videos:
            return "먼저 비디오를 스캔하세요", "❌ 비디오 없음"

        # 기본 reference annotation (현재 UI에 있는 것)
        default_annotations = {
            'foreground': self.annotations['foreground'].copy(),
            'background': self.annotations['background'].copy()
        }

        # per-video annotation 사용 가능한 비디오 수 확인
        if not self.per_video_annotations and len(default_annotations['foreground']) == 0:
            return "Annotation이 필요합니다. 최소 1개의 foreground point가 필요합니다.", "❌ Annotation 없음"

        try:
            import tempfile
            import shutil
            import torch

            batch_temp_dir = Path(tempfile.mkdtemp(prefix="sam3d_batch_"))

            # 선택된 비디오 필터링
            if selected_videos and len(selected_videos) > 0:
                videos_to_process = []
                if hasattr(self, 'batch_video_label_map'):
                    for label in selected_videos:
                        if label in self.batch_video_label_map:
                            videos_to_process.append(self.batch_video_label_map[label])
            else:
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

                # 해당 비디오의 annotation 선택
                if video_path in self.per_video_annotations:
                    video_annotations = self.per_video_annotations[video_path]
                    print(f"📹 {video_name}: 개별 annotation 사용")
                else:
                    video_annotations = default_annotations
                    print(f"📹 {video_name}: 기본 annotation 사용")

                if len(video_annotations['foreground']) == 0:
                    print(f"⚠️ {video_name}: annotation 없음, 건너뜀")
                    continue

                # 비디오 정보 찾기
                matching_info = None
                for info in self.batch_video_info:
                    if info['path'] == video_path:
                        matching_info = info
                        break

                if matching_info is None:
                    continue

                num_frames = matching_info['frames']
                calculated_stride = max(1, num_frames // target_frames)

                # 프레임 추출
                frames = self.processor.extract_frames(video_path, 0, num_frames, stride=calculated_stride)
                if not frames:
                    continue

                # 임시 디렉토리에 프레임 저장
                video_temp_dir = tempfile.mkdtemp(prefix=f"sam3d_video_{video_idx}_")

                try:
                    for idx, frame in enumerate(frames):
                        frame_path = Path(video_temp_dir) / f"{idx:05d}.jpg"
                        cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                    # SAM 2 inference
                    if self.sam2_video_predictor is not None:
                        inference_state = self.sam2_video_predictor.init_state(video_path=video_temp_dir)

                        # 해당 비디오의 annotations 적용
                        point_coords = []
                        point_labels = []

                        for px, py in video_annotations['foreground']:
                            point_coords.append([px, py])
                            point_labels.append(1)

                        for px, py in video_annotations['background']:
                            point_coords.append([px, py])
                            point_labels.append(0)

                        point_coords = np.array(point_coords, dtype=np.float32)
                        point_labels = np.array(point_labels, dtype=np.int32)

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

                        # 결과 저장
                        video_result_dir = batch_temp_dir / f"video_{video_idx:03d}"
                        video_result_dir.mkdir(exist_ok=True)

                        for frame_idx, mask in video_segments.items():
                            frame_dir = video_result_dir / f"frame_{frame_idx:04d}"
                            frame_dir.mkdir(exist_ok=True)
                            cv2.imwrite(str(frame_dir / "original.png"), cv2.cvtColor(frames[frame_idx], cv2.COLOR_RGB2BGR))
                            mask_uint8 = mask.squeeze().astype(np.uint8) * 255
                            cv2.imwrite(str(frame_dir / "mask.png"), mask_uint8)

                        total_processed_frames += len(video_segments)

                        video_results.append({
                            'video_idx': video_idx,
                            'video_name': video_name,
                            'video_path': video_path,
                            'frames': len(video_segments),
                            'result_dir': str(video_result_dir),
                            'annotation_type': 'per_video' if video_path in self.per_video_annotations else 'default'
                        })

                finally:
                    shutil.rmtree(video_temp_dir, ignore_errors=True)
                    if 'inference_state' in locals():
                        del inference_state
                    if 'video_segments' in locals():
                        del video_segments
                    del frames
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    import gc
                    gc.collect()

            # 결과 저장
            self.batch_results = {
                'temp_dir': str(batch_temp_dir),
                'videos': video_results,
                'total_frames': total_processed_frames,
                'target_frames': target_frames,
                'reference_annotations': default_annotations,
                'per_video_annotations': {k: v for k, v in self.per_video_annotations.items()}
            }

            progress(1.0, desc="Batch 처리 완료!")

            # 개별 annotation 사용 비디오 수 카운트
            per_video_count = sum(1 for v in video_results if v.get('annotation_type') == 'per_video')
            default_count = len(video_results) - per_video_count

            status = f"""
### 🎉 Batch Propagation 완료 (비디오별 Annotation) ✅

- **처리된 비디오**: {len(video_results)} / {total_videos}
  - 개별 annotation: {per_video_count}개
  - 기본 annotation: {default_count}개
- **총 프레임 수**: {total_processed_frames}
- **임시 저장 위치**: {batch_temp_dir}

### 다음 단계:
- **결과 확인**: 슬라이더로 프레임별 마스크 확인
- **Export to Fauna**: 통합 데이터셋 생성
"""

            return status, "✅ 완료"

        except Exception as e:
            import traceback
            return f"❌ Batch 처리 실패:\n{str(e)}\n{traceback.format_exc()}", "❌ 실패"

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

            # per_video_annotations 복원 (있으면)
            if 'per_video_annotations' in metadata:
                self.per_video_annotations = metadata['per_video_annotations']
                print(f"  ✓ 비디오별 annotation {len(self.per_video_annotations)}개 복원됨")
            else:
                self.init_per_video_annotations()

            print(f"\n✅ Batch 세션 로드 완료!")

            # per_video_annotations 수 확인
            per_video_count = len(self.per_video_annotations) if hasattr(self, 'per_video_annotations') else 0

            # 비디오 목록을 접을 수 있게 구성
            video_list_items = []
            for video_result in video_results:
                video_path = video_result.get('video_path', '')
                unique_id = self._generate_unique_video_id(video_path) if video_path else video_result['video_name']
                video_list_items.append(f"- **{unique_id}**: {video_result['frames']} 프레임")

            video_list_str = "\n".join(video_list_items)

            status = f"""
### 📂 Batch 세션 로드 완료 ✅

- **세션 ID**: `{metadata['session_id']}`
- **로드 경로**: `{session_dir}`
- **비디오 수**: {len(video_results)}
- **총 프레임 수**: {metadata['total_frames']}
- **목표 프레임 수**: {metadata['target_frames']} (각 비디오당)
- **비디오별 Annotation**: {per_video_count}개 복원됨

<details>
<summary><b>📋 로드된 비디오 목록 ({len(video_results)}개) - 클릭하여 펼치기/접기</b></summary>

{video_list_str}

</details>

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

    def delete_batch_session(self, session_path: str) -> Tuple[str, List[str]]:
        """
        Batch 세션 삭제

        Args:
            session_path: 세션 디렉토리 경로

        Returns:
            (상태 메시지, 업데이트된 세션 목록)
        """
        import shutil

        if not session_path:
            return "❌ 삭제할 세션을 선택하세요", []

        session_dir = Path(session_path)
        if not session_dir.exists():
            return f"❌ 세션 디렉토리가 존재하지 않습니다: {session_path}", []

        # 메타데이터 확인
        metadata_path = session_dir / "session_metadata.json"
        if not metadata_path.exists():
            return f"❌ 유효한 세션 디렉토리가 아닙니다: {session_path}", []

        try:
            session_name = session_dir.name
            shutil.rmtree(session_dir)
            print(f"🗑️ 세션 삭제됨: {session_dir}")

            # 세션 목록 새로고침
            sessions_dir = Path(self.default_output_dir) / "sessions"
            sessions = []
            if sessions_dir.exists():
                for s_dir in sessions_dir.iterdir():
                    if s_dir.is_dir() and (s_dir / "session_metadata.json").exists():
                        sessions.append(str(s_dir))

            return f"✅ 세션 '{session_name}' 삭제됨", sorted(sessions, reverse=True)

        except Exception as e:
            return f"❌ 세션 삭제 실패: {str(e)}", []

    def rename_batch_session(self, session_path: str, new_name: str) -> Tuple[str, List[str]]:
        """
        Batch 세션 이름 변경

        Args:
            session_path: 세션 디렉토리 경로
            new_name: 새 세션 이름

        Returns:
            (상태 메시지, 업데이트된 세션 목록)
        """
        import json

        if not session_path:
            return "❌ 이름을 변경할 세션을 선택하세요", []

        if not new_name or not new_name.strip():
            return "❌ 새 이름을 입력하세요", []

        new_name = new_name.strip()

        # 유효한 파일명 문자만 허용
        invalid_chars = '<>:"/\\|?*'
        if any(c in new_name for c in invalid_chars):
            return f"❌ 세션 이름에 사용할 수 없는 문자가 있습니다: {invalid_chars}", []

        session_dir = Path(session_path)
        if not session_dir.exists():
            return f"❌ 세션 디렉토리가 존재하지 않습니다: {session_path}", []

        # 메타데이터 확인
        metadata_path = session_dir / "session_metadata.json"
        if not metadata_path.exists():
            return f"❌ 유효한 세션 디렉토리가 아닙니다: {session_path}", []

        try:
            old_name = session_dir.name
            new_session_dir = session_dir.parent / new_name

            if new_session_dir.exists():
                return f"❌ 이미 같은 이름의 세션이 존재합니다: {new_name}", []

            # 디렉토리 이름 변경
            session_dir.rename(new_session_dir)

            # 메타데이터 업데이트
            new_metadata_path = new_session_dir / "session_metadata.json"
            with open(new_metadata_path, 'r') as f:
                metadata = json.load(f)
            metadata['session_id'] = new_name
            with open(new_metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"✏️ 세션 이름 변경: {old_name} → {new_name}")

            # 세션 목록 새로고침
            sessions_dir = Path(self.default_output_dir) / "sessions"
            sessions = []
            if sessions_dir.exists():
                for s_dir in sessions_dir.iterdir():
                    if s_dir.is_dir() and (s_dir / "session_metadata.json").exists():
                        sessions.append(str(s_dir))

            return f"✅ 세션 이름 변경됨: '{old_name}' → '{new_name}'", sorted(sessions, reverse=True)

        except Exception as e:
            return f"❌ 세션 이름 변경 실패: {str(e)}", []

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

            # 첫 프레임 반환 + 슬라이더 업데이트 (self.frames는 이미 RGB)
            frame_rgb = self.frames[0].copy()

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

        # 현재 프레임에 point 표시 (self.frames는 이미 RGB)
        frame_rgb = self.frames[self.current_frame_idx].copy()

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
        현재 프레임을 SAM2로 세그멘테이션
        SAM2 모델이 필수이며, 없으면 다운로드 안내 표시
        """
        if len(self.frames) == 0:
            return None, "❌ 먼저 비디오를 로드하세요"

        if len(self.annotations['foreground']) == 0:
            return None, "❌ 최소 1개의 foreground point가 필요합니다"

        # SAM2 모델 확인 - 없으면 에러
        if self.sam2_predictor is None:
            checkpoint = self.SAM2_CHECKPOINT_PATH
            if not checkpoint.exists():
                return None, f"""❌ **SAM2 모델이 필요합니다**

SAM2 체크포인트가 없습니다.
상단의 **🔄 SAM2 모델 다운로드** 버튼을 클릭하세요.

또는 터미널에서:
```
./download_checkpoints.sh
```

예상 경로: `{checkpoint}`
"""
            else:
                return None, """❌ **SAM2 모델이 로드되지 않았습니다**

상단의 **🔄 SAM2 모델 다운로드** 버튼을 클릭하여 모델을 로드하세요.
"""

        try:
            # self.frames는 이미 RGB
            frame_rgb = self.frames[self.current_frame_idx]

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

                    # stride 간격으로만 프레임 저장 (self.frames는 RGB이므로 BGR로 변환)
                    for idx, i in enumerate(frame_indices):
                        frame_path = os.path.join(temp_dir, f"{idx:05d}.jpg")
                        cv2.imwrite(frame_path, cv2.cvtColor(self.frames[i], cv2.COLOR_RGB2BGR))

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

            # 현재 프레임 시각화 (self.frames는 이미 RGB)
            self.current_frame_idx = min(self.current_frame_idx, len(self.frames) - 1)
            current_frame = self.frames[self.current_frame_idx]
            current_mask = self.masks[self.current_frame_idx]

            frame_rgb = current_frame.copy()  # 이미 RGB이므로 변환 불필요
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
            # Fallback: relative to project root (통합 구조)
            project_root = Path(__file__).parent.parent
            checkpoint_dir = project_root / "checkpoints" / "sam3d"

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

    def generate_3d_mesh(
        self,
        seed: int = 42,
        stage1_steps: int = 25,
        stage2_steps: int = 25,
        with_postprocess: bool = False,
        simplify_ratio: float = 0.95,
        with_texture_baking: bool = False,
        texture_size: int = 1024,
        use_vertex_color: bool = True,
        progress=gr.Progress()
    ) -> Tuple[str, str]:
        """
        세그멘테이션 결과로 3D mesh 생성

        Args:
            seed: 랜덤 시드 (재현성)
            stage1_steps: Stage 1 diffusion steps
            stage2_steps: Stage 2 diffusion steps
            with_postprocess: 후처리 활성화
            simplify_ratio: Face 유지 비율
            with_texture_baking: 텍스처 베이킹
            texture_size: 텍스처 해상도
            use_vertex_color: 버텍스 컬러 사용
        """
        # 메시 파라미터 설정 저장
        mesh_settings = {
            "seed": int(seed),
            "stage1_inference_steps": int(stage1_steps),
            "stage2_inference_steps": int(stage2_steps),
            "with_mesh_postprocess": with_postprocess,
            "simplify_ratio": float(simplify_ratio),
            "with_texture_baking": with_texture_baking,
            "texture_size": int(texture_size),
            "use_vertex_color": use_vertex_color
        }
        logger.info("=" * 60)
        logger.info("🔹 generate_3d_mesh() 시작")
        logger.info("=" * 60)

        # GPU 메모리 상태 로깅
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.debug(f"GPU 메모리: {mem_allocated:.2f}GB / {mem_total:.2f}GB (reserved: {mem_reserved:.2f}GB)")

        if len(self.frames) == 0 or all(m is None for m in self.masks):
            logger.error("❌ 프레임 또는 마스크 없음")
            return None, "먼저 세그멘테이션을 완료하세요"

        logger.debug(f"프레임 수: {len(self.frames)}, 마스크 수: {sum(1 for m in self.masks if m is not None)}")

        try:
            progress(0, desc="3D mesh 생성 준비 중...")

            # SAM 3D 체크포인트 확인
            if self.config:
                checkpoint_dir = Path(self.config.sam3d_checkpoint_dir).expanduser()
                logger.info(f"✓ Config에서 checkpoint 경로 로드: {checkpoint_dir}")
            else:
                # Fallback: relative to project root (통합 구조)
                project_root = Path(__file__).parent.parent
                checkpoint_dir = project_root / "checkpoints" / "sam3d"
                logger.info(f"✓ 기본 checkpoint 경로 사용: {checkpoint_dir}")

            logger.info(f"✓ Checkpoint 존재 확인 중: {checkpoint_dir}")
            logger.debug(f"   checkpoint_dir.exists(): {checkpoint_dir.exists()}")
            logger.debug(f"   pipeline.yaml 존재: {(checkpoint_dir / 'pipeline.yaml').exists()}")

            # 체크포인트 파일 목록 로깅
            if checkpoint_dir.exists():
                ckpt_files = list(checkpoint_dir.glob("*.ckpt"))
                logger.debug(f"   .ckpt 파일 수: {len(ckpt_files)}")
                for f in ckpt_files[:5]:  # 처음 5개만
                    logger.debug(f"     - {f.name}: {f.stat().st_size / 1024**2:.1f} MB")

            if not (checkpoint_dir / "pipeline.yaml").exists():
                logger.error("❌ pipeline.yaml 파일이 없음")
                progress(0.1, desc="SAM 3D 체크포인트 없음, 다운로드 시작...")

                download_success = self.download_sam3d_checkpoint(progress)

                if not download_success:
                    return None, """
### ❌ SAM 3D 체크포인트 다운로드 실패

**수동 다운로드 방법:**
```bash
# 프로젝트 루트에서 실행
./download_checkpoints.sh
```

**필요한 설정:**
1. `.env` 파일에 HuggingFace 토큰 설정: `HF_TOKEN=your_token`
2. Git LFS 설치: `sudo apt install git-lfs`
"""

            # 현재 선택된 프레임 사용
            frame_idx = self.current_frame_idx
            frame = self.frames[frame_idx]
            mask = self.masks[frame_idx] if frame_idx < len(self.masks) else None

            logger.info(f"✓ 현재 프레임 선택: {frame_idx + 1}/{len(self.frames)}")
            logger.debug(f"   Frame shape: {frame.shape}, dtype: {frame.dtype}")
            logger.debug(f"   Mask shape: {mask.shape if mask is not None else 'None'}")
            logger.debug(f"   Mask type: {type(mask)}, unique values: {np.unique(mask) if mask is not None else 'N/A'}")

            if mask is None:
                logger.error(f"❌ 프레임 {frame_idx + 1}에 마스크 없음")
                return None, f"프레임 {frame_idx + 1}에 마스크가 없습니다. 먼저 세그멘테이션을 수행하세요."

            # 3D 재구성 시도
            logger.info("✓ 3D 재구성 시작...")
            logger.info(f"   Mesh 설정: seed={mesh_settings['seed']}, steps={mesh_settings['stage1_inference_steps']}/{mesh_settings['stage2_inference_steps']}")
            logger.debug(f"   SAM3DProcessor checkpoint: {self.processor.sam3d_checkpoint}")
            progress(0.5, desc="SAM 3D 재구성 중...")

            # Unload SAM2 models to free GPU memory for SAM 3D
            # Critical for RTX 3060 12GB: SAM2 (3GB) + SAM3D (10GB) = 13GB > 12GB
            self.unload_sam2_models()

            try:
                logger.info("SAM3D inference 시작...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    mem_before = torch.cuda.memory_allocated() / 1024**3
                    logger.debug(f"   GPU 메모리 (inference 전): {mem_before:.2f} GB")

                # 파라미터를 전달하여 재구성
                reconstruction = self.processor.reconstruct_3d(
                    frame, mask,
                    seed=mesh_settings['seed'],
                    mesh_settings=mesh_settings
                )

                if torch.cuda.is_available():
                    mem_after = torch.cuda.memory_allocated() / 1024**3
                    logger.debug(f"   GPU 메모리 (inference 후): {mem_after:.2f} GB")

                logger.info(f"✓ Reconstruction 완료: {type(reconstruction)}")

                if reconstruction:
                    # PLY 저장 - 프레임 번호와 타임스탬프로 고유 파일명 생성
                    from datetime import datetime
                    import json
                    project_root = Path(__file__).parent.parent
                    output_dir = project_root / "outputs" / "3d_meshes"
                    output_dir.mkdir(parents=True, exist_ok=True)

                    timestamp = datetime.now().strftime("%H%M%S")
                    filename = f"mesh_frame{frame_idx:04d}_{timestamp}.ply"
                    output_path = output_dir / filename

                    logger.info(f"✓ Mesh 저장 중: {output_path}")
                    self.processor.export_mesh(reconstruction, str(output_path), format='ply')
                    logger.info(f"✓ Mesh 저장 완료")
                    logger.debug(f"   Output keys: {reconstruction.keys() if isinstance(reconstruction, dict) else 'N/A'}")

                    # 설정 파일 저장
                    settings_filename = f"mesh_frame{frame_idx:04d}_{timestamp}_settings.json"
                    settings_path = output_dir / settings_filename
                    settings_data = {
                        "timestamp": datetime.now().isoformat(),
                        "source": {
                            "video_path": getattr(self, 'video_path', None),
                            "frame_idx": frame_idx,
                            "total_frames": len(self.frames)
                        },
                        "parameters": mesh_settings,
                        "output": {
                            "filename": filename,
                            "format": "ply"
                        }
                    }
                    with open(settings_path, 'w', encoding='utf-8') as f:
                        json.dump(settings_data, f, indent=2, ensure_ascii=False)
                    logger.info(f"✓ 설정 저장: {settings_path}")

                    progress(1.0, desc="완료!")

                    status = f"""
### 3D Mesh 생성 완료 ✅

- **프레임**: {frame_idx + 1} / {len(self.frames)}
- **저장 위치**: `{output_path}`
- **설정 파일**: `{settings_path}`

**파라미터:**
- Seed: {mesh_settings['seed']}
- Steps: {mesh_settings['stage1_inference_steps']}/{mesh_settings['stage2_inference_steps']}
- 후처리: {'✓' if mesh_settings['with_mesh_postprocess'] else '✗'}

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

    def batch_generate_3d_mesh_current(
        self,
        video_idx: int,
        frame_idx: int,
        seed: int = 42,
        stage1_steps: int = 25,
        stage2_steps: int = 25,
        with_postprocess: bool = False,
        simplify_ratio: float = 0.95,
        with_texture_baking: bool = False,
        texture_size: int = 1024,
        use_vertex_color: bool = True,
        progress=gr.Progress()
    ) -> Tuple[str, str]:
        """
        Batch mode: 현재 선택된 비디오/프레임의 3D Mesh 생성
        """
        from datetime import datetime
        import json

        # 메시 파라미터 설정
        mesh_settings = {
            "seed": int(seed),
            "stage1_inference_steps": int(stage1_steps),
            "stage2_inference_steps": int(stage2_steps),
            "with_mesh_postprocess": with_postprocess,
            "simplify_ratio": float(simplify_ratio),
            "with_texture_baking": with_texture_baking,
            "texture_size": int(texture_size),
            "use_vertex_color": use_vertex_color
        }

        if not hasattr(self, 'batch_results') or not self.batch_results:
            return None, "먼저 Batch Propagate를 실행하세요."

        # batch_results는 {'videos': [...], 'temp_dir': ..., 'total_frames': ...} 구조
        videos = self.batch_results.get('videos', [])
        if not videos:
            return None, "비디오 결과가 없습니다."

        # video_idx로 해당 비디오 찾기
        video_result = None
        for v in videos:
            if v.get('video_idx') == video_idx:
                video_result = v
                break

        if video_result is None:
            return None, f"비디오 인덱스 {video_idx}를 찾을 수 없습니다. (총 {len(videos)}개 비디오)"

        video_name = video_result.get('video_name', f'video_{video_idx:03d}')
        video_path = video_result.get('video_path', '')
        unique_id = self._generate_unique_video_id(video_path) if video_path else video_name
        result_dir = video_result.get('result_dir', '')

        # 프레임 디렉토리에서 마스크와 이미지 로드
        if not result_dir or not Path(result_dir).exists():
            return None, f"결과 디렉토리를 찾을 수 없습니다: {result_dir}"

        frame_dirs = sorted([d for d in Path(result_dir).iterdir() if d.is_dir() and d.name.startswith('frame_')])
        if not frame_dirs:
            return None, f"비디오 {video_name}에 프레임이 없습니다."

        if frame_idx < 0 or frame_idx >= len(frame_dirs):
            return None, f"잘못된 프레임 인덱스: {frame_idx} (총 {len(frame_dirs)}개 프레임)"

        # 프레임 디렉토리에서 이미지와 마스크 로드
        frame_dir = frame_dirs[frame_idx]
        original_path = frame_dir / "original.png"
        mask_path = frame_dir / "mask.png"

        if not original_path.exists():
            return None, f"원본 이미지를 찾을 수 없습니다: {original_path}"
        if not mask_path.exists():
            return None, f"마스크를 찾을 수 없습니다: {mask_path}"

        import cv2
        frame = cv2.imread(str(original_path))
        if frame is None:
            return None, f"원본 이미지 로드 실패: {original_path}"
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None, f"마스크 로드 실패: {mask_path}"
        mask = (mask > 127).astype(np.uint8) * 255  # 이진 마스크로 변환

        # 마스크 유효성 검사
        mask_pixels = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        mask_ratio = mask_pixels / total_pixels

        logger.info(f"마스크 정보: {mask_pixels} 픽셀 ({mask_ratio*100:.2f}%), shape={mask.shape}")

        if mask_pixels == 0:
            return None, f"마스크가 비어 있습니다. 유효한 세그멘테이션이 필요합니다."

        if mask_pixels < 100:
            return None, f"마스크가 너무 작습니다 ({mask_pixels} 픽셀). 최소 100픽셀 이상 필요합니다."

        if mask_ratio < 0.001:
            return None, f"마스크 영역이 너무 작습니다 ({mask_ratio*100:.3f}%). 객체가 이미지에서 충분히 보이는지 확인하세요."

        logger.info(f"Batch 3D Mesh 생성: {unique_id} ({video_name}), frame {frame_idx}")
        logger.info(f"   Mesh 설정: seed={mesh_settings['seed']}, steps={mesh_settings['stage1_inference_steps']}/{mesh_settings['stage2_inference_steps']}")
        progress(0.3, desc="SAM 3D 초기화 중...")

        # Unload SAM2 for memory
        self.unload_sam2_models()

        try:
            progress(0.5, desc="3D 재구성 중...")
            reconstruction = self.processor.reconstruct_3d(
                frame, mask,
                seed=mesh_settings['seed'],
                mesh_settings=mesh_settings
            )

            if reconstruction:
                # 세션 폴더 내부에 저장 (있으면), 없으면 기본 경로
                output_dir = self._get_session_mesh_dir()

                timestamp = datetime.now().strftime("%H%M%S")
                filename = f"{unique_id}_frame{frame_idx:04d}_{timestamp}.ply"
                output_path = output_dir / filename

                self.processor.export_mesh(reconstruction, str(output_path), format='ply')
                logger.info(f"Mesh 저장 완료: {output_path}")

                # 설정 파일 저장
                settings_filename = f"{unique_id}_frame{frame_idx:04d}_{timestamp}_settings.json"
                settings_path = output_dir / settings_filename
                settings_data = {
                    "timestamp": datetime.now().isoformat(),
                    "source": {
                        "session_path": self.current_session_path,
                        "video_name": video_name,
                        "unique_id": unique_id,
                        "video_idx": video_idx,
                        "frame_idx": frame_idx,
                        "total_frames": len(frame_dirs)
                    },
                    "parameters": mesh_settings,
                    "output": {
                        "filename": filename,
                        "format": "ply"
                    }
                }
                with open(settings_path, 'w', encoding='utf-8') as f:
                    json.dump(settings_data, f, indent=2, ensure_ascii=False)

                # 세션 메타데이터에 mesh 정보 추가
                mesh_info = {
                    "unique_id": unique_id,
                    "video_name": video_name,
                    "video_idx": video_idx,
                    "frame_idx": frame_idx,
                    "filename": filename,
                    "settings_file": settings_filename,
                    "timestamp": datetime.now().isoformat(),
                    "parameters": mesh_settings
                }
                self._update_session_mesh_metadata(mesh_info)

                progress(1.0, desc="완료!")
                self.reload_sam2_models()

                status = f"""### 3D Mesh 생성 완료 ✅

- **비디오**: {video_name}
- **프레임**: {frame_idx + 1}
- **저장 위치**: `{output_path}`
- **설정 파일**: `{settings_path}`
- **세션 메타데이터**: 자동 업데이트됨

**파라미터:**
- Seed: {mesh_settings['seed']}
- Steps: {mesh_settings['stage1_inference_steps']}/{mesh_settings['stage2_inference_steps']}
- 후처리: {'✓' if mesh_settings['with_mesh_postprocess'] else '✗'}
"""
                return str(output_path), status
            else:
                self.reload_sam2_models()
                return None, "3D 재구성 실패"

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Batch 3D Mesh 생성 실패: {e}\n{error_details}")
            self.reload_sam2_models()

            # 더 친절한 오류 메시지
            error_msg = str(e)
            if "numel() == 0" in error_msg or "reduction dim" in error_msg:
                return None, f"""### 3D Mesh 생성 실패 ❌

**원인**: 마스크에서 유효한 3D 구조를 생성할 수 없습니다.

**가능한 해결 방법**:
1. 다른 프레임을 선택해 보세요 (객체가 더 명확한 프레임)
2. 마스크 품질 확인 - 객체가 완전히 세그멘테이션되었는지 확인
3. Seed 값을 변경해 보세요

**디버그 정보**: 마스크 {mask_pixels} 픽셀 ({mask_ratio*100:.2f}%)
"""
            else:
                return None, f"3D Mesh 생성 실패: {error_msg}\n\n자세한 로그는 터미널을 확인하세요."

    def batch_generate_3d_mesh_selected(
        self,
        selected_frames: List[dict],
        seed: int = 42,
        stage1_steps: int = 25,
        stage2_steps: int = 25,
        with_postprocess: bool = False,
        simplify_ratio: float = 0.95,
        with_texture_baking: bool = False,
        texture_size: int = 1024,
        use_vertex_color: bool = True,
        progress=gr.Progress()
    ) -> Tuple[str, str]:
        """
        선택된 프레임들의 3D Mesh 일괄 생성

        Args:
            selected_frames: 선택된 프레임 정보 리스트 [{video_idx, video_name, frame_idx}, ...]
            기타: mesh 생성 파라미터

        Returns:
            (출력 디렉토리 경로, 상태 메시지)
        """
        from datetime import datetime
        import json

        if not selected_frames:
            return None, "선택된 프레임이 없습니다. 먼저 프레임을 추가하세요."

        if not hasattr(self, 'batch_results') or not self.batch_results:
            return None, "Batch 결과가 없습니다. 먼저 Batch Propagate를 실행하세요."

        # 메시 파라미터 설정
        mesh_settings = {
            "seed": int(seed),
            "stage1_inference_steps": int(stage1_steps),
            "stage2_inference_steps": int(stage2_steps),
            "with_mesh_postprocess": with_postprocess,
            "simplify_ratio": float(simplify_ratio),
            "with_texture_baking": with_texture_baking,
            "texture_size": int(texture_size),
            "use_vertex_color": use_vertex_color
        }

        # 세션 폴더 내부에 저장
        output_dir = self._get_session_mesh_dir()

        generated_meshes = []
        failed_meshes = []
        total = len(selected_frames)

        logger.info(f"선택된 프레임 3D Mesh 생성 시작: {total}개 프레임")
        logger.info(f"   Mesh 설정: seed={mesh_settings['seed']}, steps={mesh_settings['stage1_inference_steps']}/{mesh_settings['stage2_inference_steps']}")

        # 첫 번째 프레임 전에 SAM2 언로드
        self.unload_sam2_models()

        for i, frame_info in enumerate(selected_frames):
            video_idx = frame_info['video_idx']
            frame_idx = frame_info['frame_idx']
            video_name = frame_info.get('video_name', f'video_{video_idx:03d}')

            # batch_results는 {'videos': [...], 'temp_dir': ..., 'total_frames': ...} 구조
            videos = self.batch_results.get('videos', [])

            # video_idx로 해당 비디오 찾기
            video_result = None
            for v in videos:
                if v.get('video_idx') == video_idx:
                    video_result = v
                    break

            if video_result is None:
                failed_meshes.append(f"{video_name} frame {frame_idx}: 비디오를 찾을 수 없음")
                continue

            video_path = video_result.get('video_path', '')
            unique_id = self._generate_unique_video_id(video_path) if video_path else video_name
            result_dir = video_result.get('result_dir', '')

            progress((i + 0.2) / total, desc=f"3D Mesh 생성 중: {unique_id} frame {frame_idx}")

            # 프레임 디렉토리에서 마스크와 이미지 로드
            if not result_dir or not Path(result_dir).exists():
                failed_meshes.append(f"{unique_id} frame {frame_idx}: 결과 디렉토리 없음")
                continue

            frame_dirs = sorted([d for d in Path(result_dir).iterdir() if d.is_dir() and d.name.startswith('frame_')])
            if not frame_dirs:
                failed_meshes.append(f"{unique_id} frame {frame_idx}: 프레임 디렉토리 없음")
                continue

            if frame_idx < 0 or frame_idx >= len(frame_dirs):
                failed_meshes.append(f"{unique_id} frame {frame_idx}: 잘못된 프레임 인덱스 (총 {len(frame_dirs)}개)")
                continue

            # 프레임 디렉토리에서 이미지와 마스크 로드
            frame_dir = frame_dirs[frame_idx]
            original_path = frame_dir / "original.png"
            mask_path = frame_dir / "mask.png"

            if not original_path.exists():
                failed_meshes.append(f"{unique_id} frame {frame_idx}: 원본 이미지 없음")
                continue
            if not mask_path.exists():
                failed_meshes.append(f"{unique_id} frame {frame_idx}: 마스크 파일 없음")
                continue

            import cv2
            frame = cv2.imread(str(original_path))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.uint8) * 255  # 이진 마스크로 변환

            try:
                progress((i + 0.5) / total, desc=f"3D 재구성 중: {unique_id} frame {frame_idx}")

                reconstruction = self.processor.reconstruct_3d(
                    frame, mask,
                    seed=mesh_settings['seed'],
                    mesh_settings=mesh_settings
                )

                if reconstruction:
                    timestamp = datetime.now().strftime("%H%M%S")
                    filename = f"{unique_id}_frame{frame_idx:04d}_{timestamp}.ply"
                    output_path = output_dir / filename

                    self.processor.export_mesh(reconstruction, str(output_path), format='ply')

                    # 설정 파일 저장
                    settings_filename = f"{unique_id}_frame{frame_idx:04d}_{timestamp}_settings.json"
                    settings_path = output_dir / settings_filename
                    settings_data = {
                        "timestamp": datetime.now().isoformat(),
                        "source": {
                            "session_path": self.current_session_path,
                            "unique_id": unique_id,
                            "video_name": video_name,
                            "video_idx": video_idx,
                            "frame_idx": frame_idx,
                            "total_frames": len(frame_dirs)
                        },
                        "parameters": mesh_settings,
                        "output": {
                            "filename": filename,
                            "format": "ply"
                        }
                    }
                    with open(settings_path, 'w', encoding='utf-8') as f:
                        json.dump(settings_data, f, indent=2, ensure_ascii=False)

                    # 세션 메타데이터에 mesh 정보 추가
                    mesh_info = {
                        "unique_id": unique_id,
                        "video_name": video_name,
                        "video_idx": video_idx,
                        "frame_idx": frame_idx,
                        "filename": filename,
                        "settings_file": settings_filename,
                        "timestamp": datetime.now().isoformat(),
                        "parameters": mesh_settings
                    }
                    self._update_session_mesh_metadata(mesh_info)

                    generated_meshes.append({
                        'unique_id': unique_id,
                        'video': video_name,
                        'video_idx': video_idx,
                        'frame': frame_idx,
                        'path': str(output_path)
                    })
                    logger.info(f"Generated: {filename}")
                else:
                    failed_meshes.append(f"{video_name} frame {frame_idx}: 재구성 실패")

            except Exception as e:
                logger.error(f"Failed {video_name} frame {frame_idx}: {e}")
                failed_meshes.append(f"{video_name} frame {frame_idx}: {str(e)}")
                continue

        # SAM2 다시 로드
        self.reload_sam2_models()

        progress(1.0, desc="완료!")

        if generated_meshes:
            mesh_list = "\n".join([f"- {m['video']} (frame {m['frame']})" for m in generated_meshes])
            failed_list = "\n".join([f"- {f}" for f in failed_meshes]) if failed_meshes else ""

            status = f"""### 선택 프레임 3D Mesh 생성 완료 ✅

**생성 성공**: {len(generated_meshes)}/{total}개
**저장 위치**: `{output_dir}`

**파라미터:**
- Seed: {mesh_settings['seed']}
- Steps: {mesh_settings['stage1_inference_steps']}/{mesh_settings['stage2_inference_steps']}
- 후처리: {'✓' if mesh_settings['with_mesh_postprocess'] else '✗'}

**생성된 메시:**
{mesh_list}
"""
            if failed_meshes:
                status += f"""
**실패한 프레임:**
{failed_list}
"""
            return str(output_dir), status
        else:
            return None, f"3D Mesh 생성 실패 (모든 프레임)\n\n실패 목록:\n" + "\n".join([f"- {f}" for f in failed_meshes])

    def batch_generate_3d_mesh_all(
        self,
        seed: int = 42,
        stage1_steps: int = 25,
        stage2_steps: int = 25,
        with_postprocess: bool = False,
        simplify_ratio: float = 0.95,
        with_texture_baking: bool = False,
        texture_size: int = 1024,
        use_vertex_color: bool = True,
        progress=gr.Progress()
    ) -> Tuple[str, str]:
        """
        Batch mode: 모든 비디오의 중간 프레임에서 3D Mesh 생성
        """
        from datetime import datetime
        import json

        # 메시 파라미터 설정
        mesh_settings = {
            "seed": int(seed),
            "stage1_inference_steps": int(stage1_steps),
            "stage2_inference_steps": int(stage2_steps),
            "with_mesh_postprocess": with_postprocess,
            "simplify_ratio": float(simplify_ratio),
            "with_texture_baking": with_texture_baking,
            "texture_size": int(texture_size),
            "use_vertex_color": use_vertex_color
        }

        if not hasattr(self, 'batch_results') or not self.batch_results:
            return None, "먼저 Batch Propagate를 실행하세요."

        # batch_results는 {'videos': [...], 'temp_dir': ..., 'total_frames': ...} 구조
        videos = self.batch_results.get('videos', [])
        if not videos:
            return None, "비디오 결과가 없습니다."

        # 세션 폴더 내부에 저장 (있으면), 없으면 기본 경로
        output_dir = self._get_session_mesh_dir()

        generated_meshes = []
        total = len(videos)

        logger.info(f"전체 3D Mesh 생성 시작: {total}개 비디오")
        logger.info(f"   Mesh 설정: seed={mesh_settings['seed']}, steps={mesh_settings['stage1_inference_steps']}/{mesh_settings['stage2_inference_steps']}")

        for i, video_result in enumerate(videos):
            video_idx = video_result.get('video_idx', i)
            video_name = video_result.get('video_name', f'video_{i:03d}')
            video_path = video_result.get('video_path', '')
            unique_id = self._generate_unique_video_id(video_path) if video_path else video_name
            result_dir = video_result.get('result_dir', '')

            # 프레임 디렉토리 확인
            if not result_dir or not Path(result_dir).exists():
                logger.warning(f"Skip {unique_id}: result_dir not found")
                continue

            frame_dirs = sorted([d for d in Path(result_dir).iterdir() if d.is_dir() and d.name.startswith('frame_')])
            if not frame_dirs:
                logger.warning(f"Skip {unique_id}: no frame directories")
                continue

            # 중간 프레임 선택
            mid_idx = len(frame_dirs) // 2
            frame_dir = frame_dirs[mid_idx]
            original_path = frame_dir / "original.png"
            mask_path = frame_dir / "mask.png"

            if not original_path.exists() or not mask_path.exists():
                logger.warning(f"Skip {unique_id}: no original/mask at frame {mid_idx}")
                continue

            import cv2
            frame = cv2.imread(str(original_path))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.uint8) * 255

            progress((i + 0.5) / total, desc=f"3D Mesh 생성 중: {unique_id} ({i+1}/{total})")

            # 첫 번째 비디오 전에 SAM2 언로드
            if i == 0:
                self.unload_sam2_models()

            try:
                reconstruction = self.processor.reconstruct_3d(
                    frame, mask,
                    seed=mesh_settings['seed'],
                    mesh_settings=mesh_settings
                )

                if reconstruction:
                    timestamp = datetime.now().strftime("%H%M%S")
                    filename = f"{unique_id}_frame{mid_idx:04d}_{timestamp}.ply"
                    output_path = output_dir / filename

                    self.processor.export_mesh(reconstruction, str(output_path), format='ply')

                    # 설정 파일 저장
                    settings_filename = f"{unique_id}_frame{mid_idx:04d}_{timestamp}_settings.json"
                    settings_path = output_dir / settings_filename
                    settings_data = {
                        "timestamp": datetime.now().isoformat(),
                        "source": {
                            "session_path": self.current_session_path,
                            "unique_id": unique_id,
                            "video_name": video_name,
                            "video_idx": video_idx,
                            "frame_idx": mid_idx,
                            "total_frames": len(frame_dirs)
                        },
                        "parameters": mesh_settings,
                        "output": {
                            "filename": filename,
                            "format": "ply"
                        }
                    }
                    with open(settings_path, 'w', encoding='utf-8') as f:
                        json.dump(settings_data, f, indent=2, ensure_ascii=False)

                    # 세션 메타데이터에 mesh 정보 추가
                    mesh_info = {
                        "unique_id": unique_id,
                        "video_name": video_name,
                        "video_idx": video_idx,
                        "frame_idx": mid_idx,
                        "filename": filename,
                        "settings_file": settings_filename,
                        "timestamp": datetime.now().isoformat(),
                        "parameters": mesh_settings
                    }
                    self._update_session_mesh_metadata(mesh_info)

                    generated_meshes.append({
                        'unique_id': unique_id,
                        'video': video_name,
                        'video_idx': video_idx,
                        'frame': mid_idx,
                        'path': str(output_path)
                    })
                    logger.info(f"Generated: {filename}")

            except Exception as e:
                logger.error(f"Failed {video_name}: {e}")
                continue

        # SAM2 다시 로드
        self.reload_sam2_models()

        progress(1.0, desc="완료!")

        if generated_meshes:
            mesh_list = "\n".join([f"- {m['unique_id']} (frame {m['frame']}): `{m['path']}`" for m in generated_meshes])
            status = f"""### 전체 3D Mesh 생성 완료 ✅

**생성된 메시**: {len(generated_meshes)}/{total}

**파라미터:**
- Seed: {mesh_settings['seed']}
- Steps: {mesh_settings['stage1_inference_steps']}/{mesh_settings['stage2_inference_steps']}
- 후처리: {'✓' if mesh_settings['with_mesh_postprocess'] else '✗'}

{mesh_list}
"""
            return str(output_dir), status
        else:
            return None, "3D Mesh 생성 실패 (모든 비디오)"

    def _update_session_mesh_metadata(self, mesh_info: dict) -> bool:
        """
        세션 메타데이터에 3D mesh 정보 추가/업데이트

        Args:
            mesh_info: mesh 정보 딕셔너리 {video_name, frame_idx, filename, settings, ...}

        Returns:
            성공 여부
        """
        try:
            if not self.current_session_path:
                logger.warning("현재 세션 경로가 설정되지 않음 - mesh 메타데이터 업데이트 스킵")
                return False

            session_dir = Path(self.current_session_path)
            metadata_path = session_dir / "session_metadata.json"

            if not metadata_path.exists():
                logger.warning(f"세션 메타데이터 없음: {metadata_path}")
                return False

            # 메타데이터 로드
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # 3d_meshes 필드 초기화 (없으면)
            if '3d_meshes' not in metadata:
                metadata['3d_meshes'] = []

            # 기존 항목 중복 확인 (같은 비디오/프레임이면 업데이트)
            updated = False
            for i, existing in enumerate(metadata['3d_meshes']):
                if (existing.get('video_name') == mesh_info.get('video_name') and
                    existing.get('frame_idx') == mesh_info.get('frame_idx')):
                    metadata['3d_meshes'][i] = mesh_info
                    updated = True
                    break

            if not updated:
                metadata['3d_meshes'].append(mesh_info)

            # 메타데이터 저장
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"세션 메타데이터에 mesh 정보 {'업데이트' if updated else '추가'}: {mesh_info.get('filename')}")
            return True

        except Exception as e:
            logger.error(f"세션 mesh 메타데이터 업데이트 실패: {e}")
            return False

    def _get_session_mesh_dir(self) -> Optional[Path]:
        """
        현재 세션의 3D mesh 저장 디렉토리 반환
        세션이 없으면 기본 경로 반환
        """
        if self.current_session_path:
            session_dir = Path(self.current_session_path)
            mesh_dir = session_dir / "3d_meshes"
            mesh_dir.mkdir(parents=True, exist_ok=True)
            return mesh_dir
        else:
            # 세션이 없으면 기본 경로 사용
            project_root = Path(__file__).parent.parent
            mesh_dir = project_root / "outputs" / "3d_meshes"
            mesh_dir.mkdir(parents=True, exist_ok=True)
            return mesh_dir

    def save_annotation_session(self, session_name: str = "", save_as_new: bool = False) -> str:
        """
        Annotation 세션 전체 저장 (annotation points + masks + metadata)

        Args:
            session_name: 세션 이름 (비어있으면 timestamp 사용)
            save_as_new: True면 항상 새 세션 생성, False면 기존 세션 덮어쓰기 시도
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
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 덮어쓰기 vs 새로 저장 결정
            if not save_as_new and self.current_session_path and Path(self.current_session_path).exists():
                # 기존 세션 덮어쓰기
                output_dir = Path(self.current_session_path)
                session_id = output_dir.name
                print(f"✓ 기존 세션 덮어쓰기: {session_id}")
            else:
                # 새 세션 생성
                if session_name and session_name.strip():
                    session_id = f"{session_name.strip()}_{timestamp}"
                else:
                    session_id = timestamp
                output_dir = Path(f"outputs/sessions/{session_id}")
                print(f"✓ 새 세션 ID 생성: {session_id}")

            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ 출력 디렉토리: {output_dir}")

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
                "session_type": "interactive",  # For scan_aug_sessions to find it
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

            # 현재 세션 경로 업데이트 (다음 저장 시 덮어쓰기용)
            self.current_session_path = str(output_dir)

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
        저장된 annotation 세션 로드 (단일 비디오 세션 및 batch 세션 모두 지원)
        """
        try:
            session_dir = Path(f"outputs/sessions/{session_id}")
            if not session_dir.exists():
                return None, f"세션을 찾을 수 없습니다: {session_id}"

            # 메타데이터 로드
            metadata_path = session_dir / "session_metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # 세션 타입 확인 (batch vs single)
            session_type = metadata.get("session_type", "single")

            if session_type == "batch":
                # Batch 세션 로드
                return self._load_batch_session(session_id, session_dir, metadata)
            else:
                # 단일 비디오 세션 로드
                return self._load_single_session(session_id, session_dir, metadata)

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            return None, f"로드 오류: {str(e)}\n\n```\n{error_detail}\n```"

    def _load_single_session(self, session_id: str, session_dir: Path, metadata: dict) -> Tuple[np.ndarray, str]:
        """단일 비디오 세션 로드"""
        # 프레임 및 마스크 로드
        num_frames = metadata["num_frames"]
        self.frames = []
        self.masks = []

        for i in range(num_frames):
            frame_dir = session_dir / f"frame_{i:04d}"

            # 원본 프레임 로드 (BGR→RGB 변환하여 self.frames는 항상 RGB로 유지)
            frame_path = frame_dir / "original.png"
            frame = cv2.imread(str(frame_path))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frames.append(frame_rgb)

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

        # 현재 프레임 시각화 (self.frames는 이미 RGB)
        current_frame = self.frames[self.current_frame_idx]
        current_mask = self.masks[self.current_frame_idx]

        frame_rgb = current_frame.copy()  # 이미 RGB이므로 변환 불필요
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

        # 현재 세션 경로 업데이트 (다음 저장 시 덮어쓰기용)
        self.current_session_path = str(session_dir)

        return result, status

    def _load_batch_session(self, session_id: str, session_dir: Path, metadata: dict) -> Tuple[np.ndarray, str]:
        """
        Batch 세션 로드 - batch_results 구조로 복원하여 퀄리티 체크 등에서 사용 가능
        (Quick Mode에서 세션 로드 시 호출됨)
        """
        # load_batch_session의 핵심 로직 재사용
        status_msg, _ = self.load_batch_session(str(session_dir))

        # 첫 번째 비디오의 첫 번째 프레임 미리보기 생성
        preview_image = None
        video_list = self.get_batch_video_list()

        if video_list and len(video_list) > 0:
            first_video_dir = Path(video_list[0]['result_dir'])
            frame_dirs = sorted([d for d in first_video_dir.iterdir() if d.is_dir() and d.name.startswith('frame_')])

            if frame_dirs:
                first_frame_dir = frame_dirs[0]
                original_path = first_frame_dir / "original.png"
                mask_path = first_frame_dir / "mask.png"

                if original_path.exists():
                    original = cv2.imread(str(original_path))
                    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

                    if mask_path.exists():
                        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        # 오버레이 생성
                        overlay = original.copy().astype(np.float32)
                        mask_bool = mask > 127
                        overlay[mask_bool] = overlay[mask_bool] * 0.6 + np.array([0, 255, 0]) * 0.4
                        preview_image = overlay.astype(np.uint8)
                    else:
                        preview_image = original

        # reference_annotations를 현재 annotations에도 설정 (Quick Mode 호환)
        if hasattr(self, 'batch_results') and self.batch_results:
            ref_annot = self.batch_results.get('reference_annotations', {'foreground': [], 'background': []})
            self.annotations = {
                'foreground': ref_annot.get('foreground', []),
                'background': ref_annot.get('background', [])
            }

        # 현재 세션 경로 업데이트
        self.current_session_path = str(session_dir)

        # 상태 메시지에 Quick Mode용 안내 추가
        total_videos = len(video_list) if video_list else 0
        total_frames = sum(v['num_frames'] for v in video_list) if video_list else 0

        video_list_str = ""
        for v in video_list[:10]:
            video_list_str += f"\n- **{v['video_name']}**: {v['num_frames']} 프레임"
        if len(video_list) > 10:
            video_list_str += f"\n- ... 외 {len(video_list) - 10}개 비디오"

        per_video_count = len(self.per_video_annotations) if hasattr(self, 'per_video_annotations') and self.per_video_annotations else 0

        status = f"""
### 📂 Batch 세션 로드 완료 ✅

**세션 ID**: `{session_id}`
**세션 타입**: Batch ({total_videos}개 비디오)

**로드된 데이터**:
- 🎬 비디오 수: {total_videos}개
- 📁 총 프레임: {total_frames}개
- 📍 Reference Annotations: FG {len(self.annotations.get('foreground', []))}개, BG {len(self.annotations.get('background', []))}개
- 🎯 Per-video Annotations: {per_video_count}개 비디오

**비디오 목록**:{video_list_str}

---

### 사용 가능한 기능:
1. **Batch Mode 탭**에서 **결과 시각화 & 퀄리티 체크** - 비디오별 마스크 확인
2. **Export to Fauna** - 데이터셋 내보내기
3. **프리뷰 비디오 생성** - 마스크 오버레이 영상

> 💡 **Tip**: Batch Mode 탭으로 이동하여 비디오 목록을 새로고침하세요.
"""

        return preview_image, status

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

    def delete_session(self, session_id: str) -> Tuple[str, List[str]]:
        """
        세션 삭제

        Args:
            session_id: 삭제할 세션 ID

        Returns:
            (상태 메시지, 업데이트된 세션 목록)
        """
        import shutil

        if not session_id:
            return "⚠️ 삭제할 세션을 선택하세요", self.get_session_ids()

        try:
            session_dir = Path("outputs/sessions") / session_id

            if not session_dir.exists():
                return f"❌ 세션을 찾을 수 없습니다: {session_id}", self.get_session_ids()

            # 현재 로드된 세션인지 확인
            if self.current_session_path and Path(self.current_session_path) == session_dir:
                self.current_session_path = None

            # 세션 폴더 삭제
            shutil.rmtree(session_dir)

            return f"✅ 세션 삭제 완료: `{session_id}`", self.get_session_ids()

        except Exception as e:
            import traceback
            return f"❌ 삭제 실패: {str(e)}\n{traceback.format_exc()}", self.get_session_ids()

    def rename_session(self, session_id: str, new_name: str) -> Tuple[str, List[str], str]:
        """
        세션 이름 변경

        Args:
            session_id: 변경할 세션 ID
            new_name: 새 이름

        Returns:
            (상태 메시지, 업데이트된 세션 목록, 새 세션 ID)
        """
        if not session_id:
            return "⚠️ 변경할 세션을 선택하세요", self.get_session_ids(), None

        if not new_name or not new_name.strip():
            return "⚠️ 새 이름을 입력하세요", self.get_session_ids(), session_id

        new_name = new_name.strip()

        # 특수문자 제거 (파일시스템 안전)
        import re
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', new_name)

        try:
            sessions_dir = Path("outputs/sessions")
            old_path = sessions_dir / session_id
            new_path = sessions_dir / safe_name

            if not old_path.exists():
                return f"❌ 세션을 찾을 수 없습니다: {session_id}", self.get_session_ids(), None

            if new_path.exists():
                return f"❌ 이미 존재하는 이름입니다: {safe_name}", self.get_session_ids(), session_id

            # 폴더 이름 변경
            old_path.rename(new_path)

            # 메타데이터 업데이트
            metadata_path = new_path / "session_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                metadata['session_id'] = safe_name
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

            # 현재 로드된 세션 경로 업데이트
            if self.current_session_path and Path(self.current_session_path) == old_path:
                self.current_session_path = str(new_path)

            return f"✅ 이름 변경 완료: `{session_id}` → `{safe_name}`", self.get_session_ids(), safe_name

        except Exception as e:
            import traceback
            return f"❌ 이름 변경 실패: {str(e)}\n{traceback.format_exc()}", self.get_session_ids(), session_id

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

            # 출력 디렉토리 설정 - session_id 기반으로 저장
            project_root = Path(__file__).parent.parent
            fauna_root = project_root / "outputs" / "fauna_datasets"

            # session_id 결정: current_session_path가 있으면 해당 ID 사용, 없으면 timestamp 생성
            if self.current_session_path:
                # 기존 세션 ID 사용 (폴더 이름에서 추출)
                sequence_name = Path(self.current_session_path).name
                print(f"   Using existing session ID: {sequence_name}")
            else:
                # 세션이 저장되지 않은 경우 timestamp 기반 ID 생성
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                sequence_name = f"unsaved_{timestamp}"
                print(f"   Generated new session ID: {sequence_name}")

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

            **작업 모드:**
            - 🎨 **Interactive Mode**: 단일 비디오 수동 annotation & propagation
            - 📦 **Batch Mode**: 다중 비디오 일괄 처리 및 세션 관리
            - 📝 **Lite Annotator**: 효율적 단일 프레임 annotation
            """)

            # ===== SAM2 모델 상태 (컴팩트 한 줄) =====
            with gr.Row(equal_height=True):
                sam2_status = gr.Markdown(
                    self._get_sam2_status_markdown(),
                    elem_id="sam2-status"
                )
                sam2_download_btn = gr.Button(
                    "🔄 다운로드/로드" if self.sam2_predictor is None else "✅ 로드됨",
                    variant="primary" if self.sam2_predictor is None else "secondary",
                    size="sm",
                    scale=0,
                    min_width=120
                )
                sam2_progress_text = gr.Textbox(
                    value="",
                    show_label=False,
                    scale=1,
                    max_lines=1,
                    placeholder="진행 상태..."
                )

            def download_and_load_sam2(progress=gr.Progress()):
                """SAM2 체크포인트 다운로드 및 모델 로드"""
                progress(0, desc="SAM2 확인 중...")

                # 이미 로드되어 있으면
                if self.sam2_predictor is not None and self.sam2_video_predictor is not None:
                    return self._get_sam2_status_markdown(), "✅ SAM2 모델이 이미 로드되어 있습니다."

                if not SAM2_AVAILABLE:
                    return self._get_sam2_status_markdown(), "❌ SAM2 패키지가 설치되지 않았습니다. pip install sam2"

                checkpoint = self.SAM2_CHECKPOINT_PATH
                if self.config:
                    config_checkpoint = Path(self.config.sam2_checkpoint)
                    if config_checkpoint.exists():
                        checkpoint = config_checkpoint

                # 다운로드 필요 여부
                if not checkpoint.exists():
                    progress(0.1, desc="📥 SAM2 체크포인트 다운로드 중... (약 900MB)")

                    def update_progress(p):
                        progress(0.1 + p * 0.7, desc=f"📥 다운로드 중... {p*100:.0f}%")

                    success, msg = self.download_sam2_checkpoint(update_progress)
                    if not success:
                        return self._get_sam2_status_markdown(), f"❌ 다운로드 실패: {msg}"

                # 모델 로드
                progress(0.85, desc="🔄 SAM2 모델 로딩 중...")
                success, msg = self.load_sam2_models()

                progress(1.0, desc="완료!")

                if success:
                    return self._get_sam2_status_markdown(), f"✅ {msg}"
                else:
                    return self._get_sam2_status_markdown(), f"❌ {msg}"

            sam2_download_btn.click(
                fn=download_and_load_sam2,
                outputs=[sam2_status, sam2_progress_text]
            )

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

                            with gr.Row():
                                load_session_btn = gr.Button("📂 로드", variant="primary", scale=1)
                                delete_session_btn = gr.Button("🗑️ 삭제", variant="stop", scale=1)

                            with gr.Accordion("✏️ 세션 이름 변경", open=False):
                                rename_session_input = gr.Textbox(
                                    label="새 이름",
                                    placeholder="새 세션 이름 입력",
                                    info="선택한 세션의 이름을 변경합니다"
                                )
                                rename_session_btn = gr.Button("✏️ 이름 변경", size="sm")

                            gr.Markdown("### 💾 세션 저장")

                            session_name_input = gr.Textbox(
                                label="세션 이름 (새로 저장 시)",
                                placeholder="예: mouse_experiment_1",
                                info="새로 저장 시에만 사용 (비어있으면 timestamp)"
                            )

                            with gr.Row():
                                save_session_btn = gr.Button("💾 저장", variant="secondary", scale=1)
                                save_session_new_btn = gr.Button("📝 새로 저장", variant="secondary", scale=1)

                            gr.Markdown("### 🎲 3D Mesh 설정")

                            with gr.Accordion("⚙️ Mesh 파라미터", open=False):
                                mesh_seed = gr.Number(
                                    label="Seed (재현성)",
                                    value=42,
                                    precision=0,
                                    info="동일 seed = 동일 결과"
                                )
                                with gr.Row():
                                    mesh_stage1_steps = gr.Slider(
                                        label="Stage1 Steps",
                                        minimum=5,
                                        maximum=50,
                                        value=25,
                                        step=5,
                                        info="Sparse structure 품질"
                                    )
                                    mesh_stage2_steps = gr.Slider(
                                        label="Stage2 Steps",
                                        minimum=5,
                                        maximum=50,
                                        value=25,
                                        step=5,
                                        info="Latent feature 품질"
                                    )
                                mesh_postprocess = gr.Checkbox(
                                    label="Mesh 후처리 (단순화, 홀 채우기)",
                                    value=False,
                                    info="활성화 시 처리 시간 증가"
                                )
                                mesh_simplify_ratio = gr.Slider(
                                    label="Simplify Ratio",
                                    minimum=0.5,
                                    maximum=0.99,
                                    value=0.95,
                                    step=0.05,
                                    info="Face 유지 비율 (0.95 = 5% 제거)",
                                    visible=False
                                )
                                mesh_texture_baking = gr.Checkbox(
                                    label="Texture Baking",
                                    value=False,
                                    info="텍스처 맵 생성 (추가 시간 필요)"
                                )
                                mesh_texture_size = gr.Dropdown(
                                    label="Texture Size",
                                    choices=[512, 1024, 2048],
                                    value=1024,
                                    visible=False
                                )
                                mesh_vertex_color = gr.Checkbox(
                                    label="Vertex Color 사용",
                                    value=True,
                                    info="버텍스에 색상 저장"
                                )

                                # 후처리 체크박스에 따라 simplify_ratio 표시
                                mesh_postprocess.change(
                                    fn=lambda x: gr.update(visible=x),
                                    inputs=[mesh_postprocess],
                                    outputs=[mesh_simplify_ratio]
                                )
                                # 텍스처 베이킹 체크박스에 따라 texture_size 표시
                                mesh_texture_baking.change(
                                    fn=lambda x: gr.update(visible=x),
                                    inputs=[mesh_texture_baking],
                                    outputs=[mesh_texture_size]
                                )

                            mesh_btn = gr.Button("🎲 Generate 3D Mesh", variant="primary")
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

                    # 저장 (기존 세션 덮어쓰기)
                    save_session_btn.click(
                        fn=lambda: self.save_annotation_session(save_as_new=False),
                        outputs=[status_text]
                    )

                    # 새로 저장 (새 세션 생성)
                    save_session_new_btn.click(
                        fn=lambda name: self.save_annotation_session(session_name=name, save_as_new=True),
                        inputs=[session_name_input],
                        outputs=[status_text]
                    )

                    mesh_btn.click(
                        fn=self.generate_3d_mesh,
                        inputs=[
                            mesh_seed,
                            mesh_stage1_steps,
                            mesh_stage2_steps,
                            mesh_postprocess,
                            mesh_simplify_ratio,
                            mesh_texture_baking,
                            mesh_texture_size,
                            mesh_vertex_color
                        ],
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
                    session_refresh_btn.click(
                        fn=lambda: gr.Dropdown(choices=self.get_session_ids()),
                        outputs=[session_id_dropdown]
                    )

                    # 세션 로드 핸들러 (batch/single 모두 지원)
                    def load_session_handler(session_id):
                        """세션 로드 - batch 세션인 경우 비디오 목록 정보도 상태에 포함"""
                        return self.load_annotation_session(session_id)

                    load_session_btn.click(
                        fn=load_session_handler,
                        inputs=[session_id_dropdown],
                        outputs=[image_display, status_text]
                    )

                    # 세션 삭제
                    def delete_session_handler(session_id):
                        msg, sessions = self.delete_session(session_id)
                        return msg, gr.Dropdown(choices=sessions, value=sessions[0] if sessions else None)

                    delete_session_btn.click(
                        fn=delete_session_handler,
                        inputs=[session_id_dropdown],
                        outputs=[status_text, session_id_dropdown]
                    )

                    # 세션 이름 변경
                    def rename_session_handler(session_id, new_name):
                        msg, sessions, new_id = self.rename_session(session_id, new_name)
                        return msg, gr.Dropdown(choices=sessions, value=new_id if new_id else (sessions[0] if sessions else None)), ""

                    rename_session_btn.click(
                        fn=rename_session_handler,
                        inputs=[session_id_dropdown, rename_session_input],
                        outputs=[status_text, session_id_dropdown, rename_session_input]
                    )

                    def clear_points():
                        self.annotations = {'foreground': [], 'background': []}
                        if len(self.frames) > 0:
                            frame_rgb = self.frames[self.current_frame_idx].copy()  # 이미 RGB
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
                                # 비디오 목록을 접을 수 있는 Accordion으로 감싸기
                                with gr.Accordion("📋 비디오 목록 (클릭하여 펼치기/접기)", open=True) as batch_video_accordion:
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

                                    batch_video_count_info = gr.Markdown("**선택된 비디오**: 0개")

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

                            # ===== 비디오별 개별 Annotation 섹션 =====
                            with gr.Accordion("🎬 비디오별 개별 Annotation (선택사항)", open=False):
                                gr.Markdown("""
**각 비디오마다 개별 annotation을 지정하면 더 정확한 결과를 얻을 수 있습니다.**
1. 비디오 선택 → 로드 → annotation → 저장
2. 모든 비디오 annotation 후 → "비디오별 Batch Propagate" 실행
                                """)

                                batch_per_video_select = gr.Dropdown(
                                    label="비디오 선택",
                                    choices=[],
                                    interactive=True,
                                    info="annotation할 비디오를 선택하세요"
                                )

                                with gr.Row():
                                    batch_load_video_btn = gr.Button("📹 로드", variant="secondary", size="sm")
                                    batch_save_video_anno_btn = gr.Button("💾 저장", variant="primary", size="sm")

                                batch_per_video_status = gr.Markdown("### 📋 비디오별 Annotation: 없음")

                                batch_propagate_per_video_btn = gr.Button(
                                    "🔄 비디오별 Batch Propagate",
                                    variant="primary",
                                    size="lg"
                                )

                                gr.Markdown("---")
                                gr.Markdown("**💾 Annotation 파일 저장/로드** (propagation 전에도 가능)")

                                with gr.Row():
                                    batch_save_anno_file_btn = gr.Button("💾 Annotation 파일 저장", size="sm")
                                    batch_scan_anno_files_btn = gr.Button("🔍 파일 스캔", size="sm")

                                batch_anno_file_dropdown = gr.Dropdown(
                                    label="저장된 Annotation 파일",
                                    choices=[],
                                    interactive=True
                                )

                                batch_load_anno_file_btn = gr.Button("📂 Annotation 파일 로드", variant="secondary", size="sm")

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
                            with gr.Accordion("📂 세션 불러오기 / 관리", open=False):
                                batch_session_scan_btn = gr.Button("🔍 세션 스캔", size="sm")
                                batch_load_session_dropdown = gr.Dropdown(
                                    label="세션 선택",
                                    choices=[],
                                    interactive=True
                                )
                                with gr.Row():
                                    batch_load_session_btn = gr.Button("📂 로드", variant="secondary", size="sm")
                                    batch_delete_session_btn = gr.Button("🗑️ 삭제", variant="stop", size="sm")

                                with gr.Accordion("✏️ 세션 이름 변경", open=False):
                                    batch_rename_session_input = gr.Textbox(
                                        label="새 이름",
                                        placeholder="새 세션 이름 입력"
                                    )
                                    batch_rename_session_btn = gr.Button("✏️ 이름 변경", size="sm")

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
                                label="Reference Frame / 결과 시각화",
                                type="numpy"
                            )

                            # 프레임 슬라이더 - 이미지 바로 아래에 배치
                            batch_vis_slider = gr.Slider(
                                label="🎞️ 프레임 (슬라이더로 빠르게 탐색)",
                                minimum=0,
                                maximum=1,
                                value=0,
                                step=1,
                                interactive=True
                            )

                            # 프레임 네비게이션 버튼
                            with gr.Row():
                                batch_vis_prev_btn = gr.Button("◀️ 이전", size="sm")
                                batch_vis_next_btn = gr.Button("▶️ 다음", size="sm")
                                batch_vis_first_btn = gr.Button("⏮️ 처음", size="sm")
                                batch_vis_last_btn = gr.Button("⏭️ 끝", size="sm")

                            batch_vis_info = gr.Markdown("비디오를 선택하고 프레임을 탐색하세요.")

                            batch_status_text = gr.Markdown("### 상태: 대기 중")

                            batch_output_path = gr.Textbox(
                                label="출력 경로",
                                interactive=False
                            )

                            # ===== 결과 시각화 & 퀄리티 체크 섹션 =====
                            with gr.Accordion("🎬 결과 시각화 & 퀄리티 체크", open=True):
                                gr.Markdown("**비디오별로 마스크 결과를 빠르게 확인하세요**")

                                # 비디오 선택
                                with gr.Row():
                                    batch_preview_video_select = gr.Dropdown(
                                        label="📹 비디오 선택",
                                        choices=[],
                                        interactive=True,
                                        scale=2
                                    )
                                    batch_preview_refresh_btn = gr.Button("🔄", size="sm", scale=0)

                                # 디스플레이 모드
                                batch_preview_mode = gr.Radio(
                                    label="표시 모드",
                                    choices=[
                                        ("🎭 Binary Mask", "mask"),
                                        ("🟢 Overlay", "overlay"),
                                        ("📊 Side by Side", "side_by_side")
                                    ],
                                    value="overlay",
                                    interactive=True
                                )

                                # 프레임 슬라이더와 네비게이션 버튼은 이미지 바로 아래로 이동됨 (위 참조)

                                gr.Markdown("---")
                                gr.Markdown("**🎬 프리뷰 영상 생성** (저해상도 빠른 확인)")

                                with gr.Row():
                                    batch_preview_fps = gr.Slider(
                                        label="FPS",
                                        minimum=5,
                                        maximum=30,
                                        value=15,
                                        step=5,
                                        scale=1
                                    )
                                    batch_preview_scale = gr.Slider(
                                        label="해상도 %",
                                        minimum=25,
                                        maximum=100,
                                        value=50,
                                        step=25,
                                        scale=1
                                    )

                                with gr.Row():
                                    batch_gen_preview_btn = gr.Button("🎬 프리뷰 영상 생성", variant="primary")
                                    batch_gen_all_preview_btn = gr.Button("📦 전체 비디오 프리뷰", variant="secondary")

                                # 비디오 플레이어
                                batch_preview_video = gr.Video(
                                    label="프리뷰 영상",
                                    interactive=False,
                                    autoplay=True
                                )

                                gr.Markdown("---")
                                gr.Markdown("**🎯 3D Mesh 생성**")

                                with gr.Accordion("⚙️ Mesh 파라미터", open=False):
                                    batch_mesh_seed = gr.Number(
                                        label="Seed (재현성)",
                                        value=42,
                                        precision=0,
                                        info="동일 seed = 동일 결과"
                                    )
                                    with gr.Row():
                                        batch_mesh_stage1_steps = gr.Slider(
                                            label="Stage1 Steps",
                                            minimum=5,
                                            maximum=50,
                                            value=25,
                                            step=5,
                                            info="Sparse structure 품질"
                                        )
                                        batch_mesh_stage2_steps = gr.Slider(
                                            label="Stage2 Steps",
                                            minimum=5,
                                            maximum=50,
                                            value=25,
                                            step=5,
                                            info="Latent feature 품질"
                                        )
                                    batch_mesh_postprocess = gr.Checkbox(
                                        label="Mesh 후처리 (단순화, 홀 채우기)",
                                        value=False,
                                        info="⚠️ nvdiffrast 필요 - 미설치 시 비활성화 권장"
                                    )
                                    batch_mesh_simplify_ratio = gr.Slider(
                                        label="Simplify Ratio",
                                        minimum=0.5,
                                        maximum=0.99,
                                        value=0.95,
                                        step=0.05,
                                        info="Face 유지 비율 (0.95 = 5% 제거)",
                                        visible=False
                                    )
                                    batch_mesh_texture_baking = gr.Checkbox(
                                        label="Texture Baking",
                                        value=False,
                                        info="텍스처 맵 생성 (추가 시간 필요)"
                                    )
                                    batch_mesh_texture_size = gr.Dropdown(
                                        label="Texture Size",
                                        choices=[512, 1024, 2048],
                                        value=1024,
                                        visible=False
                                    )
                                    batch_mesh_vertex_color = gr.Checkbox(
                                        label="Vertex Color 사용",
                                        value=True,
                                        info="버텍스에 색상 저장"
                                    )

                                    # 후처리 체크박스에 따라 simplify_ratio 표시
                                    batch_mesh_postprocess.change(
                                        fn=lambda x: gr.update(visible=x),
                                        inputs=[batch_mesh_postprocess],
                                        outputs=[batch_mesh_simplify_ratio]
                                    )
                                    # 텍스처 베이킹 체크박스에 따라 texture_size 표시
                                    batch_mesh_texture_baking.change(
                                        fn=lambda x: gr.update(visible=x),
                                        inputs=[batch_mesh_texture_baking],
                                        outputs=[batch_mesh_texture_size]
                                    )

                                with gr.Row():
                                    batch_gen_mesh_btn = gr.Button("🎯 현재 프레임 3D Mesh", variant="primary")
                                    batch_gen_all_mesh_btn = gr.Button("📦 전체 비디오 3D Mesh", variant="secondary")
                                batch_mesh_output = gr.File(label="생성된 3D Mesh", interactive=False)

                                # ===== 프레임 선택 리스트 기능 =====
                                gr.Markdown("---")
                                gr.Markdown("**📋 선택 프레임 일괄 3D Mesh 생성**")
                                gr.Markdown("프레임 탐색 중 원하는 프레임을 선택하여 리스트에 추가한 뒤 일괄 생성하세요.")

                                with gr.Row():
                                    batch_add_frame_btn = gr.Button("➕ 현재 프레임 추가", variant="secondary", size="sm")
                                    batch_clear_frame_list_btn = gr.Button("🗑️ 목록 초기화", size="sm")

                                batch_selected_frames_display = gr.Markdown("**선택된 프레임**: 없음")

                                # 선택된 프레임 저장용 State
                                batch_selected_frames_state = gr.State([])

                                with gr.Row():
                                    batch_gen_selected_mesh_btn = gr.Button(
                                        "🎯 선택 프레임 일괄 3D Mesh 생성",
                                        variant="primary",
                                        size="lg"
                                    )

                                batch_selected_mesh_status = gr.Markdown("")

                                gr.Markdown("---")
                                gr.Markdown("**📤 내보내기**")
                                with gr.Row():
                                    batch_gen_vis_btn = gr.Button("🎨 시각화 이미지 저장", variant="secondary")
                                    batch_gen_video_btn = gr.Button("📹 전체 영상 생성", variant="secondary")

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
                            count = len(all_labels)
                            return gr.CheckboxGroup(value=all_labels), f"**선택된 비디오**: {count}개"
                        return gr.CheckboxGroup(value=[]), "**선택된 비디오**: 0개"

                    def deselect_all_videos():
                        return gr.CheckboxGroup(value=[]), "**선택된 비디오**: 0개"

                    def update_video_count(selected):
                        """비디오 선택 수 업데이트"""
                        count = len(selected) if selected else 0
                        return f"**선택된 비디오**: {count}개"

                    batch_select_all_btn.click(
                        fn=select_all_videos,
                        outputs=[batch_video_selection, batch_video_count_info]
                    )

                    batch_deselect_all_btn.click(
                        fn=deselect_all_videos,
                        outputs=[batch_video_selection, batch_video_count_info]
                    )

                    batch_video_selection.change(
                        fn=update_video_count,
                        inputs=[batch_video_selection],
                        outputs=[batch_video_count_info]
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
                            frame_rgb = self.frames[self.current_frame_idx].copy()  # 이미 RGB
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

                    # 세션 스캔
                    def scan_batch_sessions():
                        """Batch 세션 디렉토리 스캔"""
                        sessions_dir = Path(self.default_output_dir) / "sessions"
                        if not sessions_dir.exists():
                            return gr.Dropdown(choices=[])

                        sessions = []
                        for session_dir in sessions_dir.iterdir():
                            if session_dir.is_dir():
                                # Check for batch session metadata
                                meta_file = session_dir / "session_metadata.json"
                                if meta_file.exists():
                                    sessions.append(str(session_dir))
                        return gr.Dropdown(choices=sorted(sessions, reverse=True))

                    batch_session_scan_btn.click(
                        fn=scan_batch_sessions,
                        outputs=[batch_load_session_dropdown]
                    )

                    # 세션 로드 (비디오 목록도 함께 업데이트)
                    def load_batch_session_and_refresh(session_path):
                        """Batch 세션 로드 후 비디오 목록도 업데이트"""
                        status_msg, output_path = self.load_batch_session(session_path)

                        # 비디오 목록 업데이트 (unique_id 포함: mouse+camera+frame)
                        video_list = self.get_batch_video_list()
                        if video_list:
                            choices = []
                            for v in video_list:
                                # unique_id 사용 (예: m1_cam1_0)
                                unique_id = v.get('unique_id', v['video_name'])
                                label = f"[{v['video_idx']}] {unique_id} ({v['num_frames']}f)"
                                choices.append((label, v['video_idx']))
                            video_dropdown = gr.Dropdown(choices=choices, value=choices[0][1] if choices else None)
                        else:
                            video_dropdown = gr.Dropdown(choices=[], value=None)

                        return status_msg, output_path, video_dropdown

                    batch_load_session_btn.click(
                        fn=load_batch_session_and_refresh,
                        inputs=[batch_load_session_dropdown],
                        outputs=[batch_status_text, batch_output_path, batch_preview_video_select]
                    )

                    # 세션 삭제
                    def delete_session_handler(session_path):
                        msg, sessions = self.delete_batch_session(session_path)
                        return msg, gr.Dropdown(choices=sessions, value=sessions[0] if sessions else None)

                    batch_delete_session_btn.click(
                        fn=delete_session_handler,
                        inputs=[batch_load_session_dropdown],
                        outputs=[batch_status_text, batch_load_session_dropdown]
                    )

                    # 세션 이름 변경
                    def rename_session_handler(session_path, new_name):
                        msg, sessions = self.rename_batch_session(session_path, new_name)
                        return msg, gr.Dropdown(choices=sessions, value=sessions[0] if sessions else None), ""

                    batch_rename_session_btn.click(
                        fn=rename_session_handler,
                        inputs=[batch_load_session_dropdown, batch_rename_session_input],
                        outputs=[batch_status_text, batch_load_session_dropdown, batch_rename_session_input]
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

                    # ===== 비디오별 Annotation 이벤트 핸들러 =====

                    # 비디오 스캔 시 per-video 드롭다운도 업데이트
                    def update_per_video_dropdown():
                        if hasattr(self, 'batch_video_label_map'):
                            labels = list(self.batch_video_label_map.keys())
                            return gr.Dropdown(choices=labels, value=labels[0] if labels else None)
                        return gr.Dropdown(choices=[])

                    batch_scan_btn.click(
                        fn=update_per_video_dropdown,
                        outputs=[batch_per_video_select]
                    )

                    # 비디오 로드 (개별 annotation용)
                    batch_load_video_btn.click(
                        fn=self.load_video_for_annotation,
                        inputs=[batch_per_video_select],
                        outputs=[batch_image_display, batch_status_text]
                    )

                    # 비디오별 annotation 저장
                    def save_video_anno_handler(video_label):
                        msg = self.save_current_annotation_for_video(video_label)
                        status = self.get_per_video_annotation_status()
                        return msg, status

                    batch_save_video_anno_btn.click(
                        fn=save_video_anno_handler,
                        inputs=[batch_per_video_select],
                        outputs=[batch_status_text, batch_per_video_status]
                    )

                    # ===== Annotation 파일 저장/로드 이벤트 핸들러 =====

                    # Annotation 파일 저장
                    def save_anno_file_handler():
                        path, msg = self.save_per_video_annotations_to_file()
                        status = self.get_per_video_annotation_status()
                        return msg, status

                    batch_save_anno_file_btn.click(
                        fn=save_anno_file_handler,
                        outputs=[batch_status_text, batch_per_video_status]
                    )

                    # Annotation 파일 스캔
                    def scan_anno_files_handler():
                        files = self.scan_annotation_files()
                        return gr.Dropdown(choices=files, value=files[0] if files else None)

                    batch_scan_anno_files_btn.click(
                        fn=scan_anno_files_handler,
                        outputs=[batch_anno_file_dropdown]
                    )

                    # Annotation 파일 로드
                    def load_anno_file_handler(filepath):
                        msg, status = self.load_per_video_annotations_from_file(filepath)
                        return msg, status

                    batch_load_anno_file_btn.click(
                        fn=load_anno_file_handler,
                        inputs=[batch_anno_file_dropdown],
                        outputs=[batch_status_text, batch_per_video_status]
                    )

                    # 비디오별 Batch Propagate
                    batch_propagate_per_video_btn.click(
                        fn=self.batch_propagate_with_per_video_annotations,
                        inputs=[batch_target_frames, batch_video_selection],
                        outputs=[batch_status_text, gr.State()]
                    )

                    # ===== 결과 시각화 & 퀄리티 체크 이벤트 핸들러 =====

                    # 현재 선택된 비디오 인덱스 저장
                    current_preview_video_idx = gr.State(value=0)

                    # 비디오 목록 새로고침
                    def refresh_preview_video_list():
                        """프리뷰용 비디오 목록 업데이트 (unique_id 포함: mouse+camera+frame)"""
                        video_list = self.get_batch_video_list()
                        if video_list:
                            choices = []
                            for v in video_list:
                                # unique_id 사용 (예: m1_cam1_0)
                                unique_id = v.get('unique_id', v['video_name'])
                                label = f"[{v['video_idx']}] {unique_id} ({v['num_frames']}f)"
                                choices.append((label, v['video_idx']))
                            return gr.Dropdown(choices=choices, value=choices[0][1] if choices else None)
                        return gr.Dropdown(choices=[], value=None)

                    batch_preview_refresh_btn.click(
                        fn=refresh_preview_video_list,
                        outputs=[batch_preview_video_select]
                    )

                    # 비디오 선택 시 슬라이더 범위 업데이트
                    def on_video_select(video_idx, display_mode):
                        if video_idx is None:
                            return None, "비디오를 선택하세요.", gr.Slider(maximum=1, value=0), video_idx

                        video_list = self.get_batch_video_list()
                        video_info = None
                        for v in video_list:
                            if v['video_idx'] == video_idx:
                                video_info = v
                                break

                        if video_info is None:
                            return None, "비디오를 찾을 수 없습니다.", gr.Slider(maximum=1, value=0), video_idx

                        num_frames = video_info['num_frames']
                        img, status = self.get_video_frame_for_preview(video_idx, 0, display_mode)
                        return img, status, gr.Slider(maximum=max(1, num_frames-1), value=0), video_idx

                    batch_preview_video_select.change(
                        fn=on_video_select,
                        inputs=[batch_preview_video_select, batch_preview_mode],
                        outputs=[batch_image_display, batch_vis_info, batch_vis_slider, current_preview_video_idx]
                    )

                    # 디스플레이 모드 변경 시
                    def on_mode_change(video_idx, frame_idx, display_mode):
                        if video_idx is None:
                            return None, "비디오를 선택하세요."
                        img, status = self.get_video_frame_for_preview(video_idx, int(frame_idx), display_mode)
                        return img, status

                    batch_preview_mode.change(
                        fn=on_mode_change,
                        inputs=[current_preview_video_idx, batch_vis_slider, batch_preview_mode],
                        outputs=[batch_image_display, batch_vis_info]
                    )

                    # 슬라이더 변경 시
                    def on_frame_slider_change(video_idx, frame_idx, display_mode):
                        if video_idx is None:
                            return None, "비디오를 선택하세요."
                        img, status = self.get_video_frame_for_preview(video_idx, int(frame_idx), display_mode)
                        return img, status

                    batch_vis_slider.change(
                        fn=on_frame_slider_change,
                        inputs=[current_preview_video_idx, batch_vis_slider, batch_preview_mode],
                        outputs=[batch_image_display, batch_vis_info]
                    )

                    # 프레임 네비게이션 버튼
                    def frame_nav(video_idx, current_idx, display_mode, direction):
                        if video_idx is None:
                            return None, "비디오를 선택하세요.", 0

                        video_list = self.get_batch_video_list()
                        video_info = None
                        for v in video_list:
                            if v['video_idx'] == video_idx:
                                video_info = v
                                break

                        if video_info is None:
                            return None, "비디오 없음", 0

                        max_idx = video_info['num_frames'] - 1

                        if direction == "prev":
                            new_idx = max(0, int(current_idx) - 1)
                        elif direction == "next":
                            new_idx = min(max_idx, int(current_idx) + 1)
                        elif direction == "first":
                            new_idx = 0
                        elif direction == "last":
                            new_idx = max_idx
                        else:
                            new_idx = int(current_idx)

                        img, status = self.get_video_frame_for_preview(video_idx, new_idx, display_mode)
                        return img, status, new_idx

                    batch_vis_prev_btn.click(
                        fn=lambda v, c, m: frame_nav(v, c, m, "prev"),
                        inputs=[current_preview_video_idx, batch_vis_slider, batch_preview_mode],
                        outputs=[batch_image_display, batch_vis_info, batch_vis_slider]
                    )

                    batch_vis_next_btn.click(
                        fn=lambda v, c, m: frame_nav(v, c, m, "next"),
                        inputs=[current_preview_video_idx, batch_vis_slider, batch_preview_mode],
                        outputs=[batch_image_display, batch_vis_info, batch_vis_slider]
                    )

                    batch_vis_first_btn.click(
                        fn=lambda v, c, m: frame_nav(v, c, m, "first"),
                        inputs=[current_preview_video_idx, batch_vis_slider, batch_preview_mode],
                        outputs=[batch_image_display, batch_vis_info, batch_vis_slider]
                    )

                    batch_vis_last_btn.click(
                        fn=lambda v, c, m: frame_nav(v, c, m, "last"),
                        inputs=[current_preview_video_idx, batch_vis_slider, batch_preview_mode],
                        outputs=[batch_image_display, batch_vis_info, batch_vis_slider]
                    )

                    # 프리뷰 영상 생성
                    def generate_single_preview(video_idx, display_mode, fps, scale_percent):
                        if video_idx is None:
                            return None, "비디오를 선택하세요."
                        scale = scale_percent / 100.0
                        video_path, status = self.generate_preview_video(video_idx, display_mode, int(fps), scale)
                        return video_path if video_path else None, status

                    batch_gen_preview_btn.click(
                        fn=generate_single_preview,
                        inputs=[current_preview_video_idx, batch_preview_mode, batch_preview_fps, batch_preview_scale],
                        outputs=[batch_preview_video, batch_status_text]
                    )

                    # 전체 비디오 프리뷰 생성
                    def generate_all_previews(display_mode, fps, scale_percent, progress=gr.Progress()):
                        video_list = self.get_batch_video_list()
                        if not video_list:
                            return None, "결과가 없습니다."

                        scale = scale_percent / 100.0
                        last_video_path = None
                        results = []

                        for i, video_info in enumerate(video_list):
                            # unique_id 사용 (m1_cam1_0 형식)
                            unique_id = video_info.get('unique_id', video_info['video_name'])
                            progress(i / len(video_list), desc=f"🎬 {unique_id} 생성 중... ({i+1}/{len(video_list)})")
                            video_path, status = self.generate_preview_video(
                                video_info['video_idx'], display_mode, int(fps), scale
                            )
                            if video_path:
                                last_video_path = video_path
                                results.append(unique_id)

                        progress(1.0, desc="✅ 완료!")

                        status = f"""
### 📦 전체 프리뷰 생성 완료 ✅

- **생성된 비디오**: {len(results)}개 / {len(video_list)}개
- **저장 위치**: `{Path(self.default_output_dir) / 'previews'}`

<details>
<summary><b>📋 생성된 프리뷰 목록 ({len(results)}개) - 클릭하여 펼치기</b></summary>

{chr(10).join([f'- {r}' for r in results])}

</details>
"""
                        return last_video_path, status

                    batch_gen_all_preview_btn.click(
                        fn=generate_all_previews,
                        inputs=[batch_preview_mode, batch_preview_fps, batch_preview_scale],
                        outputs=[batch_preview_video, batch_status_text]
                    )

                    # 시각화 이미지 저장
                    batch_gen_vis_btn.click(
                        fn=lambda: self.generate_batch_visualization(output_format="images"),
                        outputs=[batch_output_path, batch_status_text]
                    )

                    # 전체 시각화 영상 생성
                    batch_gen_video_btn.click(
                        fn=lambda: self.generate_batch_visualization(output_format="video"),
                        outputs=[batch_output_path, batch_status_text]
                    )

                    # 현재 프레임 3D Mesh 생성
                    batch_gen_mesh_btn.click(
                        fn=lambda video_idx, frame_idx, seed, s1, s2, pp, sr, tb, ts, vc: self.batch_generate_3d_mesh_current(
                            video_idx, int(frame_idx), seed, s1, s2, pp, sr, tb, ts, vc
                        ),
                        inputs=[
                            current_preview_video_idx, batch_vis_slider,
                            batch_mesh_seed, batch_mesh_stage1_steps, batch_mesh_stage2_steps,
                            batch_mesh_postprocess, batch_mesh_simplify_ratio,
                            batch_mesh_texture_baking, batch_mesh_texture_size, batch_mesh_vertex_color
                        ],
                        outputs=[batch_mesh_output, batch_status_text]
                    )

                    # 전체 비디오 3D Mesh 생성
                    batch_gen_all_mesh_btn.click(
                        fn=self.batch_generate_3d_mesh_all,
                        inputs=[
                            batch_mesh_seed, batch_mesh_stage1_steps, batch_mesh_stage2_steps,
                            batch_mesh_postprocess, batch_mesh_simplify_ratio,
                            batch_mesh_texture_baking, batch_mesh_texture_size, batch_mesh_vertex_color
                        ],
                        outputs=[batch_mesh_output, batch_status_text]
                    )

                    # ===== 프레임 선택 리스트 이벤트 핸들러 =====
                    def add_current_frame_to_list(video_idx, frame_idx, selected_frames):
                        """현재 프레임을 선택 목록에 추가"""
                        if video_idx is None or frame_idx is None:
                            return selected_frames, "**선택된 프레임**: 비디오/프레임을 먼저 선택하세요"

                        video_list = self.get_batch_video_list()
                        video_name = "unknown"
                        for v in video_list:
                            if v['video_idx'] == video_idx:
                                video_name = v['video_name']
                                break

                        frame_info = {
                            'video_idx': int(video_idx),
                            'video_name': video_name,
                            'frame_idx': int(frame_idx)
                        }

                        # 중복 체크
                        for existing in selected_frames:
                            if existing['video_idx'] == frame_info['video_idx'] and existing['frame_idx'] == frame_info['frame_idx']:
                                # 이미 존재함
                                display = format_selected_frames_display(selected_frames)
                                return selected_frames, display + "\n\n⚠️ 이미 추가된 프레임입니다."

                        selected_frames.append(frame_info)
                        display = format_selected_frames_display(selected_frames)
                        return selected_frames, display

                    def clear_frame_list():
                        """프레임 목록 초기화"""
                        return [], "**선택된 프레임**: 없음"

                    def format_selected_frames_display(selected_frames):
                        """선택된 프레임 목록 표시 텍스트 생성"""
                        if not selected_frames:
                            return "**선택된 프레임**: 없음"

                        display = f"**선택된 프레임**: {len(selected_frames)}개\n\n"
                        for i, f in enumerate(selected_frames):
                            display += f"{i+1}. **{f['video_name']}** - frame {f['frame_idx']}\n"
                        return display

                    def generate_selected_meshes(selected_frames, seed, s1, s2, pp, sr, tb, ts, vc, progress=gr.Progress()):
                        """선택된 프레임들의 3D Mesh 생성"""
                        output_path, status = self.batch_generate_3d_mesh_selected(
                            selected_frames, seed, s1, s2, pp, sr, tb, ts, vc, progress
                        )
                        return status

                    batch_add_frame_btn.click(
                        fn=add_current_frame_to_list,
                        inputs=[current_preview_video_idx, batch_vis_slider, batch_selected_frames_state],
                        outputs=[batch_selected_frames_state, batch_selected_frames_display]
                    )

                    batch_clear_frame_list_btn.click(
                        fn=clear_frame_list,
                        outputs=[batch_selected_frames_state, batch_selected_frames_display]
                    )

                    batch_gen_selected_mesh_btn.click(
                        fn=generate_selected_meshes,
                        inputs=[
                            batch_selected_frames_state,
                            batch_mesh_seed, batch_mesh_stage1_steps, batch_mesh_stage2_steps,
                            batch_mesh_postprocess, batch_mesh_simplify_ratio,
                            batch_mesh_texture_baking, batch_mesh_texture_size, batch_mesh_vertex_color
                        ],
                        outputs=[batch_selected_mesh_status]
                    )

                    # Batch propagate 완료 후 자동으로 프리뷰 목록 업데이트
                    def on_propagate_complete():
                        return refresh_preview_video_list()

                    batch_propagate_btn.click(
                        fn=on_propagate_complete,
                        outputs=[batch_preview_video_select]
                    )

                    batch_propagate_per_video_btn.click(
                        fn=on_propagate_complete,
                        outputs=[batch_preview_video_select]
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
                            # SAM2 모델 상태 안내
                            gr.Markdown("#### 🤖 SAM2 Model")
                            gr.Markdown("*상단의 공용 SAM2 모델을 사용합니다. 로드되지 않은 경우 상단에서 먼저 로드하세요.*")

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

                            with gr.Accordion("✂️ Crop-Based Scale Augmentation (Advanced)", open=True):
                                aug_crop_enable = gr.Checkbox(
                                    label="Enable Crop-Based Scale",
                                    value=False,
                                    info="Crop mask region, scale it, and paste on white background"
                                )
                                with gr.Row():
                                    aug_crop_scale_min = gr.Slider(
                                        label="Crop Scale Min",
                                        minimum=0.3, maximum=1.0, value=0.5, step=0.05
                                    )
                                    aug_crop_scale_max = gr.Slider(
                                        label="Crop Scale Max",
                                        minimum=1.0, maximum=3.0, value=2.0, step=0.1
                                    )

                                with gr.Row():
                                    aug_offset_x_max = gr.Slider(
                                        label="Max Horizontal Offset (ratio)",
                                        minimum=0.0, maximum=0.5, value=0.2, step=0.05,
                                        info="Offset as ratio of image width"
                                    )
                                    aug_offset_y_max = gr.Slider(
                                        label="Max Vertical Offset (ratio)",
                                        minimum=0.0, maximum=0.5, value=0.2, step=0.05,
                                        info="Offset as ratio of image height"
                                    )

                                aug_crop_padding = gr.Slider(
                                    label="Crop Padding (pixels)",
                                    minimum=0, maximum=100, value=20, step=5,
                                    info="Extra padding around mask bbox"
                                )

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

                            with gr.Accordion("🖼️ Background Replacement", open=True):
                                aug_replace_bg = gr.Checkbox(
                                    label="Enable Background Replacement",
                                    value=True,
                                    info="Replace background with images or solid color"
                                )
                                aug_bg_image_ratio = gr.Slider(
                                    label="Background Image Ratio",
                                    minimum=0.0, maximum=1.0, value=0.5, step=0.1,
                                    info="Probability of using background image (vs solid color)"
                                )
                                aug_bg_folder = gr.Textbox(
                                    label="Background Images Folder",
                                    value=self.config.augmentation_background_folder if self.config else "",
                                    placeholder="/path/to/background/images"
                                )
                                aug_load_bg_btn = gr.Button("📂 Load Background Images", size="sm")
                                aug_bg_status = gr.Markdown("No background images loaded")

                            # Safety options
                            gr.Markdown("#### 🛡️ Safety Options")
                            aug_prevent_clipping = gr.Checkbox(
                                label="Prevent Object Clipping",
                                value=True,
                                info="Auto-offset to prevent object from being clipped at image boundaries"
                            )

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

                            # Quality Report
                            gr.Markdown("#### 📊 Quality Analysis")
                            with gr.Row():
                                aug_analyze_btn = gr.Button(
                                    "📈 Generate Quality Report",
                                    variant="secondary",
                                    size="lg"
                                )

                            with gr.Accordion("⚙️ Analysis Settings", open=False):
                                aug_feature_type = gr.Dropdown(
                                    label="Feature Type",
                                    choices=["simple", "resnet"],
                                    value="simple",
                                    info="Simple: histogram-based, ResNet: deep learning features"
                                )
                                aug_cluster_method = gr.Dropdown(
                                    label="Clustering Method",
                                    choices=["kmeans", "dbscan"],
                                    value="kmeans"
                                )
                                aug_n_clusters = gr.Slider(
                                    label="Number of Clusters",
                                    minimum=2, maximum=10, value=5, step=1
                                )

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
                            sessions = set()  # Use set to avoid duplicates

                            # Search for all session_metadata.json files (both interactive and batch)
                            for session_file in session_path.rglob("session_metadata.json"):
                                try:
                                    with open(session_file, 'r') as f:
                                        metadata = json.load(f)
                                        # Include all sessions regardless of type
                                        # (interactive, batch, or unspecified)
                                        session_dir_path = str(session_file.parent)
                                        sessions.add(session_dir_path)
                                except Exception as e:
                                    # If can't read metadata, still add it
                                    session_dir_path = str(session_file.parent)
                                    sessions.add(session_dir_path)

                            # Also search for legacy session.json files (for backward compatibility)
                            for session_file in session_path.rglob("session.json"):
                                try:
                                    session_dir_path = str(session_file.parent)
                                    # Only add if not already in set from session_metadata.json
                                    if session_dir_path not in sessions:
                                        with open(session_file, 'r') as f:
                                            metadata = json.load(f)
                                            # Skip batch sessions (they should have session_metadata.json)
                                            if metadata.get('session_type') != 'batch':
                                                sessions.add(session_dir_path)
                                except:
                                    # If can't read, add it anyway
                                    session_dir_path = str(session_file.parent)
                                    if session_dir_path not in sessions:
                                        sessions.add(session_dir_path)

                            if not sessions:
                                return gr.Dropdown(choices=[]), "⚠️ No sessions found"

                            # Convert set to sorted list
                            session_list = sorted(list(sessions))
                            return gr.Dropdown(choices=session_list), f"✅ Found {len(session_list)} sessions"
                        except Exception as e:
                            return gr.Dropdown(choices=[]), f"❌ Error: {str(e)}"

                    aug_scan_btn.click(
                        fn=scan_aug_sessions,
                        inputs=[aug_session_dir],
                        outputs=[aug_session_list, aug_session_info]
                    )

                    # Load background images
                    def load_bg_images(folder_path):
                        """Load background images from folder"""
                        if not folder_path or not Path(folder_path).exists():
                            return f"❌ Folder not found: {folder_path}"

                        count = self.augmentor.load_background_images(folder_path)
                        if count > 0:
                            return f"✅ Loaded {count} background images"
                        else:
                            return "⚠️ No valid images found (jpg, jpeg, png)"

                    aug_load_bg_btn.click(
                        fn=load_bg_images,
                        inputs=[aug_bg_folder],
                        outputs=[aug_bg_status]
                    )

                    # Load session
                    def load_aug_session(session_path):
                        """Load annotation session for augmentation"""
                        try:
                            if not session_path:
                                return None, "⚠️ Please select a session"

                            session_path = Path(session_path)

                            # Try both session metadata formats
                            session_metadata_file = session_path / "session_metadata.json"
                            session_file = session_path / "session.json"

                            metadata = None
                            if session_metadata_file.exists():
                                with open(session_metadata_file, 'r') as f:
                                    metadata = json.load(f)
                            elif session_file.exists():
                                with open(session_file, 'r') as f:
                                    metadata = json.load(f)
                            else:
                                return None, f"❌ No session metadata found in {session_path}"

                            # Detect format (flat vs Fauna vs Batch)
                            flat_rgb_dir = session_path / "rgb"
                            flat_mask_dir = session_path / "masks"
                            is_flat_format = flat_rgb_dir.exists() and flat_mask_dir.exists()
                            is_batch_format = metadata.get('session_type') == 'batch'

                            # Count frames
                            frame_count = 0
                            if is_batch_format:
                                # Batch format: video_XXX/frame_XXXX/original.png
                                video_dirs = [d for d in session_path.iterdir()
                                              if d.is_dir() and d.name.startswith('video_')]
                                for video_dir in video_dirs:
                                    frame_dirs = [f for f in video_dir.iterdir()
                                                  if f.is_dir() and f.name.startswith('frame_')]
                                    # Count frames with original.png (batch format)
                                    frame_count += len([f for f in frame_dirs
                                                        if (f / "original.png").exists()])
                                format_type = "Batch (video_XXX/frame_XXXX/)"
                            elif is_flat_format:
                                frame_count = len(list(flat_rgb_dir.glob("*.png")))
                                format_type = "Flat (rgb/, masks/)"
                            else:
                                # Fauna format - count frame directories
                                frame_dirs = [d for d in session_path.iterdir() if d.is_dir()]
                                frame_count = len([d for d in frame_dirs if (d / "rgb.png").exists()])
                                format_type = "Fauna (frame directories)"

                            # Store for augmentation
                            self.aug_session_path = session_path
                            self.aug_metadata = metadata

                            session_type = metadata.get('session_type', 'unknown')
                            fauna_compat = metadata.get('fauna_compatible', False)

                            info = f"""
✅ Session loaded successfully

**Session ID:** {metadata.get('session_id', 'N/A')}
**Type:** {session_type}
**Format:** {format_type}
**Fauna Compatible:** {'✅' if fauna_compat else '⚠️'}
**Frames:** {frame_count} frames
**Created:** {metadata.get('timestamp', metadata.get('created_at', 'N/A'))}
"""
                            return None, info
                        except Exception as e:
                            import traceback
                            return None, f"❌ Error loading session: {str(e)}\n{traceback.format_exc()}"

                    aug_load_session_btn.click(
                        fn=load_aug_session,
                        inputs=[aug_session_list],
                        outputs=[aug_preview_display, aug_session_info]
                    )

                    # Generate preview
                    def generate_aug_preview(
                        rows, cols,
                        scale_enable, scale_min, scale_max, fill_color,
                        crop_enable, crop_scale_min, crop_scale_max,
                        offset_x_max, offset_y_max, crop_padding,
                        rotation_enable, rotation_min, rotation_max,
                        flip_enable,
                        noise_enable, noise_std,
                        brightness_enable, brightness_min, brightness_max,
                        contrast_enable, color_jitter_enable, blur_enable,
                        replace_bg, bg_image_ratio, prevent_clipping
                    ):
                        """Generate augmentation preview grid"""
                        try:
                            if not hasattr(self, 'aug_session_path'):
                                return None, "❌ Please load a session first"

                            # Load first frame and mask (support flat, Fauna, and Batch formats)
                            rgb_files = []
                            mask_files = []

                            # Check for flat format
                            flat_rgb_dir = self.aug_session_path / "rgb"
                            flat_mask_dir = self.aug_session_path / "masks"

                            # Check for batch format
                            is_batch = hasattr(self, 'aug_metadata') and self.aug_metadata.get('session_type') == 'batch'

                            if flat_rgb_dir.exists() and flat_mask_dir.exists():
                                # Flat format
                                rgb_files = sorted(flat_rgb_dir.glob("*.png"))
                                mask_files = sorted(flat_mask_dir.glob("*.png"))
                            elif is_batch:
                                # Batch format: video_XXX/frame_XXXX/original.png + mask.png
                                video_dirs = sorted([d for d in self.aug_session_path.iterdir()
                                                    if d.is_dir() and d.name.startswith('video_')])
                                for video_dir in video_dirs:
                                    frame_dirs = sorted([f for f in video_dir.iterdir()
                                                        if f.is_dir() and f.name.startswith('frame_')])
                                    for frame_dir in frame_dirs:
                                        rgb_file = frame_dir / "original.png"
                                        mask_file = frame_dir / "mask.png"
                                        if rgb_file.exists() and mask_file.exists():
                                            rgb_files.append(rgb_file)
                                            mask_files.append(mask_file)
                                            break  # Just need first frame for preview
                                    if rgb_files:
                                        break
                            else:
                                # Fauna format
                                frame_dirs = sorted([d for d in self.aug_session_path.iterdir() if d.is_dir()])
                                for frame_dir in frame_dirs:
                                    rgb_file = frame_dir / "rgb.png"
                                    mask_file = frame_dir / "mask.png"
                                    if rgb_file.exists() and mask_file.exists():
                                        rgb_files.append(rgb_file)
                                        mask_files.append(mask_file)
                                        break  # Just need first frame for preview

                            if not rgb_files or not mask_files:
                                return None, "❌ No RGB or mask files found in session"

                            # Load first frame
                            rgb = cv2.imread(str(rgb_files[0]))
                            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

                            mask = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE)
                            mask = mask > 127  # Convert to boolean

                            # Generate random configs
                            num_variations = int(rows * cols)
                            configs = []

                            import random
                            for _ in range(num_variations):
                                config = {'fill_color': fill_color}

                                # Crop-based scale takes precedence over regular scale
                                if crop_enable:
                                    config['crop_scale'] = random.uniform(crop_scale_min, crop_scale_max)
                                    config['crop_offset_x'] = random.uniform(-offset_x_max, offset_x_max)
                                    config['crop_offset_y'] = random.uniform(-offset_y_max, offset_y_max)
                                    config['crop_padding'] = int(crop_padding)
                                elif scale_enable:
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

                                # Background replacement
                                if replace_bg:
                                    config['replace_background'] = True
                                    config['use_bg_image'] = True
                                    config['bg_image_ratio'] = bg_image_ratio

                                # Prevent clipping
                                if prevent_clipping:
                                    config['prevent_clipping'] = True

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
                            aug_crop_enable, aug_crop_scale_min, aug_crop_scale_max,
                            aug_offset_x_max, aug_offset_y_max, aug_crop_padding,
                            aug_rotation_enable, aug_rotation_min, aug_rotation_max,
                            aug_flip_enable,
                            aug_noise_enable, aug_noise_std,
                            aug_brightness_enable, aug_brightness_min, aug_brightness_max,
                            aug_contrast_enable, aug_color_jitter_enable, aug_blur_enable,
                            aug_replace_bg, aug_bg_image_ratio, aug_prevent_clipping
                        ],
                        outputs=[aug_preview_display, aug_status]
                    )

                    # Apply batch augmentation
                    def apply_batch_augmentation(
                        multiplier, output_dir,
                        scale_enable, scale_min, scale_max, fill_color,
                        crop_enable, crop_scale_min, crop_scale_max,
                        offset_x_max, offset_y_max, crop_padding,
                        rotation_enable, rotation_min, rotation_max,
                        flip_enable,
                        noise_enable, noise_std,
                        brightness_enable, brightness_min, brightness_max,
                        contrast_enable, color_jitter_enable, blur_enable,
                        replace_bg, bg_image_ratio,
                        prevent_clipping_enable=True,
                        progress=gr.Progress()
                    ):
                        """Apply augmentation to all frames in session"""
                        from datetime import datetime

                        try:
                            if not hasattr(self, 'aug_session_path'):
                                return "❌ Please load a session first", ""

                            progress(0, desc="🔍 Loading frames...")

                            output_path = Path(output_dir)
                            output_path.mkdir(parents=True, exist_ok=True)

                            # Load all frames (support flat, Fauna, and Batch formats)
                            rgb_files = []
                            mask_files = []
                            frame_indices = []  # Track original frame index for naming

                            # Check for flat format (rgb/ and masks/ folders)
                            flat_rgb_dir = self.aug_session_path / "rgb"
                            flat_mask_dir = self.aug_session_path / "masks"

                            # Check for batch format
                            is_batch = hasattr(self, 'aug_metadata') and self.aug_metadata.get('session_type') == 'batch'

                            if flat_rgb_dir.exists() and flat_mask_dir.exists():
                                # Flat format
                                rgb_files = sorted(flat_rgb_dir.glob("*.png"))
                                mask_files = sorted(flat_mask_dir.glob("*.png"))
                                frame_indices = list(range(len(rgb_files)))
                            elif is_batch:
                                # Batch format: video_XXX/frame_XXXX/original.png + mask.png
                                global_idx = 0
                                video_dirs = sorted([d for d in self.aug_session_path.iterdir()
                                                    if d.is_dir() and d.name.startswith('video_')])
                                for video_dir in video_dirs:
                                    frame_dirs = sorted([f for f in video_dir.iterdir()
                                                        if f.is_dir() and f.name.startswith('frame_')])
                                    for frame_dir in frame_dirs:
                                        rgb_file = frame_dir / "original.png"
                                        mask_file = frame_dir / "mask.png"
                                        if rgb_file.exists() and mask_file.exists():
                                            rgb_files.append(rgb_file)
                                            mask_files.append(mask_file)
                                            frame_indices.append(global_idx)
                                            global_idx += 1
                            else:
                                # Fauna format (frame directories)
                                frame_dirs = sorted([d for d in self.aug_session_path.iterdir() if d.is_dir()])
                                for idx, frame_dir in enumerate(frame_dirs):
                                    rgb_file = frame_dir / "rgb.png"
                                    mask_file = frame_dir / "mask.png"
                                    if rgb_file.exists() and mask_file.exists():
                                        rgb_files.append(rgb_file)
                                        mask_files.append(mask_file)
                                        frame_indices.append(idx)

                            total_frames = len(rgb_files)
                            total_outputs = total_frames * int(multiplier)

                            progress(0.05, desc=f"🚀 Processing {total_frames} frames × {int(multiplier)} = {total_outputs} outputs...")

                            import random
                            processed = 0

                            for idx, (rgb_file, mask_file) in enumerate(zip(rgb_files, mask_files)):
                                # Load frame
                                rgb = cv2.imread(str(rgb_file))
                                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

                                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                                mask = mask > 127

                                # Get frame index for naming (use tracked index if available)
                                frame_idx = frame_indices[idx] if idx < len(frame_indices) else idx

                                # Generate augmentations
                                for aug_idx in range(int(multiplier)):
                                    # Random config
                                    config = {'fill_color': fill_color}

                                    # Crop-based scale takes precedence
                                    if crop_enable:
                                        config['crop_scale'] = random.uniform(crop_scale_min, crop_scale_max)
                                        config['crop_offset_x'] = random.uniform(-offset_x_max, offset_x_max)
                                        config['crop_offset_y'] = random.uniform(-offset_y_max, offset_y_max)
                                        config['crop_padding'] = int(crop_padding)
                                    elif scale_enable:
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

                                    # Background replacement
                                    if replace_bg:
                                        config['replace_background'] = True
                                        config['use_bg_image'] = True
                                        config['bg_image_ratio'] = bg_image_ratio

                                    # Prevent clipping option
                                    if prevent_clipping_enable:
                                        config['prevent_clipping'] = True

                                    # Apply augmentation
                                    aug_rgb, aug_mask, applied = self.augmentor.augment(rgb, mask, config)

                                    # Save in Fauna-compatible format (frame directories)
                                    # Use frame index for unique naming (avoids overwrite when all files are original.png)
                                    frame_dir_name = f"frame_{frame_idx:04d}_aug{aug_idx:02d}"
                                    frame_dir = output_path / frame_dir_name
                                    frame_dir.mkdir(parents=True, exist_ok=True)

                                    # Save RGB as rgb.png
                                    rgb_bgr = cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR)
                                    cv2.imwrite(str(frame_dir / "rgb.png"), rgb_bgr)

                                    # Save mask as mask.png
                                    mask_img = (aug_mask * 255).astype(np.uint8)
                                    cv2.imwrite(str(frame_dir / "mask.png"), mask_img)

                                    processed += 1

                                # Update progress
                                progress_pct = 0.05 + 0.90 * (idx + 1) / total_frames
                                progress(progress_pct, desc=f"⏳ Frame {idx + 1}/{total_frames} ({processed}/{total_outputs} outputs)")

                            # Save metadata
                            metadata = {
                                'session_id': f"augmented_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                'session_type': 'augmented',
                                'source_session': str(self.aug_session_path),
                                'original_frames': total_frames,
                                'multiplier': int(multiplier),
                                'total_augmented': processed,
                                'fauna_compatible': True,
                                'augmentation_params': {
                                    'crop_scale': {'enabled': crop_enable, 'min': crop_scale_min, 'max': crop_scale_max} if crop_enable else None,
                                    'scale': {'enabled': scale_enable, 'min': scale_min, 'max': scale_max} if scale_enable else None,
                                    'rotation': {'enabled': rotation_enable, 'min': rotation_min, 'max': rotation_max} if rotation_enable else None,
                                    'flip': flip_enable,
                                    'noise': {'enabled': noise_enable, 'std': noise_std} if noise_enable else None,
                                    'brightness': {'enabled': brightness_enable, 'min': brightness_min, 'max': brightness_max} if brightness_enable else None,
                                    'contrast': contrast_enable,
                                    'color_jitter': color_jitter_enable,
                                    'blur': blur_enable,
                                    'fill_color': fill_color,
                                    'offset_x_max': offset_x_max,
                                    'offset_y_max': offset_y_max,
                                    'crop_padding': int(crop_padding),
                                    'prevent_clipping': prevent_clipping_enable
                                },
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }

                            progress(1.0, desc="✅ Complete!")

                            # Save as both augmentation_metadata.json and session_metadata.json
                            with open(output_path / "augmentation_metadata.json", 'w') as f:
                                json.dump(metadata, f, indent=2)

                            with open(output_path / "session_metadata.json", 'w') as f:
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
                            aug_crop_enable, aug_crop_scale_min, aug_crop_scale_max,
                            aug_offset_x_max, aug_offset_y_max, aug_crop_padding,
                            aug_rotation_enable, aug_rotation_min, aug_rotation_max,
                            aug_flip_enable,
                            aug_noise_enable, aug_noise_std,
                            aug_brightness_enable, aug_brightness_min, aug_brightness_max,
                            aug_contrast_enable, aug_color_jitter_enable, aug_blur_enable,
                            aug_replace_bg, aug_bg_image_ratio,
                            aug_prevent_clipping
                        ],
                        outputs=[aug_status, aug_progress]
                    )

                    # Generate quality report
                    def generate_quality_report(
                        output_dir, feature_type, cluster_method, n_clusters
                    ):
                        """Generate HTML quality report for augmented images"""
                        try:
                            from feature_clustering import analyze_augmentation_quality
                            from html_report_generator import generate_html_report

                            output_path = Path(output_dir)
                            if not output_path.exists():
                                return "❌ Output directory not found. Please run augmentation first."

                            # Find all RGB images (support both flat and Fauna formats)
                            image_paths = []

                            # Check for flat format
                            rgb_dir = output_path / "rgb"
                            if rgb_dir.exists():
                                # Flat format
                                image_paths = list(rgb_dir.glob("*.png")) + list(rgb_dir.glob("*.jpg"))
                            else:
                                # Fauna format - collect rgb.png from all frame directories
                                frame_dirs = [d for d in output_path.iterdir() if d.is_dir()]
                                for frame_dir in frame_dirs:
                                    rgb_file = frame_dir / "rgb.png"
                                    if rgb_file.exists():
                                        image_paths.append(rgb_file)

                            if len(image_paths) < 2:
                                return f"❌ Not enough images for analysis ({len(image_paths)} found, need at least 2)"

                            # Run analysis
                            msg = f"🔍 Analyzing {len(image_paths)} images...\n"
                            results = analyze_augmentation_quality(
                                image_paths=image_paths,
                                output_dir=output_path,
                                feature_type=feature_type,
                                cluster_method=cluster_method,
                                n_clusters=int(n_clusters),
                                vis_method='tsne'
                            )

                            # Generate HTML report
                            html_path = output_path / "quality_report.html"
                            generate_html_report(
                                results=results,
                                output_path=html_path,
                                include_images=True,
                                max_images_per_cluster=5
                            )

                            metrics = results['metrics']
                            msg += f"\n✅ Analysis complete!\n\n"
                            msg += f"**Metrics:**\n"
                            msg += f"- Clusters: {metrics['n_clusters']}\n"
                            msg += f"- Silhouette Score: {metrics.get('silhouette_score', 0):.3f}\n"
                            msg += f"- Davies-Bouldin Score: {metrics.get('davies_bouldin_score', 0):.3f}\n"
                            msg += f"\n📄 **Report saved:** {html_path}\n"
                            msg += f"Open in browser to view interactive results."

                            return msg

                        except Exception as e:
                            import traceback
                            return f"❌ Error: {str(e)}\n{traceback.format_exc()}"

                    aug_analyze_btn.click(
                        fn=generate_quality_report,
                        inputs=[
                            aug_output_dir,
                            aug_feature_type,
                            aug_cluster_method,
                            aug_n_clusters
                        ],
                        outputs=[aug_status]
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
