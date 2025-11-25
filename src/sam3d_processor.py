"""
SAM 3D Object Processor
Handles video processing, segmentation, and 3D reconstruction
"""
import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from PIL import Image
import trimesh

# Determine SAM3D path (supports both submodule and standalone)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
SAM3D_SUBMODULE_PATH = PROJECT_ROOT / "external" / "sam-3d-objects"
SAM3D_STANDALONE_PATH = Path.home() / "dev" / "sam-3d-objects"

# Use submodule if available, otherwise fall back to standalone
if SAM3D_SUBMODULE_PATH.exists():
    SAM3D_PATH = str(SAM3D_SUBMODULE_PATH)
elif SAM3D_STANDALONE_PATH.exists():
    SAM3D_PATH = str(SAM3D_STANDALONE_PATH)
else:
    SAM3D_PATH = None  # Will be handled during initialization

if SAM3D_PATH:
    sys.path.insert(0, SAM3D_PATH)
    sys.path.insert(0, os.path.join(SAM3D_PATH, "notebook"))

try:
    from inference import Inference, load_image
except ImportError:
    # Will be handled during SAM3D initialization
    Inference = None
    load_image = None


@dataclass
class SegmentInfo:
    """Information about a segmented object"""
    frame_idx: int
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    center: Tuple[int, int]
    area: float


@dataclass
class TrackingResult:
    """Result of object tracking across frames"""
    start_frame: int
    end_frame: int
    segments: List[SegmentInfo]
    motion_detected: bool
    duration_seconds: float


class SAM3DProcessor:
    """Main processor for SAM 3D object segmentation and reconstruction"""

    def __init__(self, sam3d_checkpoint_path: str = None, enable_fp16: bool = True):
        """
        Initialize SAM 3D processor

        Args:
            sam3d_checkpoint_path: Path to SAM 3D checkpoint directory
            enable_fp16: Use FP16 mixed precision to reduce memory usage
        """
        if sam3d_checkpoint_path:
            self.sam3d_checkpoint = sam3d_checkpoint_path
        elif SAM3D_PATH:
            self.sam3d_checkpoint = os.path.join(SAM3D_PATH, "checkpoints/hf")
        else:
            self.sam3d_checkpoint = None
            print("Warning: SAM 3D Objects not found. 3D reconstruction will be unavailable.")
            print("Run setup.sh to install sam-3d-objects as a submodule.")

        self.inference_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enable_fp16 = enable_fp16 and torch.cuda.is_available()

        # Memory management
        self._model_loaded = False
        self._last_inference_time = None

    def initialize_sam3d(self, force_reload: bool = False):
        """
        Lazy initialization of SAM 3D model with memory optimization

        Args:
            force_reload: Force reload even if model already loaded
        """
        if self.inference_model is not None and not force_reload:
            print(f"   ✓ SAM 3D 모델 이미 로드됨 (재사용)")
            return

        # Clear GPU cache before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"\n🔹 GPU 메모리 상태 (로딩 전): {initial_memory:.2f} GB")

        print(f"\n🔹 SAM 3D 모델 초기화 중...")
        print(f"   Checkpoint 경로: {self.sam3d_checkpoint}")
        print(f"   FP16 모드: {'Enabled' if self.enable_fp16 else 'Disabled'}")

        if self.sam3d_checkpoint is None:
            raise RuntimeError(
                "SAM 3D checkpoint path is None. "
                "Please check config/model_config.yaml"
            )

        config_path = os.path.join(self.sam3d_checkpoint, "pipeline.yaml")
        print(f"   Config 파일 확인: {config_path}")

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"SAM 3D config not found at {config_path}. "
                "Please download checkpoints first."
            )

        print(f"   ✓ Config 파일 존재 확인")
        print(f"   Inference 클래스 로드 중...")

        if Inference is None:
            raise ImportError(
                "SAM 3D Inference class not imported. "
                "Check if sam-3d-objects is installed correctly."
            )

        # Load model with memory optimization
        try:
            # Set environment variables for FP16 if enabled
            if self.enable_fp16:
                os.environ['SAM3D_USE_FP16'] = '1'
                print(f"   Setting SAM3D_USE_FP16=1 for memory optimization")

            self.inference_model = Inference(config_path, compile=False)

            # Convert to FP16 if enabled (reduces memory by ~50%)
            if self.enable_fp16:
                print(f"   🔄 Converting model to FP16 (half precision)...")
                try:
                    # Convert pipeline model to FP16
                    if hasattr(self.inference_model, '_pipeline'):
                        pipeline = self.inference_model._pipeline
                        if hasattr(pipeline, 'module') and pipeline.module is not None:
                            pipeline.module.half()
                            print(f"   ✓ Model converted to FP16")
                        else:
                            print(f"   ⚠️  Pipeline module not found, skipping FP16 conversion")
                except Exception as fp16_error:
                    print(f"   ⚠️  FP16 conversion failed (using FP32): {fp16_error}")
                    self.enable_fp16 = False

            self._model_loaded = True

            # Report memory usage
            if torch.cuda.is_available():
                final_memory = torch.cuda.memory_allocated() / 1024**3
                precision = "FP16" if self.enable_fp16 else "FP32"
                print(f"   ✓ SAM 3D 모델 로드 완료 ({precision})")
                print(f"   GPU 메모리 사용: {final_memory:.2f} GB (증가: {final_memory - initial_memory:.2f} GB)")
            else:
                print(f"   ✓ SAM 3D 모델 로드 완료 (CPU mode)")

            print(f"   Model type: {type(self.inference_model)}")

        except torch.cuda.OutOfMemoryError as e:
            print(f"   ❌ GPU 메모리 부족!")
            print(f"   해결 방안:")
            print(f"     1. cleanup_model()로 이전 모델 제거")
            print(f"     2. 다른 GPU 프로세스 종료")
            print(f"     3. 시스템 재시작")
            raise RuntimeError(f"GPU OOM during model loading: {e}") from e

    def cleanup_model(self):
        """
        Clean up model and free GPU memory
        Call this after inference is complete to free up VRAM
        """
        if self.inference_model is not None:
            print(f"\n🔹 SAM 3D 모델 메모리 해제 중...")

            if torch.cuda.is_available():
                before_memory = torch.cuda.memory_allocated() / 1024**3
                print(f"   메모리 해제 전: {before_memory:.2f} GB")

            # Delete model
            del self.inference_model
            self.inference_model = None
            self._model_loaded = False

            # Force garbage collection and clear CUDA cache
            import gc
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                after_memory = torch.cuda.memory_allocated() / 1024**3
                print(f"   메모리 해제 후: {after_memory:.2f} GB")
                print(f"   ✓ {before_memory - after_memory:.2f} GB 메모리 해제됨")
            else:
                print(f"   ✓ 모델 메모리 해제 완료")

    def get_memory_status(self) -> Dict:
        """Get current GPU memory status"""
        if not torch.cuda.is_available():
            return {
                'available': False,
                'message': 'CUDA not available'
            }

        return {
            'available': True,
            'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
            'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
            'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
            'total_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
            'model_loaded': self._model_loaded
        }

    def print_memory_status(self):
        """Print current GPU memory status"""
        status = self.get_memory_status()

        if not status['available']:
            print(f"GPU: {status['message']}")
            return

        print(f"\n📊 GPU 메모리 상태:")
        print(f"   할당됨: {status['allocated_gb']:.2f} GB / {status['total_gb']:.2f} GB")
        print(f"   예약됨: {status['reserved_gb']:.2f} GB")
        print(f"   최대 사용: {status['max_allocated_gb']:.2f} GB")
        print(f"   모델 로드 여부: {'Yes' if status['model_loaded'] else 'No'}")

        # Calculate free memory
        free_gb = status['total_gb'] - status['allocated_gb']
        print(f"   사용 가능: {free_gb:.2f} GB")

        if free_gb < 1.0:
            print(f"   ⚠️  메모리 부족 - cleanup_model() 호출 권장")

    def extract_frames(
        self,
        video_path: str,
        start_frame: int = 0,
        num_frames: int = None,
        stride: int = 1
    ) -> List[np.ndarray]:
        """
        Extract frames from video

        Args:
            video_path: Path to video file
            start_frame: Starting frame index
            num_frames: Number of frames to extract (None = all)
            stride: Frame sampling stride

        Returns:
            List of frames as numpy arrays
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % stride == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

                if num_frames and len(frames) >= num_frames:
                    break

            frame_idx += 1

        cap.release()
        return frames

    def get_video_info(self, video_path: str) -> Dict:
        """Get video metadata"""
        cap = cv2.VideoCapture(video_path)
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        cap.release()
        return info

    def segment_object_interactive(
        self,
        frame: np.ndarray,
        click_point: Tuple[int, int] = None,
        bbox: Tuple[int, int, int, int] = None,
        method: str = 'simple_threshold'
    ) -> np.ndarray:
        """
        Segment object from frame using various methods

        Args:
            frame: Input frame (RGB)
            click_point: (x, y) point for segmentation
            bbox: (x, y, w, h) bounding box
            method: Segmentation method ('simple_threshold', 'grabcut', 'contour')

        Returns:
            Binary mask
        """
        if method == 'simple_threshold':
            # Simple background subtraction (assumes dark background)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
            return mask > 0

        elif method == 'grabcut' and bbox:
            # GrabCut segmentation
            mask = np.zeros(frame.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            cv2.grabCut(frame, mask, bbox, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            return mask > 0

        elif method == 'contour':
            # Contour-based segmentation
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [largest_contour], -1, 255, -1)
                return mask > 0

        # Default: return full frame as foreground
        return np.ones(frame.shape[:2], dtype=bool)

    def track_object_across_frames(
        self,
        frames: List[np.ndarray],
        initial_bbox: Tuple[int, int, int, int] = None,
        motion_threshold: float = 50.0,
        fps: float = 30.0
    ) -> TrackingResult:
        """
        Track object across multiple frames

        Args:
            frames: List of frames
            initial_bbox: Initial bounding box (x, y, w, h)
            motion_threshold: Minimum center displacement for motion detection (pixels)
            fps: Video framerate for duration calculation

        Returns:
            TrackingResult with tracking information
        """
        segments = []

        for idx, frame in enumerate(frames):
            # Segment object
            if initial_bbox:
                mask = self.segment_object_interactive(frame, bbox=initial_bbox, method='grabcut')
            else:
                mask = self.segment_object_interactive(frame, method='contour')

            # Get bounding box and center
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                center_x, center_y = x + w // 2, y + h // 2
                area = cv2.contourArea(largest_contour)

                segment_info = SegmentInfo(
                    frame_idx=idx,
                    mask=mask,
                    bbox=(x, y, w, h),
                    center=(center_x, center_y),
                    area=area
                )
                segments.append(segment_info)

        # Detect motion
        motion_detected = False
        if len(segments) > 1:
            max_displacement = 0
            for i in range(1, len(segments)):
                dx = segments[i].center[0] - segments[i-1].center[0]
                dy = segments[i].center[1] - segments[i-1].center[1]
                displacement = np.sqrt(dx**2 + dy**2)
                max_displacement = max(max_displacement, displacement)

            motion_detected = max_displacement > motion_threshold

        duration = len(frames) / fps

        return TrackingResult(
            start_frame=0,
            end_frame=len(frames) - 1,
            segments=segments,
            motion_detected=motion_detected,
            duration_seconds=duration
        )

    def reconstruct_3d(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        seed: int = 42,
        cleanup_after: bool = False
    ) -> Dict:
        """
        Reconstruct 3D object from frame and mask using SAM 3D

        Args:
            frame: Input frame (RGB)
            mask: Binary segmentation mask
            seed: Random seed for reproducibility
            cleanup_after: Clean up model after inference to free VRAM

        Returns:
            Dictionary containing reconstruction results
        """
        print("\n🔹 3D Reconstruction 시작:")
        print(f"   Frame shape: {frame.shape}")
        print(f"   Mask shape: {mask.shape}")
        print(f"   Seed: {seed}")
        print(f"   SAM3D checkpoint: {self.sam3d_checkpoint}")

        # Show memory status before initialization
        self.print_memory_status()

        try:
            self.initialize_sam3d()
            print(f"   ✓ SAM 3D 모델 초기화 완료")
        except torch.cuda.OutOfMemoryError as oom_e:
            print(f"   ❌ GPU 메모리 부족으로 모델 로드 실패")
            self.print_memory_status()
            raise RuntimeError(
                "GPU 메모리가 부족합니다. 다음을 시도하세요:\n"
                "  1. 다른 GPU 프로그램 종료\n"
                "  2. cleanup_model() 호출 후 재시도\n"
                "  3. 시스템 재시작"
            ) from oom_e
        except Exception as e:
            print(f"   ❌ SAM 3D 초기화 실패: {e}")
            raise

        if self.inference_model is None:
            error_msg = "SAM 3D inference model is None after initialization"
            print(f"   ❌ {error_msg}")
            raise RuntimeError(error_msg)

        print(f"   Inference model type: {type(self.inference_model)}")
        print(f"   Running inference...")

        # Clear cache before inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            # Run SAM 3D inference with autocast for FP16 if enabled
            if self.enable_fp16 and torch.cuda.is_available():
                print(f"   Using FP16 mixed precision")
                with torch.cuda.amp.autocast():
                    output = self.inference_model(frame, mask, seed=seed)
            else:
                output = self.inference_model(frame, mask, seed=seed)

            print(f"   ✓ Inference 완료")
            print(f"   Output keys: {output.keys() if output else 'None'}")

            # Update last inference time
            import time
            self._last_inference_time = time.time()

            # Show memory status after inference
            self.print_memory_status()

            # Optionally cleanup model
            if cleanup_after:
                print(f"   🧹 자동 정리 모드 활성화됨")
                self.cleanup_model()

            return output

        except torch.cuda.OutOfMemoryError as oom_e:
            print(f"   ❌ Inference 중 GPU 메모리 부족")
            self.print_memory_status()
            print(f"\n   자동 정리 후 재시도 중...")

            # Try cleanup and retry once
            self.cleanup_model()
            self.print_memory_status()

            raise RuntimeError(
                "Inference 중 GPU 메모리 부족. 모델을 정리했습니다.\n"
                "더 작은 이미지로 재시도하거나 시스템 재시작이 필요합니다."
            ) from oom_e

        except Exception as e:
            print(f"   ❌ Inference 실행 실패: {e}")
            import traceback
            traceback.print_exc()
            raise

    def export_mesh(
        self,
        output: Dict,
        save_path: str,
        format: str = 'ply'
    ):
        """
        Export 3D reconstruction to mesh file

        Args:
            output: SAM 3D output dictionary
            save_path: Output file path
            format: Output format ('ply', 'obj')
        """
        if 'gs' in output:
            # Export Gaussian Splatting
            ply_path = save_path if save_path.endswith('.ply') else save_path + '.ply'
            output['gs'].save_ply(ply_path)
            print(f"Gaussian splat saved to {ply_path}")

        # If mesh is available, convert and export
        if format == 'obj' and 'gs' in output:
            # Note: This is a placeholder - actual conversion would require
            # more sophisticated mesh extraction from gaussian splats
            print(f"OBJ export requires mesh extraction from gaussian splats")

    def visualize_mask_overlay(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Create visualization with mask overlay

        Args:
            frame: Input frame (RGB)
            mask: Binary mask
            alpha: Overlay transparency

        Returns:
            Visualization image
        """
        overlay = frame.copy()
        overlay[mask] = overlay[mask] * (1 - alpha) + np.array([0, 255, 0]) * alpha
        return overlay.astype(np.uint8)

    def process_video_segment(
        self,
        video_path: str,
        start_time: float = 0.0,
        duration: float = 3.0,
        output_dir: str = None,
        segmentation_method: str = 'contour',
        motion_threshold: float = 50.0
    ) -> Tuple[TrackingResult, Optional[Dict]]:
        """
        Process a video segment: extract, track, and optionally reconstruct 3D

        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            duration: Duration in seconds
            output_dir: Directory to save outputs
            segmentation_method: Segmentation method to use
            motion_threshold: Motion detection threshold

        Returns:
            Tuple of (tracking_result, reconstruction_3d)
        """
        # Get video info
        video_info = self.get_video_info(video_path)
        fps = video_info['fps']

        # Calculate frame range
        start_frame = int(start_time * fps)
        num_frames = int(duration * fps)

        # Extract frames
        print(f"Extracting {num_frames} frames starting at {start_time}s...")
        frames = self.extract_frames(video_path, start_frame, num_frames, stride=1)

        if not frames:
            raise ValueError("No frames extracted from video")

        # Track object
        print(f"Tracking object across {len(frames)} frames...")
        tracking_result = self.track_object_across_frames(
            frames,
            motion_threshold=motion_threshold,
            fps=fps
        )

        print(f"Motion detected: {tracking_result.motion_detected}")
        print(f"Duration: {tracking_result.duration_seconds:.2f}s")

        # Reconstruct 3D if motion detected and output directory provided
        reconstruction = None
        if tracking_result.motion_detected and output_dir:
            print("Performing 3D reconstruction...")
            # Use middle frame for reconstruction
            mid_idx = len(frames) // 2
            mid_frame = frames[mid_idx]
            mid_mask = tracking_result.segments[mid_idx].mask

            reconstruction = self.reconstruct_3d(mid_frame, mid_mask)

            # Save outputs
            os.makedirs(output_dir, exist_ok=True)

            # Save mask visualization
            vis = self.visualize_mask_overlay(mid_frame, mid_mask)
            Image.fromarray(vis).save(os.path.join(output_dir, 'mask_overlay.png'))

            # Export mesh
            mesh_path = os.path.join(output_dir, 'reconstruction')
            self.export_mesh(reconstruction, mesh_path, format='ply')

        return tracking_result, reconstruction


if __name__ == "__main__":
    # Test the processor
    processor = SAM3DProcessor()

    # Test video path
    test_video = "/home/joon/dev/data/markerless_mouse/mouse_1/Camera1/0.mp4"

    if os.path.exists(test_video):
        print(f"Testing with video: {test_video}")
        video_info = processor.get_video_info(test_video)
        print(f"Video info: {video_info}")

        # Extract a few frames
        frames = processor.extract_frames(test_video, start_frame=0, num_frames=10, stride=10)
        print(f"Extracted {len(frames)} frames")
    else:
        print(f"Test video not found: {test_video}")
