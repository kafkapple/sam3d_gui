"""
Lite Annotator - Efficient annotation mode integrated from SAM Annotator

Key Features:
- Direct video loading (cv2.VideoCapture) - Memory efficient
- Image folder support (glob pattern)
- Runtime input source switching
- Multi-model selection (SAM 2.1 variants)
- Automatic annotation restoration
"""

import cv2
import numpy as np
from pathlib import Path
import json
import sys
from typing import Tuple, List, Optional, Dict, Any
import torch

# SAM 2 imports
sys.path.append(str(Path.home() / 'dev/segment-anything-2'))
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2


class LiteAnnotator:
    """
    Lightweight annotation tool with direct video/image loading

    Based on SAM Annotator from mouse-super-resolution project
    Integrated into SAM 3D GUI as Tab 3: Lite Mode
    """

    # SAM 2.1 Model configurations
    SAM_MODELS = {
        'tiny': {
            'config': 'configs/sam2.1/sam2.1_hiera_t.yaml',
            'checkpoint': 'checkpoints/sam2.1_hiera_tiny.pt',
            'description': 'Fastest, lower quality'
        },
        'small': {
            'config': 'configs/sam2.1/sam2.1_hiera_s.yaml',
            'checkpoint': 'checkpoints/sam2.1_hiera_small.pt',
            'description': 'Fast, good quality'
        },
        'base_plus': {
            'config': 'configs/sam2.1/sam2.1_hiera_b+.yaml',
            'checkpoint': 'checkpoints/sam2.1_hiera_base_plus.pt',
            'description': 'Balanced'
        },
        'large': {
            'config': 'configs/sam2.1/sam2.1_hiera_l.yaml',
            'checkpoint': 'checkpoints/sam2.1_hiera_large.pt',
            'description': 'Best quality, slower'
        }
    }

    def __init__(self, sam2_base_path: Path, device: str = "cuda"):
        """
        Initialize Lite Annotator

        Args:
            sam2_base_path: Path to segment-anything-2 repository
            device: Device to run model on ("cuda", "cpu", or "auto")
        """
        self.sam2_base_path = sam2_base_path
        self.device = self._resolve_device(device)

        # Input source state
        self.input_type = None  # 'video', 'images', or None
        self.video_cap = None  # cv2.VideoCapture for video
        self.image_files = []  # List of image paths for image folder
        self.total_frames = 0

        # Current frame state
        self.current_frame_idx = 0
        self.current_frame = None  # RGB numpy array
        self.frame_path = None  # Path or virtual path for naming

        # Annotation state
        self.points = []  # List of [x, y]
        self.labels = []  # List of 1 (foreground) or 0 (background)
        self.current_mask = None  # Generated mask
        self.current_score = None  # Confidence score

        # Model state
        self.predictor = None
        self.current_model = None

        # Output directory
        self.output_dir = Path.home() / "dev/sam3d_gui/outputs/lite_annotations"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def load_model(self, model_name: str = "large") -> str:
        """
        Load SAM 2.1 model

        Args:
            model_name: One of 'tiny', 'small', 'base_plus', 'large'

        Returns:
            Status message
        """
        if model_name not in self.SAM_MODELS:
            return f"Invalid model name. Choose from: {list(self.SAM_MODELS.keys())}"

        if self.current_model == model_name and self.predictor is not None:
            return f"Model '{model_name}' already loaded"

        try:
            model_info = self.SAM_MODELS[model_name]

            # Resolve paths
            config_path = self.sam2_base_path / model_info['config']
            checkpoint_path = self.sam2_base_path / model_info['checkpoint']

            if not checkpoint_path.exists():
                return f"Checkpoint not found: {checkpoint_path}"

            # Build SAM 2 model
            sam2_model = build_sam2(
                config_file=str(config_path),
                ckpt_path=str(checkpoint_path),
                device=self.device
            )

            self.predictor = SAM2ImagePredictor(sam2_model)
            self.current_model = model_name

            msg = f"✓ Loaded SAM 2.1 {model_name} ({model_info['description']}) on {self.device}"
            return msg

        except Exception as e:
            return f"✗ Failed to load model: {str(e)}"

    def change_input_source(
        self,
        input_path: str,
        input_type: str,
        pattern: str = "*.png"
    ) -> Tuple[bool, str, int]:
        """
        Change input source (video or image folder)

        Args:
            input_path: Path to video file or image folder
            input_type: 'video' or 'images'
            pattern: Glob pattern for images (only for input_type='images')

        Returns:
            (success: bool, message: str, total_frames: int)
        """
        try:
            # Close existing video capture
            if self.video_cap is not None:
                self.video_cap.release()
                self.video_cap = None

            # Reset state
            self.image_files = []
            self.total_frames = 0
            self.current_frame_idx = 0
            self.current_frame = None
            self.points = []
            self.labels = []
            self.current_mask = None
            self.current_score = None

            path = Path(input_path).expanduser()

            if input_type == 'video':
                # Video file
                if not path.is_file():
                    return False, f"Video file not found: {input_path}", 0

                self.video_cap = cv2.VideoCapture(str(path))
                if not self.video_cap.isOpened():
                    return False, f"Cannot open video: {input_path}", 0

                self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.input_type = 'video'
                self.frame_path = path  # Store for naming

                msg = f"✓ Video loaded: {path.name} ({self.total_frames} frames)"
                return True, msg, self.total_frames

            elif input_type == 'images':
                # Image folder
                if not path.is_dir():
                    return False, f"Directory not found: {input_path}", 0

                self.image_files = sorted(list(path.glob(pattern)))
                self.total_frames = len(self.image_files)

                if self.total_frames == 0:
                    return False, f"No images found matching pattern '{pattern}'", 0

                self.input_type = 'images'

                msg = f"✓ Image folder loaded: {path.name} ({self.total_frames} images, pattern: {pattern})"
                return True, msg, self.total_frames

            else:
                return False, f"Invalid input type: {input_type}", 0

        except Exception as e:
            return False, f"✗ Error: {str(e)}", 0

    def load_frame(self, frame_idx: int) -> Tuple[Optional[np.ndarray], str]:
        """
        Load frame and restore annotation if exists

        Args:
            frame_idx: Frame index (0-based)

        Returns:
            (frame_rgb: np.ndarray or None, status_message: str)
        """
        if self.input_type is None:
            return None, "No input source loaded"

        if not (0 <= frame_idx < self.total_frames):
            return None, f"Invalid frame index: {frame_idx} (valid: 0-{self.total_frames-1})"

        try:
            # Load frame based on input type
            if self.input_type == 'video':
                # Read from video
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame_bgr = self.video_cap.read()

                if not ret:
                    return None, f"Failed to read frame {frame_idx} from video"

                self.current_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # Virtual path for annotation naming
                video_name = self.frame_path.stem
                frame_name = f"{video_name}_frame_{frame_idx:04d}"

            elif self.input_type == 'images':
                # Read from image file
                image_path = self.image_files[frame_idx]
                frame_bgr = cv2.imread(str(image_path))

                if frame_bgr is None:
                    return None, f"Failed to read image: {image_path}"

                self.current_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_name = image_path.stem

            self.current_frame_idx = frame_idx

            # Check for existing annotation
            annotation_file = self.output_dir / f'frame_{frame_idx:04d}_annotation.json'

            if annotation_file.exists():
                # Load annotation
                with open(annotation_file, 'r') as f:
                    annotation = json.load(f)

                self.points = annotation.get('points', [])
                self.labels = annotation.get('labels', [])

                # Load mask if exists
                mask_file = self.output_dir / f'frame_{frame_idx:04d}_mask.png'
                if mask_file.exists():
                    mask_uint8 = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                    self.current_mask = (mask_uint8 > 0).astype(bool)
                    self.current_score = annotation.get('confidence', 0.0)

                    # Visualize with mask
                    frame_vis = self._visualize_with_mask()
                    msg = f"Loaded {frame_name}: {len(self.points)} points, mask restored"
                    return frame_vis, msg
                else:
                    self.current_mask = None
                    self.current_score = None

                    # Visualize points only
                    frame_vis = self._visualize_points()
                    msg = f"Loaded {frame_name}: {len(self.points)} points (no mask)"
                    return frame_vis, msg
            else:
                # No annotation - fresh start
                self.points = []
                self.labels = []
                self.current_mask = None
                self.current_score = None

                msg = f"Loaded {frame_name} (no annotation)"
                return self.current_frame.copy(), msg

        except Exception as e:
            return None, f"✗ Error loading frame: {str(e)}"

    def add_point(self, x: int, y: int, point_type: str) -> Tuple[np.ndarray, str]:
        """
        Add annotation point

        Args:
            x, y: Pixel coordinates
            point_type: 'foreground' or 'background'

        Returns:
            (visualized_frame, status_message)
        """
        if self.current_frame is None:
            return self.current_frame, "Load a frame first"

        label = 1 if point_type.lower() == 'foreground' else 0
        self.points.append([x, y])
        self.labels.append(label)

        frame_vis = self._visualize_points()
        point_name = "foreground" if label == 1 else "background"
        msg = f"Added {point_name} point at ({x}, {y})"

        return frame_vis, msg

    def generate_mask(self, multimask_output: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
        """
        Generate mask using SAM

        Args:
            multimask_output: Whether to output multiple masks

        Returns:
            (visualized_frame, mask_binary, status_message)
        """
        if self.predictor is None:
            return self.current_frame, None, "Load a model first"

        if len(self.points) == 0:
            return self.current_frame, None, "Add points first"

        if self.current_frame is None:
            return None, None, "Load a frame first"

        try:
            # SAM inference
            self.predictor.set_image(self.current_frame)

            point_coords = np.array(self.points, dtype=np.float32)
            point_labels = np.array(self.labels, dtype=np.int32)

            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=multimask_output
            )

            # Select best mask
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            score = scores[best_idx]

            # Store mask
            self.current_mask = mask
            self.current_score = score

            # Visualize
            frame_vis = self._visualize_with_mask()

            # Binary mask for display
            mask_binary = (mask * 255).astype(np.uint8)

            # Statistics
            mask_area = mask.sum() / mask.size * 100
            msg = f"Mask generated! Confidence: {score:.3f}, Area: {mask_area:.1f}%"

            return frame_vis, mask_binary, msg

        except Exception as e:
            return self.current_frame, None, f"✗ Error: {str(e)}"

    def save_annotation(self) -> str:
        """
        Save current annotation (points + mask)

        Returns:
            Status message
        """
        if len(self.points) == 0:
            return "No points to save"

        if self.current_frame is None:
            return "No frame loaded"

        try:
            # Annotation JSON
            annotation = {
                'frame_idx': self.current_frame_idx,
                'input_type': self.input_type,
                'points': self.points,
                'labels': self.labels,
                'has_mask': self.current_mask is not None
            }

            if self.current_mask is not None:
                annotation['confidence'] = float(self.current_score)
                annotation['mask_area_pct'] = float(self.current_mask.sum() / self.current_mask.size * 100)

            # Save JSON
            annotation_file = self.output_dir / f'frame_{self.current_frame_idx:04d}_annotation.json'
            with open(annotation_file, 'w') as f:
                json.dump(annotation, f, indent=2)

            saved = ['annotation']

            # Save mask if exists
            if self.current_mask is not None:
                mask_file = self.output_dir / f'frame_{self.current_frame_idx:04d}_mask.png'
                mask_uint8 = (self.current_mask * 255).astype(np.uint8)
                cv2.imwrite(str(mask_file), mask_uint8)
                saved.append('mask')

            # Save visualization
            vis_file = self.output_dir / f'frame_{self.current_frame_idx:04d}_visualization.png'
            if self.current_mask is not None:
                frame_vis = self._visualize_with_mask()
            else:
                frame_vis = self._visualize_points()

            frame_vis_bgr = cv2.cvtColor(frame_vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(vis_file), frame_vis_bgr)
            saved.append('visualization')

            msg = f"✓ Saved: {', '.join(saved)} → {self.output_dir}"
            return msg

        except Exception as e:
            return f"✗ Error saving: {str(e)}"

    def clear_points(self) -> Tuple[np.ndarray, str]:
        """
        Clear all points and mask

        Returns:
            (current_frame, status_message)
        """
        self.points = []
        self.labels = []
        self.current_mask = None
        self.current_score = None

        if self.current_frame is None:
            return None, "No frame loaded"

        return self.current_frame.copy(), "Points and mask cleared"

    def _visualize_points(self, point_size: int = 10) -> np.ndarray:
        """Visualize points on current frame"""
        if self.current_frame is None:
            return None

        frame_vis = self.current_frame.copy()

        fg_color = (0, 255, 0)  # Green
        bg_color = (255, 0, 0)  # Red
        border_color = (255, 255, 255)  # White
        border_width = 2

        for (px, py), lbl in zip(self.points, self.labels):
            color = fg_color if lbl == 1 else bg_color
            cv2.circle(frame_vis, (px, py), point_size, color, -1)
            cv2.circle(frame_vis, (px, py), point_size + border_width, border_color, 2)

        return frame_vis

    def _visualize_with_mask(self, point_size: int = 10, mask_alpha: float = 0.4) -> np.ndarray:
        """Visualize mask + points on current frame"""
        if self.current_frame is None or self.current_mask is None:
            return self._visualize_points(point_size)

        frame_vis = self.current_frame.copy()

        # Mask overlay
        mask_overlay = np.zeros_like(frame_vis)
        fg_color = (0, 255, 0)  # Green
        mask_overlay[self.current_mask] = fg_color
        frame_vis = cv2.addWeighted(frame_vis, 1 - mask_alpha, mask_overlay, mask_alpha, 0)

        # Points
        bg_color = (255, 0, 0)  # Red
        border_color = (255, 255, 255)  # White
        border_width = 2

        for (px, py), lbl in zip(self.points, self.labels):
            color = fg_color if lbl == 1 else bg_color
            cv2.circle(frame_vis, (px, py), point_size, color, -1)
            cv2.circle(frame_vis, (px, py), point_size + border_width, border_color, 2)

        return frame_vis

    def get_info(self) -> Dict[str, Any]:
        """Get current state information"""
        return {
            'model': self.current_model,
            'device': self.device,
            'input_type': self.input_type,
            'total_frames': self.total_frames,
            'current_frame': self.current_frame_idx,
            'num_points': len(self.points),
            'has_mask': self.current_mask is not None,
            'output_dir': str(self.output_dir)
        }
