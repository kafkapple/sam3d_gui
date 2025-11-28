"""
데이터 증강 모듈
RGB 이미지와 마스크를 함께 증강하되, 기하학적 변환은 동일하게, 색상 변환은 RGB만 적용
"""

import cv2
import numpy as np
from typing import Tuple, Dict, List, Optional
from pathlib import Path
import random


class DataAugmentor:
    """
    데이터 증강 클래스

    핵심 원칙:
    1. Geometric transforms (회전, 크기, 이동) → RGB + Mask 동일 적용
    2. Photometric transforms (색상, 노이즈) → RGB만 적용
    """

    def __init__(self):
        self.transform_history = []
        # Background image support
        self.background_images: List[np.ndarray] = []
        self.background_folder: Optional[Path] = None
        self._bg_index_queue: List[int] = []  # Uniform sampling queue

    # ========== Background Image Methods ==========

    def load_background_images(self, folder_path: str) -> int:
        """
        배경 이미지 폴더 로드

        Args:
            folder_path: 배경 이미지 폴더 경로

        Returns:
            로드된 이미지 개수
        """
        folder = Path(folder_path)
        if not folder.exists():
            return 0

        self.background_folder = folder
        self.background_images = []

        # 이미지 파일 확장자
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in folder.iterdir()
                       if f.is_file() and f.suffix.lower() in image_extensions]

        for img_path in image_files:
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    # BGR → RGB 변환 (cv2.imread는 BGR로 로드)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.background_images.append(img_rgb)
            except Exception as e:
                print(f"Warning: Failed to load {img_path}: {e}")

        # Uniform sampling 큐 초기화
        self._reset_bg_index_queue()

        return len(self.background_images)

    def _reset_bg_index_queue(self):
        """배경 인덱스 큐를 셔플하여 초기화 (Uniform sampling)"""
        if len(self.background_images) > 0:
            self._bg_index_queue = list(range(len(self.background_images)))
            random.shuffle(self._bg_index_queue)
        else:
            self._bg_index_queue = []

    def _get_next_bg_index(self) -> int:
        """다음 배경 인덱스 반환 (Uniform sampling)"""
        if not self._bg_index_queue:
            self._reset_bg_index_queue()
        if not self._bg_index_queue:
            return 0
        return self._bg_index_queue.pop()

    def get_background(
        self,
        target_size: Tuple[int, int],
        use_image: bool = True,
        fill_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> np.ndarray:
        """
        배경 이미지 또는 단색 배경 반환

        Args:
            target_size: (height, width)
            use_image: True면 이미지 배경, False면 단색
            fill_color: 단색 배경일 경우 RGB 색상

        Returns:
            배경 이미지 (H, W, 3) RGB
        """
        h, w = target_size

        if use_image and len(self.background_images) > 0:
            # Uniform sampling으로 배경 선택
            bg_idx = self._get_next_bg_index()
            bg_img = self.background_images[bg_idx]

            # 크기 조정
            bg_h, bg_w = bg_img.shape[:2]
            if bg_h != h or bg_w != w:
                bg_img = cv2.resize(bg_img, (w, h), interpolation=cv2.INTER_LINEAR)

            return bg_img.copy()
        else:
            # 단색 배경
            return np.full((h, w, 3), fill_color, dtype=np.uint8)

    def composite_with_background(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        use_bg_image: bool = True,
        bg_image_ratio: float = 0.5,
        fill_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> np.ndarray:
        """
        마스크 영역(foreground)을 유지하고 배경을 교체

        Args:
            rgb: RGB 이미지 (H, W, 3)
            mask: 마스크 (H, W) - True인 영역이 foreground
            use_bg_image: True면 배경 이미지 사용 가능
            bg_image_ratio: 배경 이미지 사용 확률 (0.0~1.0)
            fill_color: 단색 배경 RGB

        Returns:
            합성된 이미지 (foreground + new background)
        """
        h, w = rgb.shape[:2]
        mask_bool = mask.astype(bool) if mask.dtype != bool else mask

        # 배경 선택: 이미지 또는 단색
        use_image = (
            use_bg_image and
            len(self.background_images) > 0 and
            random.random() < bg_image_ratio
        )

        background = self.get_background((h, w), use_image=use_image, fill_color=fill_color)

        # 합성: foreground(mask=True) 유지, background(mask=False) 교체
        result = background.copy()
        result[mask_bool] = rgb[mask_bool]

        return result

    def prevent_clipping(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        fill_color: Tuple[int, int, int] = (255, 255, 255),
        margin: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Object가 이미지 경계에서 잘리지 않도록 오프셋 적용

        마스크 영역이 이미지 가장자리에 닿아있으면,
        반대 방향으로 충분히 이동하여 전체가 보이도록 함

        Args:
            rgb: RGB 이미지 (H, W, 3)
            mask: 마스크 (H, W) - boolean or uint8
            fill_color: 빈 공간 채울 색상 (R, G, B)
            margin: 경계와의 최소 여백 (픽셀)

        Returns:
            (adjusted_rgb, adjusted_mask)
        """
        h, w = rgb.shape[:2]

        # 마스크를 boolean으로 변환
        mask_bool = mask.astype(bool) if mask.dtype != bool else mask

        # 마스크가 비어있으면 그대로 반환
        if not mask_bool.any():
            return rgb, mask

        # 마스크의 바운딩 박스 계산
        rows = np.any(mask_bool, axis=1)
        cols = np.any(mask_bool, axis=0)

        y_indices = np.where(rows)[0]
        x_indices = np.where(cols)[0]

        y_min, y_max = y_indices[0], y_indices[-1]
        x_min, x_max = x_indices[0], x_indices[-1]

        mask_width = x_max - x_min + 1
        mask_height = y_max - y_min + 1

        # 필요한 오프셋 계산 (마스크가 이미지 내에 완전히 들어오도록)
        offset_x = 0
        offset_y = 0

        # X축: 마스크가 이미지보다 작으면 이동 가능
        if mask_width < w - margin * 2:
            # 좌측이 잘림 (x_min < margin) → 오른쪽으로 이동
            if x_min < margin:
                offset_x = margin - x_min
            # 우측이 잘림 (x_max > w - margin - 1) → 왼쪽으로 이동
            elif x_max > w - margin - 1:
                offset_x = (w - margin - 1) - x_max

            # 이동 후에도 반대편이 잘리지 않도록 보정
            new_x_min = x_min + offset_x
            new_x_max = x_max + offset_x
            if new_x_min < 0:
                offset_x -= new_x_min  # 0으로 맞춤
            elif new_x_max >= w:
                offset_x -= (new_x_max - w + 1)

        # Y축: 마스크가 이미지보다 작으면 이동 가능
        if mask_height < h - margin * 2:
            # 상단이 잘림 (y_min < margin) → 아래로 이동
            if y_min < margin:
                offset_y = margin - y_min
            # 하단이 잘림 (y_max > h - margin - 1) → 위로 이동
            elif y_max > h - margin - 1:
                offset_y = (h - margin - 1) - y_max

            # 이동 후에도 반대편이 잘리지 않도록 보정
            new_y_min = y_min + offset_y
            new_y_max = y_max + offset_y
            if new_y_min < 0:
                offset_y -= new_y_min
            elif new_y_max >= h:
                offset_y -= (new_y_max - h + 1)

        # 오프셋이 0이면 그대로 반환
        if offset_x == 0 and offset_y == 0:
            return rgb, mask

        # 변환 행렬 생성 (평행 이동)
        M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])

        # RGB 이동
        adjusted_rgb = cv2.warpAffine(
            rgb, M, (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=fill_color
        )

        # Mask 이동
        adjusted_mask = cv2.warpAffine(
            mask_bool.astype(np.uint8), M, (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        return adjusted_rgb, adjusted_mask.astype(bool)

    def compute_bbox_from_mask(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """
        마스크로부터 바운딩 박스 계산

        Returns:
            (x_min, y_min, x_max, y_max)
        """
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not rows.any() or not cols.any():
            # 마스크가 비어있으면 전체 이미지
            return 0, 0, mask.shape[1], mask.shape[0]

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        # Convert numpy int to Python int (cv2.getRotationMatrix2D requires native types)
        return int(x_min), int(y_min), int(x_max), int(y_max)

    def scale_around_bbox(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        scale_factor: float,
        fill_color: str = 'white'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        마스크 바운딩 박스를 중심으로 확대/축소
        전체 이미지 크기는 유지

        Args:
            rgb: RGB 이미지 (H, W, 3)
            mask: 마스크 (H, W) - boolean or uint8
            scale_factor: 확대/축소 배율 (< 1.0: 축소, > 1.0: 확대)
            fill_color: 빈 공간 채울 색상 ('white', 'black', 'transparent', 'nearest')

        Returns:
            (augmented_rgb, augmented_mask)
        """
        h, w = rgb.shape[:2]

        # 1. 바운딩 박스 계산
        x_min, y_min, x_max, y_max = self.compute_bbox_from_mask(mask)

        # 바운딩 박스 중심
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2

        # 2. 변환 행렬 생성 (중심점 기준 스케일)
        # Translation to origin -> Scale -> Translation back
        M = cv2.getRotationMatrix2D((cx, cy), 0, scale_factor)

        # 3. RGB와 Mask 동일한 변환 적용
        if scale_factor < 1.0:
            # 축소: 빈 공간을 채워야 함
            if fill_color == 'white':
                borderValue_rgb = (255, 255, 255)
            elif fill_color == 'black':
                borderValue_rgb = (0, 0, 0)
            elif fill_color == 'nearest':
                # 가장자리 색상으로 채우기
                borderValue_rgb = int(rgb[cy, cx].mean())
            else:
                borderValue_rgb = (255, 255, 255)

            aug_rgb = cv2.warpAffine(
                rgb, M, (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=borderValue_rgb
            )
            aug_mask = cv2.warpAffine(
                mask.astype(np.uint8), M, (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        else:
            # 확대: 잘리는 부분 발생
            aug_rgb = cv2.warpAffine(rgb, M, (w, h))
            aug_mask = cv2.warpAffine(mask.astype(np.uint8), M, (w, h))

        return aug_rgb, aug_mask.astype(bool)

    def crop_scale_with_offset(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        scale_factor: float = 1.0,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        padding: int = 20,
        fill_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        마스크 영역을 크롭하여 확대/축소 후 위치 이동하여 원본 크기로 합성
        전체 이미지 크기는 유지하되, 마스크 영역만 변환

        Args:
            rgb: RGB 이미지 (H, W, 3)
            mask: 마스크 (H, W) - boolean or uint8
            scale_factor: 확대/축소 배율 (0.5: 50% 축소, 2.0: 200% 확대)
            offset_x: 가로 이동 비율 (-1.0 ~ 1.0, 이미지 너비 기준)
            offset_y: 세로 이동 비율 (-1.0 ~ 1.0, 이미지 높이 기준)
            padding: 크롭 시 추가 여백 (픽셀)
            fill_color: 배경 색상 (R, G, B)

        Returns:
            (augmented_rgb, augmented_mask)
        """
        h, w = rgb.shape[:2]

        # 1. 마스크 바운딩 박스 계산
        x_min, y_min, x_max, y_max = self.compute_bbox_from_mask(mask)

        # 패딩 추가
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        bbox_w = x_max - x_min
        bbox_h = y_max - y_min

        # 2. 마스크 영역 크롭
        crop_rgb = rgb[y_min:y_max, x_min:x_max].copy()
        crop_mask = mask[y_min:y_max, x_min:x_max].copy()

        # 3. 크롭된 영역 스케일 변환
        scaled_w = int(bbox_w * scale_factor)
        scaled_h = int(bbox_h * scale_factor)

        # 최소 크기 보장
        scaled_w = max(1, scaled_w)
        scaled_h = max(1, scaled_h)

        scaled_rgb = cv2.resize(crop_rgb, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        scaled_mask = cv2.resize(
            crop_mask.astype(np.uint8),
            (scaled_w, scaled_h),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        # 4. 오프셋 계산 (이미지 크기 기준)
        offset_px_x = int(w * offset_x)
        offset_px_y = int(h * offset_y)

        # 5. 원본 크기 캔버스 생성 (배경색으로 채움)
        aug_rgb = np.full((h, w, 3), fill_color, dtype=np.uint8)
        aug_mask = np.zeros((h, w), dtype=bool)

        # 6. 변환된 크롭 영역을 캔버스 중앙 + 오프셋 위치에 배치
        # 원래 bbox 중심 계산
        orig_cx = (x_min + x_max) // 2
        orig_cy = (y_min + y_max) // 2

        # 배치 위치 (중심 + 오프셋)
        paste_cx = orig_cx + offset_px_x
        paste_cy = orig_cy + offset_px_y

        # 좌상단 좌표
        paste_x = paste_cx - scaled_w // 2
        paste_y = paste_cy - scaled_h // 2

        # 7. 경계 체크 및 붙여넣기
        # 소스 영역 (스케일된 이미지에서 잘라낼 부분)
        src_x1 = max(0, -paste_x)
        src_y1 = max(0, -paste_y)
        src_x2 = min(scaled_w, w - paste_x)
        src_y2 = min(scaled_h, h - paste_y)

        # 대상 영역 (캔버스에 붙일 위치)
        dst_x1 = max(0, paste_x)
        dst_y1 = max(0, paste_y)
        dst_x2 = min(w, paste_x + scaled_w)
        dst_y2 = min(h, paste_y + scaled_h)

        # 유효한 영역이 있는 경우에만 붙여넣기
        if src_x2 > src_x1 and src_y2 > src_y1 and dst_x2 > dst_x1 and dst_y2 > dst_y1:
            aug_rgb[dst_y1:dst_y2, dst_x1:dst_x2] = scaled_rgb[src_y1:src_y2, src_x1:src_x2]
            aug_mask[dst_y1:dst_y2, dst_x1:dst_x2] = scaled_mask[src_y1:src_y2, src_x1:src_x2]

        return aug_rgb, aug_mask

    def rotate(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        angle: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        회전 (RGB + Mask 동일 적용)

        Args:
            angle: 회전 각도 (degree)
        """
        h, w = rgb.shape[:2]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        aug_rgb = cv2.warpAffine(rgb, M, (w, h))
        aug_mask = cv2.warpAffine(mask.astype(np.uint8), M, (w, h))

        return aug_rgb, aug_mask.astype(bool)

    def flip(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        mode: str = 'horizontal'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        좌우/상하 뒤집기 (RGB + Mask 동일 적용)

        Args:
            mode: 'horizontal' or 'vertical'
        """
        flip_code = 1 if mode == 'horizontal' else 0

        aug_rgb = cv2.flip(rgb, flip_code)
        aug_mask = cv2.flip(mask.astype(np.uint8), flip_code)

        return aug_rgb, aug_mask.astype(bool)

    # ========== Photometric Transforms (RGB만 적용) ==========

    def add_gaussian_noise(self, rgb: np.ndarray, std: float = 10.0) -> np.ndarray:
        """
        가우시안 노이즈 추가 (RGB만)

        Args:
            std: 노이즈 표준편차 (0~255 스케일)
        """
        noise = np.random.normal(0, std, rgb.shape).astype(np.float32)
        noisy = rgb.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy

    def adjust_brightness(self, rgb: np.ndarray, factor: float = 1.2) -> np.ndarray:
        """
        밝기 조정 (RGB만)

        Args:
            factor: 밝기 배율 (< 1.0: 어둡게, > 1.0: 밝게)
        """
        adjusted = rgb.astype(np.float32) * factor
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        return adjusted

    def adjust_contrast(self, rgb: np.ndarray, factor: float = 1.2) -> np.ndarray:
        """
        대비 조정 (RGB만)

        Args:
            factor: 대비 배율 (< 1.0: 낮게, > 1.0: 높게)
        """
        mean = rgb.mean()
        adjusted = (rgb.astype(np.float32) - mean) * factor + mean
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        return adjusted

    def color_jitter(
        self,
        rgb: np.ndarray,
        hue_shift: int = 10,
        saturation_scale: float = 1.1
    ) -> np.ndarray:
        """
        색상 변화 (RGB만)

        Args:
            hue_shift: 색조 변화량 (-180 ~ 180)
            saturation_scale: 채도 배율
        """
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Hue shift
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180

        # Saturation scale
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)

        hsv = hsv.astype(np.uint8)
        rgb_jittered = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return rgb_jittered

    def gaussian_blur(self, rgb: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        가우시안 블러 (RGB만)

        Args:
            kernel_size: 커널 크기 (홀수)
        """
        if kernel_size % 2 == 0:
            kernel_size += 1

        blurred = cv2.GaussianBlur(rgb, (kernel_size, kernel_size), 0)
        return blurred

    # ========== 통합 증강 함수 ==========

    def augment(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        config: Dict
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        설정에 따라 증강 수행

        Args:
            rgb: RGB 이미지
            mask: 마스크
            config: 증강 설정
                {
                    'scale': float or None,
                    'crop_scale': float or None,  # Crop-based scale (mutually exclusive with 'scale')
                    'crop_offset_x': float or None,  # Horizontal offset ratio
                    'crop_offset_y': float or None,  # Vertical offset ratio
                    'crop_padding': int or None,
                    'rotation': float or None,
                    'flip': str or None,
                    'noise': float or None,
                    'brightness': float or None,
                    'contrast': float or None,
                    'color_jitter': bool,
                    'blur': int or None,
                    'fill_color': str or tuple
                }

        Returns:
            (augmented_rgb, augmented_mask, applied_transforms)
        """
        aug_rgb = rgb.copy()
        aug_mask = mask.copy()
        applied = {}

        # 1. Geometric transforms (RGB + Mask)
        # Note: crop_scale and scale are mutually exclusive
        if config.get('crop_scale') is not None:
            # Crop-based scale augmentation (advanced mode)
            scale = config['crop_scale']
            offset_x = config.get('crop_offset_x', 0.0)
            offset_y = config.get('crop_offset_y', 0.0)
            padding = config.get('crop_padding', 20)
            fill_color_val = config.get('fill_color', 'white')

            # Convert fill_color string to RGB tuple
            if fill_color_val == 'white':
                fill_rgb = (255, 255, 255)
            elif fill_color_val == 'black':
                fill_rgb = (0, 0, 0)
            elif isinstance(fill_color_val, tuple):
                fill_rgb = fill_color_val
            else:
                fill_rgb = (255, 255, 255)

            aug_rgb, aug_mask = self.crop_scale_with_offset(
                aug_rgb, aug_mask, scale, offset_x, offset_y, padding, fill_rgb
            )
            applied['crop_scale'] = scale
            applied['crop_offset'] = (offset_x, offset_y)

        elif config.get('scale') is not None:
            # Traditional scale augmentation (in-place)
            scale = config['scale']
            fill_color = config.get('fill_color', 'white')
            aug_rgb, aug_mask = self.scale_around_bbox(
                aug_rgb, aug_mask, scale, fill_color
            )
            applied['scale'] = scale

        if config.get('rotation') is not None:
            angle = config['rotation']
            aug_rgb, aug_mask = self.rotate(aug_rgb, aug_mask, angle)
            applied['rotation'] = angle

        if config.get('flip') is not None:
            mode = config['flip']
            aug_rgb, aug_mask = self.flip(aug_rgb, aug_mask, mode)
            applied['flip'] = mode

        # 1.5. Prevent clipping (경계 잘림 방지)
        if config.get('prevent_clipping', False):
            # fill_color 결정
            fill_color_val = config.get('fill_color', 'white')
            if fill_color_val == 'white':
                fill_rgb = (255, 255, 255)
            elif fill_color_val == 'black':
                fill_rgb = (0, 0, 0)
            elif isinstance(fill_color_val, tuple):
                fill_rgb = fill_color_val
            else:
                fill_rgb = (255, 255, 255)

            aug_rgb, aug_mask = self.prevent_clipping(aug_rgb, aug_mask, fill_rgb)
            applied['prevent_clipping'] = True

        # 2. Photometric transforms (RGB만)
        if config.get('noise') is not None:
            std = config['noise']
            aug_rgb = self.add_gaussian_noise(aug_rgb, std)
            applied['noise'] = std

        if config.get('brightness') is not None:
            factor = config['brightness']
            aug_rgb = self.adjust_brightness(aug_rgb, factor)
            applied['brightness'] = factor

        if config.get('contrast') is not None:
            factor = config['contrast']
            aug_rgb = self.adjust_contrast(aug_rgb, factor)
            applied['contrast'] = factor

        if config.get('color_jitter'):
            aug_rgb = self.color_jitter(aug_rgb)
            applied['color_jitter'] = True

        if config.get('blur') is not None:
            kernel = config['blur']
            aug_rgb = self.gaussian_blur(aug_rgb, kernel)
            applied['blur'] = kernel

        # 3. Background replacement (최종 단계)
        if config.get('replace_background', False):
            use_bg_image = config.get('use_bg_image', True)
            bg_image_ratio = config.get('bg_image_ratio', 0.5)

            # fill_color 결정
            fill_color_val = config.get('fill_color', 'white')
            if fill_color_val == 'white':
                fill_rgb = (255, 255, 255)
            elif fill_color_val == 'black':
                fill_rgb = (0, 0, 0)
            elif isinstance(fill_color_val, tuple):
                fill_rgb = fill_color_val
            else:
                fill_rgb = (255, 255, 255)

            aug_rgb = self.composite_with_background(
                aug_rgb, aug_mask,
                use_bg_image=use_bg_image,
                bg_image_ratio=bg_image_ratio,
                fill_color=fill_rgb
            )
            applied['replace_background'] = True

        return aug_rgb, aug_mask, applied

    def generate_preview_grid(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        configs: List[Dict],
        grid_size: Tuple[int, int] = (3, 3)
    ) -> np.ndarray:
        """
        여러 증강 결과를 그리드로 시각화

        Args:
            rgb: 원본 RGB
            mask: 원본 마스크
            configs: 증강 설정 리스트
            grid_size: (rows, cols)

        Returns:
            그리드 이미지
        """
        rows, cols = grid_size
        h, w = rgb.shape[:2]

        # 그리드 이미지 생성
        grid = np.ones((rows * h, cols * w, 3), dtype=np.uint8) * 255

        for idx, config in enumerate(configs[:rows * cols]):
            row = idx // cols
            col = idx % cols

            # 증강 수행
            aug_rgb, aug_mask, applied = self.augment(rgb, mask, config)

            # 그리드에 배치 (녹색 오버레이 없이 실제 결과 표시)
            y_start = row * h
            y_end = (row + 1) * h
            x_start = col * w
            x_end = (col + 1) * w

            grid[y_start:y_end, x_start:x_end] = aug_rgb.astype(np.uint8)

        return grid


def generate_augmentation_configs(
    base_config: Dict,
    scale_range: Tuple[float, float] = (0.5, 2.0),
    rotation_range: Tuple[float, float] = (-30, 30),
    num_variations: int = 9
) -> List[Dict]:
    """
    기본 설정을 바탕으로 다양한 증강 설정 생성

    Args:
        base_config: 기본 설정
        scale_range: 스케일 범위
        rotation_range: 회전 범위
        num_variations: 생성할 변형 개수

    Returns:
        증강 설정 리스트
    """
    configs = []

    for _ in range(num_variations):
        config = base_config.copy()

        # 랜덤 스케일
        if 'scale' in base_config and base_config['scale']:
            config['scale'] = random.uniform(*scale_range)

        # 랜덤 회전
        if 'rotation' in base_config and base_config['rotation']:
            config['rotation'] = random.uniform(*rotation_range)

        # 랜덤 밝기
        if 'brightness' in base_config and base_config['brightness']:
            config['brightness'] = random.uniform(0.7, 1.3)

        # 랜덤 노이즈
        if 'noise' in base_config and base_config['noise']:
            config['noise'] = random.uniform(5, 20)

        configs.append(config)

    return configs
