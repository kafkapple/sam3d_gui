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

        return x_min, y_min, x_max, y_max

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

            # 마스크 오버레이
            overlay = aug_rgb.copy()
            overlay[aug_mask] = overlay[aug_mask] * 0.7 + np.array([0, 255, 0]) * 0.3

            # 그리드에 배치
            y_start = row * h
            y_end = (row + 1) * h
            x_start = col * w
            x_end = (col + 1) * w

            grid[y_start:y_end, x_start:x_end] = overlay.astype(np.uint8)

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
