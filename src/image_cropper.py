"""
Image Cropper Utility
Mask 기반으로 RGB/Mask 이미지 쌍을 crop하여 저장
"""
import os
from pathlib import Path
from typing import Tuple, List, Optional, Callable
import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class CropResult:
    """Result of a single image crop operation"""
    source_rgb: str
    source_mask: str
    output_rgb: str
    output_mask: str
    original_size: Tuple[int, int]  # (width, height)
    crop_bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    cropped_size: Tuple[int, int]  # (width, height)
    success: bool
    error_message: Optional[str] = None


@dataclass
class BatchCropResult:
    """Result of batch crop operation"""
    total_files: int
    successful: int
    failed: int
    results: List[CropResult]
    output_dir: str


def find_image_pairs(
    input_dir: str,
    rgb_pattern: str = "rgb",
    mask_pattern: str = "mask",
    extensions: List[str] = None
) -> List[Tuple[Path, Path]]:
    """
    Find RGB/Mask image pairs in directory.

    Matching strategies:
    1. Same filename with different subfolder (rgb/, mask/)
    2. Same filename with pattern in name (xxx_rgb.png, xxx_mask.png)
    3. Matching by index/number in filename

    Args:
        input_dir: Directory to search
        rgb_pattern: Pattern to identify RGB images
        mask_pattern: Pattern to identify mask images
        extensions: Image extensions to look for

    Returns:
        List of (rgb_path, mask_path) tuples
    """
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

    input_path = Path(input_dir)
    if not input_path.exists():
        return []

    pairs = []

    # Strategy 1: Subfolder structure (rgb/, mask/)
    rgb_dir = input_path / rgb_pattern
    mask_dir = input_path / mask_pattern

    if rgb_dir.exists() and mask_dir.exists():
        for rgb_file in rgb_dir.iterdir():
            if rgb_file.suffix.lower() in extensions:
                mask_file = mask_dir / rgb_file.name
                if mask_file.exists():
                    pairs.append((rgb_file, mask_file))
                else:
                    # Try with different extensions
                    for ext in extensions:
                        mask_candidate = mask_dir / (rgb_file.stem + ext)
                        if mask_candidate.exists():
                            pairs.append((rgb_file, mask_candidate))
                            break
        return pairs

    # Strategy 2: Pattern in filename (xxx_rgb.png, xxx_mask.png)
    all_files = [f for f in input_path.iterdir()
                 if f.is_file() and f.suffix.lower() in extensions]

    rgb_files = {}
    mask_files = {}

    for f in all_files:
        name_lower = f.stem.lower()
        if rgb_pattern.lower() in name_lower:
            # Extract base name without pattern
            base = name_lower.replace(f"_{rgb_pattern.lower()}", "").replace(f"{rgb_pattern.lower()}_", "").replace(rgb_pattern.lower(), "")
            rgb_files[base] = f
        elif mask_pattern.lower() in name_lower:
            base = name_lower.replace(f"_{mask_pattern.lower()}", "").replace(f"{mask_pattern.lower()}_", "").replace(mask_pattern.lower(), "")
            mask_files[base] = f

    for base, rgb_file in rgb_files.items():
        if base in mask_files:
            pairs.append((rgb_file, mask_files[base]))

    if pairs:
        return pairs

    # Strategy 3: Numbered files (0001.png in rgb/, 0001.png in mask/)
    # Already covered by Strategy 1

    # Strategy 4: All images as RGB, look for corresponding masks
    for f in all_files:
        # Skip if this looks like a mask
        if mask_pattern.lower() in f.stem.lower():
            continue

        # Look for mask with same name + mask pattern
        for ext in extensions:
            mask_candidates = [
                input_path / f"{f.stem}_{mask_pattern}{ext}",
                input_path / f"{mask_pattern}_{f.stem}{ext}",
                input_path / f"{f.stem}{mask_pattern}{ext}",
            ]
            for mask_candidate in mask_candidates:
                if mask_candidate.exists():
                    pairs.append((f, mask_candidate))
                    break

    return pairs


def get_mask_bbox(
    mask: np.ndarray,
    padding: int = 0,
    padding_ratio: float = 0.0,
    square: bool = False
) -> Optional[Tuple[int, int, int, int]]:
    """
    Get bounding box from binary mask.

    Args:
        mask: Binary mask image (H, W) or (H, W, C)
        padding: Fixed padding in pixels
        padding_ratio: Padding as ratio of bbox size (0.1 = 10%)
        square: If True, make bbox square (use larger dimension)

    Returns:
        (x, y, w, h) or None if no mask found
    """
    # Convert to grayscale if needed
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Threshold to binary
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Get bounding box of all contours combined
    all_points = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_points)

    # Apply padding
    pad_x = max(padding, int(w * padding_ratio))
    pad_y = max(padding, int(h * padding_ratio))

    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = w + 2 * pad_x
    h = h + 2 * pad_y

    # Make square if requested
    if square:
        size = max(w, h)
        # Center the square
        center_x = x + w // 2
        center_y = y + h // 2
        x = center_x - size // 2
        y = center_y - size // 2
        w = h = size

    # Clamp to image bounds
    img_h, img_w = mask.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, img_w - x)
    h = min(h, img_h - y)

    return (x, y, w, h)


def crop_image_pair(
    rgb_path: str,
    mask_path: str,
    output_rgb_path: str,
    output_mask_path: str,
    padding: int = 0,
    padding_ratio: float = 0.0,
    square: bool = False,
    resize: Optional[Tuple[int, int]] = None
) -> CropResult:
    """
    Crop RGB/Mask image pair based on mask bounding box.

    Args:
        rgb_path: Path to RGB image
        mask_path: Path to mask image
        output_rgb_path: Output path for cropped RGB
        output_mask_path: Output path for cropped mask
        padding: Fixed padding in pixels
        padding_ratio: Padding as ratio of bbox size
        square: Make crop square
        resize: Optional (width, height) to resize output

    Returns:
        CropResult with operation details
    """
    try:
        # Read images
        rgb = cv2.imread(rgb_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if rgb is None:
            return CropResult(
                source_rgb=rgb_path,
                source_mask=mask_path,
                output_rgb=output_rgb_path,
                output_mask=output_mask_path,
                original_size=(0, 0),
                crop_bbox=(0, 0, 0, 0),
                cropped_size=(0, 0),
                success=False,
                error_message=f"Failed to read RGB image: {rgb_path}"
            )

        if mask is None:
            return CropResult(
                source_rgb=rgb_path,
                source_mask=mask_path,
                output_rgb=output_rgb_path,
                output_mask=output_mask_path,
                original_size=(rgb.shape[1], rgb.shape[0]),
                crop_bbox=(0, 0, 0, 0),
                cropped_size=(0, 0),
                success=False,
                error_message=f"Failed to read mask image: {mask_path}"
            )

        original_size = (rgb.shape[1], rgb.shape[0])

        # Get bounding box from mask
        bbox = get_mask_bbox(mask, padding=padding, padding_ratio=padding_ratio, square=square)

        if bbox is None:
            return CropResult(
                source_rgb=rgb_path,
                source_mask=mask_path,
                output_rgb=output_rgb_path,
                output_mask=output_mask_path,
                original_size=original_size,
                crop_bbox=(0, 0, 0, 0),
                cropped_size=(0, 0),
                success=False,
                error_message="No mask content found (empty mask)"
            )

        x, y, w, h = bbox

        # Crop both images
        rgb_cropped = rgb[y:y+h, x:x+w]
        mask_cropped = mask[y:y+h, x:x+w]

        cropped_size = (w, h)

        # Resize if requested
        if resize is not None:
            rgb_cropped = cv2.resize(rgb_cropped, resize, interpolation=cv2.INTER_LANCZOS4)
            mask_cropped = cv2.resize(mask_cropped, resize, interpolation=cv2.INTER_NEAREST)
            cropped_size = resize

        # Create output directories
        os.makedirs(os.path.dirname(output_rgb_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)

        # Save cropped images
        cv2.imwrite(output_rgb_path, rgb_cropped)
        cv2.imwrite(output_mask_path, mask_cropped)

        return CropResult(
            source_rgb=rgb_path,
            source_mask=mask_path,
            output_rgb=output_rgb_path,
            output_mask=output_mask_path,
            original_size=original_size,
            crop_bbox=bbox,
            cropped_size=cropped_size,
            success=True
        )

    except Exception as e:
        return CropResult(
            source_rgb=rgb_path,
            source_mask=mask_path,
            output_rgb=output_rgb_path,
            output_mask=output_mask_path,
            original_size=(0, 0),
            crop_bbox=(0, 0, 0, 0),
            cropped_size=(0, 0),
            success=False,
            error_message=str(e)
        )


def batch_crop_images(
    input_dir: str,
    output_dir: str,
    rgb_pattern: str = "rgb",
    mask_pattern: str = "mask",
    padding: int = 0,
    padding_ratio: float = 0.0,
    square: bool = False,
    resize: Optional[Tuple[int, int]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> BatchCropResult:
    """
    Batch crop RGB/Mask image pairs from input directory.

    Args:
        input_dir: Source directory containing RGB/Mask pairs
        output_dir: Output directory for cropped images
        rgb_pattern: Pattern to identify RGB images/folder
        mask_pattern: Pattern to identify mask images/folder
        padding: Fixed padding in pixels
        padding_ratio: Padding as ratio of bbox size
        square: Make crops square
        resize: Optional (width, height) to resize outputs
        progress_callback: Optional callback(current, total, message)

    Returns:
        BatchCropResult with operation summary
    """
    # Find image pairs
    pairs = find_image_pairs(input_dir, rgb_pattern, mask_pattern)

    if not pairs:
        return BatchCropResult(
            total_files=0,
            successful=0,
            failed=0,
            results=[],
            output_dir=output_dir
        )

    # Create output directory structure
    output_path = Path(output_dir)
    output_rgb_dir = output_path / rgb_pattern
    output_mask_dir = output_path / mask_pattern
    output_rgb_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    results = []
    successful = 0
    failed = 0

    for i, (rgb_file, mask_file) in enumerate(pairs):
        if progress_callback:
            progress_callback(i + 1, len(pairs), f"Processing {rgb_file.name}")

        # Determine output paths
        output_rgb = output_rgb_dir / rgb_file.name
        output_mask = output_mask_dir / mask_file.name

        # Crop
        result = crop_image_pair(
            str(rgb_file),
            str(mask_file),
            str(output_rgb),
            str(output_mask),
            padding=padding,
            padding_ratio=padding_ratio,
            square=square,
            resize=resize
        )

        results.append(result)

        if result.success:
            successful += 1
        else:
            failed += 1

    return BatchCropResult(
        total_files=len(pairs),
        successful=successful,
        failed=failed,
        results=results,
        output_dir=output_dir
    )


def preview_crop(
    rgb_path: str,
    mask_path: str,
    padding: int = 0,
    padding_ratio: float = 0.0,
    square: bool = False
) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]]:
    """
    Preview crop without saving.

    Args:
        rgb_path: Path to RGB image
        mask_path: Path to mask image
        padding: Fixed padding in pixels
        padding_ratio: Padding as ratio of bbox size
        square: Make crop square

    Returns:
        (cropped_rgb, cropped_mask, bbox) or None if failed
    """
    try:
        rgb = cv2.imread(rgb_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if rgb is None or mask is None:
            return None

        bbox = get_mask_bbox(mask, padding=padding, padding_ratio=padding_ratio, square=square)

        if bbox is None:
            return None

        x, y, w, h = bbox
        rgb_cropped = rgb[y:y+h, x:x+w]
        mask_cropped = mask[y:y+h, x:x+w]

        # Convert RGB from BGR to RGB for display
        rgb_cropped = cv2.cvtColor(rgb_cropped, cv2.COLOR_BGR2RGB)

        return rgb_cropped, mask_cropped, bbox

    except Exception:
        return None


if __name__ == "__main__":
    # Test the module
    import sys

    if len(sys.argv) < 3:
        print("Usage: python image_cropper.py <input_dir> <output_dir>")
        print("  Finds RGB/Mask pairs and crops them based on mask bounding box")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    def progress(current, total, msg):
        print(f"[{current}/{total}] {msg}")

    result = batch_crop_images(
        input_dir=input_dir,
        output_dir=output_dir,
        padding_ratio=0.1,
        progress_callback=progress
    )

    print(f"\nBatch crop complete:")
    print(f"  Total: {result.total_files}")
    print(f"  Successful: {result.successful}")
    print(f"  Failed: {result.failed}")

    if result.failed > 0:
        print("\nFailed files:")
        for r in result.results:
            if not r.success:
                print(f"  {r.source_rgb}: {r.error_message}")
