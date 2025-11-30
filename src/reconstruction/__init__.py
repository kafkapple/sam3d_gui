"""
Multi-View 3D Reconstruction Module for Mouse Behavioral Analysis

This module provides tools for:
1. Camera calibration (intrinsic/extrinsic)
2. Template mesh deformation (Non-rigid ICP, Silhouette-based)
3. Multi-view mesh fusion (ICP alignment, Visual Hull)
4. Neural rendering (D-NeRF, 4D-GS) - optional

Priority order for implementation:
- Phase 1: Camera calibration + Keypoint triangulation
- Phase 2: Template deformation (silhouette + non-rigid ICP)
- Phase 3: Multi-view fusion
- Phase 4: Neural rendering (optional)

Usage:
    from src.reconstruction import (
        CameraCalibrator,
        MultiCameraSystem,
        NonRigidICP,
        MultiViewFusion,
    )

    # Camera calibration
    calibrator = CameraCalibrator(board_size=(5, 7))
    intrinsics = calibrator.calibrate(images)

    # Template deformation
    deformer = NonRigidICP()
    result = deformer.deform(source_vertices, source_faces, target_points)

    # Multi-view fusion
    fusion = MultiViewFusion(fusion_method='poisson')
    fused = fusion.fuse(meshes)
"""

# Camera calibration (core functionality)
from .camera_calibration import (
    CameraCalibrator,
    CameraIntrinsics,
    CameraExtrinsics,
    CameraParams,
    MultiCameraSystem,
    load_calibration,
    save_calibration,
    calibrate_stereo_pair,
    load_anipose_calibration,
)

# Template deformation
from .template_deformation import (
    DeformationResult,
    NonRigidICP,
    SilhouetteDeformer,
    TemplateMeshDeformer,
    compute_mesh_iou,
    rigid_align,
)

# Multi-view fusion
from .multiview_fusion import (
    FusionResult,
    ICPMeshAligner,
    VisualHullReconstructor,
    MultiViewFusion,
    fuse_sam3d_meshes,
    reconstruct_visual_hull,
)

__all__ = [
    # Camera calibration
    'CameraCalibrator',
    'CameraIntrinsics',
    'CameraExtrinsics',
    'CameraParams',
    'MultiCameraSystem',
    'load_calibration',
    'save_calibration',
    'calibrate_stereo_pair',
    'load_anipose_calibration',
    # Template deformation
    'DeformationResult',
    'NonRigidICP',
    'SilhouetteDeformer',
    'TemplateMeshDeformer',
    'compute_mesh_iou',
    'rigid_align',
    # Multi-view fusion
    'FusionResult',
    'ICPMeshAligner',
    'VisualHullReconstructor',
    'MultiViewFusion',
    'fuse_sam3d_meshes',
    'reconstruct_visual_hull',
]

__version__ = '0.1.0'
