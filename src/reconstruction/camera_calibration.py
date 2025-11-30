"""
Camera Calibration Module for Multi-View 3D Reconstruction

Provides:
- Single camera intrinsic calibration (ChArUco board)
- Multi-camera extrinsic calibration
- Calibration save/load utilities
- Projection/unprojection functions

References:
- OpenCV camera calibration: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- Anipose calibration: https://anipose.readthedocs.io/
- MC-Calib: Multi-camera calibration toolbox
"""

import numpy as np
import cv2
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Union
import pickle

logger = logging.getLogger(__name__)


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters"""
    camera_matrix: np.ndarray  # 3x3 intrinsic matrix (K)
    dist_coeffs: np.ndarray    # Distortion coefficients
    image_size: Tuple[int, int]  # (width, height)

    # Optional metadata
    camera_id: str = ""
    calibration_error: float = 0.0

    def to_dict(self) -> dict:
        return {
            'camera_matrix': self.camera_matrix.tolist(),
            'dist_coeffs': self.dist_coeffs.tolist(),
            'image_size': list(self.image_size),
            'camera_id': self.camera_id,
            'calibration_error': self.calibration_error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'CameraIntrinsics':
        return cls(
            camera_matrix=np.array(data['camera_matrix']),
            dist_coeffs=np.array(data['dist_coeffs']),
            image_size=tuple(data['image_size']),
            camera_id=data.get('camera_id', ''),
            calibration_error=data.get('calibration_error', 0.0),
        )


@dataclass
class CameraExtrinsics:
    """Camera extrinsic parameters (pose relative to world/reference)"""
    rotation: np.ndarray      # 3x3 rotation matrix (R)
    translation: np.ndarray   # 3x1 translation vector (t)

    # Reference info
    reference_camera_id: str = ""
    camera_id: str = ""

    @property
    def rvec(self) -> np.ndarray:
        """Rodrigues rotation vector"""
        rvec, _ = cv2.Rodrigues(self.rotation)
        return rvec.flatten()

    @property
    def projection_matrix(self) -> np.ndarray:
        """3x4 projection matrix [R|t]"""
        return np.hstack([self.rotation, self.translation.reshape(3, 1)])

    def to_dict(self) -> dict:
        return {
            'rotation': self.rotation.tolist(),
            'translation': self.translation.tolist(),
            'reference_camera_id': self.reference_camera_id,
            'camera_id': self.camera_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'CameraExtrinsics':
        return cls(
            rotation=np.array(data['rotation']),
            translation=np.array(data['translation']),
            reference_camera_id=data.get('reference_camera_id', ''),
            camera_id=data.get('camera_id', ''),
        )


@dataclass
class CameraParams:
    """Complete camera parameters (intrinsic + extrinsic)"""
    intrinsics: CameraIntrinsics
    extrinsics: Optional[CameraExtrinsics] = None

    @property
    def P(self) -> np.ndarray:
        """Full 3x4 projection matrix: K @ [R|t]"""
        if self.extrinsics is None:
            # Identity extrinsics
            return np.hstack([self.intrinsics.camera_matrix, np.zeros((3, 1))])
        return self.intrinsics.camera_matrix @ self.extrinsics.projection_matrix

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Project 3D points to 2D image coordinates

        Args:
            points_3d: Nx3 array of 3D points

        Returns:
            Nx2 array of 2D image coordinates
        """
        if points_3d.ndim == 1:
            points_3d = points_3d.reshape(1, 3)

        if self.extrinsics is not None:
            rvec = self.extrinsics.rvec
            tvec = self.extrinsics.translation
        else:
            rvec = np.zeros(3)
            tvec = np.zeros(3)

        points_2d, _ = cv2.projectPoints(
            points_3d.astype(np.float64),
            rvec,
            tvec,
            self.intrinsics.camera_matrix,
            self.intrinsics.dist_coeffs
        )
        return points_2d.reshape(-1, 2)

    def undistort_points(self, points_2d: np.ndarray) -> np.ndarray:
        """Undistort 2D points"""
        if points_2d.ndim == 1:
            points_2d = points_2d.reshape(1, 2)

        undistorted = cv2.undistortPoints(
            points_2d.reshape(-1, 1, 2).astype(np.float64),
            self.intrinsics.camera_matrix,
            self.intrinsics.dist_coeffs,
            P=self.intrinsics.camera_matrix
        )
        return undistorted.reshape(-1, 2)

    def to_dict(self) -> dict:
        data = {'intrinsics': self.intrinsics.to_dict()}
        if self.extrinsics is not None:
            data['extrinsics'] = self.extrinsics.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'CameraParams':
        intrinsics = CameraIntrinsics.from_dict(data['intrinsics'])
        extrinsics = None
        if 'extrinsics' in data:
            extrinsics = CameraExtrinsics.from_dict(data['extrinsics'])
        return cls(intrinsics=intrinsics, extrinsics=extrinsics)


class CameraCalibrator:
    """
    Single camera intrinsic calibration using ChArUco board

    ChArUco board combines checkerboard and ArUco markers for robust detection.
    """

    def __init__(
        self,
        board_size: Tuple[int, int] = (5, 7),  # (columns, rows) of squares
        square_size: float = 0.04,  # meters
        marker_size: float = 0.02,  # meters
        aruco_dict: int = cv2.aruco.DICT_4X4_50
    ):
        """
        Initialize calibrator

        Args:
            board_size: Number of squares (columns, rows)
            square_size: Size of checkerboard squares in meters
            marker_size: Size of ArUco markers in meters
            aruco_dict: ArUco dictionary type
        """
        self.board_size = board_size
        self.square_size = square_size
        self.marker_size = marker_size

        # Create ArUco dictionary and ChArUco board
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self.charuco_board = cv2.aruco.CharucoBoard(
            board_size,
            square_size,
            marker_size,
            self.aruco_dict
        )
        self.detector_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)

    def detect_charuco(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect ChArUco corners in image

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            (charuco_corners, charuco_ids) or (None, None) if detection failed
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Detect ArUco markers
        marker_corners, marker_ids, _ = self.aruco_detector.detectMarkers(gray)

        if marker_ids is None or len(marker_ids) < 4:
            return None, None

        # Interpolate ChArUco corners
        num_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, self.charuco_board
        )

        if num_corners < 4:
            return None, None

        return charuco_corners, charuco_ids

    def calibrate(
        self,
        images: List[np.ndarray],
        camera_id: str = ""
    ) -> Optional[CameraIntrinsics]:
        """
        Calibrate camera from list of images containing ChArUco board

        Args:
            images: List of calibration images
            camera_id: Optional camera identifier

        Returns:
            CameraIntrinsics or None if calibration failed
        """
        all_corners = []
        all_ids = []
        image_size = None

        for i, image in enumerate(images):
            if image_size is None:
                image_size = (image.shape[1], image.shape[0])

            corners, ids = self.detect_charuco(image)
            if corners is not None:
                all_corners.append(corners)
                all_ids.append(ids)
                logger.debug(f"Image {i}: detected {len(corners)} corners")
            else:
                logger.debug(f"Image {i}: detection failed")

        if len(all_corners) < 3:
            logger.error(f"Not enough valid images for calibration: {len(all_corners)}")
            return None

        logger.info(f"Calibrating with {len(all_corners)} images...")

        # Run calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            all_corners, all_ids, self.charuco_board, image_size, None, None
        )

        if not ret:
            logger.error("Calibration failed")
            return None

        logger.info(f"Calibration successful, RMS error: {ret:.4f}")

        return CameraIntrinsics(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            image_size=image_size,
            camera_id=camera_id,
            calibration_error=ret
        )

    def draw_detected(self, image: np.ndarray) -> np.ndarray:
        """Draw detected ChArUco corners on image"""
        output = image.copy()
        corners, ids = self.detect_charuco(image)

        if corners is not None:
            cv2.aruco.drawDetectedCornersCharuco(output, corners, ids)

        return output


class MultiCameraSystem:
    """
    Multi-camera system with calibration and triangulation

    Manages multiple cameras with shared world coordinate system.
    """

    def __init__(self, cameras: Optional[Dict[str, CameraParams]] = None):
        """
        Initialize multi-camera system

        Args:
            cameras: Dictionary of camera_id -> CameraParams
        """
        self.cameras: Dict[str, CameraParams] = cameras or {}
        self.reference_camera_id: Optional[str] = None

    def add_camera(self, camera_id: str, params: CameraParams):
        """Add camera to system"""
        self.cameras[camera_id] = params
        if self.reference_camera_id is None:
            self.reference_camera_id = camera_id

    def set_reference(self, camera_id: str):
        """Set reference camera (world origin)"""
        if camera_id not in self.cameras:
            raise ValueError(f"Camera {camera_id} not in system")
        self.reference_camera_id = camera_id

    def triangulate_point(
        self,
        observations: Dict[str, np.ndarray],
        min_views: int = 2
    ) -> Optional[np.ndarray]:
        """
        Triangulate 3D point from multi-view 2D observations

        Args:
            observations: Dict of camera_id -> 2D point (x, y)
            min_views: Minimum number of views required

        Returns:
            3D point (x, y, z) or None if triangulation failed
        """
        if len(observations) < min_views:
            return None

        # Build projection matrices and points
        proj_matrices = []
        points_2d = []

        for cam_id, point_2d in observations.items():
            if cam_id not in self.cameras:
                continue

            cam = self.cameras[cam_id]
            proj_matrices.append(cam.P)

            # Undistort point
            undistorted = cam.undistort_points(point_2d)
            points_2d.append(undistorted.flatten())

        if len(proj_matrices) < min_views:
            return None

        # DLT triangulation
        point_3d = self._triangulate_dlt(proj_matrices, points_2d)
        return point_3d

    def _triangulate_dlt(
        self,
        proj_matrices: List[np.ndarray],
        points_2d: List[np.ndarray]
    ) -> np.ndarray:
        """
        Direct Linear Transform triangulation

        Args:
            proj_matrices: List of 3x4 projection matrices
            points_2d: List of 2D points

        Returns:
            3D point
        """
        n_views = len(proj_matrices)
        A = np.zeros((2 * n_views, 4))

        for i, (P, pt) in enumerate(zip(proj_matrices, points_2d)):
            x, y = pt
            A[2*i] = x * P[2] - P[0]
            A[2*i + 1] = y * P[2] - P[1]

        # SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X[:3] / X[3]  # Dehomogenize

        return X

    def triangulate_points(
        self,
        observations: Dict[str, np.ndarray],
        min_views: int = 2
    ) -> np.ndarray:
        """
        Triangulate multiple 3D points from multi-view observations

        Args:
            observations: Dict of camera_id -> Nx2 array of points
            min_views: Minimum views per point

        Returns:
            Nx3 array of 3D points (NaN for failed triangulations)
        """
        # Get number of points from first camera
        first_cam = list(observations.keys())[0]
        n_points = len(observations[first_cam])

        points_3d = np.full((n_points, 3), np.nan)

        for i in range(n_points):
            point_obs = {}
            for cam_id, points in observations.items():
                if not np.any(np.isnan(points[i])):
                    point_obs[cam_id] = points[i]

            result = self.triangulate_point(point_obs, min_views)
            if result is not None:
                points_3d[i] = result

        return points_3d

    def compute_reprojection_error(
        self,
        point_3d: np.ndarray,
        observations: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute reprojection error for each camera

        Args:
            point_3d: 3D point
            observations: Dict of camera_id -> 2D observation

        Returns:
            Dict of camera_id -> reprojection error (pixels)
        """
        errors = {}

        for cam_id, point_2d in observations.items():
            if cam_id not in self.cameras:
                continue

            cam = self.cameras[cam_id]
            projected = cam.project(point_3d)
            error = np.linalg.norm(projected - point_2d)
            errors[cam_id] = error

        return errors

    def to_dict(self) -> dict:
        return {
            'cameras': {k: v.to_dict() for k, v in self.cameras.items()},
            'reference_camera_id': self.reference_camera_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'MultiCameraSystem':
        cameras = {k: CameraParams.from_dict(v) for k, v in data['cameras'].items()}
        system = cls(cameras)
        system.reference_camera_id = data.get('reference_camera_id')
        return system


def calibrate_stereo_pair(
    camera1: CameraParams,
    camera2: CameraParams,
    image_pairs: List[Tuple[np.ndarray, np.ndarray]],
    calibrator: CameraCalibrator
) -> Optional[CameraExtrinsics]:
    """
    Calibrate extrinsics between two cameras using ChArUco board

    Args:
        camera1: First camera (reference)
        camera2: Second camera
        image_pairs: List of (image1, image2) pairs with ChArUco board
        calibrator: ChArUco calibrator

    Returns:
        Extrinsics of camera2 relative to camera1
    """
    obj_points = []
    img_points1 = []
    img_points2 = []

    for img1, img2 in image_pairs:
        corners1, ids1 = calibrator.detect_charuco(img1)
        corners2, ids2 = calibrator.detect_charuco(img2)

        if corners1 is None or corners2 is None:
            continue

        # Find common corners
        common_ids = np.intersect1d(ids1.flatten(), ids2.flatten())
        if len(common_ids) < 6:
            continue

        # Get corresponding corners
        mask1 = np.isin(ids1.flatten(), common_ids)
        mask2 = np.isin(ids2.flatten(), common_ids)

        pts1 = corners1[mask1].reshape(-1, 2)
        pts2 = corners2[mask2].reshape(-1, 2)

        # Get 3D object points
        obj_pts = calibrator.charuco_board.getChessboardCorners()[common_ids]

        obj_points.append(obj_pts.astype(np.float32))
        img_points1.append(pts1.astype(np.float32))
        img_points2.append(pts2.astype(np.float32))

    if len(obj_points) < 3:
        logger.error("Not enough valid stereo pairs")
        return None

    # Stereo calibration
    ret, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        obj_points, img_points1, img_points2,
        camera1.intrinsics.camera_matrix, camera1.intrinsics.dist_coeffs,
        camera2.intrinsics.camera_matrix, camera2.intrinsics.dist_coeffs,
        camera1.intrinsics.image_size,
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    logger.info(f"Stereo calibration RMS error: {ret:.4f}")

    return CameraExtrinsics(
        rotation=R,
        translation=T.flatten(),
        reference_camera_id=camera1.intrinsics.camera_id,
        camera_id=camera2.intrinsics.camera_id
    )


def save_calibration(
    calibration: Union[CameraParams, MultiCameraSystem, dict],
    filepath: Union[str, Path],
    format: str = 'json'
):
    """
    Save calibration to file

    Args:
        calibration: Calibration data
        filepath: Output file path
        format: 'json' or 'pickle'
    """
    filepath = Path(filepath)

    if hasattr(calibration, 'to_dict'):
        data = calibration.to_dict()
    else:
        data = calibration

    if format == 'json':
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    elif format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    else:
        raise ValueError(f"Unknown format: {format}")

    logger.info(f"Saved calibration to {filepath}")


def load_calibration(
    filepath: Union[str, Path],
    as_type: str = 'auto'
) -> Union[CameraParams, MultiCameraSystem, dict]:
    """
    Load calibration from file

    Args:
        filepath: Input file path
        as_type: 'camera', 'system', 'dict', or 'auto'

    Returns:
        Loaded calibration data
    """
    filepath = Path(filepath)

    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
    elif filepath.suffix in ['.pkl', '.pickle']:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    else:
        # Try JSON first
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

    if as_type == 'dict':
        return data

    if as_type == 'auto':
        if 'cameras' in data:
            as_type = 'system'
        elif 'intrinsics' in data:
            as_type = 'camera'
        else:
            return data

    if as_type == 'camera':
        return CameraParams.from_dict(data)
    elif as_type == 'system':
        return MultiCameraSystem.from_dict(data)

    return data


# Utility functions for Anipose compatibility
def load_anipose_calibration(calibration_toml: Union[str, Path]) -> MultiCameraSystem:
    """
    Load calibration from Anipose TOML format

    Args:
        calibration_toml: Path to Anipose calibration.toml

    Returns:
        MultiCameraSystem
    """
    try:
        import toml
    except ImportError:
        raise ImportError("toml package required: pip install toml")

    with open(calibration_toml, 'r') as f:
        calib = toml.load(f)

    system = MultiCameraSystem()

    for cam_name, cam_data in calib.items():
        if not isinstance(cam_data, dict):
            continue

        # Parse Anipose format
        if 'matrix' in cam_data:
            K = np.array(cam_data['matrix']).reshape(3, 3)
            dist = np.array(cam_data.get('distortions', [0, 0, 0, 0, 0]))
            size = tuple(cam_data.get('size', [1920, 1080]))

            intrinsics = CameraIntrinsics(
                camera_matrix=K,
                dist_coeffs=dist,
                image_size=size,
                camera_id=cam_name
            )

            extrinsics = None
            if 'rotation' in cam_data and 'translation' in cam_data:
                R = np.array(cam_data['rotation']).reshape(3, 3)
                t = np.array(cam_data['translation'])
                extrinsics = CameraExtrinsics(
                    rotation=R,
                    translation=t,
                    camera_id=cam_name
                )

            system.add_camera(cam_name, CameraParams(intrinsics, extrinsics))

    return system
