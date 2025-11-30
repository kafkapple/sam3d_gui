"""
Multi-View Mesh Fusion Module

Provides:
- ICP-based mesh alignment and fusion
- Visual Hull reconstruction from silhouettes
- Multi-view SAM-3D mesh fusion

Key References:
- Laurentini "The Visual Hull Concept" (1994)
- Besl & McKay "ICP" (1992)
- Franco & Boyer "Carved Visual Hulls" (2003)

Theory:

1. ICP Mesh Alignment:
   - Iteratively find closest points between meshes
   - Compute optimal rigid transformation
   - Apply until convergence
   - Variants: point-to-point, point-to-plane

2. Visual Hull:
   - Back-project silhouettes to 3D cones
   - Intersect cones from all cameras
   - Result: conservative shape estimate
   - Limitation: cannot recover concavities

3. Multi-view SAM-3D Fusion:
   - Generate mesh from each view using SAM-3D
   - Align meshes using ICP
   - Fuse via averaging or Poisson reconstruction
"""

import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Union
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Optional imports with fallback
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    logger.warning("trimesh not installed. Some features will be unavailable.")

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    logger.warning("open3d not installed. Some features will be unavailable.")

try:
    from scipy.spatial import cKDTree
    from scipy.ndimage import binary_erosion, binary_dilation
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class FusionResult:
    """Result of multi-view fusion"""
    vertices: np.ndarray          # Fused mesh vertices (Nx3)
    faces: np.ndarray             # Fused mesh faces (Mx3)
    method: str = ""              # Fusion method used
    num_source_meshes: int = 0    # Number of input meshes
    quality_score: float = 0.0    # Optional quality metric

    # Optional: per-view alignment info
    transformations: Optional[List[np.ndarray]] = None

    def to_mesh(self) -> 'trimesh.Trimesh':
        """Convert to trimesh object"""
        if not HAS_TRIMESH:
            raise ImportError("trimesh required")
        return trimesh.Trimesh(vertices=self.vertices, faces=self.faces)


class ICPMeshAligner:
    """
    Iterative Closest Point alignment for mesh registration

    Supports:
    - Point-to-point ICP
    - Point-to-plane ICP (more robust)
    - Multi-scale/coarse-to-fine alignment
    """

    def __init__(
        self,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
        max_correspondence_distance: float = 0.05,  # meters
        method: str = 'point_to_plane'  # 'point_to_point' or 'point_to_plane'
    ):
        """
        Initialize ICP aligner

        Args:
            max_iterations: Maximum ICP iterations
            tolerance: Convergence tolerance (change in fitness)
            max_correspondence_distance: Max distance for valid correspondence
            method: 'point_to_point' or 'point_to_plane'
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.max_correspondence_distance = max_correspondence_distance
        self.method = method

    def align(
        self,
        source_mesh: Union['trimesh.Trimesh', Tuple[np.ndarray, np.ndarray]],
        target_mesh: Union['trimesh.Trimesh', Tuple[np.ndarray, np.ndarray]],
        initial_transform: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Align source mesh to target mesh using ICP

        Args:
            source_mesh: Source mesh (trimesh or (vertices, faces))
            target_mesh: Target mesh
            initial_transform: Initial 4x4 transformation matrix

        Returns:
            (transformation_matrix, fitness_score)
        """
        if HAS_OPEN3D:
            return self._align_open3d(source_mesh, target_mesh, initial_transform)
        else:
            return self._align_numpy(source_mesh, target_mesh, initial_transform)

    def _align_open3d(
        self,
        source_mesh,
        target_mesh,
        initial_transform
    ) -> Tuple[np.ndarray, float]:
        """Open3D-based ICP alignment"""
        # Convert to Open3D point clouds
        if HAS_TRIMESH and isinstance(source_mesh, trimesh.Trimesh):
            source_vertices = source_mesh.vertices
        else:
            source_vertices = source_mesh[0]

        if HAS_TRIMESH and isinstance(target_mesh, trimesh.Trimesh):
            target_vertices = target_mesh.vertices
        else:
            target_vertices = target_mesh[0]

        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_vertices)

        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_vertices)

        # Estimate normals for point-to-plane
        if self.method == 'point_to_plane':
            source_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            target_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )

        if initial_transform is None:
            initial_transform = np.eye(4)

        # Run ICP
        if self.method == 'point_to_plane':
            result = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd,
                self.max_correspondence_distance,
                initial_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=self.max_iterations,
                    relative_fitness=self.tolerance,
                    relative_rmse=self.tolerance
                )
            )
        else:
            result = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd,
                self.max_correspondence_distance,
                initial_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=self.max_iterations,
                    relative_fitness=self.tolerance,
                    relative_rmse=self.tolerance
                )
            )

        return result.transformation, result.fitness

    def _align_numpy(
        self,
        source_mesh,
        target_mesh,
        initial_transform
    ) -> Tuple[np.ndarray, float]:
        """Pure NumPy ICP implementation (point-to-point only)"""
        if not HAS_SCIPY:
            raise ImportError("scipy required for numpy ICP")

        # Extract vertices
        if HAS_TRIMESH and isinstance(source_mesh, trimesh.Trimesh):
            source_pts = source_mesh.vertices.copy()
        else:
            source_pts = source_mesh[0].copy()

        if HAS_TRIMESH and isinstance(target_mesh, trimesh.Trimesh):
            target_pts = target_mesh.vertices
        else:
            target_pts = target_mesh[0]

        # Apply initial transform
        if initial_transform is not None:
            source_pts = (initial_transform[:3, :3] @ source_pts.T).T + initial_transform[:3, 3]

        transform = np.eye(4)
        target_tree = cKDTree(target_pts)

        prev_error = float('inf')

        for iteration in range(self.max_iterations):
            # Find correspondences
            distances, indices = target_tree.query(source_pts, k=1)

            # Filter by distance
            valid_mask = distances < self.max_correspondence_distance
            if np.sum(valid_mask) < 10:
                logger.warning("Too few valid correspondences in ICP")
                break

            src_valid = source_pts[valid_mask]
            tgt_valid = target_pts[indices[valid_mask]]

            # Compute rigid transformation (SVD method)
            R, t, _ = self._compute_rigid_transform(src_valid, tgt_valid)

            # Apply transformation
            source_pts = (R @ source_pts.T).T + t

            # Update cumulative transform
            delta_T = np.eye(4)
            delta_T[:3, :3] = R
            delta_T[:3, 3] = t
            transform = delta_T @ transform

            # Check convergence
            mean_error = np.mean(distances[valid_mask])
            if abs(prev_error - mean_error) < self.tolerance:
                break
            prev_error = mean_error

        fitness = np.sum(distances < self.max_correspondence_distance) / len(source_pts)

        return transform, fitness

    def _compute_rigid_transform(
        self,
        source_pts: np.ndarray,
        target_pts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute rigid transform using Procrustes/SVD"""
        # Centroids
        src_centroid = np.mean(source_pts, axis=0)
        tgt_centroid = np.mean(target_pts, axis=0)

        # Center
        src_centered = source_pts - src_centroid
        tgt_centered = target_pts - tgt_centroid

        # Covariance
        H = src_centered.T @ tgt_centered

        # SVD
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Handle reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = tgt_centroid - R @ src_centroid

        return R, t, 1.0

    def align_multiple(
        self,
        meshes: List[Union['trimesh.Trimesh', Tuple[np.ndarray, np.ndarray]]],
        reference_idx: int = 0
    ) -> List[np.ndarray]:
        """
        Align multiple meshes to a reference

        Args:
            meshes: List of meshes
            reference_idx: Index of reference mesh

        Returns:
            List of 4x4 transformation matrices
        """
        transforms = [np.eye(4) for _ in meshes]
        reference = meshes[reference_idx]

        for i, mesh in enumerate(meshes):
            if i == reference_idx:
                continue

            transform, fitness = self.align(mesh, reference)
            transforms[i] = transform
            logger.info(f"Aligned mesh {i} to reference: fitness={fitness:.4f}")

        return transforms


class VisualHullReconstructor:
    """
    Visual Hull reconstruction from multi-view silhouettes

    The visual hull is the intersection of all back-projected silhouette cones.
    It provides a conservative approximation of the true shape (always contains
    the true shape, but may include extra volume in concave regions).
    """

    def __init__(
        self,
        resolution: int = 64,          # Voxel grid resolution
        padding: float = 0.1,          # Padding around bounding box (fraction)
        erosion_iterations: int = 0,   # Optional erosion for noise removal
        marching_cubes_level: float = 0.5  # Isosurface level
    ):
        """
        Initialize Visual Hull reconstructor

        Args:
            resolution: Voxel grid resolution per dimension
            padding: Padding around estimated bounding box
            erosion_iterations: Binary erosion iterations for noise removal
            marching_cubes_level: Threshold for marching cubes
        """
        self.resolution = resolution
        self.padding = padding
        self.erosion_iterations = erosion_iterations
        self.marching_cubes_level = marching_cubes_level

    def reconstruct(
        self,
        silhouettes: List[np.ndarray],
        camera_params: List['CameraParams'],
        bounding_box: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> FusionResult:
        """
        Reconstruct visual hull from silhouettes

        Args:
            silhouettes: List of binary silhouette images (HxW, values 0 or 255)
            camera_params: List of camera parameters
            bounding_box: Optional (min_corner, max_corner) of 3D region

        Returns:
            FusionResult with reconstructed mesh
        """
        if len(silhouettes) != len(camera_params):
            raise ValueError("Number of silhouettes must match number of cameras")

        # Estimate bounding box if not provided
        if bounding_box is None:
            bounding_box = self._estimate_bounding_box(silhouettes, camera_params)

        min_corner, max_corner = bounding_box

        # Add padding
        extent = max_corner - min_corner
        min_corner = min_corner - self.padding * extent
        max_corner = max_corner + self.padding * extent

        # Create voxel grid
        logger.info(f"Creating {self.resolution}^3 voxel grid...")
        voxel_grid = self._create_voxel_grid(min_corner, max_corner)

        # Initialize occupancy (all occupied)
        occupancy = np.ones(self.resolution ** 3, dtype=np.float32)

        # Carve using each silhouette
        for i, (silhouette, cam) in enumerate(zip(silhouettes, camera_params)):
            logger.debug(f"Carving with silhouette {i+1}/{len(silhouettes)}")
            occupancy = self._carve_silhouette(
                occupancy, voxel_grid, silhouette, cam
            )

        # Reshape to 3D
        occupancy = occupancy.reshape(self.resolution, self.resolution, self.resolution)

        # Optional erosion
        if self.erosion_iterations > 0:
            occupancy = binary_erosion(
                occupancy > self.marching_cubes_level,
                iterations=self.erosion_iterations
            ).astype(np.float32)

        # Extract mesh using marching cubes
        vertices, faces = self._marching_cubes(
            occupancy, min_corner, max_corner
        )

        logger.info(f"Visual hull: {len(vertices)} vertices, {len(faces)} faces")

        return FusionResult(
            vertices=vertices,
            faces=faces,
            method='visual_hull',
            num_source_meshes=len(silhouettes),
            quality_score=np.sum(occupancy > self.marching_cubes_level) / occupancy.size
        )

    def _estimate_bounding_box(
        self,
        silhouettes: List[np.ndarray],
        camera_params: List['CameraParams']
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate 3D bounding box from silhouettes"""
        # Simple heuristic: use silhouette centroids and camera positions
        # For robust estimation, would need multiple-view triangulation

        all_points = []

        for silhouette, cam in zip(silhouettes, camera_params):
            # Find silhouette centroid
            coords = np.argwhere(silhouette > 128)
            if len(coords) == 0:
                continue

            centroid_2d = coords.mean(axis=0)[::-1]  # (x, y)

            # Backproject to ray
            # Simplified: assume depth range [0.1, 2.0] meters
            for depth in [0.3, 0.5, 1.0]:
                # Undistort and unproject
                K_inv = np.linalg.inv(cam.intrinsics.camera_matrix)
                point_2d_h = np.array([centroid_2d[0], centroid_2d[1], 1.0])
                ray_dir = K_inv @ point_2d_h
                ray_dir = ray_dir / np.linalg.norm(ray_dir)

                if cam.extrinsics is not None:
                    R = cam.extrinsics.rotation
                    t = cam.extrinsics.translation
                    ray_dir = R.T @ ray_dir
                    cam_pos = -R.T @ t
                else:
                    cam_pos = np.zeros(3)

                point_3d = cam_pos + depth * ray_dir
                all_points.append(point_3d)

        if len(all_points) == 0:
            # Default bounding box
            return np.array([-0.5, -0.5, -0.5]), np.array([0.5, 0.5, 0.5])

        all_points = np.array(all_points)
        min_corner = all_points.min(axis=0)
        max_corner = all_points.max(axis=0)

        return min_corner, max_corner

    def _create_voxel_grid(
        self,
        min_corner: np.ndarray,
        max_corner: np.ndarray
    ) -> np.ndarray:
        """Create 3D voxel center coordinates"""
        x = np.linspace(min_corner[0], max_corner[0], self.resolution)
        y = np.linspace(min_corner[1], max_corner[1], self.resolution)
        z = np.linspace(min_corner[2], max_corner[2], self.resolution)

        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        voxel_centers = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)

        return voxel_centers

    def _carve_silhouette(
        self,
        occupancy: np.ndarray,
        voxel_centers: np.ndarray,
        silhouette: np.ndarray,
        camera_params: 'CameraParams'
    ) -> np.ndarray:
        """Carve voxels outside silhouette cone"""
        # Project voxel centers to image
        projected = camera_params.project(voxel_centers)

        h, w = silhouette.shape[:2]

        # Check if projections are inside image and inside silhouette
        x = projected[:, 0].astype(np.int32)
        y = projected[:, 1].astype(np.int32)

        # Bounds check
        valid_mask = (x >= 0) & (x < w) & (y >= 0) & (y < h)

        # Silhouette check
        inside_silhouette = np.zeros(len(occupancy), dtype=bool)
        inside_silhouette[valid_mask] = silhouette[y[valid_mask], x[valid_mask]] > 128

        # Carve: keep only voxels inside silhouette
        occupancy = occupancy * inside_silhouette.astype(np.float32)

        return occupancy

    def _marching_cubes(
        self,
        occupancy: np.ndarray,
        min_corner: np.ndarray,
        max_corner: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract mesh from voxel grid using marching cubes"""
        try:
            from skimage import measure
        except ImportError:
            raise ImportError("scikit-image required for marching cubes")

        # Pad for proper edge handling
        padded = np.pad(occupancy, 1, mode='constant', constant_values=0)

        # Run marching cubes
        verts, faces, normals, values = measure.marching_cubes(
            padded, level=self.marching_cubes_level
        )

        # Remove padding offset
        verts = verts - 1

        # Scale to world coordinates
        scale = (max_corner - min_corner) / (self.resolution - 1)
        verts = verts * scale + min_corner

        return verts, faces


class MultiViewFusion:
    """
    Multi-view mesh fusion combining SAM-3D outputs

    Workflow:
    1. Generate mesh from each view using SAM-3D
    2. Align meshes to common reference using ICP
    3. Fuse aligned meshes via:
       - Vertex averaging
       - Poisson reconstruction
       - Screened Poisson reconstruction
    """

    def __init__(
        self,
        aligner: Optional[ICPMeshAligner] = None,
        fusion_method: str = 'poisson',  # 'average', 'poisson', 'visual_hull_refine'
        poisson_depth: int = 8
    ):
        """
        Initialize multi-view fusion

        Args:
            aligner: ICP aligner (created if None)
            fusion_method: 'average', 'poisson', or 'visual_hull_refine'
            poisson_depth: Depth parameter for Poisson reconstruction
        """
        self.aligner = aligner or ICPMeshAligner()
        self.fusion_method = fusion_method
        self.poisson_depth = poisson_depth

    def fuse(
        self,
        meshes: List[Union['trimesh.Trimesh', Tuple[np.ndarray, np.ndarray]]],
        camera_params: Optional[List['CameraParams']] = None,
        silhouettes: Optional[List[np.ndarray]] = None,
        reference_idx: int = 0
    ) -> FusionResult:
        """
        Fuse multiple meshes from different views

        Args:
            meshes: List of meshes from each view
            camera_params: Optional camera parameters (for visual hull refinement)
            silhouettes: Optional silhouettes (for visual hull refinement)
            reference_idx: Index of reference mesh

        Returns:
            FusionResult with fused mesh
        """
        if len(meshes) < 2:
            # Single mesh: just return it
            if HAS_TRIMESH and isinstance(meshes[0], trimesh.Trimesh):
                return FusionResult(
                    vertices=meshes[0].vertices,
                    faces=meshes[0].faces,
                    method='single',
                    num_source_meshes=1
                )
            else:
                return FusionResult(
                    vertices=meshes[0][0],
                    faces=meshes[0][1],
                    method='single',
                    num_source_meshes=1
                )

        # Step 1: Align all meshes to reference
        logger.info("Step 1: Aligning meshes...")
        transforms = self.aligner.align_multiple(meshes, reference_idx)

        # Apply transformations
        aligned_meshes = []
        for mesh, transform in zip(meshes, transforms):
            if HAS_TRIMESH and isinstance(mesh, trimesh.Trimesh):
                vertices = mesh.vertices.copy()
                faces = mesh.faces.copy()
            else:
                vertices = mesh[0].copy()
                faces = mesh[1].copy()

            # Apply transform
            vertices_h = np.hstack([vertices, np.ones((len(vertices), 1))])
            vertices_transformed = (transform @ vertices_h.T).T[:, :3]
            aligned_meshes.append((vertices_transformed, faces))

        # Step 2: Fuse
        logger.info(f"Step 2: Fusing with method '{self.fusion_method}'...")

        if self.fusion_method == 'average':
            result = self._fuse_average(aligned_meshes)
        elif self.fusion_method == 'poisson':
            result = self._fuse_poisson(aligned_meshes)
        elif self.fusion_method == 'visual_hull_refine':
            if silhouettes is None or camera_params is None:
                logger.warning("Visual hull refinement requires silhouettes and camera params. "
                             "Falling back to Poisson.")
                result = self._fuse_poisson(aligned_meshes)
            else:
                result = self._fuse_visual_hull_refine(
                    aligned_meshes, silhouettes, camera_params
                )
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        result.transformations = transforms
        result.num_source_meshes = len(meshes)

        return result

    def _fuse_average(
        self,
        meshes: List[Tuple[np.ndarray, np.ndarray]]
    ) -> FusionResult:
        """
        Simple fusion by combining point clouds and remeshing

        Note: This is a simplified approach. For production, use Poisson.
        """
        # Combine all vertices
        all_vertices = np.vstack([m[0] for m in meshes])

        # Simple approach: use first mesh's faces as template
        # This only works well for very similar meshes
        reference_vertices, reference_faces = meshes[0]

        # Find corresponding vertices for each point in reference
        if HAS_SCIPY:
            from scipy.spatial import cKDTree

            all_tree = cKDTree(all_vertices)

            averaged_vertices = np.zeros_like(reference_vertices)
            for i, v in enumerate(reference_vertices):
                # Find nearby points
                distances, indices = all_tree.query(v, k=len(meshes))
                # Average
                averaged_vertices[i] = all_vertices[indices].mean(axis=0)
        else:
            averaged_vertices = reference_vertices

        return FusionResult(
            vertices=averaged_vertices,
            faces=reference_faces,
            method='average'
        )

    def _fuse_poisson(
        self,
        meshes: List[Tuple[np.ndarray, np.ndarray]]
    ) -> FusionResult:
        """
        Poisson surface reconstruction from combined point cloud
        """
        if not HAS_OPEN3D:
            logger.warning("Open3D not available. Falling back to average fusion.")
            return self._fuse_average(meshes)

        # Combine all meshes into point cloud with normals
        all_vertices = []
        all_normals = []

        for vertices, faces in meshes:
            # Compute normals
            if HAS_TRIMESH:
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                normals = mesh.vertex_normals
            else:
                # Estimate normals using Open3D
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(vertices)
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
                )
                normals = np.asarray(pcd.normals)

            all_vertices.append(vertices)
            all_normals.append(normals)

        combined_vertices = np.vstack(all_vertices)
        combined_normals = np.vstack(all_normals)

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined_vertices)
        pcd.normals = o3d.utility.Vector3dVector(combined_normals)

        # Orient normals consistently
        pcd.orient_normals_consistent_tangent_plane(k=10)

        # Poisson reconstruction
        logger.info(f"Running Poisson reconstruction (depth={self.poisson_depth})...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=self.poisson_depth
        )

        # Remove low-density vertices
        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, 0.1)
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)

        # Clean up
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)

        return FusionResult(
            vertices=vertices,
            faces=faces,
            method='poisson'
        )

    def _fuse_visual_hull_refine(
        self,
        meshes: List[Tuple[np.ndarray, np.ndarray]],
        silhouettes: List[np.ndarray],
        camera_params: List['CameraParams']
    ) -> FusionResult:
        """
        Fuse meshes with visual hull refinement

        1. Compute visual hull from silhouettes
        2. Intersect with Poisson fusion result
        """
        # First, get Poisson fusion
        poisson_result = self._fuse_poisson(meshes)

        # Compute visual hull
        vh_reconstructor = VisualHullReconstructor(resolution=64)

        # Estimate bounding box from meshes
        all_vertices = np.vstack([m[0] for m in meshes])
        min_corner = all_vertices.min(axis=0)
        max_corner = all_vertices.max(axis=0)

        vh_result = vh_reconstructor.reconstruct(
            silhouettes, camera_params,
            bounding_box=(min_corner, max_corner)
        )

        # For now, just return Poisson result
        # TODO: Implement actual intersection/refinement
        logger.warning("Visual hull refinement not fully implemented. Returning Poisson result.")

        return FusionResult(
            vertices=poisson_result.vertices,
            faces=poisson_result.faces,
            method='visual_hull_refine'
        )


# Convenience functions

def fuse_sam3d_meshes(
    meshes: List[Union['trimesh.Trimesh', Tuple[np.ndarray, np.ndarray]]],
    method: str = 'poisson'
) -> FusionResult:
    """
    Convenience function to fuse multiple SAM-3D generated meshes

    Args:
        meshes: List of meshes from SAM-3D
        method: Fusion method ('average', 'poisson')

    Returns:
        FusionResult with fused mesh
    """
    fusion = MultiViewFusion(fusion_method=method)
    return fusion.fuse(meshes)


def reconstruct_visual_hull(
    silhouettes: List[np.ndarray],
    camera_params: List['CameraParams'],
    resolution: int = 64
) -> FusionResult:
    """
    Convenience function for visual hull reconstruction

    Args:
        silhouettes: Binary silhouette masks
        camera_params: Camera parameters
        resolution: Voxel grid resolution

    Returns:
        FusionResult with visual hull mesh
    """
    reconstructor = VisualHullReconstructor(resolution=resolution)
    return reconstructor.reconstruct(silhouettes, camera_params)
