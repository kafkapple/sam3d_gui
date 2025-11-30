"""
Template Mesh Deformation Module

Provides:
- Non-rigid ICP for mesh-to-mesh/mesh-to-pointcloud registration
- Silhouette-based deformation for multi-view fitting
- Template mesh deformer combining multiple constraints

Key References:
- Amberg et al. "Optimal Step Nonrigid ICP" (2007)
- Zuffi et al. "3D Menagerie: SMAL" (CVPR 2017)
- Statistical Non-Rigid ICP (2016)

Theory:
1. Non-rigid ICP:
   - Find correspondences between template and target
   - Compute deformation minimizing distance while preserving local rigidity
   - Iteratively reduce stiffness (global → local deformation)

2. Silhouette-based fitting:
   - Render template mesh in camera view
   - Compare silhouette with target mask (IoU loss)
   - Optimize vertex positions to match silhouette

3. Combined approach:
   - SMAL-like parametric model OR free-form deformation
   - Multi-view silhouette constraints
   - Optional keypoint constraints
   - Regularization (Laplacian, ARAP)
"""

import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Union, Callable
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
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import spsolve
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("scipy not installed. Some features will be unavailable.")


@dataclass
class DeformationResult:
    """Result of mesh deformation"""
    vertices: np.ndarray          # Deformed vertices (Nx3)
    faces: np.ndarray             # Faces (Mx3), unchanged
    convergence_history: List[float] = field(default_factory=list)
    iterations: int = 0
    final_error: float = 0.0

    # Optional displacement field
    displacements: Optional[np.ndarray] = None

    def to_mesh(self) -> 'trimesh.Trimesh':
        """Convert to trimesh object"""
        if not HAS_TRIMESH:
            raise ImportError("trimesh required")
        return trimesh.Trimesh(vertices=self.vertices, faces=self.faces)


class NonRigidICP:
    """
    Non-rigid Iterative Closest Point algorithm

    Based on Amberg et al. "Optimal Step Nonrigid ICP" (2007)

    The algorithm iteratively:
    1. Find correspondences (nearest neighbors)
    2. Solve for affine transformations per vertex
    3. Apply stiffness-weighted regularization
    4. Decrease stiffness for finer deformation

    Energy function:
    E = E_data + alpha * E_stiffness + beta * E_landmark

    where:
    - E_data: sum of squared distances to correspondences
    - E_stiffness: deviation from local rigidity
    - E_landmark: optional keypoint constraints
    """

    def __init__(
        self,
        stiffness_weights: List[float] = None,
        max_iterations: int = 50,
        tolerance: float = 1e-5,
        correspondence_threshold: float = 0.1,  # meters
        use_landmarks: bool = False
    ):
        """
        Initialize Non-rigid ICP

        Args:
            stiffness_weights: List of stiffness values (high→low)
                Default: [50, 20, 5, 2, 0.8, 0.5, 0.35, 0.2]
            max_iterations: Max iterations per stiffness level
            tolerance: Convergence tolerance
            correspondence_threshold: Max distance for valid correspondence
            use_landmarks: Whether to use landmark constraints
        """
        if stiffness_weights is None:
            # Default: coarse to fine
            self.stiffness_weights = [50, 20, 5, 2, 0.8, 0.5, 0.35, 0.2]
        else:
            self.stiffness_weights = stiffness_weights

        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.correspondence_threshold = correspondence_threshold
        self.use_landmarks = use_landmarks

    def deform(
        self,
        source_vertices: np.ndarray,
        source_faces: np.ndarray,
        target_points: np.ndarray,
        source_landmarks: Optional[np.ndarray] = None,
        target_landmarks: Optional[np.ndarray] = None,
        target_normals: Optional[np.ndarray] = None
    ) -> DeformationResult:
        """
        Deform source mesh to match target point cloud

        Args:
            source_vertices: Nx3 source mesh vertices
            source_faces: Mx3 source mesh faces
            target_points: Kx3 target point cloud
            source_landmarks: Lx3 landmark positions on source (vertex indices)
            target_landmarks: Lx3 target landmark positions
            target_normals: Kx3 target normals (optional, for better correspondence)

        Returns:
            DeformationResult with deformed mesh
        """
        if not HAS_SCIPY:
            raise ImportError("scipy required for Non-rigid ICP")

        n_vertices = len(source_vertices)
        vertices = source_vertices.copy()

        # Build adjacency for stiffness regularization
        adjacency = self._build_adjacency(source_faces, n_vertices)

        # Build KD-tree for target
        target_tree = cKDTree(target_points)

        convergence_history = []
        total_iterations = 0

        # Iterate through stiffness levels
        for stiffness in self.stiffness_weights:
            logger.debug(f"Stiffness level: {stiffness}")

            for iteration in range(self.max_iterations):
                # 1. Find correspondences
                distances, indices = target_tree.query(vertices, k=1)

                # Filter by threshold
                valid_mask = distances < self.correspondence_threshold

                if np.sum(valid_mask) < 10:
                    logger.warning("Too few valid correspondences")
                    break

                # 2. Solve for deformation
                target_corr = target_points[indices]

                # Build system: (D + alpha * M) * X = D * target + alpha * M * current
                new_vertices = self._solve_step(
                    vertices, target_corr, valid_mask,
                    adjacency, stiffness,
                    source_landmarks, target_landmarks
                )

                # 3. Check convergence
                displacement = np.linalg.norm(new_vertices - vertices, axis=1)
                max_displacement = np.max(displacement)
                mean_error = np.mean(distances[valid_mask])

                vertices = new_vertices
                convergence_history.append(mean_error)
                total_iterations += 1

                if max_displacement < self.tolerance:
                    logger.debug(f"Converged at iteration {iteration}")
                    break

        # Final error
        final_distances, _ = target_tree.query(vertices, k=1)
        final_error = np.mean(final_distances)

        logger.info(f"Non-rigid ICP completed: {total_iterations} iterations, error={final_error:.4f}")

        return DeformationResult(
            vertices=vertices,
            faces=source_faces,
            convergence_history=convergence_history,
            iterations=total_iterations,
            final_error=final_error,
            displacements=vertices - source_vertices
        )

    def _build_adjacency(self, faces: np.ndarray, n_vertices: int) -> csr_matrix:
        """Build vertex adjacency matrix from faces"""
        rows = []
        cols = []

        for face in faces:
            for i in range(3):
                for j in range(3):
                    if i != j:
                        rows.append(face[i])
                        cols.append(face[j])

        data = np.ones(len(rows))
        adj = csr_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices))

        # Normalize: Laplacian-like
        degrees = np.array(adj.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1  # Avoid division by zero

        return adj

    def _solve_step(
        self,
        vertices: np.ndarray,
        target_corr: np.ndarray,
        valid_mask: np.ndarray,
        adjacency: csr_matrix,
        stiffness: float,
        source_landmarks: Optional[np.ndarray],
        target_landmarks: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Solve one step of non-rigid ICP

        Minimizes: ||V - T||^2 + stiffness * ||L * V||^2
        where L is Laplacian regularization
        """
        n_vertices = len(vertices)

        # Data term weight matrix (diagonal)
        weights = np.zeros(n_vertices)
        weights[valid_mask] = 1.0
        W = diags(weights)

        # Laplacian regularization
        degrees = np.array(adjacency.sum(axis=1)).flatten()
        D = diags(degrees)
        L = D - adjacency  # Laplacian

        # System: (W + stiffness * L^T L) * V = W * target + stiffness * L^T L * current
        # Simplified: solve for each coordinate independently

        new_vertices = np.zeros_like(vertices)

        A = W + stiffness * L.T @ L

        for dim in range(3):
            b = W @ target_corr[:, dim] + stiffness * (L.T @ L @ vertices[:, dim])

            # Add landmark constraints if available
            if self.use_landmarks and source_landmarks is not None and target_landmarks is not None:
                landmark_weight = 100.0  # Strong constraint
                for src_idx, tgt_pos in zip(source_landmarks, target_landmarks):
                    if isinstance(src_idx, (int, np.integer)):
                        A[src_idx, src_idx] += landmark_weight
                        b[src_idx] += landmark_weight * tgt_pos[dim]

            new_vertices[:, dim] = spsolve(A.tocsr(), b)

        return new_vertices


class SilhouetteDeformer:
    """
    Silhouette-based mesh deformation for multi-view fitting

    Optimizes mesh vertices to match silhouettes from multiple camera views.

    Energy function:
    E = sum_v IoU_loss(render(mesh, cam_v), mask_v) + lambda * E_reg

    where:
    - IoU_loss: 1 - IoU(predicted_silhouette, target_silhouette)
    - E_reg: Laplacian regularization to preserve local shape
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iterations: int = 100,
        lambda_laplacian: float = 0.1,
        lambda_edge: float = 0.01,
        convergence_threshold: float = 1e-4
    ):
        """
        Initialize silhouette deformer

        Args:
            learning_rate: Optimization step size
            max_iterations: Maximum optimization iterations
            lambda_laplacian: Laplacian smoothness weight
            lambda_edge: Edge length preservation weight
            convergence_threshold: Convergence criterion
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.lambda_laplacian = lambda_laplacian
        self.lambda_edge = lambda_edge
        self.convergence_threshold = convergence_threshold

    def deform(
        self,
        mesh_vertices: np.ndarray,
        mesh_faces: np.ndarray,
        target_masks: List[np.ndarray],
        camera_params: List['CameraParams'],
        initial_vertices: Optional[np.ndarray] = None
    ) -> DeformationResult:
        """
        Deform mesh to match multi-view silhouettes

        Args:
            mesh_vertices: Nx3 mesh vertices
            mesh_faces: Mx3 mesh faces
            target_masks: List of binary masks (one per camera)
            camera_params: List of camera parameters
            initial_vertices: Optional initial vertex positions

        Returns:
            DeformationResult with optimized mesh
        """
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            raise ImportError("PyTorch required for silhouette-based deformation")

        # This is a simplified implementation
        # Full implementation would use differentiable rendering (e.g., PyTorch3D)

        logger.warning("SilhouetteDeformer requires PyTorch3D for differentiable rendering. "
                      "Using simplified gradient-free optimization.")

        vertices = mesh_vertices.copy() if initial_vertices is None else initial_vertices.copy()

        # Compute initial edge lengths for regularization
        edges = self._get_edges(mesh_faces)
        initial_edge_lengths = self._compute_edge_lengths(vertices, edges)

        # Laplacian matrix for smoothness
        laplacian = self._compute_laplacian(mesh_faces, len(vertices))

        convergence_history = []

        for iteration in range(self.max_iterations):
            # Compute silhouette loss gradient (numerical)
            grad = self._compute_silhouette_gradient(
                vertices, mesh_faces, target_masks, camera_params
            )

            # Add regularization gradients
            laplacian_grad = self.lambda_laplacian * (laplacian @ vertices)
            edge_grad = self.lambda_edge * self._compute_edge_gradient(
                vertices, edges, initial_edge_lengths
            )

            total_grad = grad + laplacian_grad + edge_grad

            # Update vertices
            vertices = vertices - self.learning_rate * total_grad

            # Compute loss for convergence check
            loss = self._compute_total_loss(
                vertices, mesh_faces, target_masks, camera_params,
                laplacian, edges, initial_edge_lengths
            )
            convergence_history.append(loss)

            if iteration > 0 and abs(convergence_history[-1] - convergence_history[-2]) < self.convergence_threshold:
                logger.info(f"Converged at iteration {iteration}")
                break

        return DeformationResult(
            vertices=vertices,
            faces=mesh_faces,
            convergence_history=convergence_history,
            iterations=iteration + 1,
            final_error=convergence_history[-1] if convergence_history else 0.0,
            displacements=vertices - mesh_vertices
        )

    def _get_edges(self, faces: np.ndarray) -> np.ndarray:
        """Extract unique edges from faces"""
        edges = set()
        for face in faces:
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i+1)%3]]))
                edges.add(edge)
        return np.array(list(edges))

    def _compute_edge_lengths(self, vertices: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Compute edge lengths"""
        v0 = vertices[edges[:, 0]]
        v1 = vertices[edges[:, 1]]
        return np.linalg.norm(v1 - v0, axis=1)

    def _compute_laplacian(self, faces: np.ndarray, n_vertices: int) -> np.ndarray:
        """Compute cotangent Laplacian matrix"""
        # Simplified uniform Laplacian
        L = np.zeros((n_vertices, n_vertices))

        for face in faces:
            for i in range(3):
                v0, v1 = face[i], face[(i+1)%3]
                L[v0, v1] -= 1
                L[v1, v0] -= 1
                L[v0, v0] += 1
                L[v1, v1] += 1

        # Normalize rows
        row_sums = np.abs(L).sum(axis=1)
        row_sums[row_sums == 0] = 1
        L = L / row_sums[:, np.newaxis]

        return L

    def _compute_silhouette_gradient(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        target_masks: List[np.ndarray],
        camera_params: List['CameraParams']
    ) -> np.ndarray:
        """
        Compute gradient of silhouette loss w.r.t. vertices

        Note: This is a simplified numerical gradient.
        For production, use differentiable rendering (PyTorch3D).
        """
        eps = 1e-4
        grad = np.zeros_like(vertices)

        # Only compute for a subset of vertices for efficiency
        n_sample = min(100, len(vertices))
        sample_indices = np.random.choice(len(vertices), n_sample, replace=False)

        base_loss = self._compute_silhouette_loss(vertices, faces, target_masks, camera_params)

        for idx in sample_indices:
            for dim in range(3):
                vertices_plus = vertices.copy()
                vertices_plus[idx, dim] += eps

                loss_plus = self._compute_silhouette_loss(
                    vertices_plus, faces, target_masks, camera_params
                )

                grad[idx, dim] = (loss_plus - base_loss) / eps

        return grad

    def _compute_silhouette_loss(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        target_masks: List[np.ndarray],
        camera_params: List['CameraParams']
    ) -> float:
        """Compute IoU-based silhouette loss"""
        total_loss = 0.0

        for mask, cam in zip(target_masks, camera_params):
            # Simplified: project vertices and compute convex hull
            projected = cam.project(vertices)

            # Create simple silhouette from projected points
            h, w = mask.shape[:2]
            rendered = np.zeros((h, w), dtype=np.uint8)

            # Draw filled polygon for each face
            for face in faces:
                pts = projected[face].astype(np.int32)
                if np.all(pts >= 0) and np.all(pts[:, 0] < w) and np.all(pts[:, 1] < h):
                    try:
                        import cv2
                        cv2.fillPoly(rendered, [pts], 255)
                    except:
                        pass

            # Compute IoU
            intersection = np.sum((rendered > 0) & (mask > 0))
            union = np.sum((rendered > 0) | (mask > 0))

            if union > 0:
                iou = intersection / union
                total_loss += 1 - iou
            else:
                total_loss += 1

        return total_loss / len(target_masks)

    def _compute_edge_gradient(
        self,
        vertices: np.ndarray,
        edges: np.ndarray,
        target_lengths: np.ndarray
    ) -> np.ndarray:
        """Compute gradient of edge length preservation loss"""
        grad = np.zeros_like(vertices)

        current_lengths = self._compute_edge_lengths(vertices, edges)
        length_diff = current_lengths - target_lengths

        for i, (e0, e1) in enumerate(edges):
            direction = vertices[e1] - vertices[e0]
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 1e-8:
                direction = direction / direction_norm
                grad[e0] -= length_diff[i] * direction
                grad[e1] += length_diff[i] * direction

        return grad

    def _compute_total_loss(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        target_masks: List[np.ndarray],
        camera_params: List['CameraParams'],
        laplacian: np.ndarray,
        edges: np.ndarray,
        initial_edge_lengths: np.ndarray
    ) -> float:
        """Compute total loss including regularization"""
        silhouette_loss = self._compute_silhouette_loss(
            vertices, faces, target_masks, camera_params
        )

        laplacian_loss = np.mean(np.linalg.norm(laplacian @ vertices, axis=1))

        current_lengths = self._compute_edge_lengths(vertices, edges)
        edge_loss = np.mean((current_lengths - initial_edge_lengths) ** 2)

        return silhouette_loss + self.lambda_laplacian * laplacian_loss + self.lambda_edge * edge_loss


class TemplateMeshDeformer:
    """
    High-level template mesh deformation combining multiple methods

    Workflow:
    1. Initial alignment (rigid ICP or landmark-based)
    2. Non-rigid ICP for global shape matching
    3. Silhouette refinement for multi-view consistency
    4. Optional keypoint constraints
    """

    def __init__(
        self,
        use_nonrigid_icp: bool = True,
        use_silhouette: bool = True,
        nonrigid_icp_params: Optional[dict] = None,
        silhouette_params: Optional[dict] = None
    ):
        """
        Initialize template deformer

        Args:
            use_nonrigid_icp: Whether to use non-rigid ICP
            use_silhouette: Whether to use silhouette-based refinement
            nonrigid_icp_params: Parameters for NonRigidICP
            silhouette_params: Parameters for SilhouetteDeformer
        """
        self.use_nonrigid_icp = use_nonrigid_icp
        self.use_silhouette = use_silhouette

        if use_nonrigid_icp:
            self.nricp = NonRigidICP(**(nonrigid_icp_params or {}))
        if use_silhouette:
            self.silhouette_deformer = SilhouetteDeformer(**(silhouette_params or {}))

    def fit_to_frame(
        self,
        template_mesh: Union['trimesh.Trimesh', Tuple[np.ndarray, np.ndarray]],
        target_pointcloud: Optional[np.ndarray] = None,
        target_masks: Optional[List[np.ndarray]] = None,
        camera_params: Optional[List['CameraParams']] = None,
        keypoints_2d: Optional[Dict[str, np.ndarray]] = None,
        keypoint_vertex_mapping: Optional[Dict[str, int]] = None
    ) -> DeformationResult:
        """
        Fit template mesh to a single frame

        Args:
            template_mesh: Template mesh (trimesh or (vertices, faces))
            target_pointcloud: Target point cloud from depth/stereo
            target_masks: Target silhouette masks per camera
            camera_params: Camera parameters per view
            keypoints_2d: 2D keypoints per camera (optional)
            keypoint_vertex_mapping: Mapping from keypoint name to vertex index

        Returns:
            DeformationResult with fitted mesh
        """
        # Extract vertices and faces
        if HAS_TRIMESH and isinstance(template_mesh, trimesh.Trimesh):
            vertices = template_mesh.vertices.copy()
            faces = template_mesh.faces.copy()
        else:
            vertices, faces = template_mesh
            vertices = vertices.copy()

        result = DeformationResult(vertices=vertices, faces=faces)

        # Stage 1: Non-rigid ICP (if point cloud available)
        if self.use_nonrigid_icp and target_pointcloud is not None:
            logger.info("Stage 1: Non-rigid ICP")

            # Convert keypoints to landmarks if available
            source_landmarks = None
            target_landmarks = None

            if keypoints_2d is not None and keypoint_vertex_mapping is not None:
                # Triangulate keypoints to 3D
                # This requires camera_params
                pass  # TODO: Implement keypoint triangulation

            result = self.nricp.deform(
                result.vertices, faces, target_pointcloud,
                source_landmarks, target_landmarks
            )

        # Stage 2: Silhouette refinement (if masks available)
        if self.use_silhouette and target_masks is not None and camera_params is not None:
            logger.info("Stage 2: Silhouette refinement")
            result = self.silhouette_deformer.deform(
                result.vertices, faces, target_masks, camera_params,
                initial_vertices=result.vertices
            )

        return result

    def fit_sequence(
        self,
        template_mesh: Union['trimesh.Trimesh', Tuple[np.ndarray, np.ndarray]],
        sequence_data: List[dict],
        use_temporal_smoothing: bool = True,
        temporal_weight: float = 0.5
    ) -> List[DeformationResult]:
        """
        Fit template mesh to a sequence of frames

        Args:
            template_mesh: Initial template mesh
            sequence_data: List of dicts with keys:
                - 'target_pointcloud': Optional[np.ndarray]
                - 'target_masks': Optional[List[np.ndarray]]
                - 'camera_params': Optional[List[CameraParams]]
            use_temporal_smoothing: Whether to smooth across frames
            temporal_weight: Weight for previous frame initialization

        Returns:
            List of DeformationResult for each frame
        """
        results = []

        # Extract initial vertices
        if HAS_TRIMESH and isinstance(template_mesh, trimesh.Trimesh):
            current_vertices = template_mesh.vertices.copy()
            faces = template_mesh.faces.copy()
        else:
            current_vertices, faces = template_mesh
            current_vertices = current_vertices.copy()

        for i, frame_data in enumerate(sequence_data):
            logger.info(f"Processing frame {i+1}/{len(sequence_data)}")

            # Use previous result as initialization (temporal consistency)
            if use_temporal_smoothing and i > 0:
                init_mesh = (
                    temporal_weight * results[-1].vertices +
                    (1 - temporal_weight) * current_vertices,
                    faces
                )
            else:
                init_mesh = (current_vertices, faces)

            result = self.fit_to_frame(
                init_mesh,
                target_pointcloud=frame_data.get('target_pointcloud'),
                target_masks=frame_data.get('target_masks'),
                camera_params=frame_data.get('camera_params'),
                keypoints_2d=frame_data.get('keypoints_2d'),
                keypoint_vertex_mapping=frame_data.get('keypoint_vertex_mapping')
            )

            results.append(result)
            current_vertices = result.vertices

        return results


# Utility functions

def compute_mesh_iou(
    mesh1_vertices: np.ndarray,
    mesh1_faces: np.ndarray,
    mesh2_vertices: np.ndarray,
    mesh2_faces: np.ndarray,
    resolution: int = 64
) -> float:
    """
    Compute volumetric IoU between two meshes

    Uses voxelization for approximate IoU.
    """
    if not HAS_TRIMESH:
        raise ImportError("trimesh required for mesh IoU")

    mesh1 = trimesh.Trimesh(vertices=mesh1_vertices, faces=mesh1_faces)
    mesh2 = trimesh.Trimesh(vertices=mesh2_vertices, faces=mesh2_faces)

    # Get combined bounding box
    bounds = np.vstack([mesh1.bounds, mesh2.bounds])
    min_bound = bounds.min(axis=0)
    max_bound = bounds.max(axis=0)

    # Create voxel grid
    pitch = (max_bound - min_bound).max() / resolution

    voxels1 = mesh1.voxelized(pitch).fill()
    voxels2 = mesh2.voxelized(pitch).fill()

    # Compute IoU
    intersection = np.sum(voxels1.matrix & voxels2.matrix)
    union = np.sum(voxels1.matrix | voxels2.matrix)

    if union == 0:
        return 0.0

    return intersection / union


def rigid_align(
    source_points: np.ndarray,
    target_points: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute rigid transformation (rotation + translation) using Procrustes analysis

    Args:
        source_points: Nx3 source points
        target_points: Nx3 target points
        weights: Optional per-point weights

    Returns:
        (rotation_matrix, translation, scale)
    """
    if weights is None:
        weights = np.ones(len(source_points))

    weights = weights / weights.sum()

    # Compute centroids
    src_centroid = np.average(source_points, axis=0, weights=weights)
    tgt_centroid = np.average(target_points, axis=0, weights=weights)

    # Center points
    src_centered = source_points - src_centroid
    tgt_centered = target_points - tgt_centroid

    # Weighted covariance
    W = np.diag(weights)
    H = src_centered.T @ W @ tgt_centered

    # SVD
    U, _, Vt = np.linalg.svd(H)

    # Rotation
    R = Vt.T @ U.T

    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Translation
    t = tgt_centroid - R @ src_centroid

    # Scale (optional)
    scale = 1.0

    return R, t, scale
