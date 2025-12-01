"""
Mesh Sequence Generator
Multi-frame 3D mesh 생성, 렌더링, Blender 호환 포맷 내보내기
"""
import os
import json
import struct
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import cv2

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

try:
    import pyrender
    import pyglet
    # Headless rendering setup
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    HAS_PYRENDER = True
except ImportError:
    HAS_PYRENDER = False


@dataclass
class MeshFrame:
    """Single frame mesh data"""
    frame_idx: int
    mesh_path: str
    vertices: Optional[np.ndarray] = None
    faces: Optional[np.ndarray] = None
    vertex_colors: Optional[np.ndarray] = None
    timestamp: float = 0.0  # Time in seconds


@dataclass
class MeshSequence:
    """Sequence of meshes over time"""
    name: str
    fps: float
    frames: List[MeshFrame] = field(default_factory=list)
    output_dir: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Total duration in seconds"""
        if not self.frames:
            return 0.0
        return len(self.frames) / self.fps

    @property
    def frame_count(self) -> int:
        return len(self.frames)


def resolve_mesh_path(ply_path: str) -> Optional[str]:
    """
    Resolve actual mesh file path, handling _mesh.ply suffix variations.

    Args:
        ply_path: Original path (may not exist)

    Returns:
        Actual existing path or None
    """
    path = Path(ply_path)

    # Try original path
    if path.exists():
        return str(path)

    # Try variations
    parent = path.parent
    stem = path.stem.replace('_mesh', '')  # Remove _mesh if present

    candidates = [
        parent / f"{stem}_mesh.ply",
        parent / f"{stem}.ply",
        parent / f"{stem}_mesh.obj",
        parent / f"{stem}.obj",
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return None


def load_ply_mesh(ply_path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load PLY mesh file and return vertices, faces, and colors.

    Returns:
        (vertices, faces, vertex_colors) - colors may be None
    """
    # Resolve actual path
    actual_path = resolve_mesh_path(ply_path)
    if actual_path is None:
        raise FileNotFoundError(f"Mesh file not found: {ply_path}")

    if HAS_TRIMESH:
        mesh = trimesh.load(actual_path)
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces) if hasattr(mesh, 'faces') else np.array([])

        # Get vertex colors if available
        vertex_colors = None
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
            vertex_colors = np.array(mesh.visual.vertex_colors)[:, :3]  # RGB only

        return vertices, faces, vertex_colors
    else:
        # Basic PLY parser fallback
        return _parse_ply_basic(actual_path)


def _parse_ply_basic(ply_path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Basic PLY parser without trimesh"""
    vertices = []
    faces = []
    colors = []

    with open(ply_path, 'rb') as f:
        # Read header
        header_end = False
        vertex_count = 0
        face_count = 0
        has_color = False
        is_binary = False

        while not header_end:
            line = f.readline().decode('ascii').strip()
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line.startswith('element face'):
                face_count = int(line.split()[-1])
            elif 'red' in line or 'green' in line or 'blue' in line:
                has_color = True
            elif line.startswith('format binary'):
                is_binary = True
            elif line == 'end_header':
                header_end = True

        # Read data (ASCII only for basic parser)
        if not is_binary:
            for _ in range(vertex_count):
                parts = f.readline().decode('ascii').strip().split()
                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                if has_color and len(parts) >= 6:
                    colors.append([int(parts[3]), int(parts[4]), int(parts[5])])

            for _ in range(face_count):
                parts = f.readline().decode('ascii').strip().split()
                n = int(parts[0])
                if n >= 3:
                    faces.append([int(parts[1]), int(parts[2]), int(parts[3])])

    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32) if faces else np.array([], dtype=np.int32)
    vertex_colors = np.array(colors, dtype=np.uint8) if colors else None

    return vertices, faces, vertex_colors


def render_mesh_view(
    mesh_path: str,
    output_path: str,
    camera_pose: Optional[np.ndarray] = None,
    resolution: Tuple[int, int] = (512, 512),
    bg_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
) -> bool:
    """
    Render mesh from a specific viewpoint.

    Args:
        mesh_path: Path to PLY file
        output_path: Output image path
        camera_pose: 4x4 camera transform matrix (None = auto)
        resolution: (width, height)
        bg_color: Background color (RGBA)

    Returns:
        True if successful
    """
    if not HAS_PYRENDER or not HAS_TRIMESH:
        # Fallback: save a placeholder or use OpenCV-based simple render
        return _render_mesh_simple(mesh_path, output_path, resolution)

    try:
        # Load mesh
        mesh = trimesh.load(mesh_path)

        # Create pyrender mesh
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
            pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        else:
            pr_mesh = pyrender.Mesh.from_trimesh(mesh)

        # Create scene
        scene = pyrender.Scene(bg_color=bg_color)
        scene.add(pr_mesh)

        # Add camera
        if camera_pose is None:
            # Auto camera: look at mesh center
            center = mesh.centroid
            scale = mesh.scale
            distance = scale * 2.5

            camera_pose = np.eye(4)
            camera_pose[2, 3] = distance  # Z offset
            camera_pose[1, 3] = scale * 0.3  # Slight Y offset
            # Look at center
            camera_pose[:3, 3] += center

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        scene.add(camera, pose=camera_pose)

        # Add light
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.add(light, pose=camera_pose)

        # Render
        renderer = pyrender.OffscreenRenderer(*resolution)
        color, _ = renderer.render(scene)
        renderer.delete()

        # Save image
        cv2.imwrite(output_path, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        return True

    except Exception as e:
        print(f"Render error: {e}")
        return _render_mesh_simple(mesh_path, output_path, resolution)


def _render_mesh_simple(
    mesh_path: str,
    output_path: str,
    resolution: Tuple[int, int] = (512, 512)
) -> bool:
    """
    Simple mesh rendering fallback using orthographic projection.
    """
    try:
        vertices, faces, colors = load_ply_mesh(mesh_path)

        if len(vertices) == 0:
            return False

        # Normalize vertices to [0, 1]
        vmin = vertices.min(axis=0)
        vmax = vertices.max(axis=0)
        scale = (vmax - vmin).max()
        if scale < 1e-6:
            scale = 1.0
        vertices_norm = (vertices - vmin) / scale

        # Project to 2D (simple orthographic: XY plane)
        w, h = resolution
        margin = 0.1
        scale_factor = min(w, h) * (1 - 2 * margin)
        offset = np.array([w * margin, h * margin])

        points_2d = vertices_norm[:, :2] * scale_factor + offset
        points_2d = points_2d.astype(np.int32)

        # Create image
        img = np.ones((h, w, 3), dtype=np.uint8) * 255

        # Draw points
        if colors is not None:
            for pt, color in zip(points_2d, colors):
                if 0 <= pt[0] < w and 0 <= pt[1] < h:
                    cv2.circle(img, tuple(pt), 2, tuple(int(c) for c in color[::-1]), -1)
        else:
            for pt in points_2d:
                if 0 <= pt[0] < w and 0 <= pt[1] < h:
                    cv2.circle(img, tuple(pt), 2, (100, 100, 100), -1)

        cv2.imwrite(output_path, img)
        return True

    except Exception as e:
        print(f"Simple render error: {e}")
        return False


def create_orbit_camera_poses(
    num_frames: int,
    distance: float = 2.0,
    elevation: float = 30.0,
    center: np.ndarray = None
) -> List[np.ndarray]:
    """
    Create camera poses orbiting around a center point.

    Args:
        num_frames: Number of frames (poses)
        distance: Camera distance from center
        elevation: Elevation angle in degrees
        center: Center point (default: origin)

    Returns:
        List of 4x4 camera pose matrices
    """
    if center is None:
        center = np.zeros(3)

    poses = []
    elevation_rad = np.radians(elevation)

    for i in range(num_frames):
        angle = 2 * np.pi * i / num_frames

        # Camera position
        x = distance * np.cos(angle) * np.cos(elevation_rad)
        y = distance * np.sin(elevation_rad)
        z = distance * np.sin(angle) * np.cos(elevation_rad)

        camera_pos = center + np.array([x, y, z])

        # Look-at matrix
        forward = center - camera_pos
        forward = forward / np.linalg.norm(forward)

        up = np.array([0, 1, 0])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)

        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward
        pose[:3, 3] = camera_pos

        poses.append(pose)

    return poses


def render_mesh_sequence_video(
    mesh_sequence: MeshSequence,
    output_path: str,
    view_type: str = "orbit",
    resolution: Tuple[int, int] = (512, 512),
    orbit_frames: int = 60,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> bool:
    """
    Render mesh sequence to video.

    Args:
        mesh_sequence: MeshSequence object
        output_path: Output video path (.mp4)
        view_type: "orbit" (rotating view) or "fixed" (static view)
        resolution: Video resolution
        orbit_frames: Number of frames for orbit animation per mesh
        progress_callback: Optional progress callback(current, total, message)

    Returns:
        True if successful
    """
    if not mesh_sequence.frames:
        return False

    # Create temp directory for frames
    temp_dir = Path(output_path).parent / f".temp_render_{datetime.now().strftime('%H%M%S')}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        frame_paths = []
        total_frames = len(mesh_sequence.frames)

        if view_type == "orbit":
            # For each mesh, render from multiple viewpoints
            camera_poses = create_orbit_camera_poses(orbit_frames)

            for mesh_idx, mesh_frame in enumerate(mesh_sequence.frames):
                if progress_callback:
                    progress_callback(mesh_idx, total_frames, f"Rendering mesh {mesh_idx + 1}/{total_frames}")

                # Render from orbit viewpoints
                for cam_idx, pose in enumerate(camera_poses):
                    frame_path = temp_dir / f"frame_{mesh_idx:04d}_{cam_idx:04d}.png"
                    render_mesh_view(
                        mesh_frame.mesh_path,
                        str(frame_path),
                        camera_pose=pose,
                        resolution=resolution
                    )
                    frame_paths.append(str(frame_path))

        else:  # fixed view
            # Render each mesh from fixed viewpoint
            for mesh_idx, mesh_frame in enumerate(mesh_sequence.frames):
                if progress_callback:
                    progress_callback(mesh_idx, total_frames, f"Rendering mesh {mesh_idx + 1}/{total_frames}")

                frame_path = temp_dir / f"frame_{mesh_idx:04d}.png"
                render_mesh_view(
                    mesh_frame.mesh_path,
                    str(frame_path),
                    resolution=resolution
                )
                frame_paths.append(str(frame_path))

        # Combine frames to video
        if frame_paths:
            first_frame = cv2.imread(frame_paths[0])
            h, w = first_frame.shape[:2]

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = mesh_sequence.fps if view_type == "fixed" else 30  # 30 fps for orbit
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

            for frame_path in frame_paths:
                frame = cv2.imread(frame_path)
                out.write(frame)

            out.release()

            if progress_callback:
                progress_callback(total_frames, total_frames, "Video saved")

            return True

    except Exception as e:
        print(f"Video rendering error: {e}")
        return False

    finally:
        # Cleanup temp files
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    return False


def export_mesh_sequence_obj(
    mesh_sequence: MeshSequence,
    output_dir: str
) -> str:
    """
    Export mesh sequence as numbered OBJ files for Blender import.

    Args:
        mesh_sequence: MeshSequence object
        output_dir: Output directory

    Returns:
        Path to output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i, mesh_frame in enumerate(mesh_sequence.frames):
        # Load mesh
        vertices, faces, colors = load_ply_mesh(mesh_frame.mesh_path)

        # Write OBJ
        obj_path = output_path / f"mesh_{i:04d}.obj"
        with open(obj_path, 'w') as f:
            f.write(f"# Frame {i}\n")

            # Vertices with colors as comments
            for v_idx, v in enumerate(vertices):
                if colors is not None and v_idx < len(colors):
                    c = colors[v_idx] / 255.0
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}\n")
                else:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Faces (1-indexed)
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    # Write metadata
    metadata_path = output_path / "sequence_info.json"
    with open(metadata_path, 'w') as f:
        json.dump({
            "name": mesh_sequence.name,
            "fps": mesh_sequence.fps,
            "frame_count": mesh_sequence.frame_count,
            "duration": mesh_sequence.duration,
            "files": [f"mesh_{i:04d}.obj" for i in range(mesh_sequence.frame_count)]
        }, f, indent=2)

    return str(output_path)


def export_mesh_sequence_mdd(
    mesh_sequence: MeshSequence,
    output_path: str
) -> bool:
    """
    Export mesh sequence as MDD (Point Cache) format for Blender.

    MDD format stores vertex positions per frame. Requires all meshes
    to have the same topology (same vertex count).

    Args:
        mesh_sequence: MeshSequence object
        output_path: Output .mdd file path

    Returns:
        True if successful
    """
    if not mesh_sequence.frames:
        return False

    # Load first mesh to get vertex count
    vertices_0, faces_0, _ = load_ply_mesh(mesh_sequence.frames[0].mesh_path)
    vertex_count = len(vertices_0)

    # Check all meshes have same vertex count
    all_vertices = [vertices_0]
    for frame in mesh_sequence.frames[1:]:
        vertices, _, _ = load_ply_mesh(frame.mesh_path)
        if len(vertices) != vertex_count:
            print(f"Warning: Vertex count mismatch. MDD export requires same topology.")
            return False
        all_vertices.append(vertices)

    # Write MDD file
    # MDD format:
    # - 4 bytes: frame count (int)
    # - 4 bytes: vertex count (int)
    # - For each frame:
    #   - 4 bytes: time (float)
    #   - vertex_count * 3 * 4 bytes: vertex positions (float x, y, z)

    try:
        with open(output_path, 'wb') as f:
            # Header
            f.write(struct.pack('>i', len(mesh_sequence.frames)))  # frame count
            f.write(struct.pack('>i', vertex_count))  # vertex count

            # Frame data
            for i, vertices in enumerate(all_vertices):
                # Time
                time = i / mesh_sequence.fps
                f.write(struct.pack('>f', time))

                # Vertices
                for v in vertices:
                    f.write(struct.pack('>fff', v[0], v[1], v[2]))

        return True

    except Exception as e:
        print(f"MDD export error: {e}")
        return False


def export_mesh_sequence_alembic(
    mesh_sequence: MeshSequence,
    output_path: str
) -> bool:
    """
    Export mesh sequence as Alembic (.abc) format.

    Requires alembic python package.

    Args:
        mesh_sequence: MeshSequence object
        output_path: Output .abc file path

    Returns:
        True if successful
    """
    try:
        import alembic
        from alembic import Abc, AbcGeom
    except ImportError:
        print("Alembic package not installed. Install with: pip install alembic")
        return False

    try:
        # Create archive
        archive = Abc.OArchive(output_path)
        top = archive.getTop()

        # Create mesh object
        mesh_obj = AbcGeom.OPolyMesh(top, mesh_sequence.name)
        mesh_schema = mesh_obj.getSchema()

        # Get base topology from first frame
        vertices_0, faces_0, _ = load_ply_mesh(mesh_sequence.frames[0].mesh_path)

        # Face counts and indices
        face_counts = np.full(len(faces_0), 3, dtype=np.int32)  # All triangles
        face_indices = faces_0.flatten().astype(np.int32)

        # Write frames
        for i, mesh_frame in enumerate(mesh_sequence.frames):
            vertices, _, _ = load_ply_mesh(mesh_frame.mesh_path)

            time_sampling = Abc.TimeSampling(1.0 / mesh_sequence.fps, i / mesh_sequence.fps)

            sample = AbcGeom.OPolyMeshSchemaSample(
                vertices.flatten().astype(np.float32),
                face_indices,
                face_counts
            )
            mesh_schema.set(sample)

        return True

    except Exception as e:
        print(f"Alembic export error: {e}")
        return False


def generate_blender_import_script(
    sequence_dir: str,
    output_script_path: str,
    sequence_type: str = "obj"  # "obj", "mdd", or "alembic"
) -> str:
    """
    Generate a Blender Python script for importing the mesh sequence.

    Args:
        sequence_dir: Directory containing mesh files
        output_script_path: Path to save the script
        sequence_type: Type of sequence files

    Returns:
        Path to generated script
    """
    if sequence_type == "obj":
        script = f'''
import bpy
import os
from pathlib import Path

# Mesh sequence directory
sequence_dir = r"{sequence_dir}"

# Load sequence info
import json
info_path = Path(sequence_dir) / "sequence_info.json"
with open(info_path, 'r') as f:
    info = json.load(f)

fps = info['fps']
frame_count = info['frame_count']

# Set scene FPS
bpy.context.scene.render.fps = int(fps)
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = frame_count - 1

# Import first mesh as base
first_obj = Path(sequence_dir) / "mesh_0000.obj"
bpy.ops.wm.obj_import(filepath=str(first_obj))
base_obj = bpy.context.selected_objects[0]
base_obj.name = "MeshSequence"

# Add shape keys for animation
base_obj.shape_key_add(name="Basis", from_mix=False)

for i in range(1, frame_count):
    obj_path = Path(sequence_dir) / f"mesh_{{i:04d}}.obj"
    if obj_path.exists():
        # Import temp mesh
        bpy.ops.wm.obj_import(filepath=str(obj_path))
        temp_obj = bpy.context.selected_objects[0]

        # Join as shape key
        base_obj.select_set(True)
        bpy.context.view_layer.objects.active = base_obj
        temp_obj.select_set(True)
        bpy.ops.object.join_shapes()

        # Rename shape key
        base_obj.data.shape_keys.key_blocks[-1].name = f"Frame_{{i:04d}}"

        # Delete temp
        bpy.ops.object.select_all(action='DESELECT')
        temp_obj.select_set(True)
        bpy.ops.object.delete()

# Animate shape keys
for i, key in enumerate(base_obj.data.shape_keys.key_blocks[1:], 1):
    # Set keyframes: value 0 except at frame i
    for frame in range(frame_count):
        key.value = 1.0 if frame == i else 0.0
        key.keyframe_insert(data_path="value", frame=frame)

print(f"Imported {{frame_count}} frames as shape key animation")
'''
    elif sequence_type == "mdd":
        script = f'''
import bpy

# First import the base mesh (OBJ), then apply MDD modifier
mdd_path = r"{sequence_dir}"

# Note: MDD import requires the mesh to already exist
# 1. Import base mesh first
# 2. Select the mesh
# 3. Add Mesh Cache modifier
# 4. Set the MDD file path

print("MDD file path:", mdd_path)
print("To use:")
print("1. Import base mesh")
print("2. Select mesh > Add Modifier > Mesh Cache")
print("3. Set cache file to the MDD file")
'''
    else:  # alembic
        script = f'''
import bpy

# Alembic import
abc_path = r"{sequence_dir}"

bpy.ops.wm.alembic_import(filepath=abc_path)

print("Imported Alembic file:", abc_path)
'''

    with open(output_script_path, 'w') as f:
        f.write(script)

    return output_script_path


class MeshSequenceGenerator:
    """
    Main class for generating and exporting mesh sequences.
    """

    def __init__(self, processor=None):
        """
        Args:
            processor: SAM3DProcessor instance for mesh generation
        """
        self.processor = processor
        self.current_sequence: Optional[MeshSequence] = None

    def generate_sequence(
        self,
        frames: List[np.ndarray],
        masks: List[np.ndarray],
        output_dir: str,
        name: str = "mesh_sequence",
        fps: float = 30.0,
        mesh_settings: Dict[str, Any] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> MeshSequence:
        """
        Generate mesh sequence from frames and masks.

        Args:
            frames: List of RGB frames (numpy arrays)
            masks: List of binary masks (numpy arrays)
            output_dir: Output directory
            name: Sequence name
            fps: Frames per second
            mesh_settings: Mesh generation settings
            progress_callback: Optional callback(current, total, message)

        Returns:
            MeshSequence object
        """
        if self.processor is None:
            raise ValueError("Processor not set. Cannot generate meshes.")

        if len(frames) != len(masks):
            raise ValueError(f"Frame count ({len(frames)}) != mask count ({len(masks)})")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        mesh_settings = mesh_settings or {}
        sequence = MeshSequence(
            name=name,
            fps=fps,
            output_dir=str(output_path),
            metadata={"mesh_settings": mesh_settings}
        )

        total = len(frames)
        for i, (frame, mask) in enumerate(zip(frames, masks)):
            if progress_callback:
                progress_callback(i, total, f"Generating mesh {i + 1}/{total}")

            try:
                # Generate 3D mesh
                reconstruction = self.processor.reconstruct_3d(
                    frame, mask,
                    seed=mesh_settings.get('seed', 42),
                    mesh_settings=mesh_settings
                )

                if reconstruction:
                    # Save mesh
                    mesh_path = output_path / f"mesh_{i:04d}.ply"
                    self.processor.export_mesh(reconstruction, str(mesh_path), format='ply')

                    # Create frame entry
                    mesh_frame = MeshFrame(
                        frame_idx=i,
                        mesh_path=str(mesh_path),
                        timestamp=i / fps
                    )
                    sequence.frames.append(mesh_frame)

            except Exception as e:
                print(f"Error generating mesh for frame {i}: {e}")
                continue

        self.current_sequence = sequence

        # Save sequence metadata
        self._save_sequence_metadata(sequence)

        return sequence

    def _save_sequence_metadata(self, sequence: MeshSequence):
        """Save sequence metadata to JSON"""
        metadata_path = Path(sequence.output_dir) / "sequence_metadata.json"
        metadata = {
            "name": sequence.name,
            "fps": sequence.fps,
            "frame_count": sequence.frame_count,
            "duration": sequence.duration,
            "frames": [
                {
                    "frame_idx": f.frame_idx,
                    "mesh_path": f.mesh_path,
                    "timestamp": f.timestamp
                }
                for f in sequence.frames
            ],
            "metadata": sequence.metadata,
            "created_at": datetime.now().isoformat()
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_sequence(self, sequence_dir: str) -> MeshSequence:
        """
        Load existing mesh sequence from directory.

        Args:
            sequence_dir: Directory containing sequence files

        Returns:
            MeshSequence object
        """
        metadata_path = Path(sequence_dir) / "sequence_metadata.json"

        if not metadata_path.exists():
            # Try to auto-detect sequence
            return self._auto_detect_sequence(sequence_dir)

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        sequence = MeshSequence(
            name=metadata.get('name', 'unknown'),
            fps=metadata.get('fps', 30.0),
            output_dir=sequence_dir,
            metadata=metadata.get('metadata', {})
        )

        for frame_data in metadata.get('frames', []):
            mesh_frame = MeshFrame(
                frame_idx=frame_data['frame_idx'],
                mesh_path=frame_data['mesh_path'],
                timestamp=frame_data.get('timestamp', 0.0)
            )
            sequence.frames.append(mesh_frame)

        self.current_sequence = sequence
        return sequence

    def _auto_detect_sequence(self, sequence_dir: str) -> MeshSequence:
        """Auto-detect mesh sequence from numbered files"""
        mesh_files = sorted(Path(sequence_dir).glob("mesh_*.ply"))

        if not mesh_files:
            mesh_files = sorted(Path(sequence_dir).glob("*.ply"))

        sequence = MeshSequence(
            name=Path(sequence_dir).name,
            fps=30.0,
            output_dir=sequence_dir
        )

        for i, mesh_file in enumerate(mesh_files):
            mesh_frame = MeshFrame(
                frame_idx=i,
                mesh_path=str(mesh_file),
                timestamp=i / 30.0
            )
            sequence.frames.append(mesh_frame)

        self.current_sequence = sequence
        return sequence

    def render_video(
        self,
        output_path: str,
        view_type: str = "fixed",
        resolution: Tuple[int, int] = (512, 512),
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> bool:
        """
        Render current sequence to video.

        Args:
            output_path: Output video path
            view_type: "fixed" or "orbit"
            resolution: Video resolution
            progress_callback: Optional progress callback

        Returns:
            True if successful
        """
        if self.current_sequence is None:
            raise ValueError("No sequence loaded. Generate or load a sequence first.")

        return render_mesh_sequence_video(
            self.current_sequence,
            output_path,
            view_type=view_type,
            resolution=resolution,
            progress_callback=progress_callback
        )

    def export_for_blender(
        self,
        output_dir: str,
        format: str = "obj"  # "obj", "mdd", "alembic"
    ) -> str:
        """
        Export sequence for Blender.

        Args:
            output_dir: Output directory
            format: Export format

        Returns:
            Path to output
        """
        if self.current_sequence is None:
            raise ValueError("No sequence loaded.")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if format == "obj":
            result_dir = export_mesh_sequence_obj(self.current_sequence, str(output_path / "obj_sequence"))
            # Generate import script
            script_path = output_path / "import_to_blender.py"
            generate_blender_import_script(result_dir, str(script_path), "obj")
            return result_dir

        elif format == "mdd":
            # First export OBJ for base mesh
            obj_dir = output_path / "base_mesh"
            obj_dir.mkdir(exist_ok=True)

            # Export first frame as base
            if self.current_sequence.frames:
                first_mesh = self.current_sequence.frames[0].mesh_path
                import shutil
                base_mesh_path = obj_dir / "base_mesh.ply"
                shutil.copy(first_mesh, base_mesh_path)

            # Export MDD
            mdd_path = output_path / f"{self.current_sequence.name}.mdd"
            export_mesh_sequence_mdd(self.current_sequence, str(mdd_path))

            script_path = output_path / "import_to_blender.py"
            generate_blender_import_script(str(mdd_path), str(script_path), "mdd")
            return str(mdd_path)

        elif format == "alembic":
            abc_path = output_path / f"{self.current_sequence.name}.abc"
            export_mesh_sequence_alembic(self.current_sequence, str(abc_path))

            script_path = output_path / "import_to_blender.py"
            generate_blender_import_script(str(abc_path), str(script_path), "alembic")
            return str(abc_path)

        else:
            raise ValueError(f"Unknown format: {format}")


if __name__ == "__main__":
    # Test basic functionality
    print("Mesh Sequence Generator")
    print(f"  trimesh available: {HAS_TRIMESH}")
    print(f"  pyrender available: {HAS_PYRENDER}")

    # Test with existing PLY files if available
    test_dir = Path("/home/joon/dev/sam3d_gui/outputs")
    ply_files = list(test_dir.glob("**/*.ply"))[:3]

    if ply_files:
        print(f"\nFound {len(ply_files)} PLY files for testing")

        # Create test sequence
        sequence = MeshSequence(
            name="test_sequence",
            fps=10.0,
            output_dir=str(test_dir / "test_sequence")
        )

        for i, ply_file in enumerate(ply_files):
            mesh_frame = MeshFrame(
                frame_idx=i,
                mesh_path=str(ply_file),
                timestamp=i / 10.0
            )
            sequence.frames.append(mesh_frame)

        print(f"Created sequence with {sequence.frame_count} frames")
        print(f"Duration: {sequence.duration:.2f}s")
    else:
        print("No PLY files found for testing")
