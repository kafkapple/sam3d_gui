import torch
import utils3d.torch
import nvdiffrast.torch as dr

print("Testing utils3d rasterize_triangle_faces with CUDA...")
device = "cuda:0"

# utils3d RastContext with CUDA
print("Creating utils3d CUDA context...")
ctx = utils3d.torch.RastContext(backend="cuda")
print("Context created:", type(ctx.nvd_ctx))

# Mesh similar to decimated mesh
n_vertices = 2000
n_faces = 4000

vertices = torch.randn(1, n_vertices, 3, dtype=torch.float32, device=device) * 0.5
faces = torch.randint(0, n_vertices, (n_faces, 3), dtype=torch.int32, device=device)
uvs = torch.rand(1, n_vertices, 2, dtype=torch.float32, device=device)

# View and projection
view = torch.eye(4, dtype=torch.float32, device=device)
projection = torch.eye(4, dtype=torch.float32, device=device)

print("Mesh:", n_vertices, "vertices,", n_faces, "faces")
print("Calling rasterize_triangle_faces...")

result = utils3d.torch.rasterize_triangle_faces(
    ctx,
    vertices,
    faces,
    256, 256,
    uv=uvs,
    view=view,
    projection=projection,
)

print("Result keys:", result.keys())
mask_shape = result["mask"].shape
print("mask shape:", mask_shape)
print("SUCCESS!")
