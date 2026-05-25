import math
import random

import numpy as np
import torch
from pytorch3d.transforms import quaternion_multiply, quaternion_apply


def get_x_rotation_matrix(rx: torch.Tensor) -> torch.Tensor:
    """
    Rotation about the X‑axis by angle rx (radians).
    Supports scalar or batched rx (…,).
    Returns a tensor of shape (…, 3, 3).
    """
    c, s = torch.cos(rx), torch.sin(rx)
    zeros = torch.zeros_like(rx)
    ones = torch.ones_like(rx)

    row0 = torch.stack([ones, zeros, zeros], dim=-1)
    row1 = torch.stack([zeros, c, -s], dim=-1)
    row2 = torch.stack([zeros, s, c], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def get_y_rotation_matrix(ry: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(ry), torch.sin(ry)
    zeros = torch.zeros_like(ry)
    ones = torch.ones_like(ry)

    row0 = torch.stack([c, zeros, s], dim=-1)
    row1 = torch.stack([zeros, ones, zeros], dim=-1)
    row2 = torch.stack([-s, zeros, c], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def get_z_rotation_matrix(rz: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(rz), torch.sin(rz)
    zeros = torch.zeros_like(rz)
    ones = torch.ones_like(rz)

    row0 = torch.stack([c, -s, zeros], dim=-1)
    row1 = torch.stack([s, c, zeros], dim=-1)
    row2 = torch.stack([zeros, zeros, ones], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def compute_vertex_normals(v_pos: torch.Tensor, t_pos_idx: torch.Tensor):
    """
    v_pos: (N,3) vertex positions
    t_pos_idx: (F,3) triangle indices into v_pos
    returns: (N,3) per-vertex normals (area‐weighted)
    """
    # 1) face normals
    v0 = v_pos[t_pos_idx[:, 0]]
    v1 = v_pos[t_pos_idx[:, 1]]
    v2 = v_pos[t_pos_idx[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=1)  # (F,3)
    fn = fn / (fn.norm(dim=1, keepdim=True) + 1e-12)  # normalize

    # 2) accumulate into per‐vertex
    n = torch.zeros_like(v_pos)  # (N,3)
    # scatter‐add face normals into each vertex
    for i in range(3):
        idx = t_pos_idx[:, i]  # (F,)
        n = n.index_add(0, idx, fn)

    # 3) normalize per‐vertex
    n = n / (n.norm(dim=1, keepdim=True) + 1e-12)
    return n


def norm_quat_wxyz(q):
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(q)
    return q / (n if n > 0 else 1.0)


def quat_to_R_wxyz(q):
    w, x, y, z = norm_quat_wxyz(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def rand_translation(r_min=2.0, r_max=4.0):
    # random direction
    v = np.random.normal(size=3)
    v = v / (np.linalg.norm(v) + 1e-12)
    v[1] = abs(v[1])

    r = random.uniform(r_min, r_max)
    return (r * v).tolist()


def rand_quat_small(max_angle_deg=30.0):
    # Small random rotation around a random axis; return [w, x, y, z]
    ax = [random.gauss(0.0, 1.0) for _ in range(3)]
    norm = math.sqrt(ax[0] ** 2 + ax[1] ** 2 + ax[2] ** 2) + 1e-9
    ax = [a / norm for a in ax]
    theta = math.radians(random.uniform(0.0, max_angle_deg))
    s = math.sin(theta / 2.0)
    w = math.cos(theta / 2.0)
    x, y, z = (ax[0] * s, ax[1] * s, ax[2] * s)
    # Normalize just in case of numeric drift
    qn = math.sqrt(w * w + x * x + y * y + z * z)
    return [w / qn, x / qn, y / qn, z / qn]


def add_random_init(run_cfg, t_max=4.0, max_angle_deg=90.0):
    """
    Fill init_pos_params.translation / rotation only if missing.
    """
    if not hasattr(run_cfg.source_mesh, 'init_pos_params'):
        run_cfg.source_mesh['init_pos_params'] = {}

    t = rand_translation(r_max=t_max)
    run_cfg.source_mesh.init_pos_params['translation'] = t
    print(f"[random-init] translation={t}")

    q = rand_quat_small(max_angle_deg=max_angle_deg)
    run_cfg.source_mesh.init_pos_params['rotation'] = q
    print(f"[random-init] rotation_quat={q}")


def unit_scale_from_obj_geometry(path: str) -> float:
    """Return the unit-size scale = 2 / max side extent, using only 'v' lines."""
    import numpy as np
    vmin = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    vmax = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    with open(path, 'r', errors='ignore') as f:
        for ln in f:
            if ln.startswith('v ') or ln.startswith('v\t'):
                _, xs, ys, zs, *rest = ln.split()
                p = np.array([float(xs), float(ys), float(zs)], dtype=np.float64)
                vmin = np.minimum(vmin, p)
                vmax = np.maximum(vmax, p)
    extent = vmax - vmin
    max_side = float(np.max(extent)) if np.isfinite(extent).all() else 1.0
    s = 2.0 / max(max_side, 1e-12)
    print(f"[unit-scale] {path} -> scale={s}")
    return s

def scale_mesh_about_centroid(mesh, scale):
    """
    Uniformly scale `mesh.v_pos` about its centroid (preserves centroid location).
    Works with nvdiffmodeling Mesh objects (mesh.v_pos is a torch tensor).
    """
    s = float(scale)
    c = mesh.v_pos.mean(dim=0, keepdim=True)         # (1,3)
    mesh.v_pos = (mesh.v_pos - c) * s + c
    return mesh

