import torch
from nvdiffmodeling.src import mesh as nvmesh
import torch.nn.functional as F


def get_penetration_loss(
        mesh1: nvmesh,
        mesh2: nvmesh,
        allowable_pen: float = 0.0
) -> torch.Tensor:
    mesh1_penetration = get_mesh_penetration_loss(mesh1, mesh2, cpen=allowable_pen)
    # mesh2_penetration = get_mesh_penetration_loss(mesh2, mesh1, allowable_pen)
    return mesh1_penetration


def get_mesh_penetration_loss(
    mesh1: nvmesh,
    mesh2: nvmesh,
    sigma: float = 0.05,
    cpen:  float = 0.002
) -> torch.Tensor:
    """
    Fully GPU-safe penetration loss: for each vertex in mesh1, we compute a
    weighted sum over ALL mesh2 verts (soft nearest neighbor + normals), so
    no indexing, no out-of-bounds possible.
    """
    V1 = mesh1.v_pos   # (N1,3), CUDA
    V2 = mesh2.v_pos   # (N2,3), CUDA
    N2 = mesh2.v_nrm   # (N2,3), CUDA

    # 1) pairwise squared distances
    D2 = torch.cdist(V1, V2, p=2)**2      # (N1, N2)

    # 2) soft weights along each row
    W = torch.softmax(-D2 / (2*sigma*sigma), dim=1)  # (N1, N2)

    # 3) soft‐nearest‐vectors
    v2_soft = W @ V2    # (N1,3)
    n2_soft = W @ N2    # (N1,3)
    n2_soft = F.normalize(n2_soft, dim=1)  # re-normalize normals

    # 4) signed penetration depth along n2_soft
    #    (v2_soft − V1)⋅n2_soft > 0 when V1 is inside mesh2
    depth = ((v2_soft - V1) * n2_soft).sum(dim=1)  # (N1,)

    # 5) margin + ReLU
    pen = torch.relu(depth - cpen)  # (N1,)

    # 6) mean to scalar
    return pen.mean()


def get_fractional_soft_icp_loss(
    mesh1,         # moving Mesh, already posed; has .v_pos (N1×3) and mesh2.v_pos (N2×3)
    mesh2,
    percent: float = 0.25,
    sigma: float = 0.1,
) -> torch.Tensor:
    """
    - Picks the closest `percent` fraction of verts from mesh1 toward mesh2,
      then applies a soft‐ICP on that subset.
    - Returns a single scalar loss.
    """
    # 1) all source / target verts
    V1 = mesh1.v_pos            # (N1,3)
    V2 = mesh2.v_pos            # (N2,3)

    # 2) compute per‐vertex nearest‐neighbor distance
    dists = torch.cdist(V1, V2, p=2)          # (N1, N2)
    d_min, _ = dists.min(dim=1, keepdim=False)  # (N1,)

    # 3) pick the top‐k closest src verts
    N1 = V1.shape[0]
    k = max(1, int(percent * N1))
    _, topk_idx = torch.topk(d_min, k, largest=False)  # (k,)

    src_sel = V1[topk_idx]   # (k,3)

    # 4) build soft‐ICP between src_sel and the entire target V2
    D = torch.cdist(src_sel, V2, p=2) ** 2   # (k, N2)
    W = torch.softmax(-D / (2 * sigma**2), dim=1)  # (k, N2)

    # 5) expected squared distance per selected source, then average
    loss = (W * D).sum(dim=1).mean()
    return loss


def get_scaling_reg(scale, weight: float, tol: float = 0.2):
    """
    Penalize scale only when |s - 1| > tol.
    s is self.scaling_param (shape [1] or []).
    """
    # excess = max(|s-1|-tol, 0)
    excess = torch.clamp(torch.abs(scale - 1.0) - tol, min=0.0)
    return weight * (excess ** 2)