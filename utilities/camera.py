from typing import List, Dict, Any

from pyglm import glm
import torch
import random
import math
import numpy as np


def persp_proj(fov_x=45, ar=1, near=1.0, far=50.0):
    """
    From https://github.com/rgl-epfl/large-steps-pytorch by @bathal1 (Baptiste Nicolet)

    Build a perspective projection matrix.
    Parameters
    ----------
    fov_x : float
        Horizontal field of view (in degrees).
    ar : float
        Aspect ratio (w/h).
    near : float
        Depth of the near plane relative to the camera.
    far : float
        Depth of the far plane relative to the camera.
    """
    fov_rad = np.deg2rad(fov_x)

    tanhalffov = np.tan((fov_rad / 2))
    max_y = tanhalffov * near
    min_y = -max_y
    max_x = max_y * ar
    min_x = -max_x

    z_sign = -1.0
    proj_mat = np.array([[0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]])

    proj_mat[0, 0] = 2.0 * near / (max_x - min_x)
    proj_mat[1, 1] = 2.0 * near / (max_y - min_y)
    proj_mat[0, 2] = (max_x + min_x) / (max_x - min_x)
    proj_mat[1, 2] = (max_y + min_y) / (max_y - min_y)
    proj_mat[3, 2] = z_sign

    proj_mat[2, 2] = z_sign * far / (far - near)
    proj_mat[2, 3] = -(far * near) / (far - near)

    return proj_mat


def get_camera_params(elev_angle, azim_angle, distance, resolution, fov=60, look_at=[0, 0, 0], up=[0, -1, 0]):
    elev = np.radians(elev_angle)
    azim = np.radians(azim_angle)

    # Generate random view
    cam_z = distance * np.cos(elev) * np.sin(azim)
    cam_y = distance * np.sin(elev)
    cam_x = distance * np.cos(elev) * np.cos(azim)

    modl = glm.mat4()
    view = glm.lookAt(
        glm.vec3(cam_x, cam_y, cam_z),
        glm.vec3(look_at[0], look_at[1], look_at[2]),
        glm.vec3(up[0], up[1], up[2]),
    )

    a_mv = view * modl
    a_mv = np.array(a_mv.to_list()).T
    proj_mtx = persp_proj(fov)

    a_mvp = np.matmul(proj_mtx, a_mv).astype(np.float32)[None, ...]

    a_lightpos = np.linalg.inv(a_mv)[None, :3, 3]
    a_campos = a_lightpos

    return {
        'mvp': a_mvp,
        'lightpos': a_lightpos,
        'campos': a_campos,
        'resolution': [resolution, resolution],
    }


class CameraBatch(torch.utils.data.Dataset):
    def __init__(
            self,
            image_resolution,
            distances,
            azimuths,
            elevation_params,
            fovs,
            aug_loc,
            bs,
            look_at=[0, 0, 0], up=[0, -1, 0],
            rand_solid=False
    ):

        self.res = image_resolution

        self.dist_min = distances[0]
        self.dist_max = distances[1]

        self.azim_min = azimuths[0]
        self.azim_max = azimuths[1]

        self.fov_min = fovs[0]
        self.fov_max = fovs[1]

        self.elev_alpha = elevation_params[0]
        self.elev_beta = elevation_params[1]
        self.elev_min = elevation_params[2]
        self.elev_max = elevation_params[3]

        self.aug_loc = aug_loc

        self.look_at = look_at
        self.up = up

        self.batch_size = bs
        self.rand_solid = rand_solid

        self.focus_radius = getattr(self, 'focus_radius', 1.0)

    def __len__(self):
        return self.batch_size

    def __getitem__(self, index):
        elev = np.radians(np.random.beta(self.elev_alpha, self.elev_beta) * (self.elev_max - self.elev_min) + self.elev_min)
        azim = np.radians(np.random.uniform(self.azim_min, self.azim_max + 1.0))
        dist = np.random.uniform(self.dist_min, self.dist_max)
        fov = np.random.uniform(self.fov_min, self.fov_max)

        # --- Elevation correction relative to current target height ---
        target = np.array(self.look_at, dtype=np.float32)  # phase-set
        ty = float(target[1])
        eps = 1e-6
        gain = getattr(self, "elev_corr_gain", 0.3)  # ↓ smaller = less lowering
        sin_corr = np.sin(elev) - gain * (ty / max(dist, eps))

        # respect YAML elevation bounds
        emin = np.radians(self.elev_min)
        emax = np.radians(self.elev_max)
        sin_corr = np.clip(sin_corr, np.sin(emin), np.sin(emax))
        elev = float(np.arcsin(sin_corr))

        # --- Stay outside the source mesh + roughly fit it for this FOV ---
        r = 1.05 * float(self.focus_radius)  # 5% safety
        fit = r / np.tan(np.radians(fov) * 0.5)  # FOV fit
        d_min = max(r, fit)

        # origin-centric geometry: enforce ||dist*u - target|| >= d_min
        u = np.array([np.cos(elev) * np.cos(azim),
                      np.sin(elev),
                      np.cos(elev) * np.sin(azim)], dtype=np.float32)
        dot_ut = float(np.dot(u, target))
        t_norm2 = float(np.dot(target, target))
        disc = d_min * d_min - t_norm2 + dot_ut * dot_ut
        if disc > 0.0:
            dist = max(dist, dot_ut + np.sqrt(disc))

        # final clamp to phase bounds
        dist = float(min(max(dist, self.dist_min), self.dist_max))

        return self.get_one_camera_params(azim, dist, elev, fov)

    def get_one_camera_params(self, azim, dist, elev, fov):
        proj_mtx = persp_proj(fov)
        # Generate random view
        cam_z = dist * np.cos(elev) * np.sin(azim)
        cam_y = dist * np.sin(elev)
        cam_x = dist * np.cos(elev) * np.cos(azim)
        if self.aug_loc:
            # Random offset
            limit = self.dist_min // 2
            rand_x = np.random.uniform(-limit, limit)
            rand_y = np.random.uniform(-limit, limit)
            modl = glm.translate(glm.mat4(), glm.vec3(rand_x, rand_y, 0))
        else:
            modl = glm.mat4()

        view = glm.lookAt(
            glm.vec3(cam_x, cam_y, cam_z),
            glm.vec3(self.look_at[0], self.look_at[1], self.look_at[2]),
            glm.vec3(self.up[0], self.up[1], self.up[2]),
        )
        r_mv = view * modl
        r_mv = np.array(r_mv.to_list()).T
        mvp = np.matmul(proj_mtx, r_mv).astype(np.float32)
        campos = np.linalg.inv(r_mv)[:3, 3]

        # view direction (from camera to look_at)
        look_at_np = np.array(self.look_at, dtype=np.float32)
        view_dir = look_at_np - campos
        if np.linalg.norm(view_dir) < 1e-6:
            view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-12)

        key_scale = 0.4
        key_dist = dist * key_scale
        up_offset = np.array([0.0, 0.15 * dist, 0.12 * dist], dtype=np.float32)
        lightpos = campos + view_dir * key_dist + up_offset
        lightpos = lightpos.astype(np.float32)

        bkgs = torch.ones(self.res, self.res, 3)
        cam2world = np.linalg.inv(r_mv).astype(np.float32)
        return {
            'mvp': torch.from_numpy(mvp).float(),
            'lightpos': torch.from_numpy(lightpos).float(),
            'campos': torch.from_numpy(campos).float(),
            'bkgs': bkgs,
            'azim': torch.tensor(azim).float(),
            'elev': torch.tensor(elev).float(),
            'fov': torch.tensor(fov).int(),
            'cam2world': torch.from_numpy(cam2world).float(),
        }


@torch.no_grad()
def adapt_camera_for_phase(cams_data, dyn_mesh, fix_mesh, phase_idx: int, n_phases: int,
                           base_dist_min: float, base_dist_max: float):
    v_dyn = dyn_mesh.v_pos
    dmin, dmax = v_dyn.amin(dim=0), v_dyn.amax(dim=0)
    dyn_center = 0.5 * (dmin + dmax)
    dyn_extent_m = (dmax - dmin).max()

    v_fix = fix_mesh.v_pos
    fmin, fmax = v_fix.amin(dim=0), v_fix.amax(dim=0)
    fix_extent_m = (fmax - fmin).max()

    scene_extent = torch.max(dyn_extent_m, fix_extent_m)

    p = float((phase_idx + 1) / max(1, n_phases))  # phase progress in [0,1]
    s = 0.5 * p  # horizontal/overall shift
    s_v = 0.15 * p  # vertical shift (smaller -> lower elevation)

    base = torch.zeros_like(dyn_center)
    look_at_all = (1.0 - s) * base + s * dyn_center
    look_at_y = (1.0 - s_v) * base[1] + s_v * dyn_center[1]
    look_at = torch.stack([look_at_all[0], look_at_y, look_at_all[2]])

    # size = (1.0 - p) * fix_extent_m + p * dyn_extent_m
    # base_mult = 0.5 * size
    base_mult = 0.45 * scene_extent
    cams_data.dist_min = float(base_dist_min * base_mult * (1.0 - 0.1 * p))
    cams_data.dist_max = float(base_dist_max * base_mult * (1.0 - 0.1 * p))

    cams_data.look_at = look_at.detach().cpu().tolist()
    cams_data.focus_radius = float((0.5 * dyn_extent_m).detach().cpu().item())


def get_data_loader(cfg):
    cams_data = CameraBatch(
        cfg.train_res,
        [cfg.dist_min, cfg.dist_max],
        [cfg.azim_min, cfg.azim_max],
        [cfg.elev_alpha, cfg.elev_beta, cfg.elev_min, cfg.elev_max],
        [cfg.fov_min, cfg.fov_max],
        cfg.aug_loc,
        cfg.batch_size,
        rand_solid=True
    )
    camera_data_loader = torch.utils.data.DataLoader(cams_data, cfg.batch_size, num_workers=0, pin_memory=True)
    return camera_data_loader, cams_data


def get_next_camera_params(data_loader, device):
    params_camera = next(iter(data_loader))
    move_all_keys_to_device(device, params_camera)
    return params_camera


def move_all_keys_to_device(device, params_camera):
    for key in params_camera:
        params_camera[key] = params_camera[key].to(device)


def adapt_camera_distance(data_loader, meshes, dist_min, dist_max):
    """
    Adapt the camera distance bounds based on the combined size of a list of meshes.

    Args:
        data_loader:   a DataLoader whose .dataset has attributes `dist_min` and `dist_max`
        meshes:        list of nvdiffmodeling.Mesh instances
        dist_min:      base minimum distance (scalar)
        dist_max:      base maximum distance (scalar)

    This computes the axis‐aligned bounding box enclosing all meshes,
    takes half the longest side (i.e. the max half‐extent), and multiplies
    dist_min/dist_max by that value to set the dataset’s camera distance range.
    """
    with torch.no_grad():
        # Gather per‐mesh bounds
        mins = []
        maxs = []
        for mesh in meshes:
            v = mesh.v_pos
            mins.append(v.amin(dim=0))
            maxs.append(v.amax(dim=0))

        # Compute global AABB
        global_min = torch.stack(mins, dim=0).amin(dim=0)
        global_max = torch.stack(maxs, dim=0).amax(dim=0)

        # Half‐sizes along each axis
        half_sizes = (global_max - global_min) * 0.3

        # Use the largest half‐size as the scale multiplier
        mult = half_sizes.max().cpu().item()

        # Apply to the dataset
        data_loader.dataset.dist_min = dist_min * mult
        data_loader.dataset.dist_max = dist_max * mult


def views_to_batched_camera_dict(cams: List[Dict[str, Any]], device: torch.device = None) -> Dict[str, torch.Tensor]:
    """
    Convert a list of single-view camera dicts (each representing B=1) into a single
    batched camera params dict with batch dim B=len(cams). Move all tensors to `device`
    if provided (defaults to CUDA if available).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(cams) == 0:
        raise RuntimeError("`cams` is empty")

    batched: Dict[str, torch.Tensor] = {}
    keys = list(cams[0].keys())

    for k in keys:
        vals = [c[k] for c in cams]

        # convert non-torch types to tensors
        tensor_vals = []
        for v in vals:
            if isinstance(v, torch.Tensor):
                tv = v
            elif isinstance(v, np.ndarray):
                tv = torch.from_numpy(v)
            else:
                # handles list/tuple/scalars
                try:
                    tv = torch.tensor(v)
                except Exception as e:
                    raise TypeError(f"Cannot convert value of key '{k}' to tensor: {type(v)}") from e

            # ensure leading batch dim = 1 for each per-view tensor
            if tv.dim() == 0:
                tv = tv.unsqueeze(0)  # scalar -> (1,)
            elif tv.shape[0] != 1:
                if tv.shape[0] == 1:
                    pass
                else:
                    if tv.dim() >= 2 and tv.shape[0] != 1:
                        tv = tv.unsqueeze(0)
            tensor_vals.append(tv)

        try:
            concatenated = torch.cat(tensor_vals, dim=0).to(device)
        except RuntimeError as e:
            raise RuntimeError(f"Failed concatenation for key '{k}': {[t.shape for t in tensor_vals]}") from e

        batched[k] = concatenated

    if 'fov' in batched:
        batched['fov'] = batched['fov'].to(device).int()

    return batched


def ortho_proj(left=-2.0, right=2.0, bottom=-2.0, top=2.0, near=0.1, far=100.0):
    return np.array([
        [2.0 / (right - left), 0.0, 0.0, -(right + left) / (right - left)],
        [0.0, 2.0 / (top - bottom), 0.0, -(top + bottom) / (top - bottom)],
        [0.0, 0.0, -2.0 / (far - near), -(far + near) / (far - near)],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)


def generate_eval_cameras(
        resolution=(512, 512),
        k=4,
        elev_deg=30.0,
        dist=6.0,
        device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H, W = resolution
    cams = []

    center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    elev_rad = math.radians(float(elev_deg))

    proj_mtx = ortho_proj(
        left=-4.0,
        right=4.0,
        bottom=-4.0,
        top=4.0,
        near=0.1,
        far=100.0,
    )

    for i in range(k):
        azim_deg = i * (360.0 / float(k))
        azim_rad = math.radians(azim_deg)

        direction = np.array([
            math.cos(elev_rad) * math.cos(azim_rad),
            math.sin(elev_rad),
            math.cos(elev_rad) * math.sin(azim_rad),
        ], dtype=np.float32)

        direction = direction / (np.linalg.norm(direction) + 1e-12)
        cam_pos = center + direction * float(dist)

        view = glm.lookAt(
            glm.vec3(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])),
            glm.vec3(0.0, 0.0, 0.0),
            glm.vec3(0.0, -1.0, 0.0),
        )

        r_mv = np.array(view.to_list()).T.astype(np.float32)
        mvp = (proj_mtx @ r_mv).astype(np.float32)
        cam2world = np.linalg.inv(r_mv).astype(np.float32)
        campos = cam2world[:3, 3].astype(np.float32)

        cams.append({
            "mvp": torch.from_numpy(mvp).unsqueeze(0).to(device),
            "cam2world": torch.from_numpy(cam2world).unsqueeze(0).to(device),
            "campos": torch.from_numpy(campos).unsqueeze(0).to(device),
            "lightpos": torch.from_numpy(campos.copy()).unsqueeze(0).to(device),
            "bkgs": torch.ones(1, H, W, 3, dtype=torch.float32, device=device),
            "azim": torch.tensor([math.radians(azim_deg)], dtype=torch.float32, device=device),
            "elev": torch.tensor([math.radians(elev_deg)], dtype=torch.float32, device=device),
            "fov": torch.tensor([0.0], dtype=torch.float32, device=device),
        })

    return cams
