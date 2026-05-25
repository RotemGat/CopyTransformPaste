import pathlib
import shutil
import traceback
from math import ceil
import torch
import torch.nn.functional as F
from PIL import Image
from easydict import EasyDict
from pymeshlab import MeshSet
from nvdiffmodeling.src import obj, mesh as nvmesh, texture as nvtexture
from pytorch3d.transforms import quaternion_to_matrix
import os
import numpy as np
from pathlib import Path
import glob

from utilities.calc_utils import compute_vertex_normals, quat_to_R_wxyz, unit_scale_from_obj_geometry


def create_meshes(config, tmp_dir):
    """
    Given a list of mesh-directory paths, each containing:
      - exactly one .obj
      - optionally a .mtl and some textures,
    copy them into workspace/tmp/, then load with nvdiffmodeling.obj.load_obj.

    Args:
        config: Configuration object containing workspace details.

    Returns:
        final_meshes (list): A list of processed nvdiffmodeling mesh objects.
    """
    device = config.device
    max_faces = get_max_faces(config)

    target_mesh, target_mesh_obj, target_sim = create_mesh_from_config(config, config.target_mesh, device, max_faces, tmp_dir)
    source_mesh, source_obj, source_sim = create_mesh_from_config(config, config.source_mesh, device, max_faces, tmp_dir)

    return source_mesh, source_obj, source_sim, target_mesh, target_mesh_obj, target_sim


def create_mesh_from_config(config, mesh_data, device, max_faces: int, tmp_dir):
    sim = sim3_identity(device=device)
    mesh, mesh_obj = create_mesh(mesh_dir=mesh_data.path, tmp_dir=tmp_dir,
                                 config=config, remesh=mesh_data.get('remesh', False),
                                 target_faces=max([max_faces, mesh_data.get('target_faces', 0)]))
    if 'init_pos_params' in mesh_data.keys():
        init_pos_params = mesh_data.init_pos_params
        vec_t = torch.tensor(init_pos_params['translation'], device=device, dtype=torch.float32)
        quat = torch.tensor(init_pos_params['rotation'], device=device)
        scale = torch.tensor(init_pos_params['scale'], device=device)
        with torch.no_grad():
            mesh = apply_differentiable_transform(mesh=mesh, translation_params=vec_t, rotation_params=quat, scale=scale)
            R = torch.from_numpy(quat_to_R_wxyz(quat.cpu().numpy())).to(device, dtype=torch.float32)
            delta = {"scale": scale, "rotation": R, "translation": vec_t}
            sim = sim3_compose(sim, delta)
    return mesh, mesh_obj, sim


def get_max_faces(config):
    try:
        target_path = extract_first_obj(pathlib.Path(config.target_mesh.path))
        dyn_path = extract_first_obj(pathlib.Path(config.source_mesh.path))

        target_tris = biggest_obj_tri_count(target_path)
        dyn_tris = biggest_obj_tri_count(dyn_path)

        meshes_faces = [target_tris, dyn_tris]
        max_faces = max(meshes_faces)
    except Exception as e:
        print(f'Failed to calculate max faces; fallback to 5000. Error: {e, traceback.format_exc()}')
        max_faces = 5000
    return max_faces


def obj_tri_count(path: str) -> int:
    """Return #triangles implied by polygon faces in an OBJ (no MTL load)."""
    tri = 0
    with open(path, 'r', errors='ignore') as f:
        for ln in f:
            if ln.startswith('f ') or ln.startswith('f\t'):
                # f v1 v2 v3 ... (tokens can be v / v/vt / v//vn / v/vt/vn)
                toks = [t for t in ln.split()[1:] if t and not t.startswith('#')]
                k = len(toks)
                if k >= 3:
                    tri += (k - 2)  # fan triangulation count
    return tri


def biggest_obj_tri_count(dir_or_file: str) -> int:
    """If given a directory, scan all *.obj and take the largest triangle count."""
    if os.path.isdir(dir_or_file):
        objs = glob.glob(os.path.join(dir_or_file, '*.obj'))
        if not objs:
            raise FileNotFoundError(f"No OBJ in {dir_or_file}")
        return max(obj_tri_count(p) for p in objs)
    else:
        return obj_tri_count(dir_or_file)


def remesh_with_pymeshlab(
        in_obj: str,
        target_faces: int = 4000,
        max_subdiv_iters: int = 3,
        preserve_boundary: bool = True,
        faces_tolerance: int = 1500
):
    """
    in_obj       – path to your source .obj
    out_obj      – path where the new .obj will be written
    refine_iters – how many Loop‐style (isotropic) remeshing passes you want
    target_faces – if set, quadric‐decimate down to approx. this many triangles
    """
    ms = MeshSet()
    ms.load_new_mesh(str(in_obj))

    # 1) get initial face count
    init_faces = ms.current_mesh().face_number()
    print(f"[remesh] starting with {init_faces} faces; target is {target_faces}")

    # 2) if we need more faces, subdivide
    if target_faces > init_faces:
        subdiv_pass = 0
        while subdiv_pass < max_subdiv_iters and ms.current_mesh().face_number() + faces_tolerance < target_faces:
            ms.meshing_isotropic_explicit_remeshing(iterations=1)
            subdiv_pass += 1
            print(f"[remesh] after subdiv #{subdiv_pass}: {ms.current_mesh().face_number()} faces")
        if ms.current_mesh().face_number() + faces_tolerance < target_faces:
            print(f"[remesh] Warning: max_subdiv_iters reached but still only have {ms.current_mesh().face_number()} faces")

    # 3) if we need fewer faces, decimate
    if target_faces < ms.current_mesh().face_number() and False:
        subdiv_pass = 0
        while subdiv_pass < max_subdiv_iters and ms.current_mesh().face_number() + faces_tolerance > target_faces:
            subdiv_pass += 1
            ms.apply_filter('compute_texcoord_transfer_wedge_to_vertex')
            print("✓ UVs transferred from wedges to vertices")

            # 2) Decimate geometry only (leave UVs alone for now)
            ms.apply_filter(
                'meshing_decimation_quadric_edge_collapse',
                targetfacenum=target_faces,
                preserveboundary=preserve_boundary,
                preservetopology=True,
                optimalplacement=True,
                planarquadric=True
            )
            print(f"✓ Decimated to {ms.current_mesh().face_number()} faces")

            # 3) Push per-vertex UVs → wedge UVs
        ms.apply_filter('compute_texcoord_transfer_vertex_to_wedge')
        print("✓ UVs transferred back from vertices to wedges")

    else:
        print("[remesh] already at target, no remeshing performed")

    # 3) Save out the new OBJ (along with its MTL + textures)
    ms.save_current_mesh(str(in_obj))
    return


def create_mesh(mesh_dir, tmp_dir=None, config=None, target_faces=5000, remesh=False, unit_size: bool = True):
    if not tmp_dir:
        tmp_dir = pathlib.Path(config.workspace) / 'tmp'
        tmp_dir.mkdir(parents=True, exist_ok=True)

    mesh_dir = pathlib.Path(mesh_dir)
    if not mesh_dir.is_dir():
        raise ValueError(f"{mesh_dir} is not a directory")

    # 1) Find the OBJ inside
    src_obj = extract_first_obj(mesh_dir)

    # 2) Copy OBJ to tmp
    tmp_obj = tmp_dir / src_obj.name
    shutil.copy(src_obj, tmp_obj)

    # 3) Read mtllib (if present) to copy .mtl and map_Kd textures
    mtl_name = None
    for line in open(src_obj, 'r'):
        if line.lower().startswith('mtllib '):
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                mtl_name = parts[1].strip()
            break
    if mtl_name:
        src_mtl = mesh_dir / mtl_name
        if src_mtl.exists():
            tmp_mtl = tmp_dir / mtl_name
            shutil.copy(src_mtl, tmp_mtl)

            # copy any map_Kd references
            for mtl_line in open(src_mtl, 'r'):
                if mtl_line.lower().startswith(("map_kd", "map_ks", "bump", "map_bump", "map_refl")):
                    tex_name = mtl_line.strip().split(maxsplit=1)[1]
                    src_tex = mesh_dir / tex_name
                    if src_tex.exists():
                        # convert to RGB in case the png has alpha (RGBA)
                        img = Image.open(src_tex).convert('RGB')
                        img.save(tmp_dir / tex_name)

    # 4) Remesh
    temp_mesh = obj.load_obj(str(tmp_obj))
    if len(temp_mesh.v_pos) < target_faces and remesh:
        remesh_with_pymeshlab(in_obj=tmp_obj, target_faces=target_faces)

    # 5) Load with nvdiffmodeling (it will pick up MTL + textures automatically)
    mesh_nd = obj.load_obj(str(tmp_obj))

    # 6) unit size
    if unit_size:
        mesh_nd = nvmesh.unit_size(mesh_nd)

    print(f'Created mesh {mesh_dir} with {mesh_nd.v_pos.shape[0]} vertices and {mesh_nd.t_pos_idx.shape[0]} faces.')
    mesh_nd.v_nrm = compute_vertex_normals(mesh_nd.v_pos, mesh_nd.t_pos_idx)
    return mesh_nd, tmp_obj


def extract_first_obj(mesh_dir):
    obj_list = list(mesh_dir.glob('*.obj'))
    if not obj_list:
        raise FileNotFoundError(f"No .obj file found in {mesh_dir}")
    if len(obj_list) > 1:
        print(f"Warning: multiple OBJs in {mesh_dir}, taking the first: {obj_list}")
    src_obj = obj_list[0]
    return src_obj


def apply_differentiable_transform(mesh, rotation_params: torch.Tensor = None, translation_params: torch.Tensor = None,
                                   scale: torch.Tensor = None) -> nvmesh.Mesh:
    """
    Apply a differentiable scale, rotation, and translation to mesh vertices.

    Args:
        mesh (nvdiffmodeling mesh): original mesh
        rotation_params (torch.Tensor[4]): quaternions
        translation_params (torch.Tensor[3]): translation vector (tx, ty, tz)
        scale (torch.Tensor[1]): a learnable uniform scale factor

    Returns:
        nvmesh.Mesh: a new mesh with transformed vertices
    """
    device = mesh.v_pos.device
    rotation_params = torch.tensor((1.0, 0.0, 0.0, 0.0), device=device) if rotation_params is None else rotation_params
    translation_params = torch.tensor((0.0, 0.0, 0.0), device=device) if translation_params is None else translation_params

    scale = torch.tensor(1.0, device=device) if scale is None else scale

    # 1) build rotation from quaternion
    R = quaternion_to_matrix(rotation_params)  # [3×3]

    # 2) apply scale (Tensor), rotation and translation
    V = mesh.v_pos  # [N,3]
    Vt = scale * (V @ R.T) + translation_params  # broadcasting scale

    # 3) clone mesh and set new positions
    new_mesh = mesh.clone()
    new_mesh.v_pos = Vt
    return new_mesh


def _merge_attr_idx(a, b, a_idx, b_idx, scale_a=1.0, scale_b=1.0, add_a=0.0, add_b=0.0):
    if a is None and b is None:
        return None, None
    elif a is not None and b is None:
        return (a * scale_a) + add_a, a_idx
    elif a is None and b is not None:
        return (b * scale_b) + add_b, b_idx
    else:
        return torch.cat(((a * scale_a) + add_a, (b * scale_b) + add_b), dim=0), torch.cat((a_idx, b_idx + a.shape[0]), dim=0)


def _resize_tex(data: torch.Tensor, target: int) -> torch.Tensor:
    """
    data: [B, H, W, C]
    returns: [B, target, target, C], via bilinear interpolation
    """
    B, H, W, C = data.shape
    if H == target and W == target:
        return data
    # permute to [B, C, H, W]
    x = data.permute(0, 3, 1, 2)
    # resize
    y = F.interpolate(x, size=(target, target), mode='bilinear', align_corners=False)
    # back to [B, target, target, C]
    return y.permute(0, 2, 3, 1).contiguous()


def create_scene(meshes, sz=1024):
    scene = nvmesh.Mesh()

    tot = len(meshes) if len(meshes) % 2 == 0 else len(meshes) + 1
    nx = 2
    ny = ceil(tot / 2) if ceil(tot / 2) % 2 == 0 else ceil(tot / 2) + 1

    w = int(sz * ny)
    h = int(sz * nx)
    dev = meshes[0].v_pos.device

    kd_atlas = torch.ones((1, w, h, 4), device=dev)
    ks_atlas = torch.zeros((1, w, h, 3), device=dev)
    kn_atlas = torch.ones((1, w, h, 3), device=dev)

    for i, m in enumerate(meshes):
        # 1) Geometry merging (unchanged)
        v_pos, t_pos_idx = _merge_attr_idx(scene.v_pos, m.v_pos, scene.t_pos_idx, m.t_pos_idx)
        v_nrm, t_nrm_idx = _merge_attr_idx(scene.v_nrm, m.v_nrm, scene.t_nrm_idx, m.t_nrm_idx)
        v_tng, t_tng_idx = _merge_attr_idx(scene.v_tng, m.v_tng, scene.t_tng_idx, m.t_tng_idx)

        pos_x = i % nx
        pos_y = i // ny

        sc_x = 1.0 / nx
        sc_y = 1.0 / ny

        v_tex, t_tex_idx = _merge_attr_idx(
            scene.v_tex,
            m.v_tex,
            scene.t_tex_idx,
            m.t_tex_idx,
            scale_a=1.0,
            scale_b=torch.tensor([sc_x, sc_y], device=dev),
            add_a=0.0,
            add_b=torch.tensor([sc_x * pos_x, sc_y * pos_y], device=dev)
        )

        # 2) Resize each texture to (1, sz, sz, C)
        kd = _resize_tex(m.material['kd'].data, sz)  # [1,sz,sz,4]
        ks = _resize_tex(m.material['ks'].data, sz)  # [1,sz,sz,3]
        nm = _resize_tex(m.material['normal'].data, sz)  # [1,sz,sz,3]

        # 3) Stamp into atlas
        y0, y1 = pos_y * sz, (pos_y + 1) * sz
        x0, x1 = pos_x * sz, (pos_x + 1) * sz

        kd_atlas[:, y0:y1, x0:x1, :kd.shape[-1]] = kd[0]
        ks_atlas[:, y0:y1, x0:x1, :ks.shape[-1]] = ks[0]
        kn_atlas[:, y0:y1, x0:x1, :nm.shape[-1]] = nm[0]

        # 4) Rebuild the merged scene mesh
        scene = nvmesh.Mesh(
            v_pos=v_pos,
            t_pos_idx=t_pos_idx,
            v_nrm=v_nrm,
            t_nrm_idx=t_nrm_idx,
            v_tng=v_tng,
            t_tng_idx=t_tng_idx,
            v_tex=v_tex,
            t_tex_idx=t_tex_idx,
            base=scene
        )

    # 5) Final scene with the atlases
    scene = nvmesh.Mesh(
        material={
            'bsdf': 'diffuse',
            'kd': nvtexture.Texture2D(kd_atlas),
            'ks': nvtexture.Texture2D(ks_atlas),
            'normal': nvtexture.Texture2D(kn_atlas),
        },
        base=scene
    )

    return scene


def create_ground(mesh_dir, alt_by_mesh: bool = True, meshes: list[nvmesh] = None, tmp_dir=None, config=None, alt=0.0, scale=1.0):
    """
    Creates a ground mesh using the existing `create_mesh` function and adjusts its scale and altitude.

    Args:
        mesh_dir (str): Directory where the mesh files are located.
        tmp_dir : Temporary directory for storing intermediate files.
        alt_by_mesh (bool) - indicating if the alt is calculated by the meshes height or by alt attribute
        meshes ([nvdiffmodeling.Mesh]): The meshes to use for determining the ground altitude.
        idx (int): Index to determine if scale should be applied.
        config: Configuration object that contains workspace paths.
        alt (float): The desired altitude for the ground (z-value).
        scale (float): The scale factor to apply to the ground mesh.

    Returns:
        mesh_nd (nvdiffmodeling.Mesh): The ground mesh with updated scale and altitude.
    """

    # Create the base mesh using the existing function
    mesh_nd, mesh_obj = create_mesh(mesh_dir, tmp_dir=tmp_dir, unit_size=False, config=config, remesh=False)

    # Apply the scale to the mesh
    if scale != 1.0:
        mesh_nd.v_pos *= scale  # Scale the vertices by the given scale factor

    # Adjust the height (altitude) by modifying the y-component of the vertices
    if alt_by_mesh:
        modify_ground_alt_by_meshes(mesh_nd, meshes)

    else:
        min_height = mesh_nd.v_pos[:, 1].min()  # Get the current lowest z value
        altitude_shift = alt - min_height  # Calculate how much to shift the mesh in z direction
        torch.tensor([0.0, -altitude_shift, 0.0], device=mesh_nd.v_pos.device)  # Shift the mesh's vertices to the new altitude

    return mesh_nd


def modify_ground_alt_by_meshes(ground, meshes: list, offset: float = 0.01, up_axis: int = 1):
    # exclude the ground itself if accidentally included
    lowest_vals = [float(m.v_pos[:, up_axis].min().item()) for m in meshes]
    lowest_alt = min(lowest_vals)

    desired_ground_min = lowest_alt - float(offset)
    current_ground_min = float(ground.v_pos[:, up_axis].max().item())
    delta = desired_ground_min - current_ground_min
    ground.v_pos = ground.v_pos + torch.tensor([0.0, delta, 0.0], device=ground.v_pos.device, dtype=ground.v_pos.dtype)


def apply_sim(v, s=1.0, R=None, t=None):
    out = v
    if R is not None:
        out = R @ out
    if s is not None:
        out = s * out
    if t is not None:
        out = out + t
    return out


def _read_obj_vertices(obj_path):
    V = []
    with open(obj_path, 'r') as f:
        for ln in f:
            if ln.startswith('v '):
                _, x, y, z = ln.split()[:4]
                V.append([float(x), float(y), float(z)])
    if not V:
        raise ValueError(f"No vertex positions in {obj_path}")
    V = np.asarray(V, dtype=np.float64)
    return V, V.min(axis=0), V.max(axis=0)


def _rewrite_obj_vertices(obj_in, obj_out, transform_fn, rotate_normals_R=None):
    lines = open(obj_in, 'r').read().splitlines(True)
    out = []
    for ln in lines:
        if ln.startswith('v '):
            _, x, y, z = ln.split()[:4]
            v = np.array([float(x), float(y), float(z)], dtype=np.float64)
            v2 = transform_fn(v)
            out.append(f"v {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
        elif ln.startswith('vn ') and rotate_normals_R is not None:
            _, x, y, z = ln.split()[:4]
            n = np.array([float(x), float(y), float(z)], dtype=np.float64)
            n2 = rotate_normals_R @ n
            n2 /= (np.linalg.norm(n2) + 1e-12)
            out.append(f"vn {n2[0]:.6f} {n2[1]:.6f} {n2[2]:.6f}\n")
        else:
            out.append(ln)
    Path(os.path.dirname(obj_out)).mkdir(parents=True, exist_ok=True)
    with open(obj_out, 'w') as f:
        f.writelines(out)


def copy_mtl_and_textures(src_obj, dst_obj):
    src_dir, dst_dir = os.path.dirname(src_obj), os.path.dirname(dst_obj)
    mtl_names = []
    for ln in open(src_obj, 'r'):
        if ln.lower().startswith('mtllib '):
            mtl_names.append(ln.split(maxsplit=1)[1].strip())
    for mtl in mtl_names:
        sm = os.path.join(src_dir, mtl)
        if not os.path.exists(sm):
            continue
        dm = os.path.join(dst_dir, os.path.basename(mtl))
        if os.path.abspath(sm) != os.path.abspath(dm):
            shutil.copy(sm, dm)
        for ml in open(sm, 'r'):
            low = ml.strip().lower()
            if low.startswith(('map_kd', 'map_ks', 'bump', 'map_ns', 'map_d', 'map_bump', 'refl', 'map_refl')):
                tex = ml.split(maxsplit=1)[1].strip()
                st = os.path.join(src_dir, tex)
                if os.path.exists(st):
                    dt = os.path.join(dst_dir, os.path.basename(tex))
                    if os.path.abspath(st) != os.path.abspath(dt):
                        Path(os.path.dirname(dt)).mkdir(parents=True, exist_ok=True)
                        shutil.copy(st, dt)


def _parse_face_token(tok):
    """
    Parse one token in an OBJ face element like:
      "12", "12/34", "12//56", "12/34/56"
    Returns a tuple of ints or None: (v_idx, vt_idx, vn_idx)
    """
    parts = tok.split('/')
    if len(parts) == 1:
        return int(parts[0]) if parts[0] != '' else None, None, None
    if len(parts) == 2:
        v = int(parts[0]) if parts[0] != '' else None
        vt = int(parts[1]) if parts[1] != '' else None
        return (v, vt, None)
    if len(parts) == 3:
        v = int(parts[0]) if parts[0] != '' else None
        vt = int(parts[1]) if parts[1] != '' else None
        vn = int(parts[2]) if parts[2] != '' else None
        return v, vt, vn
    raise ValueError(f"Unsupported face token: {tok}")


def _format_face_token(v, vt, vn):
    """Format components (v,vt,vn) back to a face token string (handles None)."""
    if vt is None and vn is None:
        return f"{v}"
    if vn is None:
        return f"{v}/{vt if vt is not None else ''}"
    return f"{v}/{vt if vt is not None else ''}/{vn if vn is not None else ''}"


def _adjust_face_line(line, v_off, vt_off, vn_off):
    """
    Adjust indices in an OBJ face line by adding vertex/tex/normal offsets.
    Expects a string that starts with "f ".
    """
    toks = line.strip().split()
    assert toks[0] == 'f'
    out_toks = ['f']
    for tok in toks[1:]:
        v_idx, vt_idx, vn_idx = _parse_face_token(tok)
        if v_idx is not None:
            v_idx = v_idx + v_off
        if vt_idx is not None:
            vt_idx = vt_idx + vt_off
        if vn_idx is not None:
            vn_idx = vn_idx + vn_off
        out_toks.append(_format_face_token(v_idx, vt_idx, vn_idx))
    return ' '.join(out_toks) + '\n'


def merge_obj_files_with_merged_mtl(obj_a_path: str, obj_b_path: str, out_dir: str, merged_basename: str = None):
    """
    Merge two OBJ files A then B into a single OBJ + single merged MTL + copied textures.

    - obj_a_path, obj_b_path: input .obj files (already have their .mtl and textures copied into out_dir by _copy_mtl_and_textures)
    - out_dir: directory where merged files will be written (and where the individual MTLs/textures already live)
    - merged_basename: base name for merged outputs; defaults to '<Astem>_<Bstem>_merged'
    Returns: path to merged OBJ
    """
    out_dir = str(out_dir)
    out_dir = Path(out_dir) / 'merged'
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if merged_basename is None:
        merged_basename = f"{Path(obj_a_path).stem}_{Path(obj_b_path).stem}_merged"
    merged_obj_path = os.path.join(out_dir, merged_basename + '.obj')
    merged_mtl_name = merged_basename + '.mtl'
    merged_mtl_path = os.path.join(out_dir, merged_mtl_name)

    # read obj files
    with open(obj_a_path, 'r', encoding='utf-8') as f:
        lines_a = f.readlines()
    with open(obj_b_path, 'r', encoding='utf-8') as f:
        lines_b = f.readlines()

    # helper: find mtllib filenames referenced by an OBJ (take basenames)
    def find_mtllibs(lines):
        libs = []
        for L in lines:
            if L.strip().lower().startswith('mtllib '):
                parts = L.strip().split(maxsplit=1)
                if len(parts) == 2:
                    libs.append(parts[1].strip())
        return libs

    libs_a = find_mtllibs(lines_a)
    libs_b = find_mtllibs(lines_b)

    # process each referenced mtl: read it from out_dir (we expect _copy_mtl_and_textures placed them there)
    merged_mtl_lines = []
    map_a = {}  # old_name -> new_name (for materials from A)
    map_b = {}  # old_name -> new_name (for materials from B)

    def process_single_mtl(libname, prefix, mapping_out):
        """
        Read the MTL file named `libname` expected to be in out_dir,
        prefix material names with prefix, copy referenced textures to out_dir with prefix,
        and return modified lines; also populate mapping_out[old] = new.
        """
        src_mtl_path = os.path.join(out_dir, libname)
        if not os.path.isfile(src_mtl_path):
            # try to locate in same dir as obj files (fallback)
            # check next to obj_a_path and obj_b_path
            cand_a = os.path.join(os.path.dirname(obj_a_path), libname)
            cand_b = os.path.join(os.path.dirname(obj_b_path), libname)
            if os.path.isfile(cand_a):
                src_mtl_path = cand_a
            elif os.path.isfile(cand_b):
                src_mtl_path = cand_b
            else:
                print(f"[merge] warning: unable to find mtllib {libname} in out_dir or obj dirs; skipping")
                return []

        mod_lines = []
        dir_of_mtl = os.path.dirname(src_mtl_path) or '.'

        for L in open(src_mtl_path, 'r', encoding='utf-8'):
            s = L.strip()
            if s.lower().startswith('newmtl '):
                old = s.split(None, 1)[1]
                new = f"{prefix}{old}"
                mapping_out[old] = new
                mod_lines.append(f"newmtl {new}\n")
            elif any(s.lower().startswith(k) for k in ('map_kd', 'map_ks', 'map_bump', 'bump', 'map_d', 'map_refl', 'map_ns', 'map_ka')):
                # keep any options and take last token as filename
                parts = s.split()
                key = parts[0]
                if len(parts) >= 2:
                    texname = parts[-1]
                    # source texture likely already copied into out_dir by _copy_mtl_and_textures (basename),
                    # so try to find it there first; otherwise look relative to mtl path.
                    candidate = os.path.join(out_dir, texname)
                    if not os.path.isfile(candidate):
                        candidate = os.path.join(dir_of_mtl, texname)
                    if os.path.isfile(candidate):
                        new_basename = f"{prefix}{os.path.basename(texname)}"
                        dst_path = os.path.join(out_dir, new_basename)
                        # copy if not same path
                        try:
                            if os.path.abspath(candidate) != os.path.abspath(dst_path):
                                shutil.copy(candidate, dst_path)
                        except Exception as e:
                            print(f"[merge] warning: failed copying texture {candidate} -> {dst_path}: {e}")
                            new_basename = texname  # fallback keep original
                        # reconstruct line preserving options (everything except last token)
                        leading = " ".join(parts[1:-1])
                        if leading:
                            mod_lines.append(f"{key} {leading} {new_basename}\n")
                        else:
                            mod_lines.append(f"{key} {new_basename}\n")
                    else:
                        # texture not found: keep original line but warn
                        print(f"[merge] warning: texture {texname} referenced in {src_mtl_path} not found; leaving as-is")
                        mod_lines.append(L)
                else:
                    mod_lines.append(L)
            else:
                mod_lines.append(L)
        return mod_lines

    # process all A libs with prefix 'A_'
    for lib in libs_a:
        merged_mtl_lines.extend(process_single_mtl(lib, 'A_', map_a))
    # process all B libs with prefix 'B_'
    for lib in libs_b:
        merged_mtl_lines.extend(process_single_mtl(lib, 'B_', map_b))

    # ---- split obj content into v/vt/vn and other blocks (we reuse logic similar to merge_obj_files) ----
    v_a, vt_a, vn_a, other_a = [], [], [], []
    v_b, vt_b, vn_b, other_b = [], [], [], []
    for L in lines_a:
        if L.startswith('v '):
            v_a.append(L)
        elif L.startswith('vt '):
            vt_a.append(L)
        elif L.startswith('vn '):
            vn_a.append(L)
        elif L.lower().startswith('mtllib'):
            continue
        else:
            # adjust usemtl token if present (use map_a)
            if L.strip().lower().startswith('usemtl '):
                old = L.strip().split(maxsplit=1)[1]
                new = map_a.get(old, old)
                other_a.append(f"usemtl {new}\n")
            else:
                other_a.append(L)

    for L in lines_b:
        if L.startswith('v '):
            v_b.append(L)
        elif L.startswith('vt '):
            vt_b.append(L)
        elif L.startswith('vn '):
            vn_b.append(L)
        elif L.lower().startswith('mtllib'):
            continue
        else:
            # adjust usemtl token if present (use map_b)
            if L.strip().lower().startswith('usemtl '):
                old = L.strip().split(maxsplit=1)[1]
                new = map_b.get(old, old)
                other_b.append(f"usemtl {new}\n")
            else:
                other_b.append(L)

    # counts for offsets
    nv_a = len(v_a)
    nvt_a = len(vt_a)
    nvn_a = len(vn_a)

    # write merged.mtl then merged.obj
    try:
        with open(merged_mtl_path, 'w', encoding='utf-8') as mf:
            mf.writelines(merged_mtl_lines)
    except Exception as e:
        print(f"[merge] error writing merged mtl {merged_mtl_path}: {e}")

    with open(merged_obj_path, 'w', encoding='utf-8') as out:
        out.write(f"mtllib {merged_mtl_name}\n\n")
        # vertices & texcoords & normals
        for L in v_a:
            out.write(L)
        for L in v_b:
            out.write(L)
        for L in vt_a:
            out.write(L)
        for L in vt_b:
            out.write(L)
        for L in vn_a:
            out.write(L)
        for L in vn_b:
            out.write(L)

        out.write("\n# faces and groups from A\n")
        for L in other_a:
            out.write(L)

        out.write("\n# faces and groups from B (indices adjusted)\n")
        for L in other_b:
            if L.startswith('f '):
                new_line = _adjust_face_line(L, v_off=nv_a, vt_off=nvt_a, vn_off=nvn_a)
                out.write(new_line)
            else:
                out.write(L)

    print(f"[merge] wrote merged obj: {merged_obj_path} and mtl: {merged_mtl_path}")
    return merged_obj_path


def export_meshes(
        source_obj_in: str,
        target_obj_in: str,
        out_dir: str,
        dyn_full_params: dict = None,
        fix_full_params: dict = None,
):
    """
    Simple export:

    1) source → unit size
    2) target   → unit size
    3) source → apply dyn_full_params Sim3 (optional)
    4) target   → apply fix_full_params Sim3 (optional)
    5) merge into one OBJ+MTL

    dyn_full_params / fix_full_params:
        {'scale': float, 'rotation': 3x3 np.array, 'translation': [tx,ty,tz]}
    """
    out_dir = str(out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # --- unit-size source ---
    _, dmin, dmax = _read_obj_vertices(source_obj_in)
    c_d = 0.5 * (dmax + dmin)
    s_d = 2.0 / float(np.max(dmax - dmin))

    if dyn_full_params is None:
        dyn_full_params = {"scale": 1.0, "rotation": np.eye(3), "translation": [0.0, 0.0, 0.0]}

    s_dyn = float(dyn_full_params["scale"])
    R_dyn = np.asarray(dyn_full_params["rotation"], dtype=np.float64)  # 3x3
    t_dyn = np.asarray(dyn_full_params["translation"], dtype=np.float64)

    def T_source(v):
        # v: (3,) numpy
        v0 = v - c_d
        v1 = apply_sim(v0, s=s_d)  # unit-size
        v2 = apply_sim(v1, s=s_dyn, R=R_dyn, t=t_dyn)
        return v2

    # normals: only the rotation, no scale / translation
    R_dyn_total = R_dyn

    dyn_out = os.path.join(out_dir, Path(source_obj_in).name.replace('.obj', '_source.obj'))
    _rewrite_obj_vertices(source_obj_in, dyn_out, T_source, rotate_normals_R=R_dyn_total)
    copy_mtl_and_textures(source_obj_in, dyn_out)

    # --- unit-size target ---
    _, fmin, fmax = _read_obj_vertices(target_obj_in)
    c_f = 0.5 * (fmax + fmin)
    s_f = 2.0 / float(np.max(fmax - fmin))

    if fix_full_params is None:
        fix_full_params = {"scale": 1.0, "rotation": np.eye(3), "translation": [0.0, 0.0, 0.0]}

    s_fix = float(fix_full_params["scale"])
    R_fix = np.asarray(fix_full_params["rotation"], dtype=np.float64)
    t_fix = np.asarray(fix_full_params["translation"], dtype=np.float64)

    def T_target(v):
        v0 = v - c_f
        v1 = apply_sim(v0, s=s_f)
        v2 = apply_sim(v1, s=s_fix, R=R_fix, t=t_fix)
        return v2

    fix_out = os.path.join(out_dir, Path(target_obj_in).name.replace('.obj', '_target.obj'))
    _rewrite_obj_vertices(target_obj_in, fix_out, T_target, rotate_normals_R=R_fix)
    copy_mtl_and_textures(target_obj_in, fix_out)

    merged_path = merge_obj_files_with_merged_mtl(
        dyn_out,
        fix_out,
        out_dir,
        merged_basename=f"{Path(source_obj_in).stem}_{Path(target_obj_in).stem}_merged"
    )

    print(f'Exported meshes successfully. merged: {merged_path}')
    return


def ensure_scale_in_init(cfg: EasyDict):
    fm, dm = cfg.target_mesh.path, cfg.source_mesh.path
    if not (fm and dm):
        return

    if (cfg.source_mesh.get('init_pos_params') and cfg.source_mesh.init_pos_params.get('scale') is None) and not cfg.source_mesh.get('auto_scale', True):
        cfg.source_mesh.init_pos_params.scale = 1.0

    # don’t override if user already provided a scale
    if (cfg.source_mesh.get('init_pos_params') and cfg.source_mesh.init_pos_params.get('scale') is not None) or not cfg.source_mesh.get('auto_scale', True):
        return

    s_d = unit_scale_from_obj_geometry(extract_first_obj(pathlib.Path(dm)))
    s_f = unit_scale_from_obj_geometry(extract_first_obj(pathlib.Path(fm)))
    sf = s_f / s_d
    print(f"[scale] s_f={s_f:.6g}, s_d={s_d:.6g} -> init scale={sf:.6g}")
    if not cfg.source_mesh.get("init_pos_params"):
        cfg.source_mesh.init_pos_params = EasyDict()
    cfg.source_mesh.init_pos_params.scale = sf


# ---------- Sim3 helpers (scale, rotation, translation) ----------
def sim3_identity(device):
    dtype = torch.float32
    return {
        "scale": torch.tensor(1.0, device=device, dtype=dtype),
        "rotation": torch.eye(3, device=device, dtype=dtype),
        "translation": torch.zeros(3, device=device, dtype=dtype),
    }


def sim3_compose(base, delta):
    s1, R1, t1 = base["scale"], base["rotation"], base["translation"]
    s2, R2, t2 = delta["scale"], delta["rotation"], delta["translation"]

    # Force everything to the dtype of R1 (usually float32)
    dtype = R1.dtype
    device = R1.device
    R2 = R2.to(device=device, dtype=dtype)
    s2 = s2.to(device=device, dtype=dtype)
    t1 = t1.to(device=device, dtype=dtype)
    t2 = t2.to(device=device, dtype=dtype)
    s1 = s1.to(device=device, dtype=dtype)

    s = s2 * s1
    R = R2 @ R1
    t = s2 * (R2 @ t1) + t2

    return {"scale": s, "rotation": R, "translation": t}


def sim3_to_export(sim3: dict):
    """
    Convert Sim3 dict to pure Python for export_meshes.
    rotation is kept as a 3x3 matrix (we'll handle it there).
    """
    s = float(sim3["scale"].detach().cpu())
    R = sim3["rotation"].detach().cpu().numpy()
    t = [float(x) for x in sim3["translation"].detach().cpu()]
    return {"scale": s, "rotation": R, "translation": t}
