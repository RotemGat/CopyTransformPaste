import numpy as np
import torch
from nvdiffmodeling.src import mesh as nvmesh
from pytorch3d.transforms import matrix_to_quaternion

# This code originated from https://github.com/cube-c/Auto-Align

# Hyperparameters
ITERATION_RANSAC = 200
ITERATION_MEDIAN = 10
THRESHOLD = 5 * (np.pi / 180)
MAX_POLYS = 10000
MAX_POLYS_SUBSET = 100
SYMMETRY_PAIR_DIST = 0.03
SYMMETRY_BUCKET_SIZE = 0.1


def get_symmetry_plane(normals, positions):
    # Resample if too many vertices
    if normals.shape[0] > MAX_POLYS:
        indices = np.random.choice(
            normals.shape[0], MAX_POLYS, replace=False)
        normals = normals[indices]
        positions = positions[indices]

    if normals.shape[0] > MAX_POLYS_SUBSET:
        indices = np.random.choice(
            normals.shape[0], MAX_POLYS_SUBSET, replace=False)
        normals_subset = normals[indices]
        positions_subset = positions[indices]
    else:
        normals_subset = normals
        positions_subset = positions

    # Extract vertex pairs that satisfy symmetry condition
    positions_1 = np.tile(positions, (normals_subset.shape[0], 1))
    positions_2 = np.repeat(positions_subset, normals.shape[0], axis=0)
    normals_1 = np.tile(normals, (normals_subset.shape[0], 1))
    normals_2 = np.repeat(normals_subset, normals.shape[0], axis=0)
    plane_normals = positions_1 - positions_2
    plane_normals_scale = np.linalg.norm(plane_normals, axis=1)
    plane_normals = plane_normals / (plane_normals_scale + 1e-6).reshape(-1, 1)
    normals_3 = normals_1 - 2 * plane_normals * \
                np.sum(plane_normals * normals_1, axis=1).reshape(-1, 1)

    indices = np.nonzero((np.linalg.norm(normals_2 - normals_3, axis=1)
                          < SYMMETRY_PAIR_DIST) & (plane_normals_scale > 1e-6))[0]
    plane_normals = plane_normals[indices]
    plane_centers = np.sum((positions_1 + positions_2)
                           [indices] / 2 * plane_normals, axis=1)

    plane = np.concatenate(
        (plane_normals, plane_centers.reshape(-1, 1)), axis=1)
    plane = np.concatenate((plane, -plane), axis=0)
    plane_centers_std = np.std(plane[:, 3])
    plane[:, 3] = plane[:, 3] / (plane_centers_std + 1e-6)

    # Voting
    plane_int = np.rint(plane / SYMMETRY_BUCKET_SIZE).astype(np.int)
    plane_range = np.max(plane_int, axis=0) - np.min(plane_int, axis=0) + 1
    plane_int_hash = plane_int[:, 0] + plane_int[:, 1] * plane_range[0] \
                     + plane_int[:, 2] * plane_range[0] * plane_range[1] \
                     + plane_int[:, 3] * plane_range[0] * plane_range[1] * plane_range[2]
    value, count = np.unique(plane_int_hash, return_counts=True)
    origin = plane_int[(plane_int_hash == value[np.argmax(count)]).nonzero()[
        0][0]] * SYMMETRY_BUCKET_SIZE
    dist = np.linalg.norm(plane - origin.reshape(1, -1), axis=1)
    plane_res = np.median(
        plane[(dist < SYMMETRY_BUCKET_SIZE).nonzero()[0]], axis=0)
    plane_res[3] = plane_res[3] * (plane_centers_std + 1e-6)
    plane_res[:3] = plane_res[:3] / np.linalg.norm(plane_res[:3])

    return plane_res


def get_matrix(areas, normals, fixed_axis=None):
    # Resample if too many polygons
    if areas.size > MAX_POLYS:
        indices = np.random.choice(
            areas.size, MAX_POLYS, p=areas / sum(areas), replace=False)
        areas = areas[indices]
        normals = normals[indices]

    first_indices = np.random.choice(
        areas.size, ITERATION_RANSAC, p=areas / sum(areas))

    # RANSAC
    best_model = np.identity(3)
    best_value = -1.0

    for index in first_indices:
        model = np.zeros((3, 3))
        if fixed_axis is None:
            model[0] = normals[index]
        else:
            model[0] = fixed_axis
        next_indices = np.nonzero(
            np.abs(normals @ model[0]) < np.sin(THRESHOLD))[0]
        if next_indices.size > 0:
            next_areas = areas[next_indices]
            model[1] = normals[np.random.choice(
                next_indices, p=next_areas / sum(next_areas))]
        else:
            model[1] = np.zeros(3)
            model[1][(np.argmax(np.abs(model[0])) + 1) % 3] = 1

        model[1] = np.cross(model[0], model[1])
        model[1] = model[1] / np.linalg.norm(model[1])
        model[2] = np.cross(model[0], model[1])

        indices = np.max(np.abs(normals @ model.T), axis=1) > np.cos(THRESHOLD)
        value = np.sum(areas[indices])
        if best_value < value:
            best_value, best_model, best_indices = value, model, indices

    # Calculate median each axis, iteratively...
    areas = areas[best_indices]
    normals = normals[best_indices]
    axis = np.vstack((best_model, -best_model))
    axis_indices = np.argmax(normals @ axis.T, axis=1)
    normals_per_axis = []
    areas_per_axis = []
    xyz_axis = np.array([[[1, 2], [2, 4], [4, 5], [5, 1]], [[3, 2], [2, 0], [
        0, 5], [5, 3]], [[0, 1], [1, 3], [3, 4], [4, 0]]])
    for i in range(6):
        normals_per_axis.append(normals[axis_indices == i])
        areas_per_axis.append(areas[axis_indices == i])

    normals_area = []
    for i in range(3):
        normals_area.append(np.concatenate(
            [areas_per_axis[a] for (a, _) in xyz_axis[i]]))

    for _ in range(ITERATION_MEDIAN):
        for i in range(3):
            if fixed_axis is not None and i != 0:
                continue

            normals_proj = np.concatenate(
                [normals_per_axis[a] @ axis[b] for (a, b) in xyz_axis[i]])

            if normals_proj.size == 0:
                continue

            sort_indices = np.argsort(normals_proj)
            value = normals_proj[sort_indices]
            weight = normals_area[i][sort_indices]
            weight_cumsum = np.cumsum(weight)
            med_index = np.searchsorted(weight_cumsum, weight_cumsum[-1] / 2)

            c, s = np.cos(value[med_index]), np.sin(value[med_index])
            j, k = (i + 1) % 3, (i + 2) % 3

            transform = np.identity(3)
            transform[(j, j, k, k), (j, k, j, k)] = np.array([c, -s, s, c])
            best_model = transform.T @ best_model
            axis = np.vstack((best_model, -best_model))

    # Find minimal rotation matrix
    unit_rot = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    flip_rot = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    unit_diag = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])

    best_model_opt = best_model
    best_trace = 0
    rot = np.identity(3)
    for _ in range(3):
        rot = unit_rot @ rot
        for j in range(4):
            model_opt = np.diag(unit_diag[j]) @ rot @ best_model
            trace = np.trace(model_opt)
            if trace > best_trace:
                best_trace, best_model_opt = trace, model_opt

            model_opt = -np.diag(unit_diag[j]) @ flip_rot @ rot @ best_model
            trace = np.trace(model_opt)
            if trace > best_trace:
                best_trace, best_model_opt = trace, model_opt

    return best_model_opt


def compute_face_areas_and_normals(nvmesh: nvmesh):
    """
    Given an nvdiffmodeling.Mesh (nvmesh), return:
       - areas:  (F,) array of face areas
       - normals:(F,3) array of face normals in world coords
    """
    # v_pos: [N,3], faces: [F,3] indices
    V = nvmesh.v_pos.cpu().detach().numpy()  # (N,3)
    F = nvmesh.t_pos_idx.cpu().detach().numpy()  # (F,3)  (assuming f_pos holds face idxs)

    # grab each triangle
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    # compute normals via cross‐product
    e1 = v1 - v0
    e2 = v2 - v0
    face_normals = np.cross(e1, e2)
    # area = ½‖cross‖
    areas = 0.5 * np.linalg.norm(face_normals, axis=1)
    # normalize normals
    normals = face_normals / (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-12)

    return areas, normals


def auto_align_nvmesh(mesh: nvmesh.Mesh, symmetry: bool = False):
    """
    Returns a new Mesh whose v_pos has been rotated so that
    its principal axes align (via the get_matrix RANSAC/median pipeline).
    """
    # 1) extract areas and normals
    areas, normals = compute_face_areas_and_normals(mesh)

    # 2) optionally compute symmetry plane
    if symmetry:
        # if you pasted in get_symmetry_plane():
        verts = mesh.v_pos.cpu().numpy()
        vert_normals = mesh.v_normals.cpu().numpy()  # or compute per-vertex
        plane = get_symmetry_plane(vert_normals, verts)
        R3 = get_matrix(areas, normals, fixed_axis=plane[:3])
    else:
        R3 = get_matrix(areas, normals)

    # 3) apply R3 to your mesh.v_pos (torch)
    V = mesh.v_pos  # torch.Tensor [N,3]
    R_torch = torch.from_numpy(R3).to(V.device).to(V.dtype)

    V_new = (V @ R_torch.T)
    new_mesh = mesh.clone()
    new_mesh.v_pos = V_new

    # 4) return quaternion [w,x,y,z] (no translation since pivot=origin)
    q_align = matrix_to_quaternion(R_torch)  # shape (4,)

    return new_mesh, q_align
