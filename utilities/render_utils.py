import os
import random
import kornia
import numpy as np
import torch
from PIL import Image

from nvdiffmodeling.src import mesh, texture, render
from utilities.mesh_utils import create_scene
from utilities.resize_right import resize


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def render_meshes(meshes, data):
    """
    Combines both meshes into a scene and returns the final mesh for rendering.
    """
    render_mesh = create_scene(meshes=[mesh_obj.eval() for mesh_obj in meshes], sz=512)
    render_mesh = mesh.auto_normals(render_mesh)
    render_mesh = mesh.compute_tangents(render_mesh)

    return render_mesh.eval(data)


def create_final_mesh(bsdf, mesh_obj, textures):
    """
    Creates and returns a final mesh using the base mesh and processed textures.
    """
    final_mesh = mesh.Mesh(
        mesh_obj.v_pos,
        mesh_obj.t_pos_idx,
        material={
            'bsdf': bsdf,
            'kd': textures['kd'],
            'ks': textures['ks'],
            'normal': textures['normal'],
        },
        base=mesh_obj  # Get UVs from the original loaded mesh
    )
    return final_mesh


def process_textures(mesh_obj):
    """
    Wraps each material map in a try/except around the Gaussian blur,
    so that if the image is too small (or anything else goes wrong),
    we just use the original data.
    """
    KERNEL = (7, 7)
    SIGMA = (3, 3)

    out = {}
    for key in ('kd', 'ks', 'normal'):
        # get the Texture2D (or synthesize a default if it's missing)
        tex = mesh_obj.material.get(key)
        if tex is None:
            # create placeholder exactly at train_res
            H = W = mesh_obj.material['kd'].data.shape[1]  # assuming kd exists
            device = mesh_obj.material['kd'].data.device
            if key == 'kd':
                data = torch.ones((1, H, W, 3), device=device)
            elif key == 'ks':
                data = torch.zeros((1, H, W, 3), device=device)
            else:
                zeros = torch.zeros((1, H, W, 2), device=device)
                ones_z = torch.ones((1, H, W, 1), device=device)
                data = torch.cat([zeros, ones_z], dim=-1)
        else:
            data = tex.data

        # now attempt to blur it
        try:
            # permute to [B,C,H,W]
            x = data.permute(0, 3, 1, 2)
            y = kornia.filters.gaussian_blur2d(
                x, kernel_size=KERNEL, sigma=SIGMA
            )
            proc = y.permute(0, 2, 3, 1).contiguous()
        except RuntimeError as e:
            # if it's a padding‐too‐large error (or anything else), skip blur
            proc = data

        out[key] = texture.Texture2D(proc)

    return out


def render_image(bsdf, glctx, light_power, train_res, resize_method, meshes, data):
    """
    meshes : list[nvdiffmodeling.mesh.Mesh]
    returns   : torch.Tensor  (1, H, W, 3) in linear RGB
    """

    # Process textures for both meshes
    final_meshes = []
    for mesh_obj in meshes:
        textures = process_textures(mesh_obj)
        final_meshes.append(create_final_mesh(bsdf, mesh_obj, textures))

    # Render both meshes in the scene
    final_mesh = render_meshes(final_meshes, data)

    # Perform the actual rendering
    train_render = render.render_mesh(
        glctx,
        final_mesh,
        data['mvp'],
        data['campos'],
        data['lightpos'],
        light_power,
        train_res,
        spp=4,
        num_layers=1,
        msaa=True,
        background=data['bkgs'],
    ).permute(0, 3, 1, 2)

    # Resize the rendered output to the target resolution
    train_render = resize(train_render, out_shape=(224, 224), interp_method=resize_method)

    return train_render


def render_image_no_grad(bsdf, glctx, light_power, log_res, meshes, data, device):
    """
    meshes : list[nvdiffmodeling.mesh.Mesh]
    returns   : torch.Tensor  (1, H, W, 3) in linear RGB
    """
    with torch.no_grad():
        # Process textures for both meshes
        final_meshes = []
        for mesh_obj in meshes:
            textures = process_textures(mesh_obj)
            final_meshes.append(create_final_mesh(bsdf, mesh_obj, textures))

        # Render both meshes in the scene
        final_mesh = render_meshes(final_meshes, data)

        # Perform the actual rendering
        log_render = render.render_mesh(
            glctx,
            final_mesh,
            data['mvp'],
            data['campos'],
            data['lightpos'],
            light_power,
            log_res,
            spp=8,
            msaa=True,
            background=torch.ones(1, log_res, log_res, 3).to(device),
        )

    return log_render


def get_pil_image_from_tensor(image):
    if len(image.shape) == 4:
        image = image.squeeze(0)[..., :3].detach().cpu().numpy()
    else:
        image = image[..., :3].detach().cpu().numpy()
    image = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    # Convert image to PIL for drawing text
    pil_image = Image.fromarray(image)
    return image, pil_image
