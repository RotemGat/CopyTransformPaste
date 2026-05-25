import glob
import os
import pathlib
import time
import traceback
from pathlib import Path
from typing import Any, Union

from nvdiffmodeling.src import mesh as nvmesh
import torch
import torchvision
import tqdm
from PIL import Image
from torch import optim as optim
from torch.nn import Parameter
from torch.utils.data import DataLoader
import numpy as np

import nvdiffrast.torch as dr
from utilities.auto_align import auto_align_nvmesh
from utilities.camera import get_camera_params, get_next_camera_params, adapt_camera_distance, CameraBatch, \
    adapt_camera_for_phase, views_to_batched_camera_dict, generate_eval_cameras
from utilities.guidance import Guidance, GuidanceType
from utilities.llm import get_prior_scale, LLMSession, get_contact_ratio
from utilities.loss_utils import get_penetration_loss, get_fractional_soft_icp_loss, get_scaling_reg
from utilities.mesh_utils import create_meshes, apply_differentiable_transform, create_ground, export_meshes, \
    modify_ground_alt_by_meshes, sim3_compose, sim3_to_export
from utilities.calc_utils import scale_mesh_about_centroid, quat_to_R_wxyz
from utilities.render_utils import render_image, render_image_no_grad
from utilities.resize_right import cubic, linear, lanczos2, lanczos3
from utilities.stable_diffusion import StableDiffusion
from utilities.video import Video


class Trainer:
    def __init__(self,
                 name,  # name of this experiment
                 config,  # configuration
                 device=None,
                 ):
        self.config = config

        self._setup_workspace(name)

        self._set_initial_configs(device)

        self._initialize_guidance_model()

        self._initialize_optimizer_and_scheduler()

        self._initialize_renderer()

        self._init_meshes()

        self.stats = self._initialize_stats()

        # Log the information and load checkpoint if needed
        self._log(f'[INFO] Trainer: {self.name} | {self.device} | {self.workspace}')
        self._log(config)
        if self.workspace and config.run_checkpoint:
            self._init_checkpoint()

    # ----------------------------------- Initialization Methods ----------------------------------

    def _setup_workspace(self, name):
        self.workspace = self.config.workspace
        self.name = name
        os.makedirs(self.workspace, exist_ok=True)
        self.log_path = os.path.join(self.config.workspace, f"log_{self.name}.txt")
        self.log_ptr = open(self.log_path, "a+")
        self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
        self.best_path = f"{self.ckpt_path}/{self.name}.pth"

        self.video = Video(self.workspace)
        self.video_rotation_angle = 0.0
        os.makedirs(self.ckpt_path, exist_ok=True)

        tmp_dir = pathlib.Path(self.config.workspace) / 'tmp'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir = tmp_dir

    def _set_initial_configs(self, device):
        self.local_index = self.config.local_index
        self.eval_interval = self.config.eval_interval
        self.use_checkpoint = self.config.use_checkpoint
        self.device = device
        self.fp16 = self.config.use_fp16
        self.max_keep_ckpt = self.config.max_keep_ckpt
        self.epoch = 0
        self.max_epochs = self.config.epochs
        self.batch_size = self.config.batch_size
        self.text = self.config.text_prompt
        self.negative_text = self.config.negative_prompt
        self.ref_image = None
        self.icp_ratio = self.config.icp_ratio
        if self.config.image_path is not None and self.config.image_guidance:
            pil = Image.open(self.config.image_path).convert('RGB')
            arr = np.array(pil, dtype=np.float32) / 255.0  # H,W,3 float32 [0,1]
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
            self.ref_image = t.to(device)

    def _initialize_guidance_model(self):
        guidance_type = GuidanceType(self.config.guidance)
        if guidance_type == GuidanceType.StableDiffusion:
            self.guidance = StableDiffusion(self.device, hugging_face_token=self.config.token)
        else:
            self.guidance = Guidance(guidance_type=guidance_type, device=self.device)

        if self.config.init_scale_by_llm or self.config.init_icp_ratio_by_llm:
            self.llm = LLMSession(device=self.device)

    def _initialize_optimizer_and_scheduler(self, init_params=None):
        params = init_params if init_params else self._get_init_params()
        self.optimizer = optim.Adam(params)
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        self.scaler = torch.amp.GradScaler(enabled=self.fp16)

    def _initialize_renderer(self):
        self.glctx = dr.RasterizeCudaContext()  # CUDA context
        self.res = getattr(self.config, "render_res", 256)  # one square framebuffer size
        self.light_power = self.config.light_power
        self.train_res = self.config.train_res
        self.bsdf = self.config.bsdf

        if self.config.resize_method == 'cubic':
            self.resize_method = cubic
        elif self.config.resize_method == 'linear':
            self.resize_method = linear
        elif self.config.resize_method == 'lanczos2':
            self.resize_method = lanczos2
        elif self.config.resize_method == 'lanczos3':
            self.resize_method = lanczos3

    def _init_meshes(self):
        """ Creates target and source meshes of nvdiffmodeling """
        if not os.path.exists(self.config.source_mesh.path):
            raise ValueError("Could not find meshes in the specified paths.")

        self.mesh, self.mesh_obj, self.mesh_sim, self.target_mesh, self.target_mesh_obj, self.target_mesh_sim = create_meshes(
            self.config, self.tmp_dir)
        if self.config.use_ground:
            self.ground = create_ground(mesh_dir=self.config.ground_mesh, alt_by_mesh=True,
                                        meshes=[self.target_mesh, self.mesh], config=self.config, scale=5.0)
        return

    def _get_init_params(self):
        self.rotation_params = torch.nn.Parameter(torch.tensor((1.0, 0.0, 0.0, 0.0), device=self.device))
        self.translation_params = torch.nn.Parameter(torch.zeros(3, device=self.device))
        self.scaling_param = torch.nn.Parameter(torch.ones(1, device=self.device))
        return [
            {'params': [self.translation_params], 'lr': self.config.translation_lr},
            {'params': [self.rotation_params], 'lr': self.config.rotation_lr},
            {'params': [self.scaling_param], 'lr': self.config.scaling_lr}
        ]

    @staticmethod
    def _initialize_stats():
        return {
            "loss": [],
            "valid_loss": [],
            "results": [],
            "checkpoints": [],
            "best_result_loss": float('inf'),
            "best_result_params": {},
            "last_result_params": {},
        }

    def _init_checkpoint(self):
        if self.use_checkpoint == "scratch":
            self._log("[INFO] Training from scratch ...")
        else:
            checkpoint_dict = self._load_checkpoint()
            self.epoch = checkpoint_dict['epoch']
            self.stats = checkpoint_dict['stats']
            self._load_optimizer_and_scheduler(checkpoint_dict)

    def _load_checkpoint(self):
        checkpoint = None
        if self.use_checkpoint == "latest":
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
            self._log(f"[INFO] Loading checkpoint {checkpoint if checkpoint else 'none'}")
        elif self.use_checkpoint == "best":
            checkpoint = self.best_path if os.path.exists(self.best_path) else None
        elif self.use_checkpoint != "scratch":
            checkpoint = self.use_checkpoint

        if checkpoint:
            return torch.load(checkpoint, map_location=self.device)
        return {}

    def _load_optimizer_and_scheduler(self, checkpoint_dict):
        if 'optimizer' in checkpoint_dict:
            self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
            self._log("[INFO] Loaded optimizer.")
        if 'lr_scheduler' in checkpoint_dict:
            self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
            self._log("[INFO] Loaded scheduler.")
        if 'scaler' in checkpoint_dict:
            self.scaler.load_state_dict(checkpoint_dict['scaler'])
            self._log("[INFO] Loaded scaler.")

    def _save_checkpoint(self, name=None, full=False, best=False):
        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'stats': self.stats,
        }

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()

        if not best:
            file_path = f"{name}.pth"
            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = os.path.join(self.ckpt_path, self.stats["checkpoints"].pop(0))
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)
            torch.save(state, os.path.join(self.ckpt_path, file_path))

        else:
            torch.save(state, self.best_path)

    # ---------------------------- Training Methods --------------------------------------------
    def train(self, data_loader: DataLoader, cams_data: CameraBatch, max_epochs: int = None):
        start_t = time.time()
        self.max_epochs = max_epochs if max_epochs else self.max_epochs

        imdir = pathlib.Path(self.workspace) / 'images'
        imdir.mkdir(parents=True, exist_ok=True)

        if self.config.init_icp_ratio_by_llm:
            self.icp_ratio = get_contact_ratio(llm_session=self.llm, object1=self.config.source_mesh.name,
                                               object2=self.config.target_mesh.name,
                                               wanted_alignment=self.text)

        rotation, translation, scale = self.rotation_params, self.translation_params, self.scaling_param

        export_meshes(
            source_obj_in=self.mesh_obj,
            target_obj_in=self.target_mesh_obj,
            out_dir=str(self.tmp_dir / 'init_meshes'),
            dyn_full_params=sim3_to_export(self.mesh_sim),
            fix_full_params=sim3_to_export(self.target_mesh_sim),
        )

        n_phases = min(len(self.config.intersection_loss_w), len(self.config.icp_loss_w))
        for i_phase in range(n_phases):
            try:
                self._run_phase(cams_data, data_loader, i_phase, imdir, n_phases, rotation, scale, translation)
            except Exception as e:
                message = f"Could not run optimization due to an error: {e, traceback.format_exc()}"
                print(message)
                self._log(message)
                break

        end_t = time.time()
        self.video.close()
        self._log(f"[INFO] training takes {(end_t - start_t) / 60:.4f} minutes.")

        sim_best = {"scale": self.stats['best_result_params']['scale'].reshape(()),
                    "rotation": torch.from_numpy(
                        quat_to_R_wxyz(self.stats['best_result_params']['rotation'].cpu().numpy())).to(self.device,
                                                                                                       dtype=torch.float32),
                    "translation": self.stats['best_result_params']['translation'].float().to(self.device),
                    }

        sim_total = sim3_compose(self.mesh_sim, sim_best)

        export_meshes(
            source_obj_in=self.mesh_obj,
            target_obj_in=self.target_mesh_obj,
            out_dir=str(self.tmp_dir / 'final_meshes'),
            dyn_full_params=sim3_to_export(sim_total),
            fix_full_params=sim3_to_export(self.target_mesh_sim),
        )

        self._log(f'[INFO] Finished training: {self.name} | {self.device} | {self.workspace}')
        return self.stats['best_result_loss']

    def _run_phase(self, cams_data: CameraBatch, data_loader: DataLoader, i_phase: int, imdir: Path, n_phases: int,
                   rotation: Union[Parameter, Any],
                   scale: Union[Parameter, Any], translation: Union[Parameter, Any]):
        self.intersection_loss_w = self.config.intersection_loss_w[i_phase]
        self.icp_loss_w = self.config.icp_loss_w[i_phase]
        max_epochs = (self.max_epochs // n_phases) * (i_phase + 1)

        def as_param(x):
            return torch.nn.Parameter(x.detach().clone().to(self.device).float(), requires_grad=True)

        if i_phase != 0:
            if self.config.adapt_camera:
                adapt_camera_for_phase(cams_data=cams_data, dyn_mesh=self.mesh, fix_mesh=self.target_mesh,
                                       phase_idx=i_phase, n_phases=n_phases,
                                       base_dist_min=self.config.dist_min, base_dist_max=self.config.dist_max)

            best = self.stats['best_result_params']

            rotation, translation, scale = as_param(best['rotation']), as_param(best['translation']), as_param(
                best['scale'])
            self._log(
                f"Initializing phase {i_phase} with: translation - {translation}rotation - {rotation}, scale - {scale}")
            self.rotation_params, self.translation_params, self.scaling_param = rotation, translation, scale
            self.stats = self._initialize_stats()
            self._initialize_optimizer_and_scheduler([
                {'params': [self.translation_params], 'lr': self.config.translation_lr},
                {'params': [self.rotation_params], 'lr': self.config.rotation_lr},
                {'params': [self.scaling_param], 'lr': self.config.scaling_lr}
            ])
            auto_align = False
            init_scale = False
        else:
            auto_align = self.config.auto_align
            init_scale = self.config.scaling_lr > 0
        self._log(f"[INFO] Starting phase {i_phase + 1} of {n_phases} with max_epochs={max_epochs}, "
                  f"intersection_loss_w={self.intersection_loss_w} and icp_loss_w={self.icp_loss_w}.")

        return self.training_loop(data_loader=data_loader, imdir=imdir, mesh=self.mesh, rotation=rotation, scale=scale,
                                  translation=translation,
                                  max_epochs=max_epochs, auto_align=auto_align, name=f'phase_{i_phase}',
                                  init_scale=init_scale)

    def training_loop(self, data_loader, imdir, mesh, rotation, scale, translation, max_epochs, auto_align, name='',
                      init_scale: bool = False):
        self._evaluate_one_epoch(transformed_mesh=mesh, scale=scale, epoch='init')
        self._save_render_to_video(transformed_mesh=mesh)

        if auto_align:
            mesh, auto_align_params = auto_align_nvmesh(mesh)
            self._log(f"Auto align quaternion params: {auto_align_params}")
            self._save_render_to_video(transformed_mesh=mesh)
            self.mesh = mesh

            delta = {
                "scale": torch.tensor(1.0, device=self.device),
                "rotation": torch.from_numpy(quat_to_R_wxyz(list(auto_align_params.cpu()))).to(self.device),
                "translation": torch.zeros(3, device=self.device),
            }
            self.mesh_sim = sim3_compose(self.mesh_sim, delta)

        if init_scale:
            mesh = nvmesh.unit_size(mesh, center=False)
            if self.config.init_scale_by_llm:
                mesh = self._get_mesh_prior_scale(mesh)
            self.mesh = mesh
            self._save_render_to_video(transformed_mesh=mesh)

        while self.epoch < max_epochs:
            transformed_mesh, camera_params, loss, loss_summary = self._train_one_epoch(data_loader, mesh, rotation,
                                                                                        translation, scale, max_epochs)

            self._save_render_to_video(transformed_mesh=transformed_mesh, loss=loss)

            if self.workspace and self.local_index == 0:
                self._save_checkpoint(full=True)

            if self.epoch % self.eval_interval == 0:
                eval_loss, eval_img = self._evaluate_one_epoch(transformed_mesh=transformed_mesh, scale=scale)
                self._save_checkpoint(full=False, best=True)
                if eval_loss < self.stats['best_result_loss']:
                    self._save_best_result(imdir, loss, rotation, translation, scale, eval_img, name)

            self.epoch += 1

            # inject tiny noise
            self._inject_noise_to_params(self.lr_scheduler, self.optimizer)
        return self._evaluate_one_epoch(transformed_mesh=transformed_mesh, scale=scale, epoch='final')

    def _get_mesh_prior_scale(self, mesh) -> Any:
        prior_scale = get_prior_scale(llm_session=self.llm, object1=self.config.source_mesh.name,
                                      object2=self.config.target_mesh.name,
                                      wanted_alignment=self.text)
        self._log(f"LLM prior scale: {prior_scale}")
        prior_scale = prior_scale * 1.5
        mesh = scale_mesh_about_centroid(mesh, prior_scale)
        delta = {
            "scale": torch.tensor(float(prior_scale), device=self.device),
            "rotation": torch.eye(3, device=self.device),
            "translation": torch.zeros(3, device=self.device),
        }
        self.mesh_sim = sim3_compose(self.mesh_sim, delta)
        return mesh

    @staticmethod
    def _inject_noise_to_params(lr_scheduler, optimizer, noise_scale=0.01):
        with torch.no_grad():
            current_lrs = lr_scheduler.get_last_lr()  # one per param_group
            for (group, lr) in zip(optimizer.param_groups, current_lrs):
                sigma = noise_scale * lr
                for p in group['params']:
                    p.add_(sigma * torch.randn_like(p))

    def _save_best_result(self, imdir, loss, rotation, translation, scale, eval_img, name: str = ''):
        file_name = name + '_best_epoch.png'
        self._log(f"[INFO] New best result: {self.stats['best_result_loss']} --> {loss.item()}")
        self.stats['best_result_loss'] = loss.item()
        self.stats['best_result_params'] = {'rotation': rotation.clone().detach(),
                                            'translation': translation.clone().detach(),
                                            'scale': scale.clone().detach()}

        self._log(f"Saved best result: {self.stats['best_result_params']}")
        eval_img.save(str(imdir / file_name))

    def _train_one_epoch(self, data_loader: DataLoader, mesh, rotation, translation, scale, max_epochs):
        self._log(f"Start Training Epoch {self.epoch}, translation_lr={self.optimizer.param_groups[0]['lr']:.6f}, "
                  f"rotation_lr={self.optimizer.param_groups[1]['lr']:.6f},"
                  f"scaling_lr={self.optimizer.param_groups[2]['lr']:.6f}")

        pbar = tqdm.tqdm(
            total=max_epochs,
            desc="Epoch progress",
            bar_format=(
                "{desc}: {percentage:3.0f}% "
                "{n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}] "
                "{postfix}"
            ),
            initial=self.epoch - 1,
            dynamic_ncols=True,
            leave=True
        )

        # get one batch of camera parameters
        if self.config.adapt_dist:
            adapt_camera_distance(data_loader, [mesh, self.target_mesh], self.config.dist_min, self.config.dist_max)

        params_camera = get_next_camera_params(data_loader, self.device)

        self.optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=self.fp16):
            transformed_mesh, pred_render, loss, loss_dict = self._train_step(data=params_camera, mesh=mesh,
                                                                              rotation=rotation,
                                                                              translation=translation,
                                                                              scale=scale)

        total_loss = loss.item()

        loss_summary = ", ".join([f"{k}={v}" for k, v in loss_dict.items()])
        pbar.set_postfix_str(f"loss={total_loss:.4f}, {loss_summary}, "
                             f"translation_params: {translation.cpu().detach().numpy()}, "
                             f"rotation_params: {rotation.cpu().detach().numpy()},"
                             f"scale: {scale.cpu().detach().numpy()[0]}")
        pbar.update(1)
        pbar.close()

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.lr_scheduler.step()

        self.stats["loss"].append(total_loss)
        self._log(
            f"Finished Epoch {self.epoch}, loss: {total_loss}, loss summary: {loss_summary}, translation_params: {translation.cpu().detach().numpy()}, "
            f"rotation_params: {rotation.cpu().detach().numpy()}, scale_params: {scale.cpu().detach().numpy()}")
        return transformed_mesh, params_camera, loss, loss_dict

    def _train_step(self, data, mesh, rotation, translation, scale):
        # Apply the transformations to the mesh
        transformed_mesh = apply_differentiable_transform(mesh, rotation, translation, scale)
        pred_rgb = self._render(source_mesh=transformed_mesh, data=data)

        loss_calc, loss_dict = self._get_loss(pred_rgb, transformed_mesh, scale)
        loss = loss_calc
        return transformed_mesh, pred_rgb, loss, loss_dict

    def _get_loss(self, rendered_images, transformed_mesh, scale):
        """
        Computes the loss between the rendered image and both the positive and negative prompts.

        Args:
            rendered_image (torch.Tensor): The rendered image (batch_size, H, W, 3) in RGB.
            :param scale:
            :param rendered_images:
            :param transformed_mesh:

        Returns:
            loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0).to(self.device)

        # --- Intersection loss ---
        if self.intersection_loss_w > 0:
            intersection_loss = get_penetration_loss(mesh1=transformed_mesh,
                                                     mesh2=self.target_mesh) * self.intersection_loss_w
            loss_dict.update({'intersection_loss': round(intersection_loss.item(), 5)})
            total_loss += intersection_loss

        # --- Soft-ICP loss ---
        if self.icp_loss_w > 0:
            soft_icp_loss = get_fractional_soft_icp_loss(mesh1=transformed_mesh, mesh2=self.target_mesh,
                                                         percent=self.icp_ratio) * self.icp_loss_w
            loss_dict.update({'soft_icp_loss': round(soft_icp_loss.item(), 5)})
            total_loss += soft_icp_loss

        # --- Guidance loss ---
        if isinstance(self.guidance, StableDiffusion):
            if self.config.pos_text_w > 0:
                sds_loss = self.guidance.compute_loss(pred_rgb=rendered_images, prompt=self.text,
                                                      negative_prompt=self.negative_text, max_images=2)
                sds_loss = sds_loss.reshape(()) * self.config.pos_text_w  # force 0-dim scalar
            else:
                sds_loss = torch.tensor(0.0).to(self.device)

            loss_dict.update({'sds_loss': round(sds_loss.item(), 5)})
            total_loss = total_loss + sds_loss

        elif isinstance(self.guidance, Guidance):
            if self.config.image_guidance:
                positive_loss = self.guidance.compute_image_loss(rendered_images=rendered_images,
                                                                 ref_images=self.ref_image, device=self.device,
                                                                 penalty_weight=float(self.config.pos_text_w),
                                                                 text=self.text)
                loss_dict.update({'positive_image_loss': round(positive_loss.item(), 5)})
                total_loss += positive_loss

            else:
                if self.config.pos_text_w > 0:
                    positive_loss = self.guidance.compute_loss(images=rendered_images, text=self.text,
                                                               device=self.device,
                                                               penalty_weight=self.config.pos_text_w)
                    loss_dict.update({'positive_loss': round(positive_loss.item(), 5)})
                    total_loss += positive_loss

                if self.negative_text and self.config.neg_text_w > 0:
                    negative_loss = self.guidance.compute_loss(images=rendered_images, text=self.negative_text,
                                                               device=self.device,
                                                               penalty_weight=-self.config.neg_text_w)
                    loss_dict.update({'negative_loss': round(negative_loss.item(), 5)})
                    total_loss += negative_loss

        if self.config.scaling_reg:
            scaling_loss = get_scaling_reg(scale[0], weight=self.config.scaling_loss_w)
            loss_dict.update({'scaling_loss': round(scaling_loss.item(), 5)})
            total_loss += scaling_loss
        return total_loss, loss_dict

    def _render(self, data, source_mesh=None):
        meshes = [source_mesh, self.target_mesh] if source_mesh else [self.target_mesh]
        if self.config.use_ground:
            modify_ground_alt_by_meshes(self.ground, meshes)
            meshes.append(self.ground)
        return render_image(self.bsdf, self.glctx, self.light_power, self.train_res, self.resize_method, meshes, data)

    def _evaluate_one_epoch(self, transformed_mesh, scale, epoch=None, camera_params=None):
        epoch = epoch if epoch is not None else self.epoch
        self._log(f"Evaluate at Epoch {epoch} ...")

        eval_cam = self._build_eval_cameras() if not camera_params else camera_params

        with torch.no_grad():
            im, pred_rgb = self._get_rendered_batch(eval_cam, transformed_mesh)
            imdir = pathlib.Path(self.workspace) / 'images'
            imdir.mkdir(parents=True, exist_ok=True)
            im.save(str(imdir / f'eval_epoch_{epoch}.png'))

            loss_calc, loss_dict = self._get_loss(pred_rgb, transformed_mesh, scale)
            self._log(f"Finished evaluating Epoch {epoch}, loss: {loss_calc}, loss summary: {loss_dict}")

        return float(loss_calc.item()), im

    def _get_rendered_batch(self, camera_params, transformed_mesh):
        pred_rgb = self._render(source_mesh=transformed_mesh, data=camera_params)
        B = pred_rgb.shape[0]
        n_log = min(5, B)
        log_idx = torch.arange(n_log, device=pred_rgb.device)
        s_log = pred_rgb[log_idx]
        s_log = torchvision.utils.make_grid(s_log)

        ndarr = (s_log.detach().mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy())
        im = Image.fromarray(ndarr)
        return im, pred_rgb

    def _build_eval_cameras(self):
        device = self.mesh.v_pos.device
        res = int(self.config.log_res)
        k = int(getattr(self.config, "eval_k", 4))

        cams = generate_eval_cameras(
            resolution=(res, res),
            k=k,
            elev_deg=float(self.config.log_elev),
            dist=6.0,
            device=device,
        )

        return views_to_batched_camera_dict(cams, device=device)

    # ---------------------------- Utility Methods --------------------------------------------

    def _log(self, *args):
        if self.local_index == 0:
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush()

    def __del__(self):
        if hasattr(self, 'log_ptr'):
            self.log_ptr.close()

    def _save_render_to_video(self, transformed_mesh, loss_summary=None, loss=None):
        with torch.no_grad():
            params = get_camera_params(
                self.config.log_elev,
                self.video_rotation_angle,
                self.config.log_dist,
                self.config.log_res,
                self.config.log_fov,
            )
            self.video_rotation_angle += 1
            meshes = [transformed_mesh, self.target_mesh]
            if self.config.use_ground:
                meshes.append(self.ground)

            log_image = render_image_no_grad(bsdf=self.bsdf, glctx=self.glctx, light_power=self.config.log_light_power,
                                             log_res=self.config.log_res,
                                             meshes=meshes, data=params, device=self.device)

            self.video.ready_image(image=log_image, loss_dict=loss_summary, step_index=self.epoch, loss=loss)
