import argparse
import traceback
from copy import deepcopy
import datetime
import os.path
from easydict import EasyDict
import torch
import yaml
import numpy as np
import shutil
import glob

from trainer import Trainer
from utilities.calc_utils import add_random_init
from utilities.camera import get_data_loader
from utilities.mesh_utils import copy_mtl_and_textures, ensure_scale_in_init
from utilities.render_utils import seed_everything


def run():
    cfg = get_config()

    if cfg.get('chain_configs'):
        run_chain(cfg)
        return

    train_multi_runs(cfg)

def train_multi_runs(cfg):
    n_runs = int(cfg.get("n_runs", 1))

    best_loss, best_run_idx, best_image = float("inf"), None, None

    base_workspace = cfg.workspace
    base_seed = int(cfg.seed)

    for run_idx in range(n_runs):
        cfg_run = deepcopy(cfg)
        cfg_run.seed = base_seed + run_idx
        cfg_run.workspace = os.path.join(base_workspace, f"run_{run_idx:03d}")
        final_loss = train_once(cfg_run)

        if final_loss is None:
            continue

        if final_loss < best_loss:
            best_loss = final_loss
            best_run_idx = run_idx

    print(f"[multi-run] best run: {best_run_idx}, best eval loss: {best_loss}")

    return best_loss


def train_once(cfg):
    seed_everything(cfg.seed)
    device = torch.device(f'cuda:{cfg.local_index}') if cfg.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    camera_data_loader, cams_data = get_data_loader(cfg)
    ensure_scale_in_init(cfg)
    if cfg.source_mesh.get('random_init_params'):
        add_random_init(cfg)
    trainer = Trainer('objects_alignment', cfg, device=device)
    max_epoch = np.ceil(cfg.epochs / max(1, len(camera_data_loader))).astype(np.int32)
    os.makedirs(cfg.workspace, exist_ok=True)
    try:
        final_loss, final_image = trainer.train(max_epochs=max_epoch, data_loader=camera_data_loader, cams_data=cams_data)
        return final_loss
    except Exception as e:
        trainer._log(f"Could not run optimization due to an error: {e, traceback.format_exc()}")
        return float("inf")

# --- chained multi-object pipeline ---
def run_chain(base_cfg):
    """
    Sequentially run N configs; after each run, copy merged result into chain_data_dir
    and use its .obj as target_mesh_path for the next run.
    """
    configs = list(base_cfg.chain_configs)  # list of yaml paths
    data_dir = base_cfg.chain_data_dir
    assert configs and data_dir, "Provide --chain_configs ... and --chain_data_dir ..."

    os.makedirs(data_dir, exist_ok=True)

    # Use the existing outer workspace (already timestamped by get_config)
    outer_ws = base_cfg.workspace
    os.makedirs(outer_ws, exist_ok=True)

    merged_obj_for_next = None
    configs_list = []

    for i, cfg_path in enumerate(configs, start=1):
        with open(cfg_path, 'r') as f:
            y = yaml.safe_load(f) or {}

        run_cfg = EasyDict(deepcopy(base_cfg))
        for k, v in y.items():
            run_cfg[k] = v

        ensure_scale_in_init(run_cfg)
        base_name = os.path.splitext(os.path.basename(cfg_path))[0]
        run_cfg.workspace = os.path.join(outer_ws, f"step_{i:02d}_{base_name}")
        configs_list.append(run_cfg)

    for i, run_cfg in enumerate(configs_list, start=1):

        # For steps >1, override target_mesh_path with previous merged .obj
        if i > 1 and merged_obj_for_next is not None:
            run_cfg.target_mesh.path = merged_obj_for_next

        # Train this step
        train_once(run_cfg)

        # After training: locate merged dir
        merged_dir = os.path.join(run_cfg.workspace, "tmp", "final_meshes", "merged")
        if not os.path.isdir(merged_dir):
            merged_dir = os.path.join(run_cfg.workspace, "tmp", "final_meshes")
        if not os.path.isdir(merged_dir):
            raise FileNotFoundError(f"Could not find final meshes at {merged_dir}")

        # Choose the OBJ to carry forward (first or newest)
        src_candidates = sorted(glob.glob(os.path.join(merged_dir, "*.obj")))
        if not src_candidates:
            raise FileNotFoundError(f"No .obj found in {merged_dir}")
        src_obj = src_candidates[0]  # or max(src_candidates, key=os.path.getmtime)

        # copy OBJ + MTL + textures
        dst_iter = os.path.join(data_dir, f"merged_iter_{i}")
        os.makedirs(dst_iter, exist_ok=True)
        dst_obj = os.path.join(dst_iter, os.path.basename(src_obj))
        shutil.copy2(src_obj, dst_obj)
        copy_mtl_and_textures(src_obj, dst_obj)

        # Use the copied OBJ as next source mesh
        merged_obj_for_next = dst_iter

    print(f"[CHAIN DONE] {len(configs)} steps. Latest merged: {merged_obj_for_next}")


def get_config():
    parser = argparse.ArgumentParser()

    # ─── Config file & workspace ───────────────────────────────────────────────
    parser.add_argument('--config', '-c', type=str, default='configs/PairBench3D/hotdog.yaml', help='Path to the YAML config file')
    parser.add_argument('--workspace', '-w', type=str, default='workspace/', help='Output directory for logs, checkpoints, etc.')

    # ─── Mesh params ─────────────────────────────────────────────────────────────
    parser.add_argument('--use_ground', type=bool, default=True)
    parser.add_argument('--auto_align', type=bool, default=True, help='True for applying auto-align before running')
    parser.add_argument('--remesh', type=bool, default=True, help='True for applying remeshing to n vertices')
    parser.add_argument('--init_scale_by_llm', type=bool, default=False, help='True for initializing mesh scale with llm guidance (else - unit size)')
    parser.add_argument('--init_icp_ratio_by_llm', type=bool, default=False, help='True for initializing icp_ratio with llm guidance (else - must be in config)')

    # ─── Text prompts ────────────────────────────────────────────────────────────
    parser.add_argument('--text_prompt', '-t', type=str, default='', help='Main text prompt for guidance')
    parser.add_argument('--negative_prompt', type=str, default='', help='Negative text prompt')
    parser.add_argument('--image_path', type=str, default='', help='Image path for guidance')

    # ─── Evaluation & logging ──────────────────────────────────────────────────
    parser.add_argument('--eval_interval', type=int, default=10, help='Run evaluation every N epochs')
    parser.add_argument('--log_interval', type=int, default=5, help='Print training loss every N steps')
    parser.add_argument('--log_elev', type=float, default=30.0, help='Elevation angle to log')
    parser.add_argument('--log_fov', type=float, default=60.0, help='Field of view to log')
    parser.add_argument('--log_dist', type=float, default=3.0, help='Camera distance to log')
    parser.add_argument('--log_res', type=int, default=512, help='Resolution to log')
    parser.add_argument('--log_light_power', type=float, default=3.0, help='Light power to log')

    # ─── Guidance / CLIP settings ───────────────────────────────────────────────
    parser.add_argument('--guidance', type=str, default='clip', choices=['clip', 'align', 'structure_clip'], help='Guidance model')
    parser.add_argument('--clip_model', type=str, default='ViT-B/32', help='CLIP backbone for main loss')
    parser.add_argument('--consistency_clip_model', type=str, default='ViT-B/32', help='CLIP model for consistency loss')
    parser.add_argument('--token', type=str, default='', help="Hugging face token")
    parser.add_argument('--image_guidance', type=bool, default=False, help='Flag for image guidance instead of text')

    # ─── Random seed & device ───────────────────────────────────────────────────
    parser.add_argument('--seed', type=int, default=99, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Torch device')
    parser.add_argument('--local_index', type=int, default=0, help='Local rank for DDP')

    # ─── Training hyperparameters ──────────────────────────────────────────────
    # ─── LR ───
    parser.add_argument('--translation_lr', type=float, default=0.005, help='Translation learning rate')
    parser.add_argument('--rotation_lr', type=float, default=0.005, help='Rotation learning rate')
    parser.add_argument('--scaling_lr', type=float, default=0.005, help='Scaling learning rate')
    # ─── General ───
    parser.add_argument('--epochs', '-e', type=int, default=1000, help='Number of max training epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=25, help='Camera samples per epoch')
    parser.add_argument('--use_fp16', action='store_true', help='Enable mixed‑precision')
    # ─── Loss functions ───
    parser.add_argument('--scaling_reg', action='store_true', default=True, help='true for using scaling regularization')
    parser.add_argument('--intersection_loss', action='store_true', default=True, help='true for using intersection loss')
    parser.add_argument('--allowable_pen', default=0.002, help='Allowed penetration depth')
    # ─── Weights ───
    parser.add_argument('--pos_text_w', type=float, default=1.0, help='Weight for positive text prompt loss')
    parser.add_argument('--neg_text_w', type=float, default=1.0, help='Weight for negative text prompt loss')
    parser.add_argument('--guidance_loss_w', type=float, default=1.0, help='Weight for guidance - clip/sds loss')
    parser.add_argument('--scaling_loss_w', type=float, default=0.01, help='Weight for scaling regularization')
    parser.add_argument('--intersection_loss_w', nargs='+', default=[1.0, 10.0], help='Weight for intersection loss')
    parser.add_argument('--icp_loss_w', nargs='+', default=[1.0, 1.0], help='Weight for icp loss')

    # ─── Soft-ICP ───
    parser.add_argument('--icp_ratio', type=int, default=None, help='Ratio between vertices to perform icp on and all vertices')

    # ─── Rendering settings ─────────────────────────────────────────────────────
    parser.add_argument('--bsdf', type=str, default='diffuse', help='BSDF model (diffuse, specular, etc.)')
    parser.add_argument('--train_res', type=int, default=512, help='Internal render resolution')
    parser.add_argument('--resize_method', type=str, default='cubic', choices=['cubic', 'linear', 'lanczos2', 'lanczos3'], help='Interpolation method for resizing')

    # ─── Camera sampling ranges ────────────────────────────────────────────────
    parser.add_argument('--fov_min', type=float, default=30.0)
    parser.add_argument('--fov_max', type=float, default=90.0)
    parser.add_argument('--dist_min', type=float, default=2.5)
    parser.add_argument('--dist_max', type=float, default=3.5)
    parser.add_argument('--light_power', type=float, default=5.0)
    parser.add_argument('--elev_alpha', type=float, default=1.0)
    parser.add_argument('--elev_beta', type=float, default=5.0)
    parser.add_argument('--elev_min', type=float, default=10.0)
    parser.add_argument('--elev_max', type=float, default=60.0)
    parser.add_argument('--azim_min', type=float, default=0.0)
    parser.add_argument('--azim_max', type=float, default=360.0)
    parser.add_argument('--aug_loc', type=bool, default=True)
    parser.add_argument('--aug_light', type=bool, default=True)
    parser.add_argument('--adapt_dist', type=bool, default=True)
    parser.add_argument('--adapt_camera', type=bool, default=True)

    # ─── Checkpointing ──────────────────────────────────────────────────────────
    parser.add_argument('--run_checkpoint', help='Resume from checkpoint', default=False)
    parser.add_argument('--max_keep_ckpt', type=int, default=2, help='Number of checkpoint files to keep')
    parser.add_argument('--use_checkpoint', type=str, default='scratch', choices=['latest', 'best', 'scratch'], help='Which checkpoint to load at start')

    args = parser.parse_args()

    cfg = dict()
    if args.config is not None:
        if os.path.exists(args.config):
            with open(args.config, 'r') as f:
                try:
                    cfg = yaml.safe_load(f)
                except yaml.YAMLError as e:
                    print(e)
        else:
            cfg = args

    cfg = EasyDict(cfg)
    if not args.run_checkpoint:
        cfg.workspace = args.workspace + datetime.datetime.now().strftime("%m_%d_%Y__%H_%M_%S") + '_' + args.config.split('/')[-1].split('.')[0]
    for key in vars(args):
        if cfg.get(key) is None:
            cfg[key] = vars(args)[key]
    return cfg


if __name__ == '__main__':
    run()
