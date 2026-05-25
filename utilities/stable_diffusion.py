from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class SpecifyGradient(Function):
    """
    forward: returns `monitor_val` as a 0-dim tensor that requires_grad (so scaler & autograd are happy).
    backward: injects gt_grad * grad_output as gradient for the dummy input.
    """
    @staticmethod
    def forward(ctx, dummy, gt_grad, monitor_val=None):
        # Save the raw gradient for backward injection
        ctx.save_for_backward(gt_grad)

        # Compute monitor value:
        if monitor_val is None:
            # default: use grad energy
            monitor_val = 0.5 * gt_grad.detach().pow(2).mean()
        else:
            # ensure it's detached (we only use it for monitoring)
            monitor_val = monitor_val.detach()

        # Return a tensor that requires grad by tying to `dummy`.
        # (dummy must have requires_grad=True when calling)
        out = dummy * 0.0 + monitor_val.to(dummy.device, dummy.dtype)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (gt_grad,) = ctx.saved_tensors
        return gt_grad * grad_output, None, None


class StableDiffusion(nn.Module):
    def __init__(self, device, sd_version='2.0', hf_key=None, hugging_face_token: str = ''):
        super().__init__()
        self.device = device
        self.sd_version = sd_version

        print(f'[INFO] loading stable diffusion...')
        if hf_key is not None:
            model_key = hf_key
        elif self.sd_version == '3.5':
            model_key = 'stabilityai/stable-diffusion-3.5-medium'
        elif self.sd_version == '2.0':
            model_key = 'stabilityai/stable-diffusion-2-base'
        elif self.sd_version == '1.5':
            model_key = 'runwayml/stable-diffusion-v1-5'
        else:
            raise ValueError(f'SD version {self.sd_version} not supported.')

        # Load VAE, tokenizer, text encoder, and UNet
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae", token=hugging_face_token).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer", token=hugging_face_token)
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder", token=hugging_face_token).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet", token=hugging_face_token).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")

        # Timesteps for SDS
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)
        print(f'[INFO] loaded stable diffusion!')

    def get_text_embeds(self, prompt: str, negative_prompt: str):
        # Tokenize and encode prompts
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      truncation=True, return_tensors='pt')
        text_emb = self.text_encoder(text_input.input_ids.to(self.device))[0]
        uncond_emb = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        return torch.cat([uncond_emb, text_emb], dim=0)

    def encode_imgs(self, imgs: torch.Tensor):
        # imgs: [B,3,H,W] in [0,1]
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215
        return latents

    def compute_loss(
            self,
            pred_rgb: torch.Tensor,  # [B, 3, H, W]
            prompt: str,
            negative_prompt: str,
            guidance_scale: float = 100.0,
            max_images: int = 5,
            grad_clamp: float = None,  # e.g. 1.0 to clamp gradients to [-1,1]; None disables
            grad_scale: float = 1.0  # global multiplier for injected gradient
    ) -> torch.Tensor:
        """
        SDS guidance returned as a scalar 'loss' by injecting the SDS gradient into latents
        using SpecifyGradient.apply. This avoids averaging / normalization differences.
        """
        B = pred_rgb.shape[0]
        sub_B = min(B, max_images)

        # 1) select subset
        imgs = pred_rgb[:sub_B]  # [sub_B, 3, H, W]

        # 2) resize + encode to latents (VAE expects [-1,1])
        imgs_512 = F.interpolate(imgs, (512, 512), mode='bilinear', align_corners=False)
        latents = self.encode_imgs(imgs_512)  # [sub_B, C, h, w]

        # 3) sample timesteps per-sample (vector)
        t = torch.randint(self.min_step, self.max_step + 1, (sub_B,), device=self.device, dtype=torch.long)

        # 4) add noise
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)  # per-sample t allowed

        # 5) prepare UNet input (tile for classifier-free guidance)
        latent_model_input = torch.cat([latents_noisy, latents_noisy], dim=0)  # [2*sub_B, C, h, w]
        tt = torch.cat([t, t], dim=0)  # [2*sub_B]

        # 6) text embeds: returns [2, seq_len, hidden], expand per-sample -> [2*sub_B, seq_len, hidden]
        text_emb = self.get_text_embeds(prompt, negative_prompt)  # shape (2, seq_len, hidden)
        text_emb = text_emb.unsqueeze(0).expand(sub_B, -1, -1, -1)  # (sub_B, 2, seq_len, hidden)
        text_emb = text_emb.reshape(2 * sub_B, text_emb.size(2), text_emb.size(3))  # (2*sub_B, seq_len, hidden)

        # 7) predict noise with UNet (no grads for unet params)
        with torch.no_grad():
            noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_emb).sample  # [2*sub_B, C, h, w]

        # 8) split and perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)  # each [sub_B, C, h, w]
        # classifier-free guidance
        noise_pred_guided = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)  # [sub_B, C, h, w]

        # 9) compute SDS gradient (w * (pred - noise)) shape matches latents
        # use per-sample weights w = (1 - alpha_t)
        w = (1.0 - self.alphas[t]).view(-1, 1, 1, 1)  # [sub_B,1,1,1]
        grad_sds = w * (noise_pred_guided - noise)  # shape [sub_B, C, h, w]
        grad_sds = torch.nan_to_num(grad_sds)

        # monitoring scalar: classical SDS MSE (this is what we return)
        monitor_val = (noise_pred_guided - noise).pow(2).mean()  # scalar

        # optional clamp/scale
        if grad_clamp is not None:
            grad_sds = grad_sds.clamp(-float(grad_clamp), float(grad_clamp))
        grad_sds = grad_sds * float(grad_scale)

        # create dummy requiring grad
        dummy = torch.ones([1], device=self.device, dtype=grad_sds.dtype, requires_grad=True)

        # inject gradient and return monitor_val as scalar loss
        loss = SpecifyGradient.apply(dummy, grad_sds, monitor_val)  # forward returns monitor_val (0-d), backward injects grad_sds
        return loss

