from enum import Enum
from typing import List, Union

import math
import torch
from PIL import Image
from transformers import AlignProcessor, AlignModel, SiglipProcessor, SiglipModel
import CLIP.clip as openai_clip
from CLIP.clip import clip
import torch.nn.functional as F
import numpy as np


class GuidanceType(Enum):
    CLIP = 'clip'
    ALIGN = 'align'
    SIGLIP = 'siglip'
    StructureCLIP = 'structure_clip'
    StableDiffusion = 'sd'


cosine_sim = torch.nn.CosineSimilarity(dim=-1)


def cosine_avg(features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -cosine_sim(features, targets).mean()


class Guidance:
    """
    Unified wrapper for CLIP, ALIGN, SIGLIP.
    - score(images_pil, text) -> positive similarity (higher is better)
    - compute_loss(...)       -> negative similarity (lower is better) for training-like API
    """

    def __init__(self, guidance_type: GuidanceType, device: str = 'cpu'):
        self.type = guidance_type
        self.device = device

        if self.type == GuidanceType.CLIP:
            model, processor = openai_clip.load('ViT-B/32', device=device)
            self.model = model
            self.processor = processor
            self.mean = torch.tensor([0.48154660, 0.45782750, 0.40821073], device=device)
            self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)
            self._infer = self._infer_clip_loss  # returns negative cosine

        elif self.type == GuidanceType.ALIGN:
            self.model = AlignModel.from_pretrained("kakaobrain/align-base").to(device).eval()
            self.processor = AlignProcessor.from_pretrained("kakaobrain/align-base", device=device)
            self.mean = self.std = None
            self._infer = self._infer_align_loss  # returns negative logits_per_image

        elif self.type == GuidanceType.SIGLIP:
            name = "google/siglip2-base-patch16-224"  # or "google/siglip-so400m-patch14-384"
            # Use the dedicated classes, not AutoModel/AutoProcessor
            self.model = SiglipModel.from_pretrained(name).to(device).eval()
            self.processor = SiglipProcessor.from_pretrained(name, use_fast=True)
            self.mean = self.std = None
            self._infer = self._infer_siglip_loss  # keep loss API working

        elif self.type == GuidanceType.StructureCLIP:
            structure_clip_ckpt = './CLIP/SCLIPL_epoch_1_step0_WinoLoss.pt'
            model, processor = openai_clip.load("ViT-L/14@336px", device=device)
            ckpt = torch.load(structure_clip_ckpt, map_location=device, weights_only=True)
            model.load_state_dict(ckpt, strict=True)
            model.eval()
            self.model = model
            self.processor = processor
            self.mean = torch.tensor([0.48154660, 0.45782750, 0.40821073], device=device)
            self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)
            self._infer = self._infer_clip_loss

        else:
            raise ValueError(f"Unsupported guidance type {self.type}")

    @torch.no_grad()
    def score(self, images: List[Image.Image], text: Union[str, List[str]]) -> float:
        """
        Positive similarity (higher is better), averaged over images.
        CLIP/SigLIP → cosine; ALIGN → logits_per_image.
        """
        if self.type == GuidanceType.CLIP:
            imgs_t = [self.processor(img).unsqueeze(0).to(self.device) for img in images]
            batch = torch.cat(imgs_t, dim=0)
            txt = openai_clip.tokenize(text if isinstance(text, str) else text).to(self.device)
            img_emb = self.model.encode_image(batch)
            txt_emb = self.model.encode_text(txt)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
            # if multiple tokens provided, take the first
            if txt_emb.shape[0] > 1:
                txt_emb = txt_emb[0:1]
            return float(cosine_sim(img_emb, txt_emb.expand_as(img_emb)).mean().item())

        if self.type == GuidanceType.SIGLIP:
            inputs = self.processor(
                images=images,
                text=[text] if isinstance(text, str) else text,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            out = self.model(**inputs)
            img = out.image_embeds  # [B, D] projected embeddings
            txt = out.text_embeds  # [T, D] projected embeddings
            img = img / img.norm(dim=-1, keepdim=True)
            txt = txt / txt.norm(dim=-1, keepdim=True)

            if txt.shape[0] > 1:  # take first text if a list was passed
                txt = txt[0:1]
            sim = (img @ txt.T).mean().item()  # dot == cosine
            return float(sim)

        if self.type == GuidanceType.ALIGN:
            inputs = self.processor(images=images, text=text, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            # logits_per_image: higher => better match
            return float(outputs.logits_per_image.mean().item())

        # default (StructureCLIP same path as CLIP cosine)
        imgs_t = [self.processor(img).unsqueeze(0).to(self.device) for img in images]
        batch = torch.cat(imgs_t, dim=0)
        txt = openai_clip.tokenize(text if isinstance(text, str) else text).to(self.device)
        img_emb = self.model.encode_image(batch)
        txt_emb = self.model.encode_text(txt)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        if txt_emb.shape[0] > 1:
            txt_emb = txt_emb[0:1]
        return float(cosine_sim(img_emb, txt_emb.expand_as(img_emb)).mean().item())

    @torch.no_grad()
    def _infer_align_loss(self, images: List[Image.Image], text: Union[str, List[str]], *args, **kwargs) -> torch.Tensor:
        inputs = self.processor(images=images, text=text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        return -outputs.logits_per_image.mean()  # negative = loss

    @torch.no_grad()
    def _infer_clip_loss(self, images: torch.Tensor, text: str, device: str, penalty_weight: float = 1.0) -> torch.Tensor:
        """
        images: (B,3,H,W) tensor already in [0,1].
        """
        norm = (images - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        img_emb = self.model.encode_image(norm)
        with torch.no_grad():
            txt_tok = clip.tokenize(text).to(device)
            txt_emb = self.model.encode_text(txt_tok).detach()
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        return cosine_avg(img_emb, txt_emb) * penalty_weight

    @torch.no_grad()
    def _infer_siglip_loss(self, images, text, device, penalty_weight: float = 1.0) -> torch.Tensor:
        # Accept PIL images (like ALIGN path)
        inputs = self.processor(
            images=images,
            text=[text] if isinstance(text, str) else text,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        out = self.model(**inputs)
        img = out.image_embeds
        txt = out.text_embeds
        img = img / img.norm(dim=-1, keepdim=True)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        if txt.shape[0] > 1:
            txt = txt[0:1]
        # negative cosine (loss)
        return -torch.nn.functional.cosine_similarity(img, txt.expand_as(img), dim=-1).mean() * penalty_weight

    # ---------- Old training-style wrapper (kept for compatibility) ----------
    def compute_loss(self, images, text: str, device, penalty_weight: float) -> torch.Tensor:
        # some callers pass tensors (CLIP) or PILs (ALIGN/SigLIP) — we route to the right fn
        return self._infer(images, text, device, penalty_weight)

    def _tensor_batch_to_pil_list(self, batch: torch.Tensor) -> List[Image.Image]:
        """Convert tensor batch (B,3,H,W) in [0,1] to list[PIL.Image]."""
        if not isinstance(batch, torch.Tensor):
            raise TypeError("_tensor_batch_to_pil_list expects a torch.Tensor")
        b = batch.detach().cpu()
        if b.ndim == 3 and b.shape[0] == 3:
            b = b.unsqueeze(0)
        # convert to uint8 H,W,3
        imgs = []
        arr = (b.permute(0, 2, 3, 1).numpy() * 255.0).round().astype('uint8')
        for i in range(arr.shape[0]):
            imgs.append(Image.fromarray(arr[i]))
        return imgs

    def _ensure_pil_list(self, images) -> List[Image.Image]:
        """
        Accepts:
          - list[PIL.Image]
          - PIL.Image (single)
          - torch.Tensor (B,3,H,W) or (3,H,W)
        Returns list[PIL.Image].
        """
        if isinstance(images, Image.Image):
            return [images]
        if isinstance(images, list) and len(images) > 0 and isinstance(images[0], Image.Image):
            return images
        if isinstance(images, torch.Tensor):
            return self._tensor_batch_to_pil_list(images)
        if isinstance(images, list) and isinstance(images[0], torch.Tensor):
            # list of tensors -> convert each
            out = []
            for t in images:
                if t.ndim == 3 and t.shape[0] == 3:
                    arr = (t.detach().cpu().permute(1, 2, 0).numpy() * 255.0).round().astype('uint8')
                    out.append(Image.fromarray(arr))
                else:
                    raise ValueError("Unexpected tensor image shape in list")
            return out
        raise TypeError("Unsupported image type for guidance (expect PIL or torch.Tensor)")

    def _to_image_tensor_batch(self, images) -> torch.Tensor:
        """
        Return a torch tensor (B,3,H,W) on self.device in float [0,1].
        Accepts PIL, list[PIL], torch.Tensor (B,3,H,W) or (3,H,W).
        """
        if isinstance(images, torch.Tensor):
            t = images.to(self.device)
            if t.ndim == 3:
                t = t.unsqueeze(0)
            # if (B,H,W,3) -> permute
            if t.ndim == 4 and t.shape[-1] == 3:
                t = t.permute(0, 3, 1, 2)
            return t.float()
        # otherwise convert PIL(s)
        pil_list = self._ensure_pil_list(images)
        out = []
        for im in pil_list:
            arr = np.asarray(im.convert('RGB'), dtype=np.float32) / 255.0  # H,W,3
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)   # 1,3,H,W
            out.append(tensor)
        return torch.cat(out, dim=0).to(self.device)

    # --- new image-image guidance (put inside Guidance class) ---
    def compute_image_loss(
            self,
            rendered_images,
            ref_images,
            text: Union[str, List[str]] = None,  # <--- NEW optional text param
            device: str = None,
            penalty_weight: float = 1.0,
    ) -> torch.Tensor:

        """
        Compute an image→image guidance loss using the selected backend.
        Args:
            rendered_images: torch.Tensor (B,3,H,W) in [0,1] OR list[PIL.Image]
            ref_images:      torch.Tensor (B,3,H,W) or single (3,H,W) or PIL or list[PIL]
            device: overrides self.device
            penalty_weight: scalar multiplier
        Returns:
            scalar loss (torch.Tensor) where lower is better.
        """
        if device is None:
            device = self.device

        # convert both to batches on correct device
        R = self._to_image_tensor_batch(rendered_images).to(device)   # (B,3,H,W)
        B = self._to_image_tensor_batch(ref_images).to(device)        # (B_ref,3,H,W) or (1,...)

        # expand/trim B to match R.batch
        if B.shape[0] == 1 and R.shape[0] > 1:
            B = B.expand(R.shape[0], -1, -1, -1)
        elif B.shape[0] != R.shape[0]:
            # if sizes differ, tile or truncate to match
            if B.shape[0] < R.shape[0]:
                reps = int(np.ceil(R.shape[0] / float(B.shape[0])))
                B = B.repeat(reps, 1, 1, 1)[: R.shape[0]]
            else:
                B = B[: R.shape[0]]

        if self.type in (GuidanceType.CLIP, GuidanceType.StructureCLIP):
            # R, B are (B,3,H,W) tensors in [0,1] on device
            device = device or self.device

            # 1) detect model input resolution (some CLIP wrappers expose model.input_resolution)
            input_res = getattr(self.model, "input_resolution", None)
            if input_res is None:
                vis = getattr(self.model, "visual", None)
                input_res = getattr(vis, "input_resolution", None) if vis is not None else None
            if input_res is None:
                input_res = 224  # safe default

            # 2) resize tensors ON-TENSOR (preserves autograd for rendered images)
            # Note: align_corners=False is a safe default for bilinear resizing
            R_resized = F.interpolate(R, size=(input_res, input_res), mode="bilinear", align_corners=False)
            B_resized = F.interpolate(B, size=(input_res, input_res), mode="bilinear", align_corners=False)

            # 3) normalize using CLIP mean/std (already stored on self)
            mean = self.mean[None, :, None, None].to(device)
            std = self.std[None, :, None, None].to(device)
            norm_R = (R_resized - mean) / std
            norm_B = (B_resized - mean) / std

            # 4) encode: allow grads for norm_R, but run ref encodes in no_grad for efficiency
            img_emb = self.model.encode_image(norm_R)  # grads flow to norm_R -> to rendered pixels
            with torch.no_grad():
                ref_emb = self.model.encode_image(norm_B)

            # 5) normalize embeddings and compute negative cosine mean (loss)
            img_emb = img_emb / (img_emb.norm(dim=-1, keepdim=True) + 1e-12)
            ref_emb = ref_emb / (ref_emb.norm(dim=-1, keepdim=True) + 1e-12)
            loss = -torch.nn.functional.cosine_similarity(img_emb, ref_emb, dim=-1).mean() * penalty_weight
            return loss

        # SIGLIP path: use its processor + model outputs (image_embeds); processor accepts list[PIL] or tensors
        if self.type == GuidanceType.SIGLIP:
            pil_R = self._tensor_batch_to_pil_list(R)
            pil_B = self._tensor_batch_to_pil_list(B)
            imgs = pil_R + pil_B

            # Use provided text if available, otherwise fallback to empty strings
            if text is None:
                text_list = [""] * len(imgs)
            else:
                # support single string or list-of-strings
                if isinstance(text, str):
                    text_list = [text] * len(imgs)
                else:
                    # list provided: if len < len(imgs) we tile/truncate
                    text_list = list(text)
                    if len(text_list) == 1 and len(imgs) > 1:
                        text_list = text_list * len(imgs)
                    elif len(text_list) < len(imgs):
                        reps = int(math.ceil(len(imgs) / len(text_list)))
                        text_list = (text_list * reps)[:len(imgs)]
                    else:
                        text_list = text_list[:len(imgs)]

            inputs = self.processor(images=imgs, text=text_list, return_tensors="pt", padding=True).to(device)
            out = self.model(**inputs)
            emb = out.image_embeds
            emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-12)
            emb_R = emb[:len(pil_R)]
            emb_B = emb[len(pil_R):]
            if emb_B.shape[0] == 1 and emb_R.shape[0] > 1:
                emb_B = emb_B.expand(emb_R.shape[0], -1)
            loss = -torch.nn.functional.cosine_similarity(emb_R, emb_B, dim=-1).mean() * penalty_weight
            return loss

        # ALIGN path: processor + model returns logits_per_image when called with images & texts;
        # but when we pass only images, AlignModel returns image_embeds; we use those and compute cosine.
        if self.type == GuidanceType.ALIGN:
            pil_R = self._tensor_batch_to_pil_list(R)
            pil_B = self._tensor_batch_to_pil_list(B)
            imgs = pil_R + pil_B
            inputs = self.processor(images=imgs, return_tensors="pt", padding=True).to(device)
            outputs = self.model(**inputs)
            # AlignModel returns 'image_embeds' in outputs
            img_embs = outputs.image_embeds
            img_embs = img_embs / (img_embs.norm(dim=-1, keepdim=True) + 1e-12)
            emb_R = img_embs[:len(pil_R)]
            emb_B = img_embs[len(pil_R):]
            if emb_B.shape[0] == 1 and emb_R.shape[0] > 1:
                emb_B = emb_B.expand(emb_R.shape[0], -1)
            loss = -torch.nn.functional.cosine_similarity(emb_R, emb_B, dim=-1).mean() * penalty_weight
            return loss

        # fallback: same as CLIP behavior
        mean = self.mean[None, :, None, None].to(device)
        std = self.std[None, :, None, None].to(device)
        norm_R = (R - mean) / std
        norm_B = (B - mean) / std
        img_emb = self.model.encode_image(norm_R)
        with torch.no_grad():
            ref_emb = self.model.encode_image(norm_B)
        img_emb = img_emb / (img_emb.norm(dim=-1, keepdim=True) + 1e-12)
        ref_emb = ref_emb / (ref_emb.norm(dim=-1, keepdim=True) + 1e-12)
        loss = -torch.nn.functional.cosine_similarity(img_emb, ref_emb, dim=-1).mean() * penalty_weight
        return loss

