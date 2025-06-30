from dataclasses import dataclass, field

import numpy as np
import json
import copy
import torch
import torch.nn.functional as F
from skimage import measure
from einops import repeat
from tqdm import tqdm
from PIL import Image

from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    UniPCMultistepScheduler,
    KarrasVeScheduler,
    DPMSolverMultistepScheduler
)

import craftsman
from craftsman.systems.base import BaseSystem
from craftsman.utils.misc import get_rank
from craftsman.utils.typing import *
from diffusers import DDIMScheduler
from craftsman.systems.utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, get_sigmas, flow_sample


@craftsman.register("shape-rectified-flow-system")
class RectifiedFlowSystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        compute_metric: bool = True
        visualize_mesh: bool = True
        val_samples_json: str = None
        extract_mesh_func: str = "mc"
        remove_bg: bool = False
        octree_depth: int = 7

        # diffusion config
        z_scale_factor: float = 1.0
        guidance_scale: float = 7.5
        num_inference_steps: int = 30
        eta: float = 0.0
        snr_gamma: float = 5.0
        # flow
        weighting_scheme: str = "logit_normal"
        logit_mean: float = 0
        logit_std: float = 1.0
        mode_scale: float = 1.29
        precondition_outputs: bool = True
        precondition_t: int = 1000

        # shape vae model
        shape_model_type: str = None
        shape_model: dict = field(default_factory=dict)

        # condition model
        condition_model_type: str = None
        condition_model: dict = field(default_factory=dict)

        # diffusion model
        denoiser_model_type: str = None
        denoiser_model: dict = field(default_factory=dict)

        # noise scheduler
        noise_scheduler_type: str = None
        noise_scheduler: dict = field(default_factory=dict)

        # denoise scheduler
        denoise_scheduler_type: str = None
        denoise_scheduler: dict = field(default_factory=dict)

    cfg: Config

    def configure(self):
        super().configure()

        self.shape_model = craftsman.find(self.cfg.shape_model_type)(self.cfg.shape_model)
        self.shape_model.eval()
        self.shape_model.requires_grad_(False)

        self.condition = craftsman.find(self.cfg.condition_model_type)(self.cfg.condition_model)
        
        self.denoiser_model = craftsman.find(self.cfg.denoiser_model_type)(self.cfg.denoiser_model)

        self.noise_scheduler = craftsman.find(self.cfg.noise_scheduler_type)(**self.cfg.noise_scheduler)
        self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)

        self.denoise_scheduler = craftsman.find(self.cfg.denoise_scheduler_type)(**self.cfg.denoise_scheduler)

    def forward(self, batch: Dict[str, Any], skip_noise=False) -> Dict[str, Any]:
        # 1. encode shape latents
        if "sharp_surface" in batch.keys():
            sharp_surface = batch["sharp_surface"][..., :3 + self.cfg.shape_model.point_feats]
        else:
            sharp_surface = None
        shape_embeds, kl_embed, _ = self.shape_model.encode(
            batch["surface"][..., :3 + self.cfg.shape_model.point_feats], 
            sample_posterior=True,
            sharp_surface=sharp_surface, 
        )
        latents = kl_embed * self.cfg.z_scale_factor

        # 2. gain visual condition
        if "image" in batch and batch['image'].dim() == 5:
            if self.training:
                bs, n_images = batch['image'].shape[:2]
                batch['image'] = batch['image'].view(bs*n_images, *batch['image'].shape[-3:])
            else:
                batch['image'] = batch['image'][:, 0, ...]
                n_images = 1
                bs = batch['image'].shape[0]
            cond_latents = self.condition.forward_visual_embeds(batch).to(latents)
            latents = latents.unsqueeze(1).repeat(1, n_images, 1, 1)
            latents = latents.view(bs*n_images, *latents.shape[-2:])
        else:
            cond_latents = self.condition.forward_visual_embeds(batch).to(latents)
        ## 2.1 text condition if provided
        if "caption" in batch:
            assert bs == len(batch["caption"]), "Batch size must be the same as the caption length."
            text_cond = self.condition.encode_text(batch["caption"]).repeat_interleave(n_images, dim=0).to(latents)
            cond_latents = torch.cat([text_cond, cond_latents], dim=1)

        # 3. sample noise that we"ll add to the latents
        noise = torch.randn_like(latents).to(latents) # [batch_size, n_token, latent_dim]
        
        # 4. Sample a random timestep for each motion
        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.cfg.weighting_scheme,
            batch_size=bs*n_images,
            logit_mean=self.cfg.logit_mean,
            logit_std=self.cfg.logit_std,
            mode_scale=self.cfg.mode_scale,
        )
        indices = (u * self.cfg.noise_scheduler.num_train_timesteps).long()
        timesteps = self.noise_scheduler_copy.timesteps[indices].to(device=latents.device)

        # 5. add noise
        sigmas = get_sigmas(self.noise_scheduler_copy, timesteps, n_dim=3, dtype=latents.dtype)
        noisy_z = (1.0 - sigmas) * latents + sigmas * noise            

        # 6. diffusion model forward
        output = self.denoiser_model(noisy_z, timesteps.long(), cond_latents)

        # 7. compute loss
        if self.cfg.precondition_outputs:
            output = output * (-sigmas) + noisy_z
        # these weighting schemes use a uniform timestep sampling
        # and instead post-weight the loss
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=self.cfg.weighting_scheme, sigmas=sigmas)
        # flow matching loss
        if self.cfg.precondition_outputs:
            target = latents
        else:
            target = noise - latents

        # Compute regular loss.
        loss = torch.mean(
            (weighting.float() * (output.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()

        return {
            "loss_diffusion": loss,
            "latents": latents,
            "x_t": noisy_z,
            "noise": noise,
            "noise_pred": output,
            "timesteps": timesteps,
            }

    def training_step(self, batch, batch_idx):
        out = self(batch)

        loss = 0.
        for name, value in out.items():
            if name.startswith("loss_"):
                self.log(f"train/{name}", value)
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])

        for name, value in self.cfg.loss.items():
            if name.startswith("lambda_"):
                self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.eval()

        if get_rank() == 0:
            torch.cuda.empty_cache()
            sample_inputs = json.loads(open(self.cfg.val_samples_json).read()) # condition
            sample_inputs_ = copy.deepcopy(sample_inputs)
            sample_outputs = self.sample(sample_inputs) # list
            for i, sample_output in enumerate(sample_outputs):
                gt_rgbs = []
                sample_normals = []
                sample_depths = []
                mesh_v_f, has_surface = self.shape_model.extract_geometry(sample_output, octree_depth=self.cfg.octree_depth, extract_mesh_func=self.cfg.extract_mesh_func)
              
                for j in range(len(mesh_v_f)):
                    if "image" in sample_inputs_:
                        name = sample_inputs_["image"][j].split("/")[-1].replace(".png", "")
                    elif "mvimages" in sample_inputs_:
                        name = sample_inputs_["mvimages"][j][0].split("/")[-2].replace(".png", "")
                    self.save_mesh(
                        f"it{self.true_global_step}/{name}_{i}.obj",
                        mesh_v_f[j][0], mesh_v_f[j][1]
                    )

        out = self(batch)
        if self.global_step == 0:
            latents = self.shape_model.decode(out["latents"])
            mesh_v_f, has_surface = self.shape_model.extract_geometry(latents=latents, extract_mesh_func=self.cfg.extract_mesh_func)

            self.save_mesh(
                f"it{self.true_global_step}/{batch['uid'][0]}_{batch['sel_idx'][0] if 'sel_idx' in batch.keys() else 0}.obj",
                mesh_v_f[0][0], mesh_v_f[0][1]
            )

        return {"val/loss": out["loss_diffusion"]}
        
 
    @torch.no_grad()
    def sample(self,
               sample_inputs: Dict[str, Union[torch.FloatTensor, List[str]]],
               sample_times: int = 1,
               steps: Optional[int] = None,
               guidance_scale: Optional[float] = None,
               eta: float = 0.0,
               seed: Optional[int] = None,
               **kwargs):

        if steps is None:
            steps = self.cfg.num_inference_steps
        if guidance_scale is None:
            guidance_scale = self.cfg.guidance_scale
        do_classifier_free_guidance = guidance_scale != 1.0

        # conditional encode
        if "image" in sample_inputs:
            sample_inputs["image"] = [Image.open(img) if type(img) == str else img for img in sample_inputs["image"]]
            cond = self.condition.encode_image(sample_inputs["image"])
            if do_classifier_free_guidance:
                un_cond = self.condition.empty_image_embeds.repeat(len(sample_inputs["image"]), 1, 1).to(cond)
                cond = torch.cat([un_cond, cond], dim=0)
                
        elif "mvimages" in sample_inputs: # by default 4 views
            bs = len(sample_inputs["mvimages"])
            cond = []
            for image in sample_inputs["mvimages"]:
                if isinstance(image, list) and isinstance(image[0], str):
                    sample_inputs["image"] = [Image.open(img) for img in image] # List[PIL]
                else:
                    sample_inputs["image"] = image
                cond += [self.condition.encode_image(sample_inputs["image"])]
            cond = torch.stack(cond, dim=0).view(bs, -1, self.cfg.denoiser_model.context_dim)
            if do_classifier_free_guidance:
                un_cond = self.condition.empty_image_embeds.unsqueeze(0).repeat(len(sample_inputs["mvimages"]), 1, 1, 1).view(bs, cond.shape[1], self.cfg.denoiser_model.context_dim).to(cond) # shape 为[len(sample_inputs["mvimages"], 4*(num_latents+1), context_dim]
                cond = torch.cat([un_cond, cond], dim=0).view(bs * 2, -1, cond[0].shape[-1]) 
        else:
            raise NotImplementedError("Only image or mvimages condition is supported.")

        outputs = []
        latents = None
        
        if seed != None:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        else:
            generator = None

        for _ in range(sample_times):
            sample_loop = flow_sample(
                self.denoise_scheduler,
                self.denoiser_model.eval(),
                shape=self.shape_model.latent_shape,
                cond=cond,
                steps=steps,
                guidance_scale=guidance_scale,
                do_classifier_free_guidance=do_classifier_free_guidance,
                device=self.device,
                eta=eta,
                disable_prog=False,
                generator= generator
            )
            for sample, t in sample_loop:
                latents = sample
            outputs.append(self.shape_model.decode(latents / self.cfg.z_scale_factor, **kwargs))
        
        return outputs

    def on_validation_epoch_end(self):
        pass