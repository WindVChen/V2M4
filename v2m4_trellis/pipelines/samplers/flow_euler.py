from typing import *
import lpips
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin
from v2m4_trellis.utils import render_utils
from dreamsim import dreamsim
import torch.nn.functional as F


class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        return model(x_t, t, cond, **kwargs)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})
    
    # Remove `torch.no_grad()` to allow gradient computation.
    def gradient_sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret
    
    # Remove `torch.no_grad()` to allow gradient computation.
    def gradient_sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        optimize_uncond_noise = kwargs.get("optimize_uncond_noise", {})

        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})

        if optimize_uncond_noise != {}:  # optimize uncond noise
            start_optimize_iter = optimize_uncond_noise.get('start_optimize_iter', 10)
            refer_image = optimize_uncond_noise.get('refer_image', None)
            models = optimize_uncond_noise.get('models', None)
            normalization = optimize_uncond_noise.get('normalization', None)
            camera_params = optimize_uncond_noise.get('camera_params', None)

            if refer_image is None:
                raise ValueError("refer_image is required for optimize_uncond_noise")
            
            orig_features = kwargs['neg_cond'].detach()
            neg_cond = torch.nn.Parameter(kwargs['neg_cond'].detach().clone())

            optimizer = torch.optim.Adam([neg_cond], lr=0.1)
            loss1 = torch.nn.MSELoss()
            loss2 = lpips.LPIPS(net='vgg').cuda()
            loss3 = torch.nn.L1Loss()

            kwargs['neg_cond'] = neg_cond

        device = "cuda"
        dreamsim_model, _ = dreamsim(pretrained=True, device=device)
        mask = (refer_image.sum(dim=0) > 0).float()

        for id, (t, t_prev) in tqdm(enumerate(t_pairs), desc="Sampling", disable=not verbose):
            # Set lower bound and upper bound for t. The upper bound's setting is because at that stage, the 3D model has formed a coarse shape.
            if optimize_uncond_noise != {} and (0<=t<=0.6):
                par = tqdm(range(start_optimize_iter + id // 5), desc='Optimizing Uncond Noise', disable=False)
                for i in par:
                    optimizer.zero_grad()
                    out = self.gradient_sample_once(model, sample, t, t_prev, cond, **kwargs)

                    pred_x_0 = out.pred_x_0
                    
                    std = torch.tensor(normalization['std'])[None].to(pred_x_0.device)
                    mean = torch.tensor(normalization['mean'])[None].to(pred_x_0.device)
                    slat = pred_x_0 * std + mean

                    ret_mesh = models['slat_decoder_mesh'](slat)
                    ret_gaussian = models['slat_decoder_gs'](slat)

                    gradient_img_mesh, params = render_utils.find_closet_camera_pos(ret_mesh[0], refer_image, params=camera_params, return_optimize=True)
                    gradient_img_gs, params = render_utils.find_closet_camera_pos(ret_gaussian[0], refer_image, params=camera_params, return_optimize=True)

                    loss_mesh = dreamsim_model(F.interpolate((gradient_img_mesh * mask).unsqueeze(0), (224, 224), mode='bicubic'), F.interpolate(refer_image.unsqueeze(0), (224, 224), mode='bicubic')) + loss1(gradient_img_mesh * mask, refer_image) + loss2(gradient_img_mesh * mask, refer_image)
                    # loss_mesh = loss1(gradient_img_mesh, refer_image) + loss2(gradient_img_mesh, refer_image)
                    loss_gs = dreamsim_model(F.interpolate((gradient_img_gs * mask).unsqueeze(0), (224, 224), mode='bicubic'), F.interpolate(refer_image.unsqueeze(0), (224, 224), mode='bicubic')) + loss1(gradient_img_gs * mask, refer_image) + loss2(gradient_img_gs * mask, refer_image)
                    # loss_gs = loss1(gradient_img_gs, refer_image) + loss2(gradient_img_gs, refer_image)
                    loss_reg = 0.2 * loss1(neg_cond, orig_features) # small regularization term to avoid not optimized negative tensor
                    loss = loss_mesh + loss_gs + loss_reg
                    loss.backward()
                    optimizer.step()
                    par.set_postfix(loss_mesh=loss_mesh.item(), loss_gs=loss_gs.item(), loss_reg=loss_reg.item())

                    kwargs['neg_cond'] = neg_cond

                with torch.no_grad():
                    out = self.gradient_sample_once(model, sample, t, t_prev, cond, **kwargs)
                    
            else:
                out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)

            sample = out.pred_x_prev.detach()
            ret.pred_x_t.append(out.pred_x_prev.detach())
            ret.pred_x_0.append(out.pred_x_0.detach())

        ret.samples = sample
        return ret


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, **kwargs)


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)
    
    # Remove `torch.no_grad()` to allow gradient computation.
    def gradient_sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        return super().gradient_sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)
