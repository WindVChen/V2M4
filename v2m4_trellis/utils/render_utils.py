import sys
sys.path.append('./extensions')
import torch
import numpy as np
from tqdm import tqdm
import utils3d
from PIL import Image
import lpips
import nvdiffrast.torch as dr
import torch.nn.functional as F
import copy
import trimesh
from dreamsim import dreamsim
import random

from v2m4_trellis.utils.general_utils import load_images_custom
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

from ..renderers.mesh_renderer import intrinsics_to_projection

from ..renderers import OctreeRenderer, GaussianRenderer, MeshRenderer
from ..representations import Octree, Gaussian, MeshExtractResult
from ..modules import sparse as sp
from .random_utils import sphere_hammersley_sequence

from pytorch3d.loss import chamfer_distance


def extrinsics_to_yaw_pitch_r_lookat(R, t):
    # Compute camera origin: orig = -R^T * t
    orig = -torch.matmul(R.transpose(-1, -2), t).squeeze(-1)

    # Compute viewing direction (Z-axis of rotation matrix)
    view_dir = R[:, 2, :]  # Viewing direction (unit vector)

    # Compute r (distance from camera to look-at point)
    # Assuming the camera is looking directly along the view_dir
    r = torch.norm(orig, dim=-1)

    # Compute look-at position
    lookat = orig + r.unsqueeze(-1) * view_dir

    # Compute yaw
    yaw = torch.atan2(orig[:, 0], orig[:, 1])

    # Compute pitch
    pitch = torch.asin(orig[:, 2] / r)

    # verify the correctness of the conversion (will have some unavoidable numerical errors)
    # extr, intr = optimize_yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, 40, lookat)

    return yaw, pitch, r, lookat


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def dust3r_getting_better_initial_guess(images, prior_poses, gt_pts3d, device='cuda', params={}, using_vggt=True, save_path='', is_Hunyuan=False):
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    if not using_vggt:
        print('Using Dust3R to get better initial guess') 
        model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
        # you can put the path to a local checkpoint in model_name if needed
        model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)

        # for i, im in enumerate(images):
        #     # to pil image and save
        #     im = Image.fromarray((im.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).convert('RGB')
        #     im.save(f'input_{i}.png')
        # load_images can take a list of images or a directory
        processed_images = load_images_custom(images[:], size=images.shape[-1], square_ok=True)
        pairs = make_pairs(processed_images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=batch_size)

        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
        scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

        # retrieve useful values from scene:
        # poses = scene.get_im_poses().detach()
        pts3d = torch.stack(scene.get_pts3d(), dim=0).detach()
    else:
        print('Using VGGT to get better initial guess') 
        # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        # Initialize the model and load the pretrained weights.
        # This will automatically download the model weights the first time it's run, which may take a while.
        model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                # Predict attributes including cameras, depth maps, and point maps.
                predictions = model(images)
                # Use directly predicted world points instead of points from depth maps (which seems to be more accurate)
                pts3d = predictions["world_points"][0].detach()

    # interpolate to the size of images[0].shape[0, 1]
    pts3d = F.interpolate(pts3d.permute(0, 3, 1, 2), size=images.shape[-1], mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

    pred_pts3d = []
    for i, pts in enumerate(pts3d[1:]):
        # i + 1 because the first image is the reference image
        points = pts.view(-1, 3).detach()[images[i + 1].permute(1, 2, 0).reshape(-1, 3).sum(-1) > 0]
        pred_pts3d.append(points)

    pred_pts3d = torch.cat(pred_pts3d, dim=0)

    # Sample 10 percent of the points to speed up the optimization (10x speedup, 70s -> 7s)
    pred_pts3d = pred_pts3d[torch.randperm(pred_pts3d.shape[0])[:int(pred_pts3d.shape[0] * 0.05)]]
    gt_pts3d = gt_pts3d[torch.randperm(gt_pts3d.shape[0])[:int(gt_pts3d.shape[0] * 0.05)]]

    # sm = trimesh.PointCloud(vertices=pred_pts3d.cpu().numpy())
    # sm.export(save_path + '_pred_pts3d.ply')


    '''Align the scale, rotation, and translation between the gt point clouds and the predicted point clouds'''
    # Rotation matrix rotate -90 degree around x-axis and 180 degree around z-axis (sometimes it works, but sometimes it fails, due to different initial pose)
    # VGGT and Dust3R share this similar initial rotation matrix
    if not is_Hunyuan:
        rot6D = torch.nn.Parameter(torch.tensor([-1.0, 0.0, 0.0, 0.0, 0.0, -1.0], device=device))
    else:
        # Only -90 degree rotate around x-axis, no 180 degree around z-axis for Hunyuan
        rot6D = torch.nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0], device=device))
    # Consider the max and minimum position of 95 percent (to avoid outliers) of the gt points and that of the predicted points to decide the initial scale and translation
    gt_pts3d_95 = torch.randperm(gt_pts3d.shape[0])[:int(gt_pts3d.shape[0] * 0.95)]
    pred_pts3d_95 = torch.randperm(pred_pts3d.shape[0])[:int(pred_pts3d.shape[0] * 0.95)]

    rotation_matrix = rotation_6d_to_matrix(rot6D)
    gt_pts3d_95 = gt_pts3d[gt_pts3d_95]
    pred_pts3d_95 = pred_pts3d[pred_pts3d_95] @ rotation_matrix.T

    gt_pts3d_max = gt_pts3d_95.max(dim=0).values
    gt_pts3d_min = gt_pts3d_95.min(dim=0).values
    pred_pts3d_max = pred_pts3d_95.max(dim=0).values
    pred_pts3d_min = pred_pts3d_95.min(dim=0).values
    # divide by 4 to ensure enough space for the optimization
    scale = torch.nn.Parameter(((gt_pts3d_max - gt_pts3d_min) / (pred_pts3d_max - pred_pts3d_min) / 4).mean().unsqueeze(0))
    translation = torch.nn.Parameter((gt_pts3d_max + gt_pts3d_min) / 2 - (pred_pts3d_max + pred_pts3d_min) * scale.item() / 2)

    with torch.no_grad():
        rotation_matrix = rotation_6d_to_matrix(rot6D)
        transformed_pts3d = pred_pts3d @ rotation_matrix.T * scale + translation

        # sm = trimesh.PointCloud(vertices=transformed_pts3d.cpu().numpy())
        # sm.export(save_path + '_before_transformed_pts3d.ply')
    
    optimizer = torch.optim.Adam([rot6D, translation, scale], lr=0.01)

    pbar = tqdm(range(2000), desc=f'Aligning with {"VGGT" if using_vggt else "Dust3R"} position', disable=False)

    rotZ180 = torch.tensor([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], device=device)

    bf_flip_loss_params = {'loss': torch.inf, 'params': None}
    for i in pbar:
        optimizer.zero_grad()

        rotation_matrix = rotation_6d_to_matrix(rot6D)
        transformed_pts3d = pred_pts3d @ rotation_matrix.T * scale + translation

        # Compute the chamfer distance between the transformed point clouds and the gt point clouds
        loss = chamfer_distance(gt_pts3d.unsqueeze(0), transformed_pts3d.unsqueeze(0))[0]

        loss.backward()
        optimizer.step()

        pbar.set_postfix({'loss': loss.item()})  

        # This is to avoid the local minimum where the rotation matrix is trapped in 180 degree rotation around z-axis
        if i == len(pbar) * 3 // 4:
            bf_flip_loss_params['loss'] = loss.item()
            bf_flip_loss_params['params'] = {'rot6D': rot6D.data.clone(), 'translation': translation.data.clone(), 'scale': scale.data.clone()}

            rot6D.data = (rotZ180 @ rotation_matrix)[:2].flatten()
            translation.data = translation @ rotZ180.T
            print('Flip the rotation matrix')
            continue                      

    with torch.no_grad():
        if bf_flip_loss_params['loss'] < loss.item():
            rot6D.data = bf_flip_loss_params['params']['rot6D']
            translation.data = bf_flip_loss_params['params']['translation']
            scale.data = bf_flip_loss_params['params']['scale']
            print('Use the non-flipped parameters')

        rotation_matrix = rotation_6d_to_matrix(rot6D)
        transformed_pts3d = pred_pts3d @ rotation_matrix.T * scale + translation

        # sm = trimesh.PointCloud(vertices=transformed_pts3d.cpu().numpy())
        # sm.export(save_path + '_transformed_pts3d.ply')

        # sm = trimesh.PointCloud(vertices=gt_pts3d.cpu().numpy())
        # sm.export(save_path + '_gt_pts3d.ply')

        '''Get the converted point clouds of the reference image extracted by Dust3R'''
        reference_points = pts3d[0].view(-1, 3).detach()[images[0].permute(1, 2, 0).reshape(-1, 3).sum(-1) > 0] @ rotation_matrix.T * scale + translation

        # sm = trimesh.PointCloud(vertices=reference_points.cpu().numpy())
        # sm.export(save_path + '_reference_points.ply')

        reference_points_color = images[0].permute(1, 2, 0).reshape(-1, 3).detach()[images[0].permute(1, 2, 0).reshape(-1, 3).sum(-1) > 0]
        # get the 2D positions of images[0] that are not background
        reference_points_2d_index = torch.nonzero(images[0].permute(1, 2, 0).reshape(-1, 3).sum(-1) > 0)
        # meshgrid the 2D positions
        reference_points_2d = torch.stack(torch.meshgrid([torch.arange(images.shape[-1]), torch.arange(images.shape[-1])]), dim=-1).cuda().reshape(-1, 2)[reference_points_2d_index].squeeze().float()

        reference_points_2d = reference_points_2d * 2 / images.shape[-1] - 1

    '''Given the scale aligned 3D points, optimize R, T to find the similar rendering image aligned with the reference image (note that here we not align the color, but the 2D pixel positions). That R, T will be the extrinsics of the reference image. Note: In the optimization phase, we directly optimize the R[:, 2, :], T, as the optimize_yaw_pitch_r_fov_to_extrinsics_intrinsics function only relates to Z-axis and T.'''
    # Use the first rendering camera position as the initial guess to avoid the local minimum (eg, pure black)
    rot3D = torch.nn.Parameter(prior_poses[0, 2, :3].detach())
    translation = torch.nn.Parameter(prior_poses[0, :3, 3].detach())

    optimizer = torch.optim.Adam([rot3D, translation], lr=0.01)

    pbar = tqdm(range(500), desc='Re-projecting reference image points', disable=False)

    intrinsic = params['intrinsics']
    renderer = params['renderer']
    near = renderer.rendering_options["near"]
    far = renderer.rendering_options["far"]
    resolution = renderer.rendering_options["resolution"]

    for i in pbar:
        optimizer.zero_grad()

        z = rot3D
        x = torch.cross(-torch.tensor([0, 0, 1.]).cuda(), z, dim=-1)
        y = torch.cross(z, x, dim=-1)
        x = x / x.norm(dim=-1, keepdim=True)
        y = y / y.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)
        rotation_matrix = torch.stack([x, y, z], dim=-2)

        extrinsic = torch.cat([rotation_matrix, translation.unsqueeze(1)], dim=-1)
        extrinsic = torch.cat([extrinsic, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=device)], dim=0)

        perspective = intrinsics_to_projection(intrinsic, near, far)
        full_proj = torch.matmul(perspective, extrinsic)

        # Project the reference_points from world coordinate system to the camera coordinate system (actually NDC coordinate system)
        pt3 = torch.cat([reference_points, torch.ones_like(reference_points[..., :1])], dim=-1)
        pt3 = torch.matmul(pt3, full_proj.transpose(-1, -2))

        optim_converted_pcd = pt3[..., :2] / pt3[..., -1:]

        # with torch.no_grad():
        #     converted_pcd = (optim_converted_pcd + 1) * resolution / 2
        #     converted_pcd = converted_pcd.clamp(min=0, max=resolution)

        #     # init a black image and then asign the color of the point cloud
        #     cimg = torch.zeros((resolution, resolution, 3))
        #     converted_pcd = converted_pcd.reshape(-1, 2).cpu().long().numpy()
        #     # use converted_pcd.long() to index the image and then assign the corresponding color from images[0] to the indexed position
        #     cimg[converted_pcd[..., 1], converted_pcd[..., 0]] = reference_points_color.cpu()
        #     cimg = cimg.reshape(resolution, resolution, 3)

        #     from torchvision.transforms import ToPILImage
        #     to_pil = ToPILImage()
        #     to_pil(cimg.permute(2, 0, 1).detach().cpu()).save('reprojected_image.png')

        # exchange the 2 dimensions in y-axis to ensure it is in the order of W, H
        loss = torch.nn.MSELoss()(optim_converted_pcd, reference_points_2d[:, [1, 0]])

        loss.backward()
        optimizer.step()

        pbar.set_postfix({'loss': loss.item()})

    # with torch.no_grad():
    #     converted_pcd = (optim_converted_pcd + 1) * resolution / 2
    #     converted_pcd = converted_pcd.clamp(min=0, max=resolution)

    #     # init a black image and then asign the color of the point cloud
    #     cimg = torch.zeros((resolution, resolution, 3))
    #     converted_pcd = converted_pcd.reshape(-1, 2).cpu().long().numpy()
    #     # use converted_pcd.long() to index the image and then assign the corresponding color from images[0] to the indexed position
    #     cimg[converted_pcd[..., 1], converted_pcd[..., 0]] = reference_points_color.cpu()
    #     cimg = cimg.reshape(resolution, resolution, 3)

    #     from torchvision.transforms import ToPILImage
    #     to_pil = ToPILImage()
    #     to_pil(cimg.permute(2, 0, 1).detach().cpu()).save('reprojected_image.png')

    with torch.no_grad():
        z = rot3D
        x = torch.cross(-torch.tensor([0, 0, 1.]).cuda(), z, dim=-1)
        y = torch.cross(z, x, dim=-1)
        x = x / x.norm(dim=-1, keepdim=True)
        y = y / y.norm(dim=-1, keepdim=True)
        z = z / z.norm(dim=-1, keepdim=True)
        rotation_matrix = torch.stack([x, y, z], dim=-2)

    return rotation_matrix.unsqueeze(0).detach(), translation.unsqueeze(1).unsqueeze(0).detach()


def sample_camera_positions(num_samples=100, device='cuda'):
    """
    Samples camera positions uniformly distributed in 3D space, avoiding clustering issues.
    
    Motivation:
    -----------
    A naive approach of sampling yaw, pitch, and radius uniformly leads to uneven distributions:
    - **Pitch (elevation)**: A uniform sample results in more points near the poles.
    - **Radius**: A uniform sample causes clustering near smaller radii since sphere surface area grows with r².
    
    Solution:
    ---------
    1. **Equal-area pitch sampling**: Use inverse transform sampling to ensure uniform coverage on the sphere.
       - Instead of directly sampling pitch uniformly, we sample `z = sin(pitch)` uniformly in [-1,1] and recover `pitch = asin(2u - 1)`.
    2. **Corrected radius sampling**: Ensure uniform distribution in 3D space by sampling radius using:
       - `r = sqrt((r_max^2 - r_min^2) * u + r_min^2)`
       - This accounts for increasing surface area as radius grows.
    """
    # Define bounds
    r_min, r_max = 1.0, 5.0  # Radius range
    lookat_min, lookat_max = -1.0, 1.0  # Look-at position bounds

    # radius sampling using quadratic distribution
    u = torch.rand(num_samples, device=device)
    radius = torch.sqrt((r_max**2 - r_min**2) * u + r_min**2)

    # Yaw: Uniform azimuth sampling
    yaw = torch.rand(num_samples, device=device) * (2 * torch.pi) - torch.pi

    # Pitch: Equal-area elevation sampling
    v = torch.rand(num_samples, device=device)
    pitch = torch.asin(2 * v - 1)

    # Sample look-at positions uniformly
    lookat_x = torch.rand(num_samples, device=device) * (lookat_max - lookat_min) + lookat_min
    lookat_y = torch.rand(num_samples, device=device) * (lookat_max - lookat_min) + lookat_min
    lookat_z = torch.rand(num_samples, device=device) * (lookat_max - lookat_min) + lookat_min

    # Combine into final tensor
    cameras = torch.stack([yaw, pitch, radius, lookat_x, lookat_y, lookat_z], dim=1)
    return cameras


def yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs, fovs):
    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]
    if not isinstance(rs, list):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, list):
        fovs = [fovs] * len(yaws)

    yaws = torch.tensor(yaws, dtype=torch.float32).cuda()
    pitchs = torch.tensor(pitchs, dtype=torch.float32).cuda()
    rs = torch.tensor(rs, dtype=torch.float32).cuda()
    fovs = torch.deg2rad(torch.tensor(fovs, dtype=torch.float32)).cuda()

    origs = torch.stack([
        torch.sin(yaws) * torch.cos(pitchs),
        torch.cos(yaws) * torch.cos(pitchs),
        torch.sin(pitchs),
    ], dim=1) * rs[:, None]

    lookat = torch.tensor([0, 0, 0], dtype=torch.float32).cuda()
    up = torch.tensor([0, 0, 1], dtype=torch.float32).cuda()

    extrinsics = utils3d.torch.extrinsics_look_at(origs, lookat.expand(origs.shape[0], -1), up.expand(origs.shape[0], -1))
    intrinsics = utils3d.torch.intrinsics_from_fov_xy(fovs, fovs)
    if not is_list:
        extrinsics = extrinsics[0]
        intrinsics = intrinsics[0]
    return extrinsics, intrinsics


# Particle Swarm Optimization implementation
def particle_swarm_optimization(fitness_function, bounds, num_particles=200, max_iter=25, init_samples=2000, use_init_samples=True, rmbg_image=None, renderer=None, prior_params=None, save_path=None, is_Hunyuan=False, use_vggt=False):
    dim = 6 # yaw, pitch, r, lookat_x, lookat_y, lookat_z

    if use_init_samples:
        '''Initialize with init_samples number, and then select the best num_particles number of samples for good inputs to the later dust3R optimization'''
        # Initialize swarm with random values within bounds
        # swarm = lower_bounds + (upper_bounds - lower_bounds) * torch.rand((init_samples, dim), device='cuda')
        swarm = sample_camera_positions(init_samples, device='cuda')
        if prior_params is not None:
            prior_params = torch.from_numpy(prior_params).cuda()
            swarm = torch.cat([prior_params.unsqueeze(0), swarm], dim=0)

        scores, _ = fitness_function(swarm, renderer)
        _, top_indices = torch.topk(scores, num_particles, largest=False)  
        swarm = swarm[top_indices]

        scores, renderings, extr, intr, pts3d = fitness_function(swarm[:7], renderer, return_more=True) # Total 8 samples including the reference image, thus select 8-1 samples

        # save the rendering for debug
        # for i, r in enumerate(renderings[:7]):
        #     Image.fromarray((r.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(save_path + f"_rendering_{i}.png")

        sample_images = torch.cat([rmbg_image.unsqueeze(0), renderings.to('cuda')], dim=0)

        if save_path is not None:
            Image.fromarray((renderings[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(save_path + f"_1_after_large_sampling.png")

        '''Leverage dust3R / VGGT to get a better initial guess'''
        try:
            aligned_R, aligned_T = dust3r_getting_better_initial_guess(sample_images, extr.detach(), pts3d.cuda(), device='cuda', params={'intrinsics': intr[0], 'renderer': renderer}, save_path=save_path, is_Hunyuan=is_Hunyuan, using_vggt=use_vggt)

            # Convert the aligned_R and aligned_T to yaw, pitch, r, lookat_x, lookat_y, lookat_z
            yaw, pitch, r, lookat = extrinsics_to_yaw_pitch_r_lookat(aligned_R, aligned_T)

            aligned_swarm = torch.cat([yaw.unsqueeze(1), pitch.unsqueeze(1), r.unsqueeze(1), lookat], dim=1)

            # return aligned_swarm[0].cpu().numpy()

            '''Concat with the initial samples, that is 1 + (num_samples - 1) = num_samples samples'''
            swarm = torch.cat([aligned_swarm, swarm[:num_particles-1]], dim=0)
        except:
            print({'Dust3R' if not use_vggt else 'VGGT'} + 'failed to get a better initial guess, use the initial samples')

        scores, renderings = fitness_function(swarm, renderer)

        if save_path is not None:
            Image.fromarray((renderings[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(save_path + f"_2_after_{'dust3r' if not use_vggt else 'vggt'}.png")

    else:
        # As init samples may sample all in the same region
        swarm = sample_camera_positions(num_particles, device='cuda')

    personal_best = swarm.clone()
    personal_best_scores = torch.full((num_particles,), float('inf'), device='cuda')
    global_best = None
    global_best_score = float('inf')

    par = tqdm(range(max_iter), desc='PSO Optimization', disable=False)
    for _ in par:
        velocities = torch.empty((num_particles, dim), device='cuda').uniform_(-1, 1)

        # Compute fitness scores for all particles in parallel
        scores, _ = fitness_function(swarm, renderer)  # Ensure fitness_function supports batch input
        
        # Update personal best
        better_scores = scores < personal_best_scores
        personal_best_scores[better_scores] = scores[better_scores]
        personal_best[better_scores] = swarm[better_scores]
        
        # Update global best
        min_score, min_idx = torch.min(scores, dim=0)
        if min_score < global_best_score:
            global_best_score = min_score.item()
            global_best = swarm[min_idx].clone()

        # Update velocities and positions
        r1, r2 = torch.rand((2, num_particles, dim), device='cuda')
        velocities = 0.5 * velocities + r1 * (personal_best - swarm) + r2 * (global_best - swarm)
        swarm += velocities

        par.set_postfix({'best_score': global_best_score})

    scores, renderings = fitness_function(global_best.unsqueeze(0).repeat(2,1), renderer)

    if save_path is not None:
        Image.fromarray((renderings[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(save_path + f"_3_after_PSO.png")

    return global_best.cpu().numpy()  # Return result to CPU for further processing


def optimize_yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov, lookat):
    lookat = lookat.cuda()
    r = r.cuda()
    fov = torch.deg2rad(torch.tensor(float(fov))).cuda()
    yaw = yaw.cuda()
    pitch = pitch.cuda()
    orig = torch.stack([
        torch.sin(yaw) * torch.cos(pitch),
        torch.cos(yaw) * torch.cos(pitch),
        torch.sin(pitch),
    ]).squeeze() * r
    extr = utils3d.torch.extrinsics_look_at(orig, lookat, torch.tensor([0, 0, 1]).float().cuda())
    intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
    extrinsics = extr
    intrinsics = intr
    return extrinsics, intrinsics


# Batch version of the above function
def batch_optimize_yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs, fovs, lookats):
    """
    yaws: torch.Tensor of shape [B]
    pitchs: torch.Tensor of shape [B]
    rs: torch.Tensor of shape [B]
    fovs: torch.Tensor of shape [B]
    lookats: torch.Tensor of shape [B, 3]
    """
    B = yaws.shape[0]
    lookats = lookats.cuda()
    rs = rs.cuda()
    fovs = torch.deg2rad(torch.tensor(fovs).expand(B)).cuda()
    yaws = yaws.cuda()
    pitchs = pitchs.cuda()
    origs = torch.cat([
        torch.sin(yaws) * torch.cos(pitchs),
        torch.cos(yaws) * torch.cos(pitchs),
        torch.sin(pitchs),
    ], dim=1) * rs
    extrs = utils3d.torch.extrinsics_look_at(origs, lookats, torch.tensor([0, 0, 1]).float().cuda())
    intrs = utils3d.torch.intrinsics_from_fov_xy(fovs, fovs)
    extrinsics = extrs
    intrinsics = intrs
    return extrinsics, intrinsics


def render_frames(sample, extrinsics, intrinsics, options={}, colors_overwrite=None, verbose=True, **kwargs):
    if isinstance(sample, Octree):
        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = options.get('resolution', 512)
        renderer.rendering_options.near = options.get('near', 0.8)
        renderer.rendering_options.far = options.get('far', 1.6)
        renderer.rendering_options.bg_color = options.get('bg_color', (0, 0, 0))
        renderer.rendering_options.ssaa = options.get('ssaa', 4)
        renderer.pipe.primitive = sample.primitive
    elif isinstance(sample, Gaussian):
        renderer = GaussianRenderer()
        renderer.rendering_options.resolution = options.get('resolution', 512)
        renderer.rendering_options.near = options.get('near', 0.8)
        renderer.rendering_options.far = options.get('far', 1.6)
        renderer.rendering_options.bg_color = options.get('bg_color', (0, 0, 0))
        renderer.rendering_options.ssaa = options.get('ssaa', 1)
        renderer.pipe.kernel_size = kwargs.get('kernel_size', 0.1)
        renderer.pipe.use_mip_gaussian = True
    elif isinstance(sample, MeshExtractResult):
        renderer = MeshRenderer()
        renderer.rendering_options.resolution = options.get('resolution', 512)
        renderer.rendering_options.near = options.get('near', 1)
        renderer.rendering_options.far = options.get('far', 100)
        renderer.rendering_options.ssaa = options.get('ssaa', 4)
        # white background
        renderer.rendering_options.bg_color = options.get('bg_color', (0, 0, 0))
    else:
        raise ValueError(f'Unsupported sample type: {type(sample)}')
    
    rets = {}
    for j, (extr, intr) in tqdm(enumerate(zip(extrinsics, intrinsics)), desc='Rendering', disable=not verbose):
        if not isinstance(sample, MeshExtractResult):
            res = renderer.render(sample, extr, intr, colors_overwrite=colors_overwrite)
            if 'color' not in rets: rets['color'] = []
            if 'depth' not in rets: rets['depth'] = []
            rets['color'].append(np.clip(res['color'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
            if 'percent_depth' in res:
                rets['depth'].append(res['percent_depth'].detach().cpu().numpy())
            elif 'depth' in res:
                rets['depth'].append(res['depth'].detach().cpu().numpy())
            else:
                rets['depth'].append(None)
        else:
            res = renderer.render(sample, extr, intr)
            if 'normal' not in rets: rets['normal'] = []
            rets['normal'].append(np.clip(res['normal'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
            if 'color' not in rets: rets['color'] = []
            rets['color'].append(np.clip(res['color'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
            if 'texture' not in rets: rets['texture'] = []
            try:
                rets['texture'].append(np.clip(res['texture'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
            except:
                pass
    return rets


def render_video(sample, resolution=512, bg_color=(0, 0, 0), num_frames=300, r=2, fov=40, **kwargs):
    yaws = torch.linspace(0, 2 * 3.1415, num_frames)
    pitch = 0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * 3.1415, num_frames))
    yaws = yaws.tolist()
    pitch = pitch.tolist()
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, r, fov)
    if 'init_extrinsics' in kwargs and kwargs['init_extrinsics'] is not None:
        init_extrinsics_inv = kwargs['init_extrinsics'].squeeze().transpose(-1, -2).inverse().transpose(-1, -2)
        extrinsics = extrinsics @ init_extrinsics_inv

    return render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, **kwargs)


def render_multiview(sample, resolution=512, nviews=30, init_extrinsics=None, return_canonical=False):
    if return_canonical:
        fov = 40
        _, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics([0], [0], 0, fov)
        extrinsics = torch.eye(4).unsqueeze(0).expand(nviews, -1, -1).cuda()
    else:
        r = 2
        fov = 40
        cams = [sphere_hammersley_sequence(i, nviews) for i in range(nviews)]
        yaws = [cam[0] for cam in cams]
        pitchs = [cam[1] for cam in cams]
        extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
        if init_extrinsics is not None:
            init_extrinsics_inv = init_extrinsics.squeeze().transpose(-1, -2).inverse().transpose(-1, -2)
            extrinsics = extrinsics @ init_extrinsics_inv

    res = render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': (0, 0, 0)})
    return res['color'], extrinsics, intrinsics


def render_multiview_gradient(sample, resolution=512, nviews=30, init_extrinsics=None, random_indices=[], envmap=None, return_type='color', input_intr_extrincs=None, return_intr_extrincs=False, return_rast_vertices=False):
    assert return_type in ['color', 'envmap', 'normal_map', 'normal', 'mask'], f"Unsupported return type: {return_type}"
    assert envmap is not None or return_type != 'envmap', "envmap must be provided for rendering envmap"

    if input_intr_extrincs is None:
        r = 2
        fov = 40
        cams = [sphere_hammersley_sequence(i, nviews) for i in range(nviews)]
        yaws = [cam[0] for cam in cams]
        pitchs = [cam[1] for cam in cams]
        extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
        if init_extrinsics is not None:
            init_extrinsics_inv = init_extrinsics.squeeze().transpose(-1, -2).inverse().transpose(-1, -2)
            extrinsics = extrinsics @ init_extrinsics_inv
    else:
        intrinsics, extrinsics = input_intr_extrincs

    if return_intr_extrincs:
        return intrinsics, extrinsics

    extrinsics = extrinsics[random_indices]
    intrinsics = intrinsics[random_indices]

    options = {'resolution': resolution, 'bg_color': (0, 0, 0)}

    renderer = MeshRenderer()
    renderer.rendering_options.resolution = options.get('resolution', 512)
    renderer.rendering_options.near = options.get('near', 1)
    renderer.rendering_options.far = options.get('far', 100)
    '''Since not compare to reference image here (the compared two renderings adopt the same rendering setting), we use SSAA = 1 to save time'''
    renderer.rendering_options.ssaa = options.get('ssaa', 1)

    if return_type == 'envmap':
        sh = SphericalHarmonics(envmap)
        # Render background for all viewpoints once
        bgs = sh.render_backgrounds(envmap, resolution * options.get('ssaa', 1), fov, extrinsics)

    if return_rast_vertices:
        rast_batch, full_proj_batch = renderer.render_batch(sample, extrinsics, intrinsics, return_types = [return_type], params={"sh": sh, "bgs": bgs} if return_type == 'envmap' else {}, return_rast_vertices=return_rast_vertices)
        return rast_batch, full_proj_batch

    res = renderer.render_batch(sample, extrinsics, intrinsics, return_types = [return_type], params={"sh": sh, "bgs": bgs} if return_type == 'envmap' else {})

    return torch.clip(res[return_type], 0., 1.)


def render_snapshot(samples, resolution=512, bg_color=(0, 0, 0), offset=(-16 / 180 * np.pi, 20 / 180 * np.pi), r=10, fov=8, **kwargs):
    yaw = [0, np.pi/2, np.pi, 3*np.pi/2]
    yaw_offset = offset[0]
    yaw = [y + yaw_offset for y in yaw]
    pitch = [offset[1] for _ in range(4)]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov)
    return render_frames(samples, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, **kwargs)


# Optimize the camera paramters to find the most aligned camera pose with the rmbg image
def find_closet_camera_pos(sample, rmbg_image, resolution=518, bg_color=(0, 0, 0), iterations=100, params=None, return_optimize=False, prior_params=None, save_path=None, is_Hunyuan=False, use_vggt=False):
    fov = 40

    options = {'resolution': resolution, 'bg_color': bg_color}
    if isinstance(sample, Gaussian):
        renderer = GaussianRenderer()
        renderer.rendering_options.resolution = options.get('resolution', 512)
        renderer.rendering_options.near = options.get('near', 0.8)
        renderer.rendering_options.far = options.get('far', 1.6)
        renderer.rendering_options.bg_color = options.get('bg_color', (0, 0, 0))
        renderer.rendering_options.ssaa = options.get('ssaa', 1)
        renderer.pipe.kernel_size = 0.1
        renderer.pipe.use_mip_gaussian = True
    elif isinstance(sample, MeshExtractResult):
        renderer = MeshRenderer()
        renderer.rendering_options.resolution = options.get('resolution', 512)
        renderer.rendering_options.near = options.get('near', 1)
        renderer.rendering_options.far = options.get('far', 100)
        '''Use SSAA = 4 will help speed up the optimization, however also longer the time for each iteration. Since the reference image is also smoothed. thus we use SSAA = 4 here'''
        renderer.rendering_options.ssaa = options.get('ssaa', 4)
    else:
        raise ValueError(f'Unsupported sample type: {type(sample)}')

    # If params is provided, directly render the image with the given camera parameters
    if params is not None:
        yaw, pitch, r, lookat_x, lookat_y, lookat_z = params
        yaw = torch.tensor([yaw], dtype=torch.float32).cuda()
        pitch = torch.tensor([pitch], dtype=torch.float32).cuda()
        r = torch.tensor([r], dtype=torch.float32).cuda()
        lookat = torch.tensor([lookat_x, lookat_y, lookat_z], dtype=torch.float32).cuda()

        # Get the final rendering
        extr, intr = optimize_yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov, lookat)
        if not isinstance(sample, MeshExtractResult):
            res = renderer.render(sample, extr, intr, colors_overwrite=None)
        else:
            res = renderer.render(sample, extr, intr, return_types = ["color"])
        if return_optimize:
            ret = res['color']
        else:
            ret = np.clip(res['color'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        
        return ret, params
    
    device = "cuda"
    dreamsim_model, _ = dreamsim(pretrained=True, device=device)

    rmbg_image = torch.tensor(np.array(rmbg_image)).float().cuda().permute(2, 0, 1) / 255

    # Get foreground mask from the rmbg image (which is not black)
    mask = (rmbg_image.sum(dim=0) > 0).float()

    # Batch version of the above function. Set torch.no_grad() to avoid memory leak, as PSO does not require gradients
    @torch.no_grad()
    def fitness_batch(params, renderer, sub_batch_size=10, return_more=False):
        '''
        params: batch of camera parameters [yaw, pitch, r, lookat_x, lookat_y, lookat_z], shape: [B, 6]
        sub_batch_size: batch size for rendering, too large batch size may cause OOM and overflow
        '''
        params = torch.tensor(params, dtype=torch.float32).cuda()
        yaw, pitch, r, lookat_x, lookat_y, lookat_z = params.chunk(6, dim=1)
        lookat = torch.cat([lookat_x, lookat_y, lookat_z], dim=1)
        extr, intr = batch_optimize_yaw_pitch_r_fov_to_extrinsics_intrinsics(
            yaw, pitch, r, fov, lookat
        )
        losses = []
        renderings = []
        pts3d = []
        for i in range(0, params.shape[0], sub_batch_size):
            sub_extr = extr[i:i+sub_batch_size]
            sub_intr = intr[i:i+sub_batch_size]
            if not isinstance(sample, MeshExtractResult):
                raise ValueError("Only mesh is supported for now")
            else:
                res = renderer.render_batch(sample, sub_extr, sub_intr, return_types = ["mask", "color"])

            rendering = torch.clip(res['color'], 0., 1.)

            # convert rendering and rmbg_image to PIL image
            # use mask to avoid overfitting, since some little translation of the image may lead to better metric, but not the real alignment
            loss = dreamsim_model(F.interpolate(rendering * mask, (224, 224), mode='bicubic'), F.interpolate(rmbg_image.unsqueeze(0), (224, 224), mode='bicubic'))

            # loss_boundary: ensure the mask region of rendering and rmbg_image are the same with the loss weight of total_img_size / mask_size[0, 1, : n]
            loss_boundary = torch.nn.MSELoss(reduction="none")(res['mask'], mask.unsqueeze(0).expand(res['mask'].shape)).mean(dim=(1, 2)) * (mask.numel() / mask.sum()) * 0.1

            loss = loss + loss_boundary
            
            losses.append(loss)
            renderings.append(rendering.detach().cpu())

            if return_more:
                res = renderer.render_batch(sample, sub_extr, sub_intr, return_types = ["points3Dpos"])
                pts3d.append(res['points3Dpos'].detach().cpu())
        
        renderings = torch.cat(renderings)
        losses = torch.cat(losses)
        # save the rendering for debug
        # for i, r in enumerate(renderings[:10]):
        #     Image.fromarray((r.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(f"rendering_{i}.png")
        if return_more:
            return losses, renderings, extr, intr, torch.cat(pts3d)
        return losses, renderings
    
    "========== PSO Optimization =========="
    bounds = None
    
    loss1 = torch.nn.MSELoss()
    loss2 = lpips.LPIPS(net='vgg').cuda()

    # Coarse optimization with PSO
    coarse_params = particle_swarm_optimization(fitness_batch, bounds, rmbg_image=rmbg_image, renderer=renderer, prior_params=prior_params, save_path=save_path, is_Hunyuan=is_Hunyuan, use_vggt=use_vggt)
    yaw, pitch, r, lookat_x, lookat_y, lookat_z = coarse_params

    "========== Gradient Descent Optimization =========="
    # Fine-tune with gradient descent
    yaw = torch.nn.Parameter(torch.tensor([yaw], dtype=torch.float32))
    pitch = torch.nn.Parameter(torch.tensor([pitch], dtype=torch.float32))
    r = torch.nn.Parameter(torch.tensor([r], dtype=torch.float32))
    lookat = torch.nn.Parameter(torch.tensor([lookat_x, lookat_y, lookat_z], dtype=torch.float32))

    optimizer = torch.optim.Adam([yaw, pitch, lookat, r], lr=0.01)
    
    '''Gradient Descent Optimization is unstable here, sometimes will totally destroy the good position found by PSO, thus disable it'''
    par = tqdm(range(300), desc='Aligning', disable=False)
    for iter in par:
        extr, intr = optimize_yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov, lookat)

        if not isinstance(sample, MeshExtractResult):
            res = renderer.render(sample, extr, intr, colors_overwrite=None)
        else:
            res = renderer.render(sample, extr, intr, return_types = ["mask"])

        # loss_boundary: ensure the mask region of rendering and rmbg_image are the same with the loss weight of total_img_size / mask_size
        loss = torch.nn.MSELoss()(res['mask'], mask.expand(res['mask'].shape)) * (mask.numel() / mask.sum())

        par.set_postfix({'loss': loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print("Yaw: ", yaw.item(), "Pitch: ", pitch.item(), "Lookat: ", lookat, "Radius: ", r.item())

    # Get the final rendering
    extr, intr = optimize_yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov, lookat)
    if not isinstance(sample, MeshExtractResult):
        res = renderer.render(sample, extr, intr, colors_overwrite=None)
        ret = np.clip(res['color'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
    else:
        res = renderer.render(sample, extr, intr)
        ret = np.clip(res['color'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
    
    return ret, torch.stack([yaw[0], pitch[0], r[0], lookat[0], lookat[1], lookat[2]]).detach().squeeze().cpu().numpy()


class SphericalHarmonics:
    """
    Environment map approximation using spherical harmonics.

    This class implements the spherical harmonics lighting model of [Ramamoorthi
    and Hanrahan 2001], that approximates diffuse lighting by an environment map.
    """

    def __init__(self, envmap):
        """
        Precompute the coefficients given an envmap.

        Parameters
        ----------
        envmap : torch.Tensor
            The environment map to approximate.
        """
        h,w = envmap.shape[:2]

        # Compute the grid of theta, phi values
        theta = (torch.linspace(0, np.pi, h, device='cuda')).repeat(w, 1).t()
        phi = (torch.linspace(3*np.pi, np.pi, w, device='cuda')).repeat(h,1)

        # Compute the value of sin(theta) once
        sin_theta = torch.sin(theta)
        # Compute x,y,z
        # This differs from the original formulation as here the up axis is Y
        x = sin_theta * torch.cos(phi)
        z = -sin_theta * torch.sin(phi)
        y = torch.cos(theta)

        # Compute the polynomials
        Y_0 = 0.282095
        # The following are indexed so that using Y_n[-p]...Y_n[p] gives the proper polynomials
        Y_1 = [
            0.488603 * z,
            0.488603 * x,
            0.488603 * y
            ]
        Y_2 = [
            0.315392 * (3*z.square() - 1),
            1.092548 * x*z,
            0.546274 * (x.square() - y.square()),
            1.092548 * x*y,
            1.092548 * y*z
        ]
        import matplotlib.pyplot as plt
        area = w*h
        radiance = envmap[..., :3]
        dt_dp = 2.0 * np.pi**2 / area

        # Compute the L coefficients
        L = [ [(radiance * Y_0 * (sin_theta)[..., None] * dt_dp).sum(dim=(0,1))],
            [(radiance * (y * sin_theta)[..., None] * dt_dp).sum(dim=(0,1)) for y in Y_1],
            [(radiance * (y * sin_theta)[..., None] * dt_dp).sum(dim=(0,1)) for y in Y_2]]

        # Compute the R,G and B matrices
        c1 = 0.429043
        c2 = 0.511664
        c3 = 0.743125
        c4 = 0.886227
        c5 = 0.247708

        self.M = torch.stack([
            torch.stack([ c1 * L[2][2] , c1 * L[2][-2], c1 * L[2][1] , c2 * L[1][1]           ]),
            torch.stack([ c1 * L[2][-2], -c1 * L[2][2], c1 * L[2][-1], c2 * L[1][-1]          ]),
            torch.stack([ c1 * L[2][1] , c1 * L[2][-1], c3 * L[2][0] , c2 * L[1][0]           ]),
            torch.stack([ c2 * L[1][1] , c2 * L[1][-1], c2 * L[1][0] , c4 * L[0][0] - c5 * L[2][0]])
        ]).movedim(2,0)

    def eval(self, n):
        """
        Evaluate the shading using the precomputed coefficients.

        Parameters
        ----------
        n : torch.Tensor
            Array of normals at which to evaluate lighting.
        """
        normal_array = n.view((-1, 3))
        h_n = torch.nn.functional.pad(normal_array, (0,1), 'constant', 1.0)
        l = (h_n.t() * (self.M @ h_n.t())).sum(dim=1)
        return l.t().view(n.shape)
    
    def render_backgrounds(self, envmap, res, fov_x, view_mats):
        """
        Precompute the background of each input viewpoint with the envmap.

        Params
        ------
        envmap : torch.Tensor
            The environment map used in the scene.
        """
        h = w = res
        pos_int = torch.arange(w*h, dtype = torch.int32, device='cuda')
        pos = 0.5 - torch.stack((pos_int % w, pos_int // w), dim=1) / torch.tensor((w,h), device='cuda')
        a = np.deg2rad(fov_x)/2
        r = w/h
        f = torch.tensor((2*np.tan(a),  2*np.tan(a)/r), device='cuda', dtype=torch.float32)
        rays = torch.cat((pos*f, torch.ones((w*h,1), device='cuda'), torch.zeros((w*h,1), device='cuda')), dim=1)
        rays_norm = (rays.transpose(0,1) / torch.norm(rays, dim=1)).transpose(0,1)
        rays_view = torch.matmul(rays_norm, view_mats.inverse().transpose(1,2)).reshape((view_mats.shape[0],h,w,-1))
        theta = torch.acos(rays_view[..., 1])
        phi = torch.atan2(rays_view[..., 0], rays_view[..., 2])
        envmap_uvs = torch.stack([0.75-phi/(2*np.pi), theta / np.pi], dim=-1)
        bgs = dr.texture(envmap[None, ...], envmap_uvs, filter_mode='linear').flip(1)
        bgs[..., -1] = 0 # Set alpha to 0

        return bgs