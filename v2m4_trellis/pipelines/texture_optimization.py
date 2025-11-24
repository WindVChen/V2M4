import torch
import numpy as np
import imageio
import xatlas
import utils3d
from tqdm import tqdm
import nvdiffrast.torch as dr
from v2m4_trellis.utils import render_utils
from v2m4_trellis.utils.general_utils import *


def collect_mesh_views(args, texture_optim_input_list, extrinsics_list, output_path, base_name_list, outputs_list):
    # Rendering multiple views for the first mesh, while one view (aligned with the reference image) for the rest meshes. Assinging larger weights to the reference image aligned renderings.

    near = 0.1,
    far = 10.0,

    vertices = texture_optim_input_list[0]['mesh'][0].vertices.cpu().numpy()
    faces = texture_optim_input_list[0]['mesh'][0].faces.cpu().numpy()

    # Compute in advance, since all the meshes share the same topology (as they optimized from the same first mesh)
    # parametrize mesh (this will destroy the watertightness of the mesh)
    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)

    '''Collect all the data for texture optimization'''
    add_weight_index = []
    all_masks = []
    all_observations = []
    all_uv = []
    all_uv_dr = []


    for i, (extr, outputs) in enumerate(zip(extrinsics_list, texture_optim_input_list)):
        fix_geometry = True

        vertices = outputs['mesh'][0].vertices.cpu().numpy()
        faces = outputs['mesh'][0].faces.cpu().numpy()

        vertices = vertices[vmapping]
        faces = indices

        # bake texture
        # render the mesh from the aligned view
        observations, extrinsics, intrinsics = render_utils.render_multiview(outputs['gaussian'][0] if args.model == "TRELLIS" else outputs['mesh'][0], resolution=1024,  nviews=1, init_extrinsics=extr, return_canonical=True)
        if args.model == "TRELLIS":
            imageio.imsave(output_path + "/" + base_name_list[i] + "_gs_view_after_slat_optimization.png", (observations[0]).astype(np.uint8))
        if i == 0:
            # render the mesh from multiple views
            observations_mul, extrinsics_mul, intrinsics_mul = render_utils.render_multiview(outputs['gaussian'][0] if args.model == "TRELLIS" else outputs['mesh'][0], resolution=1024, nviews=100, init_extrinsics=extr)

            observations.extend(observations_mul)
            extrinsics = torch.cat([extrinsics, extrinsics_mul], dim=0)
            intrinsics = torch.cat([intrinsics, intrinsics_mul], dim=0)

        add_weight_index.extend([1] + [0] * (len(observations) - 1))
            
        masks = [np.any(observation > 0, axis=-1) for observation in observations]
        extrinsics = [extrinsics[id].cpu().numpy() for id in range(len(extrinsics))]
        intrinsics = [intrinsics[id].cpu().numpy() for id in range(len(intrinsics))]

        vertices = torch.tensor(vertices).cuda()
        faces = torch.tensor(faces.astype(np.int32)).cuda()
        uvs = torch.tensor(uvs).cuda()
        observations = [torch.tensor(obs / 255.0).float().cuda() for obs in observations]
        masks = [torch.tensor(m>0).bool().cuda() for m in masks]
        views = [utils3d.torch.extrinsics_to_view(torch.tensor(extr).cuda()) for extr in extrinsics]
        projections = [utils3d.torch.intrinsics_to_perspective(torch.tensor(intr).cuda(), near, far) for intr in intrinsics]

        rastctx = utils3d.torch.RastContext(backend='cuda')
        observations = [observations.flip(0) for observations in observations]
        masks = [m.flip(0) for m in masks]

        if args.model == "Hunyuan" or args.model == "TripoSG" or args.model == "Craftsman":
            for idnx, (view, projection) in tqdm(enumerate(zip(views, projections)), total=len(views), disable=False, desc=f'Re-gain observations and masks for {args.model}'):
                with torch.no_grad():
                    rast = utils3d.torch.rasterize_triangle_faces(
                        rastctx, outputs_list[i]['mesh_genTex'][0].vertices[None], outputs_list[i]['mesh_genTex'][0].faces.int(), observations[0].shape[1], observations[0].shape[0], uv=outputs_list[i]['mesh_genTex'][0].uv[None], view=view, projection=projection
                    )
                observations[idnx] = dr.texture(outputs_list[i]['mesh_genTex'][0].texture.flip(0).unsqueeze(0), rast['uv'].detach(), rast['uv_dr'].detach())[0].float().cuda()
                masks[idnx] = rast['mask'].detach().bool().cuda().squeeze().flip(0)

        _uv = []
        _uv_dr = []
        for observation, view, projection in tqdm(zip(observations, views, projections), total=len(views), disable=False, desc='Collect all UV data for texture optimization'):
            with torch.no_grad():
                rast = utils3d.torch.rasterize_triangle_faces(
                    rastctx, vertices[None], faces, observation.shape[1], observation.shape[0], uv=uvs[None], view=view, projection=projection
                )
                _uv.append(rast['uv'].detach())
                _uv_dr.append(rast['uv_dr'].detach())

        all_observations.extend(observations)
        all_masks.extend(masks)
        all_uv.extend(_uv)
        all_uv_dr.extend(_uv_dr)

    return all_observations, all_masks, all_uv, all_uv_dr, add_weight_index, vmapping, indices, uvs, rastctx, faces


def texture_optimization(all_observations, all_masks, all_uv, all_uv_dr, add_weight_index, uvs, texture_size, rastctx, faces):
    lambda_tv=0.01
    texture = torch.nn.Parameter(torch.zeros((1, texture_size, texture_size, 3), dtype=torch.float32).cuda())
    optimizer = torch.optim.Adam([texture], betas=(0.5, 0.9), lr=1e-2)

    total_steps = 2500
    with tqdm(total=total_steps, disable=False, desc='Texture Optimization') as pbar:
        for step in range(total_steps):
            optimizer.zero_grad()
            selected = np.random.randint(0, len(all_uv))

            uv, uv_dr, observation, mask = all_uv[selected], all_uv_dr[selected], all_observations[selected], all_masks[selected]
            render = dr.texture(texture, uv, uv_dr)[0]
            loss = torch.nn.functional.l1_loss(render[mask], observation[mask])
            if lambda_tv > 0:
                loss += lambda_tv * tv_loss(texture)

            # Apply weight to the loss
            loss *= 1 + add_weight_index[selected]
            loss.backward()
            optimizer.step()
            # annealing
            optimizer.param_groups[0]['lr'] = cosine_anealing(optimizer, step, total_steps, 1e-2, 1e-5)
            pbar.set_postfix({'loss': loss.item()})
            pbar.update()
    texture = np.clip(texture[0].flip(0).detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
    mask = 1 - utils3d.torch.rasterize_triangle_faces(
        rastctx, (uvs * 2 - 1)[None], faces, texture_size, texture_size
    )['mask'][0].detach().cpu().numpy().astype(np.uint8)
    texture = cv2.inpaint(texture, mask, 3, cv2.INPAINT_TELEA)
    texture = Image.fromarray(texture)

    return texture