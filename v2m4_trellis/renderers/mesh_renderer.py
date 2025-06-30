import torch
try:
    import kaolin as kal
    import nvdiffrast.torch as dr
except :
    print("Kaolin and nvdiffrast are not installed. Please install them to use the mesh renderer.")
from easydict import EasyDict as edict
from ..representations.mesh import MeshExtractResult
import torch.nn.functional as F
import numpy as np


def intrinsics_to_projection(
        intrinsics: torch.Tensor,
        near: float,
        far: float,
    ) -> torch.Tensor:
    """
    OpenCV intrinsics to OpenGL perspective matrix

    Args:
        intrinsics (torch.Tensor): [3, 3] OpenCV intrinsics matrix
        near (float): near plane to clip
        far (float): far plane to clip
    Returns:
        (torch.Tensor): [4, 4] OpenGL perspective matrix
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = - 2 * cy + 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.
    return ret


class MeshRenderer:
    """
    Renderer for the Mesh representation.

    Args:
        rendering_options (dict): Rendering options.
        glctx (nvdiffrast.torch.RasterizeGLContext): RasterizeGLContext object for CUDA/OpenGL interop.
        """
    def __init__(self, rendering_options={}, device='cuda'):
        self.rendering_options = edict({
            "resolution": None,
            "near": None,
            "far": None,
            "ssaa": 1
        })
        self.rendering_options.update(rendering_options)
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.device=device

    def deepcopy(self):
        """
        Create a deepcopy of the renderer.

        Returns:
            MeshRenderer: Deepcopy of the renderer.
        """
        return MeshRenderer(self.rendering_options, self.device)
        
    def render(
            self,
            mesh : MeshExtractResult,
            extrinsics: torch.Tensor,
            intrinsics: torch.Tensor,
            return_types = ["mask", "normal", "depth", "color", "texture"]
        ) -> edict:
        """
        Render the mesh.

        Args:
            mesh : meshmodel
            extrinsics (torch.Tensor): (4, 4) camera extrinsics
            intrinsics (torch.Tensor): (3, 3) camera intrinsics
            return_types (list): list of return types, can be "mask", "depth", "normal_map", "normal", "color"

        Returns:
            edict based on return_types containing:
                color (torch.Tensor): [3, H, W] rendered color image
                depth (torch.Tensor): [H, W] rendered depth image
                normal (torch.Tensor): [3, H, W] rendered normal image
                normal_map (torch.Tensor): [3, H, W] rendered normal map image
                mask (torch.Tensor): [H, W] rendered mask image
        """
        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        ssaa = self.rendering_options["ssaa"]
        
        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            default_img = torch.zeros((1, resolution, resolution, 3), dtype=torch.float32, device=self.device)
            ret_dict = {k : default_img if k in ['normal', 'normal_map', 'color'] else default_img[..., :1] for k in return_types}
            return ret_dict
        
        perspective = intrinsics_to_projection(intrinsics, near, far)
        
        RT = extrinsics.unsqueeze(0)
        full_proj = (perspective @ extrinsics).unsqueeze(0)
        
        vertices = mesh.vertices.unsqueeze(0)

        vertices_homo = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
        vertices_camera = torch.bmm(vertices_homo, RT.transpose(-1, -2))
        vertices_clip = torch.bmm(vertices_homo, full_proj.transpose(-1, -2))
        faces_int = mesh.faces.int()
        rast, rast_db = dr.rasterize(
            self.glctx, vertices_clip, faces_int, (resolution * ssaa, resolution * ssaa), grad_db=True)
        
        out_dict = edict()
        for type in return_types:
            img = None
            if type == "mask" :
                img = dr.antialias((rast[..., -1:] > 0).float(), rast, vertices_clip, faces_int)
            elif type == "depth":
                img = dr.interpolate(vertices_camera[..., 2:3].contiguous(), rast, faces_int)[0]
                img = dr.antialias(img, rast, vertices_clip, faces_int)
            elif type == "normal" :
                img = dr.interpolate(
                    mesh.face_normal.reshape(1, -1, 3), rast,
                    torch.arange(mesh.faces.shape[0] * 3, device=self.device, dtype=torch.int).reshape(-1, 3)
                )[0]
                img = dr.antialias(img, rast, vertices_clip, faces_int)
                # normalize norm pictures
                img = (img + 1) / 2
                mask = dr.antialias((rast[..., -1:] > 0).float(), rast, vertices_clip, faces_int)
                img = torch.where(mask > 0, img, torch.ones_like(img))
            elif type == "normal_map" :
                img = dr.interpolate(mesh.vertex_attrs[:, 3:].contiguous(), rast, faces_int)[0]
                img = dr.antialias(img, rast, vertices_clip, faces_int)
            elif type == "color" :
                img = dr.interpolate(mesh.vertex_attrs[:, :3].contiguous(), rast, faces_int)[0]
                img = dr.antialias(img, rast, vertices_clip, faces_int)
            elif type == "texture":
                try:
                    uv_map, uv_map_dr = dr.interpolate(mesh.uv, rast, faces_int, rast_db, diff_attrs='all')
                    img = dr.texture(mesh.texture.unsqueeze(0), uv_map, uv_map_dr)
                    # using mask to filter out the texture, set the background to pure white
                    mask = dr.antialias((rast[..., -1:] > 0).float(), rast, vertices_clip, faces_int)
                    img = torch.where(mask > 0, img, torch.ones_like(img))
                except Exception as e:
                    print(e)
                    continue

            if ssaa > 1:
                img = F.interpolate(img.permute(0, 3, 1, 2), (resolution, resolution), mode='bilinear', align_corners=False, antialias=True)
                img = img.squeeze()
            else:
                img = img.permute(0, 3, 1, 2).squeeze()
            out_dict[type] = img

        return out_dict

    # Allow for batch differentiable rendering
    def render_batch(
        self,
        mesh: MeshExtractResult,
        extrinsics_batch: torch.Tensor,
        intrinsics_batch: torch.Tensor,
        return_types=["mask", "normal", "depth"],
        params={},
        return_rast_vertices=False,
    ) -> list:
        """
        Render the mesh for a batch of camera extrinsics and intrinsics.

        Args:
            mesh: MeshExtractResult object representing the mesh.
            extrinsics_batch (torch.Tensor): [B, 4, 4] batch of camera extrinsics.
            intrinsics_batch (torch.Tensor): [B, 3, 3] batch of camera intrinsics.
            return_types (list): List of return types; can include "mask", "depth", "normal_map", "normal", "color".

        Returns:
            list[edict]: A list of results, one edict per batch item, containing the requested image types.
        """
        resolution = self.rendering_options["resolution"]
        near = self.rendering_options["near"]
        far = self.rendering_options["far"]
        ssaa = self.rendering_options["ssaa"]
        batch_size = extrinsics_batch.shape[0]

        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            default_img = torch.zeros((batch_size, resolution, resolution, 3), dtype=torch.float32, device=self.device)
            ret_list = [{k: default_img if k in ['normal', 'normal_map', 'color'] else default_img[..., :1] for k in return_types} for _ in range(batch_size)]
            return ret_list

        perspective_batch = torch.stack([intrinsics_to_projection(intrinsics, near, far) for intrinsics in intrinsics_batch])
        full_proj_batch = torch.bmm(perspective_batch, extrinsics_batch)

        vertices = mesh.vertices.unsqueeze(0).expand(batch_size, -1, -1)
        vertices_homo = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
        vertices_camera_batch = torch.bmm(vertices_homo, extrinsics_batch.transpose(-1, -2))
        vertices_clip_batch = torch.bmm(vertices_homo, full_proj_batch.transpose(-1, -2))
        faces_int = mesh.faces.int()

        rast_batch, _ = dr.rasterize(self.glctx, vertices_clip_batch, faces_int, (resolution * ssaa, resolution * ssaa))
        if return_rast_vertices:
            return rast_batch, full_proj_batch

        out_dict = edict()

        for type in return_types:
            img = None
            if type == "mask":
                img = dr.antialias((rast_batch[..., -1:] > 0).float(), rast_batch, vertices_clip_batch, faces_int)
            elif type == "depth":
                img = dr.interpolate(vertices_camera_batch[..., 2:3].contiguous(), rast_batch, faces_int)[0]
                img = dr.antialias(img, rast_batch, vertices_clip_batch, faces_int)
            elif type == "normal":
                img = dr.interpolate(
                    mesh.face_normal.reshape(1, -1, 3), rast_batch,
                    torch.arange(mesh.faces.shape[0] * 3, device=self.device, dtype=torch.int).reshape(-1, 3)
                )[0]
                img = (img + 1) / 2
                img = dr.antialias(img, rast_batch, vertices_clip_batch, faces_int, pos_gradient_boost=3)
            elif type == "normal_map":
                img = dr.interpolate(mesh.vertex_attrs[:, 3:].contiguous(), rast_batch, faces_int)[0]
                img = (img + 1) / 2
                img = dr.antialias(img, rast_batch, vertices_clip_batch, faces_int, pos_gradient_boost=3)
            elif type == "color":
                img = dr.interpolate(mesh.vertex_attrs[:, :3].contiguous(), rast_batch, faces_int)[0]
                img = dr.antialias(img, rast_batch, vertices_clip_batch, faces_int, pos_gradient_boost=3)
            elif type == "envmap":
                # Sample envmap at each vertex using the SH approximation
                vert_light = params['sh'].eval(mesh.vertex_attrs[:, 3:].contiguous()).contiguous()
                # Sample incoming radiance
                light = dr.interpolate(vert_light[None, ...], rast_batch, faces_int)[0]

                col = torch.cat((light / torch.pi, torch.ones((*light.shape[:-1],1), device='cuda')), dim=-1)
                img = dr.antialias(torch.where(rast_batch[..., -1:] != 0, col, params['bgs']), rast_batch, vertices_clip_batch, faces_int, pos_gradient_boost=3)[..., :-1]

                # vert_light = torch.ones_like(mesh.vertex_attrs[:, 3:])
                # col = dr.interpolate(vert_light[None, ...], rast_batch, faces_int)[0]
                # img = dr.antialias(col, rast_batch, vertices_clip_batch, faces_int, pos_gradient_boost=3)                
            
            '''Manually calculate the rendering process to visualize the rendering process'''
            # from PIL import Image
            # from torchvision.transforms import ToPILImage
            # to_pil = ToPILImage()

            # timg = to_pil(img[0].permute(2, 0, 1).detach().cpu())
            # timg.save("test.png")

            # mask = dr.antialias((rast_batch[..., -1:] > 0).float(), rast_batch, vertices_clip_batch, faces_int)
            # mimg = to_pil(mask[0].permute(2, 0, 1).detach().cpu())
            # mimg.save("mask.png")

            # img = dr.interpolate(mesh.vertices, rast_batch, faces_int)[0]
            # img = dr.antialias(img, rast_batch, vertices_clip_batch, faces_int, pos_gradient_boost=3)

            # pt3 = img[0][mask[0].squeeze() > 0].reshape(-1, 3)

            # pt3 = torch.cat([pt3, torch.ones_like(pt3[..., :1])], dim=-1)
            # pt3 = torch.matmul(pt3, full_proj_batch[0].transpose(-1, -2))

            # converted_pcd = pt3[..., :2] / pt3[..., -1:]
            # converted_pcd = (converted_pcd + 1) * mimg.size[0] / 2
            # converted_pcd = converted_pcd.clamp(min=0, max=mimg.size[0])
            # # init a black image and then asign the color of the point cloud
            # cimg = torch.ones((mimg.size[0], mimg.size[0], 3))
            # converted_pcd = converted_pcd.reshape(-1, 2).cpu().long().numpy()

            # # use converted_pcd.long() to index the image and then assign the corresponding color from images[0] to the indexed position
            # cimg[converted_pcd[..., 1], converted_pcd[..., 0]] = torch.tensor(np.array(timg))[mask[0].cpu().squeeze() > 0].reshape(-1, 3).float() / 255

            # cimg = cimg.reshape(mimg.size[0], mimg.size[0], 3)

            # cimg = to_pil(cimg.permute(2, 0, 1).detach().cpu())
            # cimg.save("cimg.png")
            
            '''Return the 3D points in the world coordinate space of the rendered image'''
            if type == "points3Dpos":
                img = dr.interpolate(mesh.vertices, rast_batch, faces_int)[0]
                img = dr.antialias(img, rast_batch, vertices_clip_batch, faces_int, pos_gradient_boost=3)

                mask = dr.antialias((rast_batch[..., -1:] > 0).float(), rast_batch, vertices_clip_batch, faces_int)

                # Since the znear zfar and etc has been determined, resolution here is on the fine-grained level of sampling, not spatial resolution
                if ssaa > 1:
                    img = F.interpolate(img.permute(0, 3, 1, 2), (resolution, resolution), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                    mask = F.interpolate(mask.permute(0, 3, 1, 2), (resolution, resolution), mode='nearest').permute(0, 2, 3, 1)

                pt3 = img[mask.squeeze() > 0].reshape(-1, 3)

                out_dict[type] = pt3
            else:
                if ssaa > 1:
                    img = F.interpolate(img.permute(0, 3, 1, 2), (resolution, resolution), mode='bilinear', align_corners=False, antialias=True)
                    img = img.squeeze()
                else:
                    img = img.permute(0, 3, 1, 2).squeeze()
                out_dict[type] = img

        return out_dict
