import os
import imageio
import torch
import numpy as np
from v2m4_trellis.utils.render_utils import render_frames
from natsort import ns, natsorted
import trimesh
import argparse
import dill
from v2m4_trellis.representations.mesh import MeshExtractResult
import utils3d.torch as utils3d


def parse_args():
    parser = argparse.ArgumentParser(description='Trellis Benchmark')
    parser.add_argument('--baseline', action='store_true', help='Run the baseline model')
    parser.add_argument('--normal', action='store_true', help='Run the normal model')
    parser.add_argument('--interpolate', type=int, default=1, help='Interpolation steps between frames')
    parser.add_argument('--result_path', type=str, default='results', help='Path to the results folder')
    return parser.parse_args()


def get_folder_size(folder):
    """Returns the number of image files in the given folder."""
    return len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])


if __name__ == "__main__":
    args = parse_args()
    root = args.result_path
    animations = [os.path.join(root, dir) for dir in os.listdir(root) if os.path.isdir(os.path.join(root, dir))]

    animations = natsorted(animations, alg=ns.PATH)

    assigned_animations = animations

    for animation in assigned_animations:
        # create folder output_path + "/" + "output_final_rendering_images"
        # create folder output_path + "/" + "output_final_rendering_images_baseline"
        os.makedirs(animation + "/" + "output_final_rendering_images", exist_ok=True)
        os.makedirs(animation + "/" + "output_final_rendering_images_baseline", exist_ok=True)

        renderings = []
        renderings_normal = []
        renderings_baseline = []
        renderings_baseline_normal = []

        renderings_interpolated = []
        renderings_interpolated_normal = []

        source_path = animation
        output_path = animation

        print("/n/n ============= Start processing: ", animation, " =============/n")

        glbs_list = os.listdir(source_path)
        # only keep file with suffix "_baseline_sample.glb"
        glbs_list = [glb for glb in glbs_list if "_texture_consistency_sample.glb" in glb]
        glbs_list = natsorted(glbs_list, alg=ns.PATH)

        if args.baseline:
            glbs_list_baseline = os.listdir(source_path)
            # only keep file with suffix "_baseline_sample.glb"
            glbs_list_baseline = [glb for glb in glbs_list_baseline if "_baseline_sample.glb" in glb]
            glbs_list_baseline = natsorted(glbs_list_baseline, alg=ns.PATH)
        else:
            glbs_list_baseline = glbs_list

        with open(output_path + "/extrinsics_list.pkl", 'rb') as f:
            extrinsics_list = dill.load(f)
            extr = extrinsics_list[0]

        last_mesh = None
        
        for ind, (glb, glb_baseline) in enumerate(zip(glbs_list, glbs_list_baseline)):
            mesh_file = source_path + "/" + glb
            mesh = trimesh.load(mesh_file, process=False)
            mesh = mesh.geometry["geometry_0"]
            mesh.vertices = mesh.vertices @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

            mesh_repre = MeshExtractResult(
                vertices=torch.tensor(mesh.vertices, dtype=torch.float32).cuda(),
                faces=torch.tensor(mesh.faces, dtype=torch.int64).cuda(),
                vertex_attrs=torch.tensor(mesh.visual.to_color().vertex_colors[..., :3], dtype=torch.float32).repeat(1, 2).cuda() / 255,
                res=512, texture=torch.tensor(np.array(mesh.visual.material.baseColorTexture), dtype=torch.float32).flip(0).cuda() / 255, 
                uv=torch.tensor(mesh.visual.uv, dtype=torch.float32).cuda()
            )

            fovs = torch.deg2rad(torch.tensor(40, dtype=torch.float32)).cuda()
            intrinsics = [utils3d.intrinsics_from_fov_xy(fovs, fovs)]
            extrinsics = [torch.eye(4).cuda()]

            if (ind % args.interpolate == 0) and last_mesh is not None:
                # interpolation mesh
                for id in range(args.interpolate - 1):
                    temp_mesh_repre = MeshExtractResult(
                        vertices=last_mesh.vertices * (1 - id / args.interpolate) + mesh_repre.vertices * (id / args.interpolate),
                        faces=mesh_repre.faces,
                        vertex_attrs=torch.tensor(mesh.visual.to_color().vertex_colors[..., :3], dtype=torch.float32).repeat(1, 2).cuda() / 255,
                        res=512, texture=torch.tensor(np.array(mesh.visual.material.baseColorTexture), dtype=torch.float32).flip(0).cuda() / 255, 
                        uv=torch.tensor(mesh.visual.uv, dtype=torch.float32).cuda()
                    )

                    img = render_frames(temp_mesh_repre, extrinsics, intrinsics, {'resolution': 1024, 'bg_color': (1, 1, 1)})['texture'][0]
                    renderings_interpolated.append(img)

                    if args.normal:
                        img_normal = render_frames(temp_mesh_repre, extrinsics, intrinsics, {'resolution': 1024, 'bg_color': (1, 1, 1)})['normal'][0]
                        renderings_interpolated_normal.append(img_normal)

            img = render_frames(mesh_repre, extrinsics, intrinsics, {'resolution': 1024, 'bg_color': (1, 1, 1)})['texture'][0]

            renderings.append(img)
            renderings_interpolated.append(img)

            if args.normal:
                img_normal = render_frames(mesh_repre, extrinsics, intrinsics, {'resolution': 1024, 'bg_color': (1, 1, 1)})['normal'][0]
                renderings_normal.append(img_normal)
                renderings_interpolated_normal.append(img_normal)
            
            imageio.imwrite(output_path + "/" + "output_final_rendering_images" + "/" + f"{ind:04d}.png", img)

            last_mesh = mesh_repre.deepcopy()

            "======== Re-canonicalization of the Mesh and Gaussian (*exclude RF*) ========"

            if args.baseline:
                mesh_file_baseline = source_path + "/" + glb_baseline
                mesh_baseline = trimesh.load(mesh_file_baseline, process=False)
                mesh_baseline = mesh_baseline.geometry["geometry_0"]
                mesh_baseline.vertices = mesh_baseline.vertices @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
                
                vertices = torch.from_numpy(mesh_baseline.vertices).unsqueeze(0).float().cuda()

                vertices_homo = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
                vertices_camera = torch.bmm(vertices_homo, extr.transpose(-1, -2)).squeeze()

                mesh_baseline.vertices = vertices_camera[:, :3].cpu().numpy()
                
                mesh_repre_baseline = MeshExtractResult(
                    vertices=torch.tensor(mesh_baseline.vertices, dtype=torch.float32).cuda(),
                    faces=torch.tensor(mesh_baseline.faces, dtype=torch.int64).cuda(),
                    vertex_attrs=torch.tensor(mesh_baseline.visual.to_color().vertex_colors[..., :3], dtype=torch.float32).repeat(1, 2).cuda() / 255,
                    res=512, texture=torch.tensor(np.array(mesh_baseline.visual.material.baseColorTexture), dtype=torch.float32).flip(0).cuda() / 255, 
                    uv=torch.tensor(mesh_baseline.visual.uv, dtype=torch.float32).cuda()
                )

                img_baseline = render_frames(mesh_repre_baseline, extrinsics, intrinsics, {'resolution': 1024, 'bg_color': (1, 1, 1)})['texture'][0]
                renderings_baseline.append(img_baseline)  
                imageio.imwrite(output_path + "/" + "output_final_rendering_images_baseline" + "/" + f"{ind:04d}.png", img_baseline)

                if args.normal:
                    img_baseline_normal = render_frames(mesh_repre_baseline, extrinsics, intrinsics, {'resolution': 1024, 'bg_color': (1, 1, 1)})['normal'][0]
                    renderings_baseline_normal.append(img_baseline_normal)

        imageio.mimsave(output_path + "/" + "output_final_rendering_video.mp4", renderings, fps=30)
        imageio.mimsave(output_path + "/" + f"output_final_rendering_video_interpolated_{args.interpolate}.mp4", renderings_interpolated, fps=30)

        if args.normal:
            imageio.mimsave(output_path + "/" + "output_final_rendering_video_normal.mp4", renderings_normal, fps=30)

        if args.baseline:
            imageio.mimsave(output_path + "/" + "output_final_rendering_video_baseline.mp4", renderings_baseline, fps=30)

            if args.normal:
                imageio.mimsave(output_path + "/" + "output_final_rendering_video_baseline_normal.mp4", renderings_baseline_normal, fps=30)

        print("/n/n ============= Finish processing: ", animation, " =============/n")