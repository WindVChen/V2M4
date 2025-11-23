import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
import rembg
import math
import subprocess
import shutil
import copy
import xatlas
import imageio
import lpips
import torch
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import utils3d
from v2m4_trellis.pipelines import TrellisImageTo3DPipeline
from v2m4_trellis.utils import render_utils, postprocessing_utils
from v2m4_trellis.utils.general_utils import *
from v2m4_trellis.representations.mesh import MeshExtractResult
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency
from natsort import ns, natsorted
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from largesteps.parameterize import from_differential, to_differential
from largesteps.geometry import compute_matrix
from largesteps.optimize import AdamUniform
from v2m4_trellis.utils.render_utils import rotation_6d_to_matrix
import nvdiffrast.torch as dr
import argparse
from natsort import ns, natsorted
import dill
import tripoSG.app as tripoSG_app
from tripoSG.triposg.scripts.briarmbg import BriaRMBG
from tripoSG.triposg.triposg.pipelines.pipeline_triposg import TripoSGPipeline
from tripoSG.mv_adapter.scripts.inference_ig2mv_sdxl import prepare_pipeline
from tripoSG.texture import TexturePipeline, ModProcessConfig

import craftsman.app as craftsman_app
from craftsman.pipeline import CraftsManPipeline


def face_area_consistency_loss(original_vertices, deformed_vertices, faces):
    """
    Ensures that face areas remain the same between the original and deformed meshes.
    
    Args:
        original_vertices: (V, 3) Tensor of original vertex positions.
        deformed_vertices: (V, 3) Tensor of deformed vertex positions.
        faces: (F, 3) Tensor containing face indices.
    
    Returns:
        Scalar loss value enforcing area preservation.
    """
    def compute_face_areas(vertices, faces):
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        return 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0), dim=1)  # (F,)

    original_areas = compute_face_areas(original_vertices, faces)
    deformed_areas = compute_face_areas(deformed_vertices, faces)

    return torch.mean((deformed_areas - original_areas) ** 2)  # MSE between areas

def edge_length_consistency_loss(original_vertices, deformed_vertices, faces):
    """
    Ensures that edge lengths remain close to their original values.
    
    Args:
        original_vertices: (V, 3) Tensor of original vertex positions.
        deformed_vertices: (V, 3) Tensor of deformed vertex positions.
        faces: (F, 3) Tensor containing face indices.
    
    Returns:
        Scalar loss value enforcing edge length preservation.
    """
    def compute_edge_lengths(vertices, faces):
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        return torch.cat([
            torch.norm(v1 - v0, dim=1, keepdim=True),  # Edge v0-v1
            torch.norm(v2 - v1, dim=1, keepdim=True),  # Edge v1-v2
            torch.norm(v0 - v2, dim=1, keepdim=True)   # Edge v2-v0
        ], dim=1).view(-1)  # Flatten

    original_lengths = compute_edge_lengths(original_vertices, faces)
    deformed_lengths = compute_edge_lengths(deformed_vertices, faces)

    return torch.mean((deformed_lengths - original_lengths) ** 2)  # MSE between edge lengths


def outputs_to_files_for_blender(input_glb_folder):
    # Input and output paths
    output_npy_path = input_glb_folder + "/output_vertex_offsets.npy"
    output_texture_path = input_glb_folder + "/output_texture.png"  # Path to save the texture

    # Get all GLB files and sort them by time order
    glb_files = sorted([f for f in os.listdir(input_glb_folder) if f.endswith("_texture_consistency_sample.glb")])
    glb_files = natsorted(glb_files, alg=ns.PATH)

    # Read the first frame as a reference
    first_mesh = trimesh.load(os.path.join(input_glb_folder, glb_files[0]), process=False)
    ref_vertices = np.array(first_mesh.geometry['geometry_0'].vertices)

    # copy and rename the first frame using shutil
    shutil.copy(os.path.join(input_glb_folder, glb_files[0]), os.path.join(input_glb_folder, "output_mesh.glb"))

    # Try to save the texture (only executed the first time)
    if hasattr(first_mesh.geometry['geometry_0'].visual, "material") and hasattr(first_mesh.geometry['geometry_0'].visual.material, "baseColorTexture"):
        texture = first_mesh.geometry['geometry_0'].visual.material.baseColorTexture
        texture.save(output_texture_path)

    # Store vertex offsets for all frames
    vertex_offsets = []

    for glb_file in glb_files:
        mesh = trimesh.load(os.path.join(input_glb_folder, glb_file), process=False)
        current_vertices = np.array(mesh.geometry['geometry_0'].vertices)
        offset = current_vertices - ref_vertices
        vertex_offsets.append(offset)

        ref_vertices = current_vertices

    # Convert to NumPy array (num_frames, num_vertices, 3)
    vertex_offsets = np.array(vertex_offsets)
    np.save(output_npy_path, vertex_offsets)

def compute_edges_and_weights(faces, V):
    """
    Compute edges (edge list) and edge_weights (weights based on edge lengths)
    :param faces: Triangle mesh indices (M, 3)
    :param V: Vertex coordinates (N, 3)
    :return: edges (E, 2), edge_weights (E,)
    """
    # Retrieve the edges of the triangles
    edges = torch.cat([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ], dim=0)  # (3M, 2)

    # Remove duplicate edges (undirected graph)
    edges = torch.sort(edges, dim=1)[0]  # Sort each edge's endpoints to ensure undirected consistency
    edges = torch.unique(edges, dim=0)  # Remove duplicates

    # Compute edge lengths
    edge_lengths = torch.norm(V[edges[:, 0]] - V[edges[:, 1]], dim=1)

    # Compute weights based on edge lengths (to prevent division by zero)
    edge_weights = 1.0 / (edge_lengths + 1e-8)

    return edges, edge_weights


def arap_loss(V, V_opt, faces):
    """
    Compute parallel ARAP Loss using the full neighborhood to compute R
    :param V: Original vertex positions (N, 3)
    :param V_opt: Deformed vertex positions (N, 3)
    :param faces: Triangle indices of the mesh (M, 3)
    :return: ARAP Loss
    """
    # Compute edges and edge_weights
    edges, edge_weights = compute_edges_and_weights(faces, V)

    # Compute displacement before and after deformation for each edge
    V_i = V[edges[:, 0]]  # (E, 3)
    V_j = V[edges[:, 1]]  # (E, 3)
    V_opt_i = V_opt[edges[:, 0]]  # (E, 3)
    V_opt_j = V_opt[edges[:, 1]]  # (E, 3)

    # Compute local transformation matrices S_i of the original mesh
    S_i = (V_j - V_i).unsqueeze(-1) @ (V_opt_j - V_opt_i).unsqueeze(1)  # (E, 3, 3)

    # Compute local S matrices for each vertex (N, 3, 3)
    N = V.shape[0]  # Number of vertices
    S = torch.zeros((N, 3, 3), device=V.device)
    counts = torch.zeros(N, device=V.device)

    # Accumulate neighborhood contributions
    S.index_add_(0, edges[:, 0], S_i * edge_weights.view(-1, 1, 1))
    S.index_add_(0, edges[:, 1], S_i * edge_weights.view(-1, 1, 1))  # Accumulate for the other endpoint as well
    counts.index_add_(0, edges[:, 0], edge_weights)
    counts.index_add_(0, edges[:, 1], edge_weights)

    # Normalize S (to prevent numerical instability)
    S = S / (counts.view(-1, 1, 1) + 1e-8)

    # Compute batch SVD
    U, _, Vh = torch.linalg.svd(S)  # (N, 3, 3)
    R = U @ Vh  # (N, 3, 3)

    # Compute ARAP Loss
    arap_term = (V_opt_j - V_opt_i) - torch.bmm(R[edges[:, 0]], (V_j - V_i).unsqueeze(-1)).squeeze(-1)
    loss = torch.mean(edge_weights * torch.norm(arap_term, dim=-1) ** 2)

    return loss


def seed_torch(seed=0):
    print("Seed Fixed!")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def render_video_and_glbs(outputs, base_name, output_path, init_extrinsics=None, fix_geometry=False, inplace_mesh_change=False):
    # GLB files can be extracted from the outputs
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        # Optional parameters
        simplify=0 if fix_geometry else 0.95,          # Ratio of triangles to remove in the simplification process
        texture_size=1024,      # Size of the texture used for the GLB
        init_extrinsics=init_extrinsics, # Initial extrinsics for the camera
        fill_holes=False if fix_geometry else True, # Fill holes in the mesh
        return_watertight=inplace_mesh_change, # Return a watertight mesh (no texture)
        bake_v_color=True if inplace_mesh_change else False, # Bake vertex colors into the texture
    )
    glb.export(output_path + "/" + base_name + "_sample.glb")

    if inplace_mesh_change:
        # rotate as glb has a rotation pre-processing
        outputs['mesh'][0].vertices = torch.from_numpy(glb.vertices @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])).float().cuda()
        outputs['mesh'][0].faces = torch.from_numpy(glb.faces).long().cuda()

        vertex_normals = torch.from_numpy(glb.vertex_normals)
        outputs['mesh'][0].vertex_attrs = vertex_normals.repeat(1, 2).float().cuda()
        outputs['mesh'][0].vertex_attrs[:, :3] = torch.from_numpy(glb.visual.vertex_colors)[:, :3] / 255.
        
        outputs['mesh'][0].face_normal = outputs['mesh'][0].comput_face_normals(outputs['mesh'][0].vertices, outputs['mesh'][0].faces)

    return glb.visual

def parse_args():
    parser = argparse.ArgumentParser(description='V2M4 Pipeline')
    parser.add_argument('--root', type=str, default='', help='Root directory of the dataset')
    parser.add_argument('--output', type=str, default='', help='Output directory of the results')
    parser.add_argument('--N', type=int, default=1, help='Total number of parallel processes')
    parser.add_argument('--n', type=int, default=0, help='Index of the current process')
    parser.add_argument('--model', type=str, default='Hunyuan', help='Base model, TRELLIS, Craftsman, TripoSG or Hunyuan2.0', choices=['TRELLIS', 'Hunyuan', 'TripoSG', 'Craftsman'])
    parser.add_argument('--baseline', action='store_true', help='Run the baseline model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--skip', type=int, default=5, help='Skip every N frames for large movement of the object (default: 5)')
    parser.add_argument('--use_vggt', action='store_true', help='Use VGGT for camera search, otherwise use dust3R (default: True)')
    parser.add_argument('--use_tracking', action='store_true', help='Use point tracking for mesh registration guidance (!!New Feature!!)')
    parser.add_argument('--tracking_camera_radius', type=int, default=8, help='Adapt the camera radius to ensure the object motion is within the camera view (this argument is only used when --use_tracking is set)')
    parser.add_argument('--blender_path', type=str, default='blender-4.2.1-linux-x64/', help='Path to the Blender executable')
    parser.add_argument('--max_faces', type=int, default=10000, help='Maximum number of faces for the generated mesh (default: 10000). Lower value can speed up the process for all the 3D generation models but not affect TRELLIS, which leverages a different processing pipeline')
    return parser.parse_args()

def get_folder_size(folder):
    """Returns the number of image files in the given folder."""
    return len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])


if __name__ == "__main__":
    args = parse_args()
    root = "examples" if args.root == "" else args.root
    default_output_root = "results_examples"
    animations = [os.path.join(root, dir) for dir in os.listdir(root) if os.path.isdir(os.path.join(root, dir))]

    # filter out animations where the corresponding output folder has file "output_animation.glb"
    if not args.baseline:
        animations = [anim for anim in animations if not os.path.exists(os.path.join(default_output_root if args.output == "" else args.output, anim.split("/")[-1], "output_animation.glb"))]
    # Filter out animations where the corresponding output folder has file "_baseline_sample.glb"
    # animations = [anim for anim in animations if not os.path.exists(os.path.join("results_benchmark_final_seed42" if args.output == "" else args.output, anim.split("/")[-1], "0001_baseline_sample.glb"))]

    # sort
    animations = natsorted(animations, alg=ns.PATH)

    '''Parallel Processing - Average assignment according to the image numbers within each folder'''
    # Get folder sizes
    folder_sizes = [(anim, get_folder_size(anim)) for anim in animations]

    # Sort folders by size in descending order (largest first)
    folder_sizes.sort(key=lambda x: x[1], reverse=True)

    # Assign folders to processes in a balanced way
    assignments = [[] for _ in range(args.N)]
    workload = [0] * args.N  # Track workload for each process

    for folder, size in folder_sizes:
        # Assign the folder to the process with the least workload
        min_index = workload.index(min(workload))
        assignments[min_index].append(folder)
        workload[min_index] += size

    # Get the folders assigned to the current process
    assigned_animations = assignments[args.n]

    print(f"Process {args.n} assigned {len(assigned_animations)} folders with total images: {sum(get_folder_size(f) for f in assigned_animations)}")

    # Load a pipeline from a model folder or a Hugging Face model hub.
    if args.model == "TRELLIS":
        pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        pipeline.cuda()
    elif args.model == "Hunyuan":
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
        pipeline_paint = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
    elif args.model == "TripoSG" or args.model == "Craftsman":
        checkpoints_dir = "./models/checkpoints/"

        if args.model == "TripoSG":
            RMBG_PRETRAINED_MODEL = f"{checkpoints_dir}/RMBG-1.4"

            pipeline_rmbg_net = BriaRMBG.from_pretrained(RMBG_PRETRAINED_MODEL).to("cuda")
            pipeline_rmbg_net.eval()

            TRIPOSG_PRETRAINED_MODEL = f"{checkpoints_dir}/TripoSG"
            pipeline_triposg_pipe = TripoSGPipeline.from_pretrained(TRIPOSG_PRETRAINED_MODEL).to("cuda", torch.float16)
        elif args.model == "Craftsman":
            checkpoints_dir_CraftsMan = f"/{checkpoints_dir}/craftsman-DoraVAE"
            pipeline_crafts = CraftsManPipeline.from_pretrained(checkpoints_dir_CraftsMan, device="cuda", torch_dtype=torch.bfloat16) # bf16 for fast inference

        pipeline_mv_adapter_pipe = prepare_pipeline(
            base_model="stabilityai/stable-diffusion-xl-base-1.0",
            vae_model="madebyollin/sdxl-vae-fp16-fix",
            unet_model=None,
            lora_model=None,
            adapter_path="huanngzh/mv-adapter",
            scheduler=None,
            num_views=6,
            device="cuda",
            dtype=torch.float16,
        )

        pipeline_texture = TexturePipeline(
            upscaler_ckpt_path=f"{checkpoints_dir}/RealESRGAN_x2plus.pth",
            inpaint_ckpt_path=f"{checkpoints_dir}/big-lama.pt",
            device="cuda",
        )

        mod_config = ModProcessConfig(view_upscale=True, inpaint_mode="view")

    # All the models in pipeline are disabled requiring_grad.
    if args.model == "TRELLIS":
        for model in pipeline.models.values():
            for param in model.parameters():
                param.requires_grad = False
    elif args.model == "Hunyuan":
        pipeline.model.requires_grad = False
        pipeline.vae.requires_grad = False
        pipeline.conditioner.requires_grad = False
        pipeline_paint.models['delight_model'].pipeline.feature_extractor.requires_grad = False
        pipeline_paint.models['delight_model'].pipeline.text_encoder.requires_grad = False
        pipeline_paint.models['delight_model'].pipeline.unet.requires_grad = False
        pipeline_paint.models['delight_model'].pipeline.vae.requires_grad = False
        pipeline_paint.models['multiview_model'].pipeline.unet.requires_grad = False
        pipeline_paint.models['multiview_model'].pipeline.vae.requires_grad = False
        pipeline_paint.models['multiview_model'].pipeline.text_encoder.requires_grad = False
        pipeline_paint.models['multiview_model'].pipeline.feature_extractor.requires_grad = False

    rembg_session = rembg.new_session('birefnet-massive')

    for animation in assigned_animations:
        source_path = animation
        output_path = os.path.join(default_output_root if args.output == "" else args.output, animation.split("/")[-1])

        print("/n/n ============= Start processing: ", animation, " =============/n")

        # New folder for the output
        os.makedirs(output_path, exist_ok=True)

        # Fix the seed for reproducibility
        seed = args.seed
        seed_torch(seed)
            
        imgs_list = os.listdir(source_path)
        # exclude folders
        imgs_list = [img for img in imgs_list if not os.path.isdir(source_path + "/" + img)]
        imgs_list = natsorted(imgs_list, alg=ns.PATH)

        existing_outputs = os.listdir(output_path)
        # exclude folders
        existing_outputs = [img for img in existing_outputs if not os.path.isdir(output_path + "/" + img)]
        existing_outputs = natsorted(existing_outputs, alg=ns.PATH)

        outputs_list = []
        base_name_list = []
        extrinsics_list = []
        visual_list = []
        params = None
        for ind, img in enumerate(imgs_list):            
            # Skip every N frames for large movement of the object
            if not args.baseline and ind % args.skip != 0 and ind != len(imgs_list) - 1:
                continue

            # Load an image
            image = Image.open(source_path + "/" + img)

            # Get base name of the image
            base_name = image.filename.split("/")[-1].split(".")[0]

            "======================= 3D Mesh Generation per video frame ======================="
            # Run the pipeline
            if args.model == "TRELLIS":
                rmbg_image, outputs, slat, coords, cond = pipeline.run(
                    image,
                    # Optional parameters
                    seed=seed,
                    save_path=output_path + "/" + base_name + "_rmbg.png", rembg_session=rembg_session
                )
            elif args.model == "Hunyuan":
                save_path = output_path + "/" + base_name + "_rmbg.png"
                cropped_image, rmbg_image = TrellisImageTo3DPipeline.preprocess_image(image, return_rgba=True, rembg_session=rembg_session)
                rmbg_image.save(save_path)
                cropped_image.save(save_path.replace(".png", "_cropped.png"))

                torch.manual_seed(seed)
                mesh = pipeline(image=cropped_image)[0]

                for cleaner in [FloaterRemover(), DegenerateFaceRemover()]:
                    mesh = cleaner(mesh)

                # more facenum, more cost time. The distribution median is ~15000
                mesh = FaceReducer()(mesh, max_facenum=args.max_faces)

                # since in Hunyuan2.0 texture paint, they use xatlas to generate the texture, which may destroy the watertightness. Thus save the attributes of the mesh before painting.
                vertices_watertight = mesh.vertices @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
                faces_watertight = mesh.faces
                mean_point = mesh.vertices.mean(axis=0)
                vertices_watertight = (vertices_watertight - mean_point) * 0.5 + mean_point

                mesh = pipeline_paint(mesh, image=cropped_image)

                # rotate mesh (from y-up to z-up) and scale it to half size to align with the TRELLIS
                mesh.vertices = mesh.vertices @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
                mesh.vertices = (mesh.vertices - mean_point) * 0.5 + mean_point

                outputs = {'mesh': [None], 'mesh_genTex': [None]}

                # !NOTICE: The 'mesh.vertex_normals' below may not be accurate since the vertices are customly rotated, not by applying transfrormation matrix in Trimesh.
                '!!!Fingdings!!!: GLB file expects the vertex colors are in linear RGB format, not sRGB format. So if export GLB with vertex color by trimesh, and then import to other viewers like Blender and Windows 3D viewer, the visualized result will be more brighten. (Trimesh itself does not automatically convert the color space, so import again to trimesh will display correctly.)'
                outputs['mesh_genTex'][0] = MeshExtractResult(
                    vertices=torch.tensor(mesh.vertices, dtype=torch.float32).cuda(),
                    faces=torch.tensor(mesh.faces, dtype=torch.int64).cuda(),
                    vertex_attrs=torch.cat([torch.tensor(mesh.visual.to_color().vertex_colors[..., :3], dtype=torch.float32).cuda() / 255, torch.from_numpy(mesh.vertex_normals).float().cuda()], dim=-1),
                    # Hunyuan's texture is 2048x2048, so we resize it to 1024x1024
                    res=512, texture=torch.tensor(np.array(mesh.visual.material.image.resize((1024, 1024), Image.Resampling.LANCZOS)), dtype=torch.float32).cuda() / 255, 
                    uv=torch.tensor(mesh.visual.uv, dtype=torch.float32).cuda()
                )

                # Since hunyaun2.0 texture generation involves xatlas which destroy watertightness, we need to manually calculate the texture
                vertices = vertices_watertight.astype(np.float32)
                faces = faces_watertight.astype(np.int64)
                # bake texture
                observations, extrinsics, intrinsics = render_utils.render_multiview(outputs['mesh_genTex'][0], resolution=1024, nviews=300)
                masks = [np.any(observation > 0, axis=-1) for observation in observations]
                extrinsics = [extrinsics[i].cpu().numpy() for i in range(len(extrinsics))]
                intrinsics = [intrinsics[i].cpu().numpy() for i in range(len(intrinsics))]

                vertices_color = postprocessing_utils.bake_vertice_color(
                    vertices, faces,
                    observations, masks, extrinsics, intrinsics,
                    verbose=True
                )

                mesh = trimesh.Trimesh(vertices, faces, process=False, vertex_colors=vertices_color)
                outputs['mesh'][0] = MeshExtractResult(
                    vertices=torch.tensor(mesh.vertices, dtype=torch.float32).cuda(),
                    faces=torch.tensor(mesh.faces, dtype=torch.int64).cuda(),
                    vertex_attrs=torch.cat([torch.tensor(mesh.visual.vertex_colors[..., :3], dtype=torch.float32).cuda() / 255, torch.from_numpy(mesh.vertex_normals).float().cuda()], dim=-1),
                    res=512
                )
            elif args.model == "TripoSG" or args.model == "Craftsman":
                save_path = output_path + "/" + base_name + "_rmbg.png"
                _, rmbg_image_rgba, rmbg_image = TrellisImageTo3DPipeline.preprocess_image(image, return_all_rbga=True, rembg_session=rembg_session)
                rmbg_image.save(save_path)

                torch.manual_seed(seed)

                if args.model == "TripoSG":
                    vertices, faces, mesh = tripoSG_app.run_full(source_path + "/" + img, rmbg_image_rgba, pipeline_rmbg_net, pipeline_triposg_pipe, pipeline_mv_adapter_pipe, True, seed, pipeline_texture, mod_config, max_faces=args.max_faces)
                elif args.model == "Craftsman":
                    vertices, faces, mesh = craftsman_app.run_full(source_path + "/" + img, rmbg_image_rgba, pipeline_crafts, pipeline_mv_adapter_pipe, True, seed, pipeline_texture, mod_config, max_faces=args.max_faces)

                outputs = {'mesh': [None], 'mesh_genTex': [None]}

                outputs['mesh_genTex'][0] = MeshExtractResult(
                    vertices=torch.tensor(mesh.vertices, dtype=torch.float32).cuda(),
                    faces=torch.tensor(mesh.faces, dtype=torch.int64).cuda(),
                    vertex_attrs=torch.cat([torch.tensor(mesh.visual.to_color().vertex_colors[..., :3], dtype=torch.float32).cuda() / 255, torch.from_numpy(mesh.vertex_normals).float().cuda()], dim=-1),
                    # Hunyuan's texture is 2048x2048, so we resize it to 1024x1024
                    res=512, texture=torch.tensor(np.array(mesh.visual.material.baseColorTexture.resize((1024, 1024), Image.Resampling.LANCZOS)), dtype=torch.float32).cuda() / 255, 
                    uv=torch.tensor(mesh.visual.uv, dtype=torch.float32).cuda()
                )

                # bake texture
                observations, extrinsics, intrinsics = render_utils.render_multiview(outputs['mesh_genTex'][0], resolution=1024, nviews=300)
                masks = [np.any(observation > 0, axis=-1) for observation in observations]
                extrinsics = [extrinsics[i].cpu().numpy() for i in range(len(extrinsics))]
                intrinsics = [intrinsics[i].cpu().numpy() for i in range(len(intrinsics))]

                vertices_color = postprocessing_utils.bake_vertice_color(
                    vertices, faces,
                    observations, masks, extrinsics, intrinsics,
                    verbose=True
                )

                mesh = trimesh.Trimesh(vertices, faces, process=False, vertex_colors=vertices_color)
                outputs['mesh'][0] = MeshExtractResult(
                    vertices=torch.tensor(mesh.vertices, dtype=torch.float32).cuda(),
                    faces=torch.tensor(mesh.faces, dtype=torch.int64).cuda(),
                    vertex_attrs=torch.cat([torch.tensor(mesh.visual.vertex_colors[..., :3], dtype=torch.float32).cuda() / 255, torch.from_numpy(mesh.vertex_normals).float().cuda()], dim=-1),
                    res=512
                )

            if args.baseline:
                if args.model == "TRELLIS":
                    render_video_and_glbs(outputs, base_name + "_baseline", output_path)
                    continue
                elif args.model == "Hunyuan" or args.model == "TripoSG" or args.model == "Craftsman":
                    # convert MeshExtractResult to trimesh (use uv and texture)
                    mesh = outputs['mesh_genTex'][0]
                    mesh = trimesh.Trimesh(vertices=mesh.vertices.cpu().numpy() @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]), faces=mesh.faces.cpu().numpy(), visual=trimesh.visual.TextureVisuals(uv=mesh.uv.cpu().numpy(), image=Image.fromarray((mesh.texture.cpu().numpy() * 255).astype(np.uint8))), process=False)
                    mesh.export(output_path + "/" + base_name + "_baseline" + "_sample.glb")
                    continue

            "============================================================"

            "======== Section 4.1 - Camera Search and Mesh Re-Pose ========"
            rend_img, params = render_utils.find_closet_camera_pos(outputs['mesh'][0], rmbg_image, save_path=output_path + "/" + base_name, is_Hunyuan=(args.model == "Hunyuan" or args.model == "TripoSG"), use_vggt=args.use_vggt)  # Craftsman is similar to TRELLIS canonical pose
            imageio.imsave(output_path + "/" + base_name + "_sample_mesh_align.png", rend_img)
            if args.model == "TRELLIS":
                rend_img, params = render_utils.find_closet_camera_pos(outputs['gaussian'][0], rmbg_image, params=params, use_vggt=args.use_vggt)
                imageio.imsave(output_path + "/" + base_name + "_sample_gs_align.png", rend_img)
            else:
                rend_img, params = render_utils.find_closet_camera_pos(outputs['mesh_genTex'][0], rmbg_image, params=params, use_vggt=args.use_vggt)
                imageio.imsave(output_path + "/" + base_name + "_sample_genTex_align.png", rend_img)
            "============================================================"

            "======== Section 4.2 - Mesh Appearance Refinement via Negative Condition Embedding Optimization ========"
            if args.model == "TRELLIS":
                refer_image = torch.tensor(np.array(rmbg_image)).float().cuda().permute(2, 0, 1) / 255
                start_optimize_iter = 5  # Optimization iteration for the first denoise step T.

                optim_params = {'start_optimize_iter': start_optimize_iter, 'refer_image': refer_image, 'models': pipeline.models, 'normalization': pipeline.slat_normalization, 'camera_params': params}
                # Set the interval as the whole process [0, 1.0] to allow fine-tuning the later steps (which have more details, the default is only [0.5, 1.0]).
                slat = pipeline.sample_slat(cond, coords, {"cfg_interval": [0., 1.0],}, optimize_uncond_noise=optim_params)
                outputs = pipeline.decode_slat(slat, ['mesh', 'gaussian'])

            "============================================================"

            "======== Section 4.1 - Re-canonicalization of the Mesh and Gaussian ========"
            # ------ Mesh Part -------
            yaw, pitch, r, lookat_x, lookat_y, lookat_z = params
            yaw = torch.tensor([yaw], dtype=torch.float32).cuda()
            pitch = torch.tensor([pitch], dtype=torch.float32).cuda()
            r = torch.tensor([r], dtype=torch.float32).cuda()
            lookat = torch.tensor([lookat_x, lookat_y, lookat_z], dtype=torch.float32).cuda()

            # Get the extrinsics from the camera parameters
            orig = torch.stack([
                torch.sin(yaw) * torch.cos(pitch),
                torch.cos(yaw) * torch.cos(pitch),
                torch.sin(pitch),
            ]).squeeze() * r
            extr = utils3d.torch.extrinsics_look_at(orig, lookat, torch.tensor([0, 0, 1]).float().cuda())
            extr = extr.unsqueeze(0)

            vertices = outputs['mesh'][0].vertices.unsqueeze(0)

            vertices_homo = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
            vertices_camera = torch.bmm(vertices_homo, extr.transpose(-1, -2)).squeeze()

            # Replace the vertices of the mesh (note that there is also Normal that needs transform, but here we temporarily skip that)
            outputs['mesh'][0].vertices = vertices_camera[:, :3]
            
            if args.model == "TRELLIS":
                # ------ Gaussian Part ------
                gs_pos = outputs['gaussian'][0].get_xyz.unsqueeze(0)

                gs_pos_homo = torch.cat([gs_pos, torch.ones_like(gs_pos[..., :1])], dim=-1)
                gs_pos_camera = torch.bmm(gs_pos_homo, extr.transpose(-1, -2)).squeeze()

                outputs['gaussian'][0].from_xyz(gs_pos_camera[:, :3])

                gs_rot = outputs['gaussian'][0].get_rotation

                q_batch = gs_rot / torch.norm(gs_rot, dim=1, keepdim=True)
                # Convert rotation matrix to a single quaternion
                q_matrix = matrix_to_quaternion_batched(extr[0, :3, :3])
                # Perform batched quaternion multiplication
                q_result = reverse_quaternion_multiply_batched(q_matrix, q_batch)
                # Normalize each quaternion in the batch
                q_result = q_result / torch.norm(q_result, dim=1, keepdim=True)

                outputs['gaussian'][0].from_rotation(q_result)

                visual = render_video_and_glbs(outputs, base_name + "_re-canonicalization", output_path, init_extrinsics=extr, inplace_mesh_change=True)

            elif args.model == "Hunyuan" or args.model == "TripoSG" or args.model == "Craftsman":
                mesh = outputs['mesh'][0]
                mesh = trimesh.Trimesh(vertices=mesh.vertices.cpu().numpy() @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]), faces=mesh.faces.cpu().numpy(), vertex_colors=np.clip(mesh.vertex_attrs.cpu().numpy()[:, :3] * 255, 0, 255).astype(np.uint8), process=False)
                mesh.export(output_path + "/" + base_name + "_re-canonicalization" + "_sample.glb")
                visual = mesh.visual

                # For GenTex
                vertices = outputs['mesh_genTex'][0].vertices.unsqueeze(0)
                vertices_homo = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
                vertices_camera = torch.bmm(vertices_homo, extr.transpose(-1, -2)).squeeze()
                outputs['mesh_genTex'][0].vertices = vertices_camera[:, :3]

            outputs_list.append(outputs)
            base_name_list.append(base_name)
            extrinsics_list.append(extr)
            visual_list.append(visual)

            "============================================================"

        if args.baseline:
            continue

        # save the lists for the intermediate results
        with open(output_path + "/outputs_list.pkl", 'wb') as f:
            dill.dump(outputs_list, f)
        with open(output_path + "/base_name_list.pkl", 'wb') as f:
            dill.dump(base_name_list, f)
        with open(output_path + "/extrinsics_list.pkl", 'wb') as f:
            dill.dump(extrinsics_list, f)
        with open(output_path + "/visual_list.pkl", 'wb') as f:
            dill.dump(visual_list, f)

        # Empty cuda 
        torch.cuda.empty_cache()

        "======== Section 4.3 - Consistent Topology via Iterative Pairwise Registration ========"
        # Load the lists (this is for debugging purpose, you can comment it out if you don't need to load the lists)
        # with open(output_path + "/outputs_list.pkl", 'rb') as f:
        #     outputs_list = dill.load(f)
        # with open(output_path + "/base_name_list.pkl", 'rb') as f:
        #     base_name_list = dill.load(f)
        # with open(output_path + "/extrinsics_list.pkl", 'rb') as f:
        #     extrinsics_list = dill.load(f)
        # with open(output_path + "/visual_list.pkl", 'rb') as f:
        #     visual_list = dill.load(f)

        # detach each attribute in outputs_list[0]['mesh'][0]
        outputs_list[0]['mesh'][0].vertices = outputs_list[0]['mesh'][0].vertices.detach()
        outputs_list[0]['mesh'][0].faces = outputs_list[0]['mesh'][0].faces.detach()
        outputs_list[0]['mesh'][0].vertex_attrs = outputs_list[0]['mesh'][0].vertex_attrs.detach()
        outputs_list[0]['mesh'][0].face_normal = outputs_list[0]['mesh'][0].face_normal.detach()

        '------  !!!New Feature!!!: Point Tracking for Mesh Registration Guidance ---------'
        ' Select N views for calculating the points tracking for mesh registration guidance in the second dense alignment phase '
        # Load the cotracker model for the point tracking
        if args.use_tracking:
            cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").cuda()

            with torch.no_grad():
                nviews_track = 20
                # use the first frame as the extrinsics since these meshes are in the similar position
                intr_track, extr_track = render_utils.render_multiview_gradient(None, nviews=nviews_track, init_extrinsics=extrinsics_list[0], return_intr_extrincs=True, radius=args.tracking_camera_radius)

                track_video = [None] * nviews_track

                mask_tracks = render_utils.render_multiview_gradient(outputs_list[0]['mesh'][0], resolution=512, nviews=nviews_track, init_extrinsics=extrinsics_list[0], random_indices=[iid for iid in range(nviews_track)], return_type='mask', input_intr_extrincs=(intr_track, extr_track))

                rast_tracks, full_proj_batchs = render_utils.render_multiview_gradient(outputs_list[0]['mesh'][0], resolution=512, nviews=nviews_track, init_extrinsics=extrinsics_list[0], random_indices=[iid for iid in range(nviews_track)], input_intr_extrincs=(intr_track, extr_track), return_rast_vertices=True)

                # [0, 1, 3] -> [u, v, face index]
                uv_tracking_points = [rast_track[..., [0, 1, 3]] for rast_track in rast_tracks]

                # ensure the foreground points in mask_tracks[i] are less equal than 10000 (otherwise OOM). For the points with more than 10000, randomly sample 10000 points by masking out the other points with mask value = 0
                for i in range(nviews_track):
                    if mask_tracks[i].sum() > 10000:
                        # Get indices of nonzero elements
                        nonzero_indices = torch.nonzero(mask_tracks[i]).squeeze()
                        # Randomly select 10000 indices from nonzero indices
                        selected_indices = nonzero_indices[torch.randperm(len(nonzero_indices))[:10000]]
                        # Create new mask with only selected indices set to 1
                        new_mask = torch.zeros_like(mask_tracks[i])
                        new_mask[selected_indices[:, 0], selected_indices[:, 1]] = 1
                        mask_tracks[i] = new_mask

                # filter out points with mask_track == 0
                for i in range(nviews_track):
                    uv_tracking_points[i] = uv_tracking_points[i][mask_tracks[i] > 0]

                for i in range(0, len(outputs_list)):
                    observations_target_all_track = render_utils.render_multiview_gradient(outputs_list[i]['mesh'][0], resolution=512, nviews=nviews_track, init_extrinsics=extrinsics_list[0], random_indices=[iid for iid in range(nviews_track)], return_type='color', input_intr_extrincs=(intr_track, extr_track))
                    for j in range(nviews_track):
                        if track_video[j] is None:
                            track_video[j] = observations_target_all_track[j][None]
                        else:
                            track_video[j] = torch.cat((track_video[j], observations_target_all_track[j][None]), dim=0)

                # save the track_video
                for i in range(nviews_track):
                    imageio.mimsave(os.path.join(output_path, f"track_video{i}.mp4"), track_video[i].permute(0, 2, 3, 1).cpu().numpy() * 255, fps=30)
                    if i == 5:
                        break  # only save the first 6 views for debugging

                # concatenate the video
                for i in range(nviews_track):
                    track_video[i] = track_video[i][None].float()

                pred_tracks = [None] * nviews_track
                pred_visibilities = [None] * nviews_track
                
                max_frames_per_chunk = 32  # Adjust based on GPU memory to avoud OOM
                for i in range(nviews_track):
                    # query points are the foreground points (value = 1) in the mask_track[i]
                    queries = torch.nonzero(mask_tracks[i], as_tuple=False).float().cuda()
                    # flip the x and y coordinates as cotracker query format is W, H
                    queries = queries.flip(1)
                    queries = torch.cat([torch.zeros_like(queries[:, :1]), queries], dim=1)  # add the fixed frame 

                    # Initialize results
                    num_frames = track_video[i].shape[1]
                    all_tracks = []
                    all_visibilities = []
                    
                    # Process video in chunks
                    for chunk_start in range(0, num_frames, max_frames_per_chunk):
                        # +1 because we need to include the last frame in the chunk for the query
                        chunk_end = min(chunk_start + max_frames_per_chunk + 1, num_frames)
                        video_chunk = track_video[i][:, chunk_start:chunk_end].clone()
                        
                        # For the first chunk, use the original queries
                        if chunk_start == 0:
                            chunk_queries = queries[None]
                        # For later chunks, use the last predictions from the previous chunk as starting points
                        else:
                            # Extract the last frame predictions from previous chunk
                            last_positions = all_tracks[-1][-1:].clone()  # Shape: [1, N, 2]
                            # Create new queries that start from these positions
                            chunk_queries = torch.cat([torch.zeros_like(last_positions[:, :, :1]), last_positions], dim=2)
                            
                        # Process the current chunk
                        pred_track_chunk, pred_visibility_chunk = cotracker(video_chunk, queries=chunk_queries)
                        
                        # Store results
                        all_tracks.append(pred_track_chunk[0])
                        all_visibilities.append(pred_visibility_chunk[0])
                    
                    # Concatenate results from all chunks
                    if len(all_tracks) > 1:
                        # For tracks after the first chunk, we need to skip the first frame (as it's duplicated)
                        concatenated_tracks = [all_tracks[0]]
                        concatenated_visibilities = [all_visibilities[0]]
                        
                        for j in range(1, len(all_tracks)):
                            concatenated_tracks.append(all_tracks[j][1:])  # Skip the first frame
                            concatenated_visibilities.append(all_visibilities[j][1:])  # Skip the first frame
                            
                        pred_tracks[i] = torch.cat(concatenated_tracks, dim=0)
                        pred_visibilities[i] = torch.cat(concatenated_visibilities, dim=0)
                    else:
                        # If only one chunk, just use it directly
                        pred_tracks[i] = all_tracks[0]
                        pred_visibilities[i] = all_visibilities[0]

                    if i < 6: # only display the first 6 views for debugging
                        # only display 1000 query points
                        random_indices = torch.randperm(pred_tracks[i].shape[1])[:1000]

                        vis = Visualizer(
                            save_dir=output_path,
                            linewidth=1,
                            mode='rainbow',
                            tracks_leave_trace=-1
                        )
                        vis.visualize(
                            video=track_video[i],
                            tracks=pred_tracks[i][:, random_indices].unsqueeze(0),
                            visibility=pred_visibilities[i][:, random_indices].unsqueeze(0),
                            filename='track_result' + str(i))


        '----------------------------------------------------------------------------------'

        temp = copy.deepcopy(outputs_list[0])
        temp['mesh'][0] = temp['mesh'][0].deepcopy()
        texture_optim_input_list = [temp]

        for i in range(0, len(outputs_list) - 1):
            # Load reference shape
            v_ref = outputs_list[i+1]['mesh'][0].vertices
            f_ref = outputs_list[i+1]['mesh'][0].faces
            n_ref = outputs_list[i+1]['mesh'][0].comput_v_normals(v_ref, f_ref)
            outputs_list[i+1]['mesh'][0].vertex_attrs[:, 3:] = n_ref

            # Load source shape
            v = outputs_list[i]['mesh'][0].vertices
            f = outputs_list[i]['mesh'][0].faces
            n = outputs_list[i]['mesh'][0].comput_v_normals(v, f)
            outputs_list[i]['mesh'][0].vertex_attrs[:, 3:] = n

            envmap_path = os.path.join("kloppenheim_06_2k.hdr")
            envmap = torch.tensor(imageio.imread(envmap_path, format='HDR-FI'), device='cuda')
            # Add alpha channel
            alpha = torch.ones((*envmap.shape[:2],1), device='cuda')
            envmap = torch.cat((envmap, alpha), dim=-1)
            
            ' -------------- First, coarsely align the last frame to the current frame, optimize 500 iterations first, then optimize 250 iterations for each of the 2 principal axes to ensure the mesh is aligned in the correct direction, not flip ----------------'
            device = 'cuda'

            rot6D = torch.nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=device))
            translation = torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0], device=device))
            scale = torch.nn.Parameter(torch.tensor([1.0, 1.0, 1.0], device=device))
            optimizer = torch.optim.Adam([rot6D, translation, scale], lr=0.01)

            # Total iterations for all optimization stages, set pbar to 1250 to use flip version around first and second principal axis of point clouds, if want to disable, set to 500
            pbar = tqdm(range(500), desc='Coarsely align last to current frame', disable=False)

            nviews = 1000

            # get all views' intrinsic and extrinsic parameters in advance
            intr, extr = render_utils.render_multiview_gradient(None, nviews=nviews, init_extrinsics=extrinsics_list[i+1], return_intr_extrincs=True)

            # To avoid OOM for rendering too many views one time.
            observations_target_all = None
            split_nviews = 10
            views_per_split = nviews // split_nviews
            for split in range(split_nviews):
                start_idx = split * views_per_split
                end_idx = start_idx + views_per_split
                observations_split = render_utils.render_multiview_gradient(
                    outputs_list[i+1]['mesh'][0],
                    resolution=512,
                    nviews=views_per_split,
                    init_extrinsics=extrinsics_list[i+1],
                    random_indices=[iid for iid in range(start_idx, end_idx)],
                    envmap=envmap,
                    return_type='normal_map',
                    input_intr_extrincs=(intr, extr)
                )
                if observations_target_all is None:
                    observations_target_all = observations_split
                else:
                    observations_target_all = torch.cat((observations_target_all, observations_split), dim=0)

            selected_views = 20
            best_loss = float('inf')
            best_params = None

            for iter in pbar:
                optimizer.zero_grad()

                rotation_matrix = rotation_6d_to_matrix(rot6D)
                transformed_pts3d = v @ rotation_matrix.T * scale + translation

                # Compute the chamfer distance between the transformed point clouds and the gt point clouds
                loss_chamfer = chamfer_distance(v_ref.unsqueeze(0), transformed_pts3d.unsqueeze(0))[0] * 1e-1

                random_indices = torch.randperm(nviews)[:selected_views]

                outputs_list[i]['mesh'][0].vertices = transformed_pts3d.squeeze()

                # Recompute vertex normals
                outputs_list[i]['mesh'][0].face_normal = outputs_list[i]['mesh'][0].comput_face_normals(v, f)

                # Avoid backward the second time error
                outputs_list[i]['mesh'][0].vertex_attrs = outputs_list[i]['mesh'][0].vertex_attrs.detach()
                outputs_list[i]['mesh'][0].vertex_attrs[:, 3:] = outputs_list[i]['mesh'][0].comput_v_normals(v, f)        

                # Render images
                observations_target = observations_target_all[random_indices]
                observations_init = render_utils.render_multiview_gradient(outputs_list[i]['mesh'][0], resolution=512, nviews=nviews, init_extrinsics=extrinsics_list[i+1], random_indices=random_indices, envmap=envmap, return_type='normal_map', input_intr_extrincs=(intr, extr))

                # Compute L1 image loss
                # No masked loss to ensure mesh can well register to the target mesh
                loss_l1 = (observations_init - observations_target).abs().mean()

                loss = loss_chamfer + loss_l1

                loss.backward()
                optimizer.step()

                pbar.set_postfix({'loss': loss.item(), 'chamfer': loss_chamfer.item(), 'l1': loss_l1.item()}) 

                # Stage 1 (0-500): Initial optimization
                # At iteration 500, store result and start Stage 2 with first principal axis rotation
                if iter == 500:
                    best_loss = loss.item()
                    best_params = {
                        'rot6D': rot6D.data.clone(),
                        'translation': translation.data.clone(),
                        'scale': scale.data.clone(),
                        'observation_init': observations_init.detach().clone(),
                        'observation_target': observations_target.detach().clone()
                    }

                    # Get principal axes and rotate around first axis
                    with torch.no_grad():
                        rotation_matrix = rotation_6d_to_matrix(rot6D)
                        transformed_pts3d = v @ rotation_matrix.T * scale + translation
                        centered_pts = transformed_pts3d - transformed_pts3d.mean(dim=0)
                        _, _, V = torch.pca_lowrank(centered_pts)
                        principal_axes = V[:, :2]
                        
                        # Stage 2 (500-750): Rotate 180° around first principal axis
                        axis = principal_axes[:, 0] / torch.norm(principal_axes[:, 0])
                        angle = torch.tensor(torch.pi, device=device)
                        # Compute cross product matrix [k]×
                        kx, ky, kz = axis
                        cross_matrix = torch.tensor([
                            [0, -kz, ky],
                            [kz, 0, -kx],
                            [-ky, kx, 0]
                        ], device=device)
                        
                        # Compute [k]×²
                        cross_matrix_squared = cross_matrix @ cross_matrix
                        
                        # Apply Rodrigues formula
                        R = torch.eye(3, device=device) + torch.sin(angle) * cross_matrix + (1 - torch.cos(angle)) * cross_matrix_squared
                        
                        # Get the current transformation parameters
                        current_rotation = rotation_6d_to_matrix(rot6D).clone()
                        current_translation = translation.data.clone()
                        current_scale = scale.data.clone()
                        
                        # Compute the mean point of the transformed points
                        mean_point = transformed_pts3d.mean(dim=0).clone()

                        # Update translation to account for the rotation around the mean point
                        translation.data = mean_point - ((mean_point - current_translation) @ R.T)
                        
                        # Update rotation matrix and scale matrix by Polar decomposition of (current_rotation.T @ diag(S) @ R.T)
                        # Create diagonal scale matrix
                        scale_matrix = torch.diag(current_scale)
                        
                        # Compute M = current_rotation.T @ diag(S) @ R.T
                        M = current_rotation.T @ scale_matrix @ R.T
                        
                        # Perform polar decomposition: M = UP where U is rotation and P is symmetric positive-definite
                        U, S, Vh = torch.linalg.svd(M)
                        rotation_part = (U @ Vh).T  # This is the rotation component
                        
                        # Convert rotation matrix to 6D representation
                        rot6D.data = torch.cat([rotation_part[0, :], rotation_part[1, :]], dim=0)
                        
                        # Update scale by extracting singular values
                        scale.data = torch.diag(Vh.T @ torch.diag(S) @ Vh)

                # Stage 3 (750-1000): Check Stage 2 result and try second axis rotation
                elif iter == 750:
                    # Store result from first axis rotation if better
                    if loss.item() < best_loss:
                        # indicate this is better than the previous best result
                        print('180 degree rotation around first principal axis is better than the previous best result!')
                        best_loss = loss.item()
                        best_params = {
                            'rot6D': rot6D.data.clone(),
                            'translation': translation.data.clone(),
                            'scale': scale.data.clone(),
                            'observation_init': observations_init.detach().clone(),
                            'observation_target': observations_target.detach().clone()
                        }

                    with torch.no_grad():
                        # Rotate 180° around second principal axis
                        axis = principal_axes[:, 1] / torch.norm(principal_axes[:, 1])
                        angle = torch.tensor(torch.pi, device=device)
                        # Compute cross product matrix [k]×
                        kx, ky, kz = axis
                        cross_matrix = torch.tensor([
                            [0, -kz, ky],
                            [kz, 0, -kx],
                            [-ky, kx, 0]
                        ], device=device)
                        
                        # Compute [k]×²
                        cross_matrix_squared = cross_matrix @ cross_matrix
                        
                        # Apply Rodrigues formula
                        R = torch.eye(3, device=device) + torch.sin(angle) * cross_matrix + (1 - torch.cos(angle)) * cross_matrix_squared
                        
                        # Update translation to account for the rotation around the mean point
                        translation.data = mean_point - ((mean_point - current_translation) @ R.T)
                        
                        # Compute M = current_rotation.T @ diag(S) @ R.T
                        M = current_rotation.T @ scale_matrix @ R.T
                        
                        # Perform polar decomposition: M = UP where U is rotation and P is symmetric positive-definite
                        U, S, Vh = torch.linalg.svd(M)
                        rotation_part = (U @ Vh).T  # This is the rotation component
                        
                        # Convert rotation matrix to 6D representation
                        rot6D.data = torch.cat([rotation_part[0, :], rotation_part[1, :]], dim=0)
                        
                        # Update scale by extracting singular values
                        scale.data = torch.diag(Vh.T @ torch.diag(S) @ Vh)
                
                # Stage 4 (1000-1250): Check Stage 3 result and try both first and second axis rotation
                elif iter == 1000:
                    # Store result from first axis rotation if better
                    if loss.item() < best_loss:
                        # indicate this is better than the previous best result
                        print('180 degree rotation around second principal axis is better than the previous best result!')
                        best_loss = loss.item()
                        best_params = {
                            'rot6D': rot6D.data.clone(),
                            'translation': translation.data.clone(),
                            'scale': scale.data.clone(),
                            'observation_init': observations_init.detach().clone(),
                            'observation_target': observations_target.detach().clone()
                        }

                    with torch.no_grad():
                        # Rotate 180° around both first and second principal axes
                        axis1 = principal_axes[:, 0] / torch.norm(principal_axes[:, 0])
                        axis2 = principal_axes[:, 1] / torch.norm(principal_axes[:, 1])
                        angle = torch.tensor(torch.pi, device=device)

                        # Compute cross product matrices [k]× for both axes
                        kx1, ky1, kz1 = axis1
                        cross_matrix1 = torch.tensor([
                            [0, -kz1, ky1],
                            [kz1, 0, -kx1],
                            [-ky1, kx1, 0]
                        ], device=device)

                        kx2, ky2, kz2 = axis2  
                        cross_matrix2 = torch.tensor([
                            [0, -kz2, ky2],
                            [kz2, 0, -kx2],
                            [-ky2, kx2, 0]
                        ], device=device)

                        # Compute [k]×² for both axes
                        cross_matrix1_squared = cross_matrix1 @ cross_matrix1
                        cross_matrix2_squared = cross_matrix2 @ cross_matrix2

                        # Apply Rodrigues formula for both rotations
                        R1 = torch.eye(3, device=device) + torch.sin(angle) * cross_matrix1 + (1 - torch.cos(angle)) * cross_matrix1_squared
                        R2 = torch.eye(3, device=device) + torch.sin(angle) * cross_matrix2 + (1 - torch.cos(angle)) * cross_matrix2_squared

                        # Combine rotations
                        R = R2 @ R1

                        # Update translation to account for rotation around mean point
                        translation.data = mean_point - ((mean_point - current_translation) @ R.T)
                        
                        # Compute M = current_rotation.T @ diag(S) @ R.T
                        M = current_rotation.T @ scale_matrix @ R.T
                        
                        # Perform polar decomposition: M = UP where U is rotation and P is symmetric positive-definite
                        U, S, Vh = torch.linalg.svd(M)
                        rotation_part = (U @ Vh).T  # This is the rotation component
                        
                        # Convert rotation matrix to 6D representation
                        rot6D.data = torch.cat([rotation_part[0, :], rotation_part[1, :]], dim=0)
                        
                        # Update scale by extracting singular values
                        scale.data = torch.diag(Vh.T @ torch.diag(S) @ Vh)

            if best_params is not None:
                # After all stages, check if final result was best
                if loss.item() < best_loss:
                    # indicate this is better than the previous best result
                    print('Final result after all rotations is better than previous best results!')
                    best_loss = loss.item()
                    best_params = {
                        'rot6D': rot6D.data.clone(),
                        'translation': translation.data.clone(),
                        'scale': scale.data.clone(),
                        'observation_init': observations_init.detach().clone(),
                        'observation_target': observations_target.detach().clone()
                    }

                # Set parameters to best found across all stages
                rot6D.data = best_params['rot6D']
                translation.data = best_params['translation']
                scale.data = best_params['scale']
                observations_init = best_params['observation_init']
                observations_target = best_params['observation_target']
            
            with torch.no_grad():
                # save the rendered images
                for j in range(3):
                    imageio.imsave(output_path + "/" + base_name_list[i] + "_mesh_view_after_coarse_geometry_consistency" + str(j) + ".png", (observations_init[j].detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                    imageio.imsave(output_path + "/" + base_name_list[i] + "_mesh_refview_after_coarse_geometry_consistency" + str(j) + ".png", (observations_target[j].detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)) 

                rotation_matrix = rotation_6d_to_matrix(rot6D)
                transformed_pts3d = v @ rotation_matrix.T * scale + translation

                outputs_list[i]['mesh'][0].vertices = transformed_pts3d

            # Load source shape
            v = outputs_list[i]['mesh'][0].vertices
            f = outputs_list[i]['mesh'][0].faces
            n = outputs_list[i]['mesh'][0].comput_v_normals(v, f)
            outputs_list[i]['mesh'][0].vertex_attrs[:, 3:] = n

            init_copy = outputs_list[i]['mesh'][0].deepcopy()

            '------------------------------------------------------------------------------------------'

            '------------- Second, densely align the last frame to the current frame -----------------'
            steps = 1000 # Number of optimization steps
            step_size = 1e-1 # Step size
            lambda_ = 99 # Hyperparameter lambda of our method, used to compute the matrix (I + lambda_*L)

            # Compute the system matrix
            M = compute_matrix(v, f, lambda_)

            # Parameterize
            u = to_differential(M, v)

            u.requires_grad = True
            opt = AdamUniform([u], step_size)

            par = tqdm(range(steps), desc='Optimizing Mesh for Geometry Consistency', disable=False)

            nviews = 1000

            # get all views' intrinsic and extrinsic parameters and target observations in advance
            intr, extr = render_utils.render_multiview_gradient(None, nviews=nviews, init_extrinsics=extrinsics_list[i+1], return_intr_extrincs=True)

            observations_target_all = None
            split_nviews = 10
            views_per_split = nviews // split_nviews
            for split in range(split_nviews):
                start_idx = split * views_per_split
                end_idx = start_idx + views_per_split
                observations_split = render_utils.render_multiview_gradient(
                    outputs_list[i+1]['mesh'][0],
                    resolution=512,
                    nviews=views_per_split,
                    init_extrinsics=extrinsics_list[i+1],
                    random_indices=[iid for iid in range(start_idx, end_idx)],
                    envmap=envmap,
                    return_type='normal_map',
                    input_intr_extrincs=(intr, extr)
                )
                if observations_target_all is None:
                    observations_target_all = observations_split
                else:
                    observations_target_all = torch.cat((observations_target_all, observations_split), dim=0)

            selected_views = 50

            for iter in par:
                random_indices = torch.randperm(nviews)[:selected_views]

                # Get cartesian coordinates for parameterization
                v = from_differential(M, u, 'Cholesky')

                outputs_list[i]['mesh'][0].vertices = v.squeeze()

                # Recompute vertex normals
                outputs_list[i]['mesh'][0].face_normal = outputs_list[i]['mesh'][0].comput_face_normals(v, f)

                # Avoid backward the second time error
                outputs_list[i]['mesh'][0].vertex_attrs = outputs_list[i]['mesh'][0].vertex_attrs.detach()
                outputs_list[i]['mesh'][0].vertex_attrs[:, 3:] = outputs_list[i]['mesh'][0].comput_v_normals(v, f)        

                # Render images
                observations_target = observations_target_all[random_indices]
                observations_init = render_utils.render_multiview_gradient(outputs_list[i]['mesh'][0], resolution=512, nviews=nviews, init_extrinsics=extrinsics_list[i+1], random_indices=random_indices, envmap=envmap, return_type='normal_map', input_intr_extrincs=(intr, extr))

                # Compute L1 image loss
                loss_l1 = (observations_init - observations_target).abs().mean() * 1

                # ARAP loss
                arap_loss_ = arap_loss(init_copy.vertices, v, f) * 1

                # chamfer loss
                chamfer_loss_ = chamfer_distance(v.unsqueeze(0), v_ref.unsqueeze(0))[0] * 1e1

                face_area_loss_ = face_area_consistency_loss(init_copy.vertices, v, f) * 1e6

                edge_length_loss_ = edge_length_consistency_loss(init_copy.vertices, v, f) * 1e2

                # Tracking loss constraint
                if args.use_tracking:
                    tracking_loss = 0

                    vertices_track = outputs_list[i]['mesh'][0].vertices

                    for j in range(nviews_track):
                        full_proj_batch = full_proj_batchs[j][None]

                        u_track, v_track, faces_track = uv_tracking_points[j].split(1, dim=1)
                        # rast start from index 1
                        faces_track = faces_track.int() - 1
                        v0, v1, v2 = vertices_track[outputs_list[i]['mesh'][0].faces.int()[faces_track.int().squeeze()]].split(1, dim=1)
                        uv_inter_pts = u_track * v0.squeeze() + v_track * v1.squeeze() + (1 - u_track - v_track) * v2.squeeze()

                        uv_inter_pts = uv_inter_pts.squeeze().unsqueeze(0).expand(1, -1, -1)

                        vertices_homo = torch.cat([uv_inter_pts, torch.ones_like(uv_inter_pts[..., :1])], dim=-1)
                        pt_track = torch.bmm(vertices_homo, full_proj_batch.transpose(-1, -2))

                        converted_pcd = pt_track[..., :2] / pt_track[..., -1:]
                        converted_pcd = (converted_pcd + 1) * observations_target[0].shape[1] / 2
                        pt_track = converted_pcd.clamp(min=0, max=observations_target[0].shape[1])
                        
                        assert pred_tracks[j][i+1][..., :2].shape == pt_track[0][..., :2].shape
                        tracking_loss += (pred_tracks[j][i+1][..., :2].clamp(min=0, max=observations_target[0].shape[1]) - pt_track[0][..., :2]).abs().mean() * 1e-3

                    tracking_loss = tracking_loss / nviews_track

                loss = loss_l1 + arap_loss_ + chamfer_loss_ + face_area_loss_ + edge_length_loss_ + (tracking_loss if args.use_tracking else 0)

                # set the progress bar
                par.set_description(f"Geometry Consistency Loss: {loss.item():.4f}, L1 Loss: {loss_l1.item():.4f}, ARAP Loss: {arap_loss_.item():.4f}, Chamfer Loss: {chamfer_loss_.item():.4f}, Face Area Loss: {face_area_loss_.item():.4f}, Edge Length Loss: {edge_length_loss_.item():.4f}" + (f", Tracking Loss: {tracking_loss.item():.4f}" if args.use_tracking else ""))

                # Backpropagate
                opt.zero_grad()
                loss.backward()
                
                # Update parameters
                opt.step() 

            '------------------------------------------------------------------------------------------'

            with torch.no_grad():
                # save the rendered images
                for j in range(5):
                    imageio.imsave(output_path + "/" + base_name_list[i] + "_mesh_view_after_geometry_consistency" + str(j) + ".png", (observations_init[j].detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
                    imageio.imsave(output_path + "/" + base_name_list[i] + "_mesh_refview_after_geometry_consistency" + str(j) + ".png", (observations_target[j].detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)) 

                outputs_list[i+1]['mesh'][0] = outputs_list[i]['mesh'][0]
                # detach each attribute in outputs_list[i+1]['mesh'][0]
                outputs_list[i+1]['mesh'][0].vertices = outputs_list[i+1]['mesh'][0].vertices.detach()
                outputs_list[i+1]['mesh'][0].faces = outputs_list[i+1]['mesh'][0].faces.detach()
                outputs_list[i+1]['mesh'][0].vertex_attrs = outputs_list[i+1]['mesh'][0].vertex_attrs.detach()
                outputs_list[i+1]['mesh'][0].face_normal = outputs_list[i+1]['mesh'][0].face_normal.detach()

                temp = copy.deepcopy(outputs_list[i+1])
                temp['mesh'][0] = temp['mesh'][0].deepcopy()
                texture_optim_input_list.append(temp)

                vertices = outputs_list[i+1]['mesh'][0].vertices.cpu().numpy()
                faces = outputs_list[i+1]['mesh'][0].faces.cpu().numpy()

                vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
                glb = trimesh.Trimesh(vertices, faces, visual=visual_list[0], process=False)
                glb.export(output_path + "/" + base_name_list[i+1] + "_consistency" + "_sample.glb")

        "============================================================"

        "======== Section 4.4 - Texture Consistency ========"

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

        texture_size=1024

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

        '''Texture Optimization'''
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

        "============================================================"

        "======== Section 4.5 - Mesh Interpolation and 4D Asset Conversion ========"

        last_base_name = None
        last_vertices = None
        for i, (base_name, outputs) in enumerate(zip(base_name_list, texture_optim_input_list)):
            vertices = outputs['mesh'][0].vertices.cpu().numpy()
            faces = outputs['mesh'][0].faces.cpu().numpy()

            vertices = vertices[vmapping]
            faces = indices

            vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
            mesh = trimesh.Trimesh(vertices, faces, visual=trimesh.visual.TextureVisuals(uv=uvs.cpu().numpy(), image=texture), process=False)

            mesh.export(output_path + "/" + base_name + "_texture_consistency" + "_sample.glb")

            # Interpolate the vertices
            if last_base_name is not None:
                for iiid in range(int(last_base_name) + 1, int(base_name)):
                    inter_vertices = last_vertices + (vertices - last_vertices) * (iiid - int(last_base_name)) / (int(base_name) - int(last_base_name))
                    mesh = trimesh.Trimesh(inter_vertices, faces, visual=trimesh.visual.TextureVisuals(uv=uvs.cpu().numpy(), image=texture), process=False)
                    mesh.export(output_path + "/" + str(iiid).zfill(4) + "_texture_consistency" + "_sample.glb")
            
            last_base_name = base_name
            last_vertices = vertices

        # Convert to offset files between each frame
        outputs_to_files_for_blender(output_path)

        # Execute the blender script
        command = [
            f"{args.blender_path}/blender", "--background", "--python", "blender_script_merge_to_GLB.py", "--",
            "--base_mesh", output_path + "/output_mesh.glb",
            "--vertex_offsets", output_path + "/output_vertex_offsets.npy",
            "--output_glb", output_path + "/output_animation.glb"
        ]

        result = subprocess.run(command, capture_output=True, text=True)

        "============================================================"

        print(result.stdout)
        print(result.stderr)

        print("\n\n ============= Finish processing: ", animation, " =============\n")