import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
import rembg
import subprocess
import copy
import imageio
import torch
import random
import numpy as np
from PIL import Image
import trimesh
import argparse
import dill
from natsort import ns, natsorted

from v2m4_trellis.pipelines import TrellisImageTo3DPipeline, mesh_generation, registration, texture_optimization
from v2m4_trellis.utils import render_utils, postprocessing_utils
from v2m4_trellis.utils.render_utils import rotation_6d_to_matrix


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

    '''======================= Intialization of models in the pipeline ======================='''
    # Load a pipeline from a model folder or a Hugging Face model hub.
    if args.model == "TRELLIS":
        pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        pipeline.cuda()
    elif args.model == "Hunyuan":
        try:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

            pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
            pipeline_paint = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
        except:
            raise ImportError("Please install Hunyuan3D related dependencies according to the instructions in Install.md")
    elif args.model == "TripoSG" or args.model == "Craftsman":
        checkpoints_dir = "./models/checkpoints/"

        try:
            # we use TripoSG's texture gen for Craftsman3D as well
            from tripoSG.mv_adapter.scripts.inference_ig2mv_sdxl import prepare_pipeline
            from tripoSG.texture import TexturePipeline, ModProcessConfig

            if args.model == "TripoSG":
                import tripoSG.app as tripoSG_app
                from tripoSG.triposg.scripts.briarmbg import BriaRMBG
                from tripoSG.triposg.triposg.pipelines.pipeline_triposg import TripoSGPipeline

                RMBG_PRETRAINED_MODEL = f"{checkpoints_dir}/RMBG-1.4"

                pipeline_rmbg_net = BriaRMBG.from_pretrained(RMBG_PRETRAINED_MODEL).to("cuda")
                pipeline_rmbg_net.eval()

                TRIPOSG_PRETRAINED_MODEL = f"{checkpoints_dir}/TripoSG"
                pipeline_triposg_pipe = TripoSGPipeline.from_pretrained(TRIPOSG_PRETRAINED_MODEL).to("cuda", torch.float16)
            elif args.model == "Craftsman":
                import craftsman.app as craftsman_app
                from craftsman.pipeline import CraftsManPipeline

                checkpoints_dir_CraftsMan = f"{checkpoints_dir}/craftsman-DoraVAE"
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
        except:
            raise ImportError("Please install TripoSG or Craftsman related dependencies according to the instructions in Install.md, note for Craftsman, you need to both install Craftsman and TripoSG dependencies as Craftsman leverages TripoSG's texture generation module")

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

    '''========================================================'''

    '''======================= Start Processing each animation ======================='''
    for animation in assigned_animations:
        source_path = animation
        output_path = os.path.join(default_output_root if args.output == "" else args.output, animation.split("/")[-1])

        print("\n\n ============= Start processing: ", animation, f"(base model: {args.model})", " =============\n")

        # New folder for the output
        os.makedirs(output_path, exist_ok=True)

        # Fix the seed for reproducibility
        seed = args.seed
        seed_torch(seed)
            
        imgs_list = os.listdir(source_path)
        # exclude folders
        imgs_list = [img for img in imgs_list if not os.path.isdir(source_path + "/" + img)]
        imgs_list = natsorted(imgs_list, alg=ns.PATH)

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
                outputs = mesh_generation.hunyuan_mesh_gen(pipeline, pipeline_paint, cropped_image, args)
                
            elif args.model == "TripoSG" or args.model == "Craftsman":
                save_path = output_path + "/" + base_name + "_rmbg.png"
                _, rmbg_image_rgba, rmbg_image = TrellisImageTo3DPipeline.preprocess_image(image, return_all_rbga=True, rembg_session=rembg_session)
                rmbg_image.save(save_path)

                torch.manual_seed(seed)

                if args.model == "TripoSG":
                    vertices, faces, mesh = tripoSG_app.run_full(source_path + "/" + img, rmbg_image_rgba, pipeline_rmbg_net, pipeline_triposg_pipe, pipeline_mv_adapter_pipe, True, seed, pipeline_texture, mod_config, max_faces=args.max_faces)
                elif args.model == "Craftsman":
                    vertices, faces, mesh = craftsman_app.run_full(source_path + "/" + img, rmbg_image_rgba, pipeline_crafts, pipeline_mv_adapter_pipe, True, seed, pipeline_texture, mod_config, max_faces=args.max_faces)

                outputs = mesh_generation.tripo_craftsman_mesh_process(vertices, faces, mesh)

            if args.baseline:
                if args.model == "TRELLIS":
                    postprocessing_utils.render_video_and_glbs(outputs, base_name + "_baseline", output_path)
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
            outputs, extr, visual = mesh_generation.repose(args, outputs, params, output_path, base_name)

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
            nviews_track = 20
            uv_tracking_points, pred_tracks, full_proj_batchs = registration.point_tracking(args, outputs_list, extrinsics_list, output_path, nviews_track)
        else:
            nviews_track = 0
            uv_tracking_points = None
            pred_tracks = None
            full_proj_batchs = None

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
            
            ' -------------- First, coarsely align the last frame to the current frame ----------------'
            rot6D, translation, scale, observations_init, observations_target = registration.coarse_registration(outputs_list, extrinsics_list, i, envmap, v_ref, v, f)
            
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

            '------------- Second, densely align the last frame to the current frame (outputs_list updated inside the function) -----------------'
            observations_init, observations_target = registration.fine_registration(args, outputs_list, extrinsics_list, i, envmap, nviews_track, uv_tracking_points, pred_tracks, full_proj_batchs, v_ref, v, f, init_copy)

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

            '------------------------------------------------------------------------------------------'

        "============================================================"

        "======== Section 4.4 - Texture Consistency ========"
        texture_size=1024

        # collect all the observations and corresponding uv coordinates
        all_observations, all_masks, all_uv, all_uv_dr, add_weight_index, vmapping, indices, uvs, rastctx, faces = texture_optimization.collect_mesh_views(args, texture_optim_input_list, extrinsics_list, output_path, base_name_list, outputs_list)

        '''Texture Optimization'''
        texture = texture_optimization.texture_optimization(all_observations, all_masks, all_uv, all_uv_dr, add_weight_index, uvs, texture_size, rastctx, faces)

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
        postprocessing_utils.outputs_to_files_for_blender(output_path)

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

    '''========================================================'''