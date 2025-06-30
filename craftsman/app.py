from .pipeline import (
    CraftsManPipeline
)

import sys
import os
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../tripoSG"))

import os
import numpy as np
import torch
from PIL import Image
import trimesh
import random

import uuid
# import ipdb


# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "triposg"))

from tripoSG.utils import simplify_mesh

from tripoSG.mv_adapter.scripts.inference_ig2mv_sdxl import prepare_pipeline, preprocess_image
from tripoSG.mv_adapter.mvadapter.utils import get_orthogonal_camera, make_image_grid
from tripoSG.mv_adapter.mvadapter.utils.render import NVDiffRastContextWrapper, load_mesh, render
import time
from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover


checkpoints_dir = os.path.join(os.path.dirname(__file__), '../models/checkpoints')

checkpoints_dir_CraftsMan = os.path.join(os.path.dirname(__file__), '../models/checkpoints/craftsman-DoraVAE')

file_absolute_path = os.path.abspath(__file__)

def get_random_hex():
    random_bytes = os.urandom(8)
    random_hex = random_bytes.hex()
    timestamp = int(time.time() * 1000)  # Add millisecond precision timestamp
    return f"{random_hex}_{timestamp}"

def get_unique_filename():
    return f"{uuid.uuid4()}_{int(time.time() * 1000)}"

def run_full(image: str, rmbg_img=None, crafts_pipeline=None, mv_adapter_pipe=None, align_scale_with_Trellis=False, seed=0, texture_pipe=None, mod_config=None):
    TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), get_unique_filename())
    os.makedirs(TMP_DIR, exist_ok=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_VIEWS = 6

    if crafts_pipeline is None:
        crafts_pipeline = CraftsManPipeline.from_pretrained(checkpoints_dir_CraftsMan, device=DEVICE, torch_dtype=torch.bfloat16) # bf16 for fast inference

    if mv_adapter_pipe is None:
        mv_adapter_pipe = prepare_pipeline(
            base_model="stabilityai/stable-diffusion-xl-base-1.0",
            vae_model="madebyollin/sdxl-vae-fp16-fix",
            unet_model=None,
            lora_model=None,
            adapter_path="huanngzh/mv-adapter",
            scheduler=None,
            num_views=NUM_VIEWS,
            device=DEVICE,
            dtype=torch.float16,
        )

    # Prepare cameras
    cameras = get_orthogonal_camera(
        elevation_deg=[0, 0, 0, 0, 89.99, -89.99],
        distance=[1.8] * NUM_VIEWS,
        left=-0.55,
        right=0.55,
        bottom=-0.55,
        top=0.55,
        azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
        device=DEVICE,
    )
    ctx = NVDiffRastContextWrapper(device=DEVICE, context_type="cuda")

    for _ in range(5):
        try:
            # rmbg_img save to TMP_DIR
            if rmbg_img is not None:
                rmbg_path = os.path.join(TMP_DIR, f"rmbg_img_{get_random_hex()}.png")
                rmbg_img.save(rmbg_path)

                mesh = crafts_pipeline(rmbg_path).meshes[0]
            
            else:
                mesh = crafts_pipeline(image).meshes[0]
                
            target_face_num = 35000

            # mesh = simplify_mesh(mesh, target_face_num)
            for cleaner in [FloaterRemover(), DegenerateFaceRemover()]:
                mesh = cleaner(mesh)

            # more facenum, more cost time. The distribution median is ~15000
            mesh = FaceReducer()(mesh, max_facenum=20000)

            if align_scale_with_Trellis:
                # ipdb.set_trace()
                vertices_watertight = mesh.vertices @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
                faces_watertight = mesh.faces
                mean_point = mesh.vertices.mean(axis=0)
                vertices_watertight = (vertices_watertight - mean_point) * 0.5 + mean_point
            
            save_dir = os.path.join(TMP_DIR, "examples")
            os.makedirs(save_dir, exist_ok=True)
            mesh_path = os.path.join(save_dir, f"craftsman_{get_random_hex()}.glb")
            mesh.export(mesh_path)

            torch.cuda.empty_cache()

            height, width = 768, 768

            # mv adapter

            mesh = load_mesh(mesh_path, rescale=True, device=DEVICE)
            render_out = render(
                ctx,
                mesh,
                cameras,
                height=height,
                width=width,
                render_attr=False,
                normal_background=0.0,
            )
            control_images = (
                torch.cat(
                    [
                        (render_out.pos + 0.5).clamp(0, 1),
                        (render_out.normal / 2 + 0.5).clamp(0, 1),
                    ],
                    dim=-1,
                )
                .permute(0, 3, 1, 2)
                .to(DEVICE)
            )
            if rmbg_img is None:
                rmbg_img = Image.open(image)
            image = preprocess_image(rmbg_img, height, width)

            pipe_kwargs = {}
            pipe_kwargs["generator"] = torch.Generator(device=DEVICE).manual_seed(seed)

            images = mv_adapter_pipe(
                "high quality",
                height=height,
                width=width,
                num_inference_steps=15,
                guidance_scale=3.0,
                num_images_per_prompt=NUM_VIEWS,
                control_image=control_images,
                control_conditioning_scale=1.0,
                reference_image=image,
                reference_conditioning_scale=1.0,
                negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
                cross_attention_kwargs={"scale": 1.0},
                **pipe_kwargs,
            ).images

            torch.cuda.empty_cache()

            mv_image_path = os.path.join(save_dir, f"mv_adapter_{get_random_hex()}.png")
            make_image_grid(images, rows=1).save(mv_image_path)

            if texture_pipe is None:
                from tripoSG.texture import TexturePipeline
                texture_pipe = TexturePipeline(
                    upscaler_ckpt_path=f"{checkpoints_dir}/RealESRGAN_x2plus.pth",
                    inpaint_ckpt_path=f"{checkpoints_dir}/big-lama.pt",
                    device=DEVICE,
                )

            if mod_config is None:
                from tripoSG.texture import ModProcessConfig
                mod_config = ModProcessConfig(view_upscale=True, inpaint_mode="view")

            textured_glb_path = texture_pipe(
                mesh_path=mesh_path,
                save_dir=save_dir,
                save_name=f"texture_mesh_{get_random_hex()}.glb",
                uv_unwarp=True,
                uv_size=4096,
                rgb_path=mv_image_path,
                rgb_process_config=mod_config,
                camera_azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
            )
            break
        except Exception as e:
            print("Craftsman Mesh generation Error: ", e)
            print("Retrying...")
            seed = random.randint(0, 1000000)

    glb_mesh = trimesh.load(textured_glb_path, process=False).geometry.popitem()[1]

    if align_scale_with_Trellis:
        # ipdb.set_trace()
        glb_mesh.vertices = glb_mesh.vertices @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        glb_mesh.vertices = (glb_mesh.vertices - mean_point) * 0.5 + mean_point

    # rm the tmp dir
    os.system(f"rm -rf {TMP_DIR}")

    if align_scale_with_Trellis:
        return vertices_watertight.astype(np.float32), faces_watertight.astype(np.int64), glb_mesh
    return glb_mesh

if __name__ == "__main__":
    image = "consistent_all_data/aurorus-copy/0.png"
    _, _, mesh = run_full(image, align_scale_with_Trellis=True)
    mesh.export("test.glb")