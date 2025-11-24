import torch
import numpy as np
import trimesh
import utils3d
from v2m4_trellis.utils.postprocessing_utils import render_video_and_glbs
from v2m4_trellis.utils.general_utils import *
from v2m4_trellis.representations.mesh import MeshExtractResult
from v2m4_trellis.utils import render_utils, postprocessing_utils


def repose(args, outputs, params, output_path, base_name):
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

    return outputs, extr, visual


def hunyuan_mesh_gen(pipeline, pipeline_paint, cropped_image, args):
    from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover

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
    return outputs


def tripo_craftsman_mesh_process(vertices, faces, mesh):
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
    return outputs