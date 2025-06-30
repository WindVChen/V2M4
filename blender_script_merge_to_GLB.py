import bpy
import argparse
import sys
import numpy as np
from mathutils import Vector

# Function to parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Blender animation script with shape keys.")
    
    parser.add_argument("--base_mesh", type=str, required=True, help="Path to the base .glb mesh")
    parser.add_argument("--vertex_offsets", type=str, required=True, help="Path to the vertex offsets .npy file")
    parser.add_argument("--output_glb", type=str, required=True, help="Path to save the output .glb file")
    
    # Extract only arguments after `--`
    if "--" in sys.argv:
        args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    else:
        parser.print_help()
        sys.exit(1)
    
    return args

# Parse arguments
args = parse_args()

# File paths
base_mesh_path = args.base_mesh
vertex_offsets_path = args.vertex_offsets
output_glb_path = args.output_glb

print(f"📂 Using:\n  Mesh: {base_mesh_path}\n  Offsets: {vertex_offsets_path}\n  Output: {output_glb_path}")

# Load `.npy` data
vertex_offsets = np.load(vertex_offsets_path)  # shape: (num_frames, num_vertices, 3)
num_frames, num_vertices, _ = vertex_offsets.shape

# Clear the scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import `.glb`
bpy.ops.import_scene.gltf(filepath=base_mesh_path)

# Get the mesh object
obj = next((o for o in bpy.context.selected_objects if o.type == 'MESH'), None)
if obj is None:
    raise RuntimeError("⚠️ No MESH object found, please check the `.glb` file!")

obj.name = "AnimatedMesh"

# Ensure object is active
bpy.context.view_layer.objects.active = obj
bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)

# Add Shape Keys
if obj.data.shape_keys is None:
    obj.shape_key_add(name="Basis")  # Basis shape key (default)

# Create shape keys for each frame
for frame_idx in range(num_frames):
    bpy.context.scene.frame_set(frame_idx)

    # Create a new shape key for the frame
    shape_key = obj.shape_key_add(name=f"Frame_{frame_idx}")

    # Apply vertex displacements
    current_offsets = vertex_offsets[frame_idx]
    
    for vert_idx, vert in enumerate(obj.data.vertices):
        shape_key.data[vert_idx].co = vert.co + Vector((current_offsets[vert_idx][0], -current_offsets[vert_idx][2], current_offsets[vert_idx][1]))

    # Animate shape key influence
    shape_key.value = 1.0
    shape_key.keyframe_insert(data_path="value", frame=frame_idx)
    if frame_idx > 0:
        shape_key.value = 0.0
        shape_key.keyframe_insert(data_path="value", frame=frame_idx - 1)

# Adjust timeline frame range
bpy.context.scene.frame_end = num_frames - 1

# # Export FBX (will loss texture, maybe due to their type difference. GLB - PBR, FBX - Phong)
# bpy.ops.export_scene.fbx(filepath=output_fbx_path, 
#                          use_selection=True, 
#                          use_mesh_modifiers=True, 
#                          add_leaf_bones=False, 
#                          bake_anim=True, 
#                          bake_anim_use_all_bones=False, 
#                          bake_anim_use_nla_strips=False, 
#                          bake_anim_use_all_actions=False, 
#                          path_mode='COPY', 
#                          embed_textures=True)

# Export GLB
bpy.ops.export_scene.gltf(filepath=output_glb_path, export_format='GLB', export_texcoords=True, export_materials='EXPORT')

print(f"✅ GLB animation exported: {output_glb_path}")
