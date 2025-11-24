import os
import torch
import imageio
from tqdm import tqdm
from pytorch3d.loss import chamfer_distance

from largesteps.parameterize import from_differential, to_differential
from largesteps.geometry import compute_matrix
from largesteps.optimize import AdamUniform

from v2m4_trellis.utils import render_utils
from v2m4_trellis.utils.general_utils import *
from v2m4_trellis.utils.render_utils import rotation_6d_to_matrix
from v2m4_trellis.utils.optimize_loss import face_area_consistency_loss, edge_length_consistency_loss, arap_loss


def point_tracking(args, outputs_list, extrinsics_list, output_path, nviews_track):
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").cuda()

    with torch.no_grad():
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
    
    return uv_tracking_points, pred_tracks, full_proj_batchs


def coarse_registration(outputs_list, extrinsics_list, i, envmap, v_ref, v, f):
    ' -------------- Coarsely align the last frame to the current frame, optimize 500 iterations first, then optimize 250 iterations for each of the 2 principal axes to ensure the mesh is aligned in the correct direction, not flip ----------------'
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

    return rot6D, translation, scale, observations_init, observations_target


def fine_registration(args, outputs_list, extrinsics_list, i, envmap, nviews_track, uv_tracking_points, pred_tracks, full_proj_batchs, v_ref, v, f, init_copy):
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

    return observations_init, observations_target