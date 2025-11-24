import torch

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