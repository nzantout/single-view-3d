import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch3d.ops.knn as knn
import pytorch3d.ops
from pytorch3d.structures import Meshes

# define losses
def voxel_loss(voxel_src: torch.Tensor, voxel_tgt: torch.Tensor):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# loss = 
	# implement some loss for binary voxel grids

	loss = nn.BCELoss()(
		torch.clamp(voxel_src, min=0.0, max=1.0), 
		torch.clamp(voxel_tgt, min=0.0, max=1.0))

	return loss

def chamfer_loss(point_cloud_src: torch.Tensor, point_cloud_tgt: torch.Tensor):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 
	# implement chamfer loss from scratch

	dists_12, idxs_12, _ = knn.knn_points(point_cloud_src, point_cloud_tgt)
	dists_21, idxs_21, _ = knn.knn_points(point_cloud_tgt, point_cloud_src)

	loss_chamfer = dists_12.sum() + dists_21.sum()

	return loss_chamfer

def smoothness_loss(mesh_src: Meshes):
	# loss_laplacian = 
	# implement laplacian smoothening loss
	
	V = mesh_src.verts_packed()
	num_verts = V.shape[0]
	edges = mesh_src.edges_packed()
	neg_indices = torch.cat([edges, edges.flip(1)], dim=0).T
	neg_values = -torch.ones(neg_indices.shape[1], device=mesh_src.device)
	L_neg = torch.sparse_coo_tensor(
		neg_indices, 
		neg_values,
		(num_verts, num_verts),
		device=mesh_src.device)
	
	sums = -L_neg.sum(dim=1).to_dense()
	values = torch.cat([neg_values, sums])
	indices = torch.cat([
		neg_indices, 
		torch.range(0, num_verts-1, device=mesh_src.device).repeat(2, 1)],
		dim=1)

	L = torch.sparse_coo_tensor(indices, values, (num_verts, num_verts), device=mesh_src.device)

	loss_laplacian = torch.norm(L @ V)

	return loss_laplacian