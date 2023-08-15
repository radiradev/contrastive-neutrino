import torch
from MinkowskiEngine.utils import sparse_collate

def clr_sparse_collate(data, dtype=torch.int32, device=None):
    # Unzip the dataset into separate coordinate and feature tuples for i and j
    x_i, x_j = zip(*data)
    coordinates_i, features_i = zip(*x_i)
    coordinates_j, features_j = zip(*x_j)
    # Collate the coordinate and feature tuples separately
    collated_i = sparse_collate(coords=coordinates_i, feats=features_i, dtype=dtype, device=device)
    collated_j = sparse_collate(coords=coordinates_j, feats=features_j, dtype=dtype, device=device)
    return collated_i, collated_j