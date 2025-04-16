import torch
import yaml
import os
import wandb
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

def clr_labels_sparse_collate(data, dtype=torch.int32, device=None):
    # Unzip the dataset into separate coordinate and feature tuples for i and j
    x_i, x_j, labels = zip(*data)
    labels = torch.cat(labels)
    coordinates_i, features_i = zip(*x_i)
    coordinates_j, features_j = zip(*x_j)
    # Collate the coordinate and feature tuples separately
    collated_i = sparse_collate(coords=coordinates_i, feats=features_i, dtype=dtype, device=device)
    collated_j = sparse_collate(coords=coordinates_j, feats=features_j, dtype=dtype, device=device)
    return collated_i, collated_j, labels

def get_smaller_dataset(original_dataset, num_imgs_per_label, num_labels=5):
    new_dataset = torch.utils.data.TensorDataset(
        *[t.unflatten(0, (num_labels, -1))[:,:num_imgs_per_label].flatten(0, 1) for t in original_dataset.tensors]
    )
    return new_dataset

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def get_wandb_ckpt(artifact_name: str, return_name=False):
    api = wandb.Api()
    artifact = api.artifact(f"{artifact_name}")    
    artifact_dir = artifact.download(os.path.join(os.environ.get('SCRATCH'), artifact_name))
    artifact_dir = os.path.join(artifact_dir, 'model.ckpt')

    if return_name:
        return artifact_dir, artifact.name
    return artifact_dir
