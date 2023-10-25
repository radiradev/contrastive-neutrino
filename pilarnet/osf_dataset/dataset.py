import torch 

from torch.utils.data import Dataset, DataLoader
from pytorch3d.transforms import random_rotation
from MinkowskiEngine.utils import batch_sparse_collate, sparse_quantize


class ClassifierDatasetOSF(Dataset):
    def __init__(self, data_path):
        super().__init__()
        # only import the image reader when we need it
        from .osf.image_api import image_reader_3d    
        self.image_reader = image_reader_3d(data_path)

    def __len__(self) -> int:
        return self.image_reader.entry_count()
    
    def __getitem__(self, index: int):
        reader_output = self.image_reader.get_image(index)
        voxels, energy, classes  = (torch.from_numpy(item) for item in reader_output)
        return voxels, energy.unsqueeze(-1), classes

import os, sys
import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import MinkowskiEngine as ME

# this must be faster to do when batched 
def rotate(coords, feats):
    coords = coords @ random_rotation(dtype=coords.dtype, device=coords.device)
    return coords, feats

def drop(coords, feats, p=0.1):
    mask = torch.rand(coords.shape[0]) > p
    return coords[mask], feats[mask]

def shift_energy(coords, feats, max_scale_factor=0.1):
    shift = 1 - torch.rand(1, dtype=feats.dtype, device=feats.device) * max_scale_factor
    return coords, feats * shift

def translate(coords, feats, cube_size=512):
    normalized_shift = torch.rand(3, dtype=coords.dtype, device=coords.device)
    translation = normalized_shift * (cube_size / 10)
    return coords + translation, feats

def identity(coords, feats):
    return coords, feats


#### Eventually the datasets should be grouped together
class ConvertedDataset(torchvision.datasets.DatasetFolder):
    def __init__(
            self,
            root='/mnt/rradev/osf_data_512px/converted_data/train',
            extensions='.npz',
            take_log = False,
            take_sqrt = True,
            clip = True,

    ):
        super().__init__(root=root, extensions=extensions, loader=self.loader)
        self.take_log = take_log
        self.take_sqrt = take_sqrt
        self.clip = clip
        
    def loader(self, path):
        return np.load(path)
    
    def transform_energy(self, energy):
        if self.take_log:
            energy = torch.log(energy)
        if self.take_sqrt:
            energy = torch.sqrt(energy)
        if self.clip:
            energy = torch.clip(energy, -1, 1)
        return energy

    def preprocessing(self, sample):
        # split the energy and the coordinates
        coords, feat = np.split(sample['points'], [3], axis=1)

        # convert the label to an integer
        label = self.class_to_idx[str(sample['label'])]

        # convert to torch tensors
        coords = torch.from_numpy(coords).float()
        feat = torch.from_numpy(feat).float()
        label = torch.tensor([label]).long()

        return coords, feat, label 
    
    def apply_augmentations(self, coords_i, feat_i, label):
        funcs = [rotate, drop, shift_energy, translate] 
        
        
        # draw functions and augment i
        funcs_i = np.random.choice(funcs, 2)

        for func in funcs_i:
            coords_i, feat_i = func(coords_i, feat_i)

        feat_i = self.transform_energy(feat_i)
        coords_i, feat_i = sparse_quantize(coords_i, feat_i)

        return coords_i, feat_i, label
        


    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, _ = self.samples[index]
        sample = self.loader(path)
        coords, feat, label = self.preprocessing(sample)
        coords, feat, label = self.apply_augmentations(coords, feat, label)
        return coords, feat, label



import zipfile
class DebugDataset(ConvertedDataset):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except zipfile.BadZipFile as e:
            print(f"Error accessing file at index {index}: {self.samples[index][0]}")
            raise e



class MaskedDatasetOSF(Dataset):
    def __init__(self, data_path, patch_size=32, masking_ratio=0.5) -> None:
        super().__init__()
        from .osf.image_api import image_reader_3d    
        self.image_reader = image_reader_3d(data_path)
        self.patch_size = patch_size
        self.masking_ratio = masking_ratio

    def __len__(self) -> int:
        return self.image_reader.entry_count()

    def __getitem__(self, index: int):
        reader_output = self.image_reader.get_image(index)
        voxels, energy, _ = (torch.from_numpy(item) for item in reader_output)
        target = voxels, energy.unsqueeze(-1)

        # TODO find a way to do this without creating a new tensor


        masked_input = drop_random_cubes(voxels, energy, self.masking_ratio, self.patch_size)
        return masked_input, target


def drop_random_cubes(coordinates, features, drop_rate, quantization_size):
    if features.ndim == 1:
        features = features.unsqueeze(-1)
    quant_coords, quant_feats = sparse_quantize(coordinates, features, quantization_size=quantization_size)

    # Determine the number of cubes to drop
    num_cubes_to_drop = int(len(quant_coords) * drop_rate)

    # Randomly select cubes to drop
    drop_indices = torch.randperm(len(quant_coords))[:num_cubes_to_drop]
    drop_coords = quant_coords[drop_indices]

    # Create a mask for dropped cubes
    drop_mask = torch.ones(len(quant_coords), dtype=torch.bool)
    drop_mask[drop_indices] = 0

    # Get the quantized coords of all voxels
    quantized_voxel_coords = (coordinates / quantization_size).floor().int()

    # Remove voxels in dropped cubes from voxels
    voxel_drop_mask = (quantized_voxel_coords[:, None, :] == drop_coords[None, :, :]).all(-1).any(-1)
    new_voxels = coordinates[~voxel_drop_mask]
    new_features = features[~voxel_drop_mask]


    return new_voxels, new_features


if __name__ == "__main__":
    # create dataset
    paths = ['/workspace/osf_data/sample.root', '/workspace/osf_data/sample_2.root']
    dataset = MaskedDatasetOSF(paths)

    # create dataloader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=batch_sparse_collate)
    batch = next(iter(dataloader))

    #asert that elements of the batch are not identical
    assert not all([torch.equal(batch[0], batch[i]) for i in range(len(batch))]), "All elements of the batch are identical"

    # print batch
    voxels, energy, classes = batch
    print(voxels)
    print(voxels.shape, energy.shape, classes.shape)
