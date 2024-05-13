import os
from enum import Enum
from glob import glob

import numpy as np

import torch
import torchvision

from MinkowskiEngine.utils import sparse_quantize

from rotation_conversions import random_rotation

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

class ThrowsDataset(torchvision.datasets.DatasetFolder):
    quantization_size = 0.38
    
    def __init__(self, dataroot, dataset_type, extensions='.npz', train_mode=True):
        super().__init__(root=dataroot, extensions=extensions, loader=self.loader)
        self.dataset_type = dataset_type
        self.train_mode = train_mode

    def loader(self, path):
        return np.load(path)

    def contrastive_augmentations(self, xi, xj):
        funcs = [rotate, drop, shift_energy, translate] 

        coords_i, feat_i = xi
        coords_j, feat_j = xj
    
        # draw functions and augment i
        funcs_i = np.random.choice(funcs, 2)
        funcs_j = np.random.choice(funcs, 2)

        for func in funcs_i:
            coords_i, feat_i = func(coords_i, feat_i)
        
        for func in funcs_j:
            coords_j, feat_j = func(coords_j, feat_j)
        
        coords_i, feat_i = sparse_quantize(
            coords_i, feat_i, quantization_size=self.quantization_size
        )
        coords_j, feat_j = sparse_quantize(
            coords_j, feat_j, quantization_size=self.quantization_size
        )

        return (coords_i, feat_i), (coords_j, feat_j) 
    
    def augment_single(self, coords, feats):
        funcs = [rotate, drop, shift_energy, translate]
        funcs = np.random.choice(funcs, 2)
        for func in funcs:
            coords, feats = func(coords, feats)
        return coords, feats

    def __getitem__(self, index: int):
        if (
            self.dataset_type == DataPrepType.CLASSIFICATION or
            self.dataset_type == DataPrepType.CLASSIFICATION_AUG
        ):
            path, label = self.samples[index]
            sample = self.loader(path)
            coords, feats = sample['coordinates'], sample['adc']
            if self.train_mode and self.dataset_type == DataPrepType.CLASSIFICATION_AUG:
                coords, feats = self.augment_single(
                    torch.tensor(coords, dtype=torch.float), torch.tensor(feats, dtype=torch.float)
                )
            coords, feats = sparse_quantize(
                coords, np.expand_dims(feats, axis=1), quantization_size=self.quantization_size
            )
            return coords, feats, torch.tensor(label).long().unsqueeze(0)

        elif self.dataset_type == DataPrepType.CONTRASTIVE_AUG:
            path, _ = self.samples[index]
            sample = self.loader(path)
            coords, feats = sample['coordinates'], np.expand_dims(sample['adc'], axis=1)
            # no idea why coords is a tensor while feats is a numpy array
            xi = (torch.tensor(coords), torch.tensor(feats, dtype=torch.float))
            xj = (torch.tensor(coords), torch.tensor(feats, dtype=torch.float))
            xi, xj = self.contrastive_augmentations(xi, xj)
            return xi, xj


class DataPrepType(Enum):
    CONTRASTIVE_AUG = 1
    CLASSIFICATION = 2
    CLASSIFICATION_AUG = 2
