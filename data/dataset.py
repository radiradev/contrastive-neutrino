import numpy as np
import torchvision
import torch
import os
from glob import glob
from utils.rotation_conversions import random_rotation
from MinkowskiEngine.utils import sparse_quantize

# this must be faster to do when batched 
def rotate(coords, feats):
    coords = coords @ random_rotation(dtype=coords.dtype, device=coords.device)
    return coords, feats

def drop(coords, feats, p=0.1):
    mask = torch.rand(coords.shape[0]) > p
    #don't drop all coordinates
    if mask.sum() == 0:
        return coords, feats
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


def contrastive_augmentations(xi, xj, quantization_size=0.38):
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
    
    coords_i, feat_i = sparse_quantize(coords_i, feat_i, quantization_size=quantization_size)
    coords_j, feat_j = sparse_quantize(coords_j, feat_j, quantization_size=quantization_size)

    return (coords_i, feat_i), (coords_j, feat_j) 


class ClassifierBaseDataset(torchvision.datasets.DatasetFolder):
    def __init__(self, root, extensions='.npz'):
        super().__init__(root=root, extensions=extensions, loader=self.loader)

    def loader(self, path):
        return np.load(path)
    
    def __getitem__(self, index: int):
        path, label = self.samples[index]
        sample = self.loader(path)
        coords, feats = sample['coordinates'], sample['adc']
        coords, feats = sparse_quantize(torch.tensor(coords, dtype=torch.float), torch.tensor(np.expand_dims(feats, axis=1), dtype=torch.float), quantization_size=self.quantization_size)
        return coords, feats, torch.tensor(label).long().unsqueeze(0)


class ClassifierAugmentedDataset(torchvision.datasets.DatasetFolder):
    def __init__(self, train_mode, root, extensions='.npz'):
        super().__init__(root=root, extensions=extensions, loader=self.loader)
        self.train_mode = train_mode

    def loader(self, path):
        return np.load(path)

    def augment_single(self, coords, feats):
        funcs = [rotate, drop, shift_energy, translate]
        funcs = np.random.choice(funcs, 2)
        for func in funcs:
            coords, feats = func(coords, feats)
        return coords, feats

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        sample = self.loader(path)
        
        coords, feats = sample['coordinates'], sample['adc']
        coords, feats = torch.tensor(coords, dtype=torch.float), torch.tensor(np.expand_dims(feats, axis=1), dtype=torch.float32)
        
        if self.train_mode:
            coords, feats = self.augment_single(coords, feats)
        
        coords, feats = sparse_quantize(coords, feats, quantization_size=self.quantization_size)
        return coords, feats, torch.tensor(label).long().unsqueeze(0)


class ContrastiveAugmentationsDataset(torchvision.datasets.DatasetFolder):
    def __init__(self, root, extensions='.npz'):
        super().__init__(root=root, extensions=extensions, loader=self.loader)
        
    def loader(self, path):
        return np.load(path)

    def __getitem__(self, index: int):
        path, _ = self.samples[index]
        sample = self.loader(path)
        coords, feats = sample['coordinates'], np.expand_dims(sample['adc'], axis=1)
        # no idea why coords is a tensor while feats is a numpy array
        xi = (torch.tensor(coords, dtype=torch.float), torch.tensor(feats, dtype=torch.float))
        xj = (torch.tensor(coords, dtype=torch.float), torch.tensor(feats, dtype=torch.float))
        xi, xj = contrastive_augmentations(xi, xj)
        return xi, xj
    
        
class ContrastiveThrowsAugmentationsDataset(torchvision.datasets.DatasetFolder):
    def __init__(self, root, extensions='.npz'):
        super().__init__(root=root, extensions=extensions, loader=self.loader)
        self.create_path_cache()

    def loader(self, path):
        return np.load(path)

    def create_path_cache(self):
        """Prepares a cache dictionary based on file identifiers."""
        self.path_cache = {}
        
        for path, _ in self.samples:
            # Generate the identifier for the current path
            parts = os.path.basename(path).split('_')
            identifier_parts = [part for part in parts if "throw" not in part and "nominal" not in part]
            identifier = "_".join(identifier_parts)
            
            # Append the path to its identifier's list in the cache
            if identifier not in self.path_cache:
                self.path_cache[identifier] = []
            self.path_cache[identifier].append(path)

    def grab_other_path(self, path):
        """Given a path, select another file with the same eventID and file number."""
        # Generate the identifier for the given path
        parts = os.path.basename(path).split('_')
        identifier_parts = [part for part in parts if "throw" not in part and "nominal" not in part]
        identifier = "_".join(identifier_parts)
        
        # Retrieve potential matches from cache
        potential_matches = [p for p in self.path_cache[identifier] if p != path]
        
        # Randomly select one
        if potential_matches:
            return np.random.choice(potential_matches)
        else:
            print(f"Did not find matching file for {path}")
            return path

    def __getitem__(self, index: int):
        path, _ = self.samples[index]
        other_path = self.grab_other_path(path)  # This will now use the cache
        sample, other_sample = self.loader(path), self.loader(other_path)

        coords_i, feats_i = sample['coordinates'], np.expand_dims(sample['adc'], axis=1)
        coords_j, feats_j = other_sample['coordinates'], np.expand_dims(other_sample['adc'], axis=1)

        xi = (torch.tensor(coords_i, dtype=torch.float), torch.tensor(feats_i, dtype=torch.float))
        xj = (torch.tensor(coords_j, dtype=torch.float), torch.tensor(feats_j, dtype=torch.float))
        xi, xj = contrastive_augmentations(xi, xj)
        return xi, xj 


if __name__ == '__main__':
    dataset = ContrastiveThrowsAugmentationsDataset(root='/pscratch/sd/r/rradev/larndsim_throws_converted_jun7/train')
    print(dataset[0])
