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
    
    def __init__(self, dataset_type, root, extensions='.npz',train_mode=True):
        super().__init__(root=root, extensions=extensions, loader=self.loader)
        self.dataset_type = dataset_type
        assert dataset_type in ['single_particle', 'contrastive', 'augmentations', 'single_particle_augmented'], f"Unknown dataset type {dataset_type}"
        # Create the cache
        if dataset_type == 'contrastive':
            self.create_path_cache()
        self.train_mode = train_mode

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
        
        coords_i, feat_i = sparse_quantize(coords_i, feat_i, quantization_size=self.quantization_size)
        coords_j, feat_j = sparse_quantize(coords_j, feat_j, quantization_size=self.quantization_size)

        return (coords_i, feat_i), (coords_j, feat_j) 
    
    def augment_single(self, coords, feats):
        funcs = [rotate, drop, shift_energy, translate]
        funcs = np.random.choice(funcs, 2)
        for func in funcs:
            coords, feats = func(coords, feats)
        return coords, feats

    def __getitem__(self, index: int):
        if self.dataset_type == 'single_particle':
            path, label = self.samples[index]
            sample = self.loader(path)
            coords, feats = sample['coordinates'], sample['adc']
            coords, feats = sparse_quantize(coords, np.expand_dims(feats, axis=1), quantization_size=self.quantization_size)
            return coords, feats, torch.tensor(label).long().unsqueeze(0)

        elif self.dataset_type == 'single_particle_augmented':
            path, label = self.samples[index]
            sample = self.loader(path)
            
            coords, feats = sample['coordinates'], sample['adc']
            coords, feats = torch.tensor(coords), torch.tensor(np.expand_dims(feats, axis=1), dtype=torch.float32)
            
            if self.train_mode:
                coords, feats = self.augment_single(coords, feats)
            
            coords, feats = sparse_quantize(coords, feats, quantization_size=self.quantization_size)
            return coords, feats, torch.tensor(label).long().unsqueeze(0)
        
        elif self.dataset_type == 'contrastive':
            path, _ = self.samples[index]
            other_path = self.grab_other_path(path)  # This will now use the cache
            sample, other_sample = self.loader(path), self.loader(other_path)

            coords_i, feats_i = sample['coordinates'], np.expand_dims(sample['charge'], axis=1)
            coords_j, feats_j = other_sample['coordinates'], np.expand_dims(other_sample['charge'], axis=1)

            xi, xj = (torch.tensor(coords_i), torch.tensor(feats_i)), (torch.tensor(coords_j), torch.tensor(feats_j))
            xi, xj = self.contrastive_augmentations(xi, xj)
            return xi, xj 
        
        elif self.dataset_type == 'augmentations':
            path, _ = self.samples[index]
            sample = self.loader(path)
            coords, feats = sample['coordinates'], np.expand_dims(sample['charge'], axis=1)
            # no idea why coords is a tensor while feats is a numpy array
            xi, xj = (torch.tensor(coords), torch.tensor(feats)), (torch.tensor(coords), torch.tensor(feats))
            xi, xj = self.contrastive_augmentations(xi, xj)
            return xi, xj

class CLRDataset(torchvision.datasets.DatasetFolder):
    def __init__(
            self,
            dataset_type,
            root='/pscratch/sd/r/rradev/converted_data/train',
            extensions='.npz',
            take_log = False,
            take_sqrt = True,
            clip = True,

    ):
        super().__init__(root=root, extensions=extensions, loader=self.loader)
        self.dataset_type = dataset_type
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
        label = torch.tensor(label).long()

        return coords, feat, label 
    
    def classifier_augmentations(self, coord, feat, label):
        funcs = np.random.choice([identity, drop, shift_energy, translate], 2)
        for func in funcs:
            coord, feat = func(coord, feat) 
        
        feat = self.transform_energy(feat)
        coord, feat = sparse_quantize(coord, feat)
        return coord, feat, label
    
    def contrastive_augmentations(self, coords_i, feat_i, label):
        coords_j, feat_j = coords_i.clone(), feat_i.clone()
        funcs = [rotate, drop, shift_energy, translate] 
        
        
        # draw functions and augment i
        funcs_i = np.random.choice(funcs, 2)
        funcs_j = np.random.choice(funcs, 2)

        for func in funcs_i:
            coords_i, feat_i = func(coords_i, feat_i)
        
        for func in funcs_j:
            coords_j, feat_j = func(coords_j, feat_j)

        feat_i = self.transform_energy(feat_i)
        feat_j = self.transform_energy(feat_j)
        coords_i, feat_i = sparse_quantize(coords_i, feat_i)
        coords_j, feat_j = sparse_quantize(coords_j, feat_j)

        return (coords_i, feat_i), (coords_j, feat_j) 
        


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
        if self.dataset_type == 'contrastive':
            x_i, x_j = self.contrastive_augmentations(coords, feat, label)
            return x_i, x_j
        elif self.dataset_type == 'single_particle_augmented':
            coords, feat, label = self.classifier_augmentations(coords, feat, label)
            return coords, feat, label.unsqueeze(0)
        elif self.dataset_type == 'single_particle_base':
            feat = self.transform_energy(feat)
            coords, feat = sparse_quantize(coords, feat)
            return coords, feat, label.unsqueeze(0)
        else:
            raise ValueError(f'Unknown dataset type {self.dataset_type}')
        

if __name__ == '__main__':
    dataset = CLRDataset('contrastive')
    xi, xj = dataset[0]
    print(xi[0].shape, xj[0].shape)

    dataset_larnd = ThrowsDataset(dataset_type='single_particle_augmented', root='/pscratch/sd/r/rradev/larndsim_throws_converted_new/val')
    coords, feats, label = dataset_larnd[0]
    print(coords.shape, feats.shape, label.shape)
