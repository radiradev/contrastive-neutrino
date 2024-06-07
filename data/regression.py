import numpy as np
import torchvision
import torch
import os
from glob import glob
from MinkowskiEngine.utils import sparse_quantize
import uproot
import pickle
import particle
import re

class Regression(torchvision.datasets.DatasetFolder):
    quantization_size = 0.38
    
    def __init__(self, root, extensions='.npz', energies='particle_energy_train.pkl', regress='energy'):
        super().__init__(root=root, extensions=extensions, loader=self.loader)
        self.energies = self.load_energies(energies)
        self.regression_value = regress


    def load_energies(self, filename):
        # use the current python file path
        # get the directory from the abs path

        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        print(f'Loading energies from {filepath}')
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def loader(self, path):
        return np.load(path)

    def idx_to_class(self, idx):
        # invert the class_to_idx dictionary
        return {v: k for k, v in self.class_to_idx.items()}[idx]
    
    def label_to_pdg(self, label):
        name = self.idx_to_class(label)
        pdg_map = {
            'electron': 11,
            'gamma': 22,
            'muon': 13,
            'pion': 211,
            'proton': 2212
        }
        return pdg_map[name]
    
    def replace_throw_part(self, search_key):
        # Define the pattern to match the throw part
        pattern = r'_throw\d+_'
        # Replace the matched pattern with '_throw15_'
        replaced_key = re.sub(pattern, r'_throw15_', search_key)
        return replaced_key

    def correct_energy(self, energy, label):
        # correct energy with the mass of the particle
        pdg = self.label_to_pdg(label)
        mass = particle.Particle.from_pdgid(pdg).mass
        return energy - mass
    

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        sample = self.loader(path)
        coords, feats = sample['coordinates'], sample['adc']
        energy = self.energies[self.replace_throw_part(os.path.basename(path))][self.regression_value]
        # correct energy with the mass of the particle
        energy = self.correct_energy(energy, label)
        coords, feats = sparse_quantize(coords, np.expand_dims(feats, axis=1), quantization_size=self.quantization_size)

        return coords, feats, torch.tensor(energy).float().unsqueeze(0)
