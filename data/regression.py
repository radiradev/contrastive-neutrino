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
    
    def __init__(self, root, extensions='.npz', energies='particle_energy_train.pkl'):
        super().__init__(root=root, extensions=extensions, loader=self.loader)
        self.energies = self.load_energies(energies)


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
        energy = self.energies[self.replace_throw_part(os.path.basename(path))]['energy'] # summed_deps or energy
        # correct energy with the mass of the particle
        #energy = self.correct_energy(energy, label)
        coords, feats = sparse_quantize(coords, np.expand_dims(feats, axis=1), quantization_size=self.quantization_size)

        return coords, feats, torch.tensor(energy).float().unsqueeze(0)


class RegressionProtons(torch.utils.data.Dataset):
    quantization_size = 0.38
    
    def __init__(self, 
                 root='/global/cfs/cdirs/dune/users/rradev/contrastive/edep_2thousand_per_particle/larndsim_converted/all_nominal/proton',  
                 extensions='.npz', 
                 train_mode=True):
        super().__init__()
        self.train_mode = train_mode
        root_path = '/global/cfs/cdirs/dune/users/rradev/contrastive/edep_2thousand_per_particle/edep_root'
        self.root_evt_ids, self.energies = self.load_root(root_path)
        self.samples = glob(os.path.join(root, f'*{extensions}'))

    def load_root(self, path):
        particle_name = 'proton'
        filename = f'{particle_name}_edeps_out_2000.root'
        file = uproot.open(os.path.join(path, filename))['EDepSimEvents']
        event_id = file['Event/EventId'].array()

        # get the energy of the first particle in the event
        energies = file['Trajectories']['Trajectories.InitialMomentum'].array()[:, 0]['fE']
        energies = np.array(energies)
        # check that it is a proton
        pdg = file['Trajectories']['Trajectories.PDGCode'].array()[:, 0]
        assert np.all(pdg == 2212), "Not all particles are protons"
        proton_mass = particle.Particle.from_pdgid(2212).mass
        energies -= proton_mass
        return np.array(event_id), energies

    def __len__(self):
        return len(self.samples)

    def loader(self, path):
         # get eventID from the path
        # get the filename
        filename = os.path.basename(path)
        # get the eventID
        eventID = filename.split('_')[-1].split('.')[0] # filename in the form {particle}_*_throw{num}_eventID_{id}.npz
        assert eventID.isdigit(), f"EventID {eventID} is not a number"
        return np.load(path), int(eventID)

    def __getitem__(self, index: int):
        path = self.samples[index]
        sample, eventID = self.loader(path)
        # use the eventID to get the energy
        root_evt_id = self.root_evt_ids[eventID]
        energy = self.energies[eventID]
        assert eventID == root_evt_id, f"EventID {eventID} does not match the root evt_id {root_evt_id}"


        coords, feats = sample['coordinates'], sample['adc']
        coords, feats = sparse_quantize(coords, np.expand_dims(feats, axis=1), quantization_size=self.quantization_size)
        return coords, feats, torch.tensor(energy).long().unsqueeze(0)
        
if __name__ == '__main__':
    dataset = RegressionProtons()
    coords, feats, label = dataset[0]
    print(coords.shape, feats.shape, label.shape)