from tqdm import tqdm
import uproot
import numpy as np
import pickle
import h5py
import os
from multiprocessing import Pool, Manager
from functools import partial

def get_npz_files(phase, particle):
    return sorted(os.listdir(f'{larnd_directory}/{phase}/{particle}'))

# find the corresponding root file 
def get_edep_file(npz_file):
    npz_file = os.path.basename(npz_file).split('_')[:4]
    npz_file = '_'.join(npz_file)
    return os.path.join(edeps_directory, npz_file + '.root')

def load_npz_file(filename, phase, particle):
    return np.load(os.path.join(larnd_directory, phase, particle, filename))

def extract_energy(filename, event_id):
    try:
        file = uproot.open(filename)['EDepSimEvents']
    except:
        print(f'Could not open {filename}')
        return None
    energies = file['Trajectories']['Trajectories.InitialMomentum'].array()[:, 0]['fE']
    file.close()
    return np.array(energies)[event_id]

def extract_vertex(filename, event_id):
    file = uproot.open(filename)['EDepSimEvents']
    vertex = file['Primaries']['Primaries.Position'].array()['fP'][event_id]
    x, y, z = vertex['fX'], vertex['fY'], vertex['fZ']
    return np.array([x, y, z])

def get_h5py_file(npz_file):
    npz_file = os.path.basename(npz_file).split('_')[:4]
    npz_file = '_'.join(npz_file)
    return os.path.join(edeps_directory, npz_file + '.h5')

def get_energy_deposits(npz_file, event_id):
    h5py_file = get_h5py_file(npz_file)
    f = h5py.File(h5py_file, 'r')
    mask = f['segments']['eventID'] == event_id
    dE = f['segments']['dE'][mask]
    f.close()
    return dE
    

def energy_vertex(npz_file, phase, particle):
    # find the corresponding edep file
    # grab the event id from the npz file
    event_id = npz_file.split('_')[-1]
    # remove .npz
    event_id = event_id.split('.')[0]
    assert event_id.isdigit()

    edep_file = get_edep_file(npz_file)

    # get the energy
    energy = extract_energy(edep_file, int(event_id))
    # get the vertex
    vertex = extract_vertex(edep_file, int(event_id))
    
    
    edeps = get_energy_deposits(npz_file, int(event_id))

    # check the vertex is matching
    npz = load_npz_file(npz_file, phase, particle)
    assert np.allclose(vertex, npz['vertex'] * 10), f' Vertices did not match {vertex} {npz["vertex"]}'
    particle_energy[npz_file] = {
        "energy": energy,
        "summed_deps": np.sum(edeps)
    }

def process_npz_files(phase, particles):
    def process_particle(particle):
        npz_files = get_npz_files(phase, particle)
        energy_vertex_local = partial(energy_vertex, phase=phase, particle=particle)
        with Pool() as pool:
            for _ in tqdm(pool.imap_unordered(energy_vertex_local, [f for f in npz_files]), total=len(npz_files)):
                pass

    for particle in particles:
        process_particle(particle)

edeps_directory = '/global/cfs/cdirs/dune/users/rradev/contrastive/edep_2thousand_per_particle/edeps/'
larnd_directory = '/global/cfs/cdirs/dune/users/rradev/contrastive/edep_2thousand_per_particle/larndsim_converted/'

particle_energy = Manager().dict()

phases = [
    'all_nominal'
]

for phase in phases:
    process_npz_files(phase, ['pion', 'proton', 'gamma', 'electron', 'muon'])

with open('particle_energy_throws.pkl', 'wb') as f:
    pickle.dump(dict(particle_energy), f)