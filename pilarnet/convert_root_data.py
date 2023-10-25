# modified from https://github.com/NuTufts/pilarnet_w_larcv1/blob/main/extractdata.py

import numpy as np
import concurrent.futures

from larcv import larcv
from tqdm import tqdm
from pathlib import Path
from ROOT import TChain


def pdg_to_label(pdg):
    mapping = {
        11: "Electron",
        -11: "Electron",
        13: "Muon",
        -13: "Muon",
        22: "Gamma",
        211: "Pion",
        -211: "Pion",
        2212: "Proton",
        321: "other",
        1000010020: "other",
        1000020040: "other",
        1000010030: "other",
        1000020030: "other",
        2112: "other",
    }
    return mapping[pdg]


def convert_root_file(index, minimum_voxel_count=10):
    filename = str(files_dir / f'converted_data/dlprod_512px_0{index}.root')
    
    # add particle information
    particle_tree = TChain("particle_mcst_tree")
    particle_tree.AddFile(filename)
    # add cluster information
    cluster_tree=TChain("cluster3d_mcst_tree")
    cluster_tree.AddFile(filename)
    

    for cluster_entry in range(cluster_tree.GetEntries()):
        cluster_tree.GetEntry(cluster_entry)
        particle_tree.GetEntry(cluster_entry)
        clusters = cluster_tree.cluster3d_mcst_branch
        particles = particle_tree.particle_mcst_branch

        # Loop over an array of "array of voxels (for one particle instance)"
        for index,cluster in enumerate(clusters.as_vector()):
            
            if index + 1 > particles.as_vector().size():
                # the last cluster is not associated with a particle
                break

            # Create a numpy array and fill with (x,y,z) coordinate
            voxels = np.zeros(shape=(cluster.size(), 3),dtype=np.int32)
            larcv.fill_3d_voxels(cluster, clusters.meta(), voxels)

            energy = np.zeros(shape=(cluster.size(), 1),dtype=np.float32)
            larcv.fill_3d_pcloud(cluster, clusters.meta(), energy)

            points = np.concatenate((voxels, energy), axis=1)

            label = pdg_to_label(particles.as_vector()[index].pdg_code())

            if len(voxels) < minimum_voxel_count or label == "other":
                continue

            # save the label and points to a file
            save_dir = files_dir / f'{label}_file{index}_cluster{cluster_entry}.npz'
            np.savez(save_dir, points=points, label=label)  
            
            
        
NUM_FILES = 10
files_dir = Path('/mnt/rradev/osf_data_512px')



with concurrent.futures.ProcessPoolExecutor() as executor:
    for _ in tqdm(executor.map(convert_root_file, range(NUM_FILES)), total=NUM_FILES):
        pass





