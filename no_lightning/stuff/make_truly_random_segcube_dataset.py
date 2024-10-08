"""
Messy hardcoded script to make a "truly" random dataset by sampling the unique initial conditions
separately between train, test, and val. This way there will not be the same set of initial
conditions across train, test, val to stop the model "cheating".
"""
import os, shutil, random

import numpy as np
from tqdm import tqdm

# The train and test have the same set of fixed initial points so can use data from either
SEGCUBE_FULL_DATASET = (
    "/share/rcifdata/awilkins/contrastive-neutrino_data/datasets/segmentedcube_full/test"
)
SEGCUBE_OUT_DATASET = (
    "/share/rcifdata/awilkins/contrastive-neutrino_data/datasets/segmentedcube_truly_randomised"
)
OUT_DATASET_TRAIN_SIZE = 30000
OUT_DATASET_VAL_SIZE = 6000
OUT_DATASET_TEST_SIZE = 100000

electron_x = np.load("test_electron_x.npy")
muon_x = np.load("test_muon_x.npy")
pion_x = np.load("test_pion_x.npy")
electron_y = np.load("test_electron_y.npy")
muon_y = np.load("test_muon_y.npy")
pion_y = np.load("test_pion_y.npy")
electron_z = np.load("test_electron_z.npy")
muon_z = np.load("test_muon_z.npy")
pion_z = np.load("test_pion_z.npy")
electron_dirx = np.load("test_electron_dirx.npy")
muon_dirx = np.load("test_muon_dirx.npy")
pion_dirx = np.load("test_pion_dirx.npy")
electron_diry = np.load("test_electron_diry.npy")
muon_diry = np.load("test_muon_diry.npy")
pion_diry = np.load("test_pion_diry.npy")
electron_dirz = np.load("test_electron_dirz.npy")
muon_dirz = np.load("test_muon_dirz.npy")
pion_dirz = np.load("test_pion_dirz.npy")
electron_p = np.load("test_electron_p.npy")
muon_p = np.load("test_muon_p.npy")
pion_p = np.load("test_pion_p.npy")

electron_all_info = np.column_stack(
    [electron_x, electron_y, electron_z, electron_dirx, electron_diry, electron_dirz, electron_p]
)
electron_unique_info = [ tuple(row) for row in np.unique(electron_all_info, axis=0) ]
muon_all_info = np.column_stack(
    [muon_x, muon_y, muon_z, muon_dirx, muon_diry, muon_dirz, muon_p]
)
muon_unique_info = [ tuple(row) for row in np.unique(muon_all_info, axis=0) ]
pion_all_info = np.column_stack(
    [pion_x, pion_y, pion_z, pion_dirx, pion_diry, pion_dirz, pion_p]
)
pion_unique_info = [ tuple(row) for row in np.unique(pion_all_info, axis=0) ]

electron_fnames_info = {
    fname : tuple(electron_all_info[i])
    for i, fname in enumerate(os.listdir(os.path.join(SEGCUBE_FULL_DATASET, "electron")))
}
muon_fnames_info = {
    fname : tuple(muon_all_info[i])
    for i, fname in enumerate(os.listdir(os.path.join(SEGCUBE_FULL_DATASET, "muon")))
}
pion_fnames_info = {
    fname : tuple(pion_all_info[i])
    for i, fname in enumerate(os.listdir(os.path.join(SEGCUBE_FULL_DATASET, "pion")))
}

denom = OUT_DATASET_TRAIN_SIZE + OUT_DATASET_VAL_SIZE + OUT_DATASET_TEST_SIZE
train_frac = OUT_DATASET_TRAIN_SIZE / denom
val_frac = OUT_DATASET_VAL_SIZE / denom

electron_train_infos = set(electron_unique_info[:int(train_frac * len(electron_unique_info))])
electron_val_infos = set(electron_unique_info[len(electron_train_infos):len(electron_train_infos) + int(val_frac * len(electron_unique_info))])
electron_test_infos = set(electron_unique_info[len(electron_train_infos) + len(electron_val_infos):])
fnames = list(electron_fnames_info)
random.shuffle(fnames)
train_cntr, val_cntr, test_cntr = 0, 0, 0
for fname in tqdm(fnames):
    info = electron_fnames_info[fname]
    if train_cntr < OUT_DATASET_TRAIN_SIZE and info in electron_train_infos:
        shutil.copy(
            os.path.join(SEGCUBE_FULL_DATASET, "electron", fname),
            os.path.join(SEGCUBE_OUT_DATASET, "train", "electron", fname)
        )
        train_cntr += 1
    elif val_cntr < OUT_DATASET_VAL_SIZE and info in electron_val_infos:
        shutil.copy(
            os.path.join(SEGCUBE_FULL_DATASET, "electron", fname),
            os.path.join(SEGCUBE_OUT_DATASET, "val", "electron", fname)
        )
        val_cntr += 1
    elif test_cntr < OUT_DATASET_TEST_SIZE and info in electron_test_infos:
        shutil.copy(
            os.path.join(SEGCUBE_FULL_DATASET, "electron", fname),
            os.path.join(SEGCUBE_OUT_DATASET, "test", "electron", fname)
        )
        test_cntr += 1

muon_train_infos = set(muon_unique_info[:int(train_frac * len(muon_unique_info))])
muon_val_infos = set(muon_unique_info[len(muon_train_infos):len(muon_train_infos) + int(val_frac * len(muon_unique_info))])
muon_test_infos = set(muon_unique_info[len(muon_train_infos) + len(muon_val_infos):])
fnames = list(muon_fnames_info)
random.shuffle(fnames)
train_cntr, val_cntr, test_cntr = 0, 0, 0
for fname in tqdm(fnames):
    info = muon_fnames_info[fname]
    if train_cntr < OUT_DATASET_TRAIN_SIZE and info in muon_train_infos:
        shutil.copy(
            os.path.join(SEGCUBE_FULL_DATASET, "muon", fname),
            os.path.join(SEGCUBE_OUT_DATASET, "train", "muon", fname)
        )
        train_cntr += 1
    elif val_cntr < OUT_DATASET_VAL_SIZE and info in muon_val_infos:
        shutil.copy(
            os.path.join(SEGCUBE_FULL_DATASET, "muon", fname),
            os.path.join(SEGCUBE_OUT_DATASET, "val", "muon", fname)
        )
        val_cntr += 1
    elif test_cntr < OUT_DATASET_TEST_SIZE and info in muon_test_infos:
        shutil.copy(
            os.path.join(SEGCUBE_FULL_DATASET, "muon", fname),
            os.path.join(SEGCUBE_OUT_DATASET, "test", "muon", fname)
        )
        test_cntr += 1

pion_train_infos = set(pion_unique_info[:int(train_frac * len(pion_unique_info))])
pion_val_infos = set(pion_unique_info[len(pion_train_infos):len(pion_train_infos) + int(val_frac * len(pion_unique_info))])
pion_test_infos = set(pion_unique_info[len(pion_train_infos) + len(pion_val_infos):])
fnames = list(pion_fnames_info)
random.shuffle(fnames)
train_cntr, val_cntr, test_cntr = 0, 0, 0
for fname in tqdm(fnames):
    info = pion_fnames_info[fname]
    if train_cntr < OUT_DATASET_TRAIN_SIZE and info in pion_train_infos:
        shutil.copy(
            os.path.join(SEGCUBE_FULL_DATASET, "pion", fname),
            os.path.join(SEGCUBE_OUT_DATASET, "train", "pion", fname)
        )
        train_cntr += 1
    elif val_cntr < OUT_DATASET_VAL_SIZE and info in pion_val_infos:
        shutil.copy(
            os.path.join(SEGCUBE_FULL_DATASET, "pion", fname),
            os.path.join(SEGCUBE_OUT_DATASET, "val", "pion", fname)
        )
        val_cntr += 1
    elif test_cntr < OUT_DATASET_TEST_SIZE and info in pion_test_infos:
        shutil.copy(
            os.path.join(SEGCUBE_FULL_DATASET, "pion", fname),
            os.path.join(SEGCUBE_OUT_DATASET, "test", "pion", fname)
        )
        test_cntr += 1

# The proton already has properly randomised initial conditions... somehow
fnames = [ fname for fname in os.listdir(os.path.join(SEGCUBE_FULL_DATASET, "proton")) ]
random.shuffle(fnames)
train_cntr, val_cntr, test_cntr = 0, 0, 0
for fname in tqdm(fnames):
    if train_cntr < OUT_DATASET_TRAIN_SIZE:
        shutil.copy(
            os.path.join(SEGCUBE_FULL_DATASET, "proton", fname),
            os.path.join(SEGCUBE_OUT_DATASET, "train", "proton", fname)
        )
        train_cntr += 1
    elif val_cntr < OUT_DATASET_VAL_SIZE:
        shutil.copy(
            os.path.join(SEGCUBE_FULL_DATASET, "proton", fname),
            os.path.join(SEGCUBE_OUT_DATASET, "val", "proton", fname)
        )
        val_cntr += 1
    elif test_cntr < OUT_DATASET_TEST_SIZE:
        shutil.copy(
            os.path.join(SEGCUBE_FULL_DATASET, "proton", fname),
            os.path.join(SEGCUBE_OUT_DATASET, "test", "proton", fname)
        )
        test_cntr += 1
