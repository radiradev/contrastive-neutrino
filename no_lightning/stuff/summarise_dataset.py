import os

import torch
import numpy as np
from tqdm import tqdm

TRAIN_ELECTRON_DIR = "/share/rcifdata/awilkins/contrastive-neutrino_data/datasets/segmentedcube_truly_randomised/train/electron"
TRAIN_MUON_DIR = "/share/rcifdata/awilkins/contrastive-neutrino_data/datasets/segmentedcube_truly_randomised/train/muon"
TRAIN_PION_DIR = "/share/rcifdata/awilkins/contrastive-neutrino_data/datasets/segmentedcube_truly_randomised/train/pion"
TRAIN_PROTON_DIR = "/share/rcifdata/awilkins/contrastive-neutrino_data/datasets/segmentedcube_truly_randomised/train/proton"
TEST_ELECTRON_DIR = "/share/rcifdata/awilkins/contrastive-neutrino_data/datasets/segmentedcube_truly_randomised/test/electron"
TEST_MUON_DIR = "/share/rcifdata/awilkins/contrastive-neutrino_data/datasets/segmentedcube_truly_randomised/test/muon"
TEST_PION_DIR = "/share/rcifdata/awilkins/contrastive-neutrino_data/datasets/segmentedcube_truly_randomised/test/pion"
TEST_PROTON_DIR = "/share/rcifdata/awilkins/contrastive-neutrino_data/datasets/segmentedcube_truly_randomised/test/proton"
# TRAIN_ELECTRON_DIR = "/share/rcifdata/awilkins/contrastive-neutrino_data/datasets/segmentedcube_full/train/electron"
# TRAIN_MUON_DIR = "/share/rcifdata/awilkins/contrastive-neutrino_data/datasets/segmentedcube_full/train/muon"
# TRAIN_PION_DIR = "/share/rcifdata/awilkins/contrastive-neutrino_data/datasets/segmentedcube_full/train/pion"
# TRAIN_PROTON_DIR = "/share/rcifdata/awilkins/contrastive-neutrino_data/datasets/segmentedcube_full/train/proton"
# TEST_ELECTRON_DIR = "/share/rcifdata/awilkins/contrastive-neutrino_data/datasets/segmentedcube_full/test/electron"
# TEST_MUON_DIR = "/share/rcifdata/awilkins/contrastive-neutrino_data/datasets/segmentedcube_full/test/muon"
# TEST_PION_DIR = "/share/rcifdata/awilkins/contrastive-neutrino_data/datasets/segmentedcube_full/test/pion"
# TEST_PROTON_DIR = "/share/rcifdata/awilkins/contrastive-neutrino_data/datasets/segmentedcube_full/test/proton"

def get_data(
    path,
    ixs_outpath,
    iys_outpath,
    izs_outpath,
    idirxs_outpath,
    idirys_outpath,
    idirzs_outpath,
    ips_outpath
):
    ixs = []
    iys = []
    izs = []
    idirxs = []
    idirys = []
    idirzs = []
    ips = []
    for fname in tqdm(os.listdir(path)):
        data = torch.load(os.path.join(path, fname))
        ixs.append(data["TrueIniPos"][0])
        iys.append(data["TrueIniPos"][1])
        izs.append(data["TrueIniPos"][2])
        idirxs.append(data["TrueIniDir"][0])
        idirys.append(data["TrueIniDir"][1])
        idirzs.append(data["TrueIniDir"][2])
        ips.append(data["TrueIniP"])
    np.save(ixs_outpath, np.array(ixs))
    np.save(iys_outpath, np.array(iys))
    np.save(izs_outpath, np.array(izs))
    np.save(idirxs_outpath, np.array(idirxs))
    np.save(idirys_outpath, np.array(idirys))
    np.save(idirzs_outpath, np.array(idirzs))
    np.save(ips_outpath, np.array(ips))

print("train electron...")
get_data(
    TRAIN_ELECTRON_DIR,
    "train_electron_x.npy",
    "train_electron_y.npy",
    "train_electron_z.npy",
    "train_electron_dirx.npy",
    "train_electron_diry.npy",
    "train_electron_dirz.npy",
    "train_electron_p.npy"
)
print("train muon...")
get_data(
    TRAIN_MUON_DIR,
    "train_muon_x.npy",
    "train_muon_y.npy",
    "train_muon_z.npy",
    "train_muon_dirx.npy",
    "train_muon_diry.npy",
    "train_muon_dirz.npy",
    "train_muon_p.npy"
)
print("train pion...")
get_data(
    TRAIN_PION_DIR,
    "train_pion_x.npy",
    "train_pion_y.npy",
    "train_pion_z.npy",
    "train_pion_dirx.npy",
    "train_pion_diry.npy",
    "train_pion_dirz.npy",
    "train_pion_p.npy"
)
print("train proton...")
get_data(
    TRAIN_PROTON_DIR,
    "train_proton_x.npy",
    "train_proton_y.npy",
    "train_proton_z.npy",
    "train_proton_dirx.npy",
    "train_proton_diry.npy",
    "train_proton_dirz.npy",
    "train_proton_p.npy"
)
print("test electron...")
get_data(
    TEST_ELECTRON_DIR,
    "test_electron_x.npy",
    "test_electron_y.npy",
    "test_electron_z.npy",
    "test_electron_dirx.npy",
    "test_electron_diry.npy",
    "test_electron_dirz.npy",
    "test_electron_p.npy"
)
print("test muon...")
get_data(
    TEST_MUON_DIR,
    "test_muon_x.npy",
    "test_muon_y.npy",
    "test_muon_z.npy",
    "test_muon_dirx.npy",
    "test_muon_diry.npy",
    "test_muon_dirz.npy",
    "test_muon_p.npy"
)
print("test pion...")
get_data(
    TEST_PION_DIR,
    "test_pion_x.npy",
    "test_pion_y.npy",
    "test_pion_z.npy",
    "test_pion_dirx.npy",
    "test_pion_diry.npy",
    "test_pion_dirz.npy",
    "test_pion_p.npy"
)
print("test proton...")
get_data(
    TEST_PROTON_DIR,
    "test_proton_x.npy",
    "test_proton_y.npy",
    "test_proton_z.npy",
    "test_proton_dirx.npy",
    "test_proton_diry.npy",
    "test_proton_dirz.npy",
    "test_proton_p.npy"
)
print("done.")
