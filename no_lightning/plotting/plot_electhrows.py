import sys, os
sys.path.append("/home/awilkins/contrastive-neutrino/no_lightning")

import numpy as np
import matplotlib; from matplotlib import pyplot as plt

import torch
from MinkowskiEngine.utils import sparse_quantize

from config_parser import get_config
from dataset import ThrowsDataset, DataPrepType

matplotlib.use("pdf") # remove if using plt.show()
TEST_CONF="/home/awilkins/contrastive-neutrino/no_lightning/experiments/clr/exp_test.yaml"
NOM_PATH="/share/rcifdata/awilkins/contrastive-neutrino_data/datasets/nominal/train/electron/electron_edeps_out_0_larndsim_nominal_eventID_10.npz"
ELEC3_PATH="/share/rcifdata/awilkins/contrastive-neutrino_data/datasets/electronics_throw3/test/electron/electron_edeps_out_0_larndsim_electronics_throw3_eventID_10.npz"
ELEC4_PATH="/share/rcifdata/awilkins/contrastive-neutrino_data/datasets/electronics_throw4/val/electron/electron_edeps_out_0_larndsim_electronics_throw4_eventID_10.npz"
X_MIN_MAX=(-175, -25)
Y_MIN_MAX=(-100, -40)
Z_MIN_MAX=(1750, 1900)

def main():
    conf = get_config(TEST_CONF)
    torch.manual_seed(2)

    dataset = ThrowsDataset(
        os.path.join(conf.data_path, "val"),
        conf.data_prep_type,
        conf.augs, conf.n_augs,
        conf.quantisation_size
    )

    sample_nom = np.load(NOM_PATH)
    feats_nom = sample_nom["adc"]
    norm_adc = matplotlib.colors.Normalize(vmin=min(feats_nom)*0.5, vmax=max(feats_nom)*1.5)
    m_adc = matplotlib.cm.ScalarMappable(norm=norm_adc, cmap=matplotlib.cm.viridis)

    for i, fpath in enumerate([NOM_PATH, ELEC3_PATH, ELEC4_PATH]):
        sample = np.load(fpath)
        coords, feats = sample["coordinates"], sample["adc"]
        coords, feats = sparse_quantize(
            torch.tensor(coords, dtype=torch.float),
            np.expand_dims(torch.tensor(feats, dtype=torch.float), axis=1),
            quantization_size=dataset.quantization_size
        )

        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(projection="3d")

        for coord, feat in zip(coords, feats):
            x, y, z = get_cube()
            x = x + coord[0].item()
            y = y + coord[1].item()
            z = z + coord[2].item()
            ax.plot_surface(x, y, z, color=m_adc.to_rgba(feat.item()))

        ax.set_xlim(X_MIN_MAX[0], X_MIN_MAX[1])
        ax.set_ylim(Y_MIN_MAX[0], Y_MIN_MAX[1])
        ax.set_zlim(Z_MIN_MAX[0], Z_MIN_MAX[1])
        ax.grid(True)
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.axes.zaxis.set_ticklabels([])

        fig.tight_layout()
        plt.savefig(f"throw_{i}.pdf")
        plt.close()
        # plt.show()

def get_cube():
    """Get coords for plotting cuboid surface with Axes3D.plot_surface"""
    phi = np.arange(1, 10, 2) * np.pi / 4
    Phi, Theta = np.meshgrid(phi, phi)

    x = np.cos(Phi) * np.sin(Theta)
    y = np.sin(Phi) * np.sin(Theta)
    z = np.cos(Theta) / np.sqrt(2)

    return x,y,z

if __name__ == "__main__":
    main()
