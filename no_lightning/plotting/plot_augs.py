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
IDX=2
# X_MIN_MAX=(-80, 20)
# Y_MIN_MAX=(-100, 300)
# Z_MIN_MAX=(1700, 1800)
X_MIN_MAX=(700, 850)
Y_MIN_MAX=(400, 700)
Z_MIN_MAX=(1400, 1600)

def main():
    conf = get_config(TEST_CONF)
    torch.manual_seed(2)

    dataset = ThrowsDataset(
        os.path.join(conf.data_path, "val"), conf.data_prep_type, conf.augs, conf.n_augs
    )

    path, _ = dataset.samples[IDX]
    print(path)
    sample = np.load(path)
    coords, feats = sample["coordinates"], sample["adc"]

    norm_adc = matplotlib.colors.Normalize(vmin=min(feats)*0.9, vmax=max(feats)*1.1)
    m_adc = matplotlib.cm.ScalarMappable(norm=norm_adc, cmap=matplotlib.cm.jet)

    funcs = dataset.augs
    for i, func in enumerate(funcs):
        print(func)
        coords_aug, feats_aug = func(torch.tensor(coords, dtype=torch.float), torch.tensor(feats, dtype=torch.float))
        coords_aug, feats_aug = sparse_quantize(
            coords_aug, np.expand_dims(feats_aug, axis=1), quantization_size=dataset.quantization_size
        )

        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(projection="3d")

        for coord, feat in zip(coords_aug, feats_aug):
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
        plt.savefig(f"augmentation_{i}.pdf")
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
