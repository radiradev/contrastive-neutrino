import argparse

import numpy as np
import matplotlib; from matplotlib import pyplot as plt

import MinkowskiEngine as ME

QUANTIZATION_SIZE = 0.38

def main(args):
    data = np.load(args.input_path)
    coords = data["coordinates"]
    feats = data["adc"]

    coords, feats = ME.utils.sparse_quantize(
        coords, np.expand_dims(feats, axis=1), quantization_size=QUANTIZATION_SIZE
    )
    coords = coords.numpy()

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    norm_feats = matplotlib.colors.Normalize(
        vmin=min(feat[0] for feat in feats), vmax=max(feat[0] for feat in feats)
    )
    m_feats = matplotlib.cm.ScalarMappable(norm=norm_feats, cmap=matplotlib.cm.viridis)

    for coords, feat in zip(coords, feats):
        feat = int(feat[0])

        x, y, z = get_cube()
        x = x + coords[0]
        y = y + coords[1]
        z = z + coords[2]

        c = m_feats.to_rgba(feat)

        ax.plot_surface(x, z, y, color=c, shade=False)

    ax.set_box_aspect((4,8,4))
    ax.grid(False)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.show()

def get_cube():
    """Get coords for plotting cuboid surface with Axes3D.plot_surface"""
    phi = np.arange(1, 10, 2) * np.pi / 4
    Phi, Theta = np.meshgrid(phi, phi)

    x = np.cos(Phi) * np.sin(Theta)
    y = np.sin(Phi) * np.sin(Theta)
    z = np.cos(Theta) / np.sqrt(2)

    return x,y,z

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_path", type=str)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main(parse_arguments())
