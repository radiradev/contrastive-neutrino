import argparse

import numpy as np
from tqdm import tqdm
import matplotlib # remove if using plt.show()
matplotlib.use("pdf") # remove if using plt.show()
from matplotlib import pyplot as plt
import umap

LABELS = { 0 : "electron", 1 : "gamma", 2 : "muon", 3 : "pion", 4 : "proton" }

def main(args):
    data = np.load(args.preds)
    labels, feats = data[:, -1], data[:, :-1]

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(feats)

    fig, ax = plt.subplots()
    for label_num, name in LABELS.items():
        embedding_label = embedding[(labels == label_num)]
        ax.scatter(embedding_label[:, 0], embedding_label[:, 1], label=name, s=1)

    ax.set_title(args.title)
    plt.legend()
    fig.tight_layout()
    plt.savefig("umap.pdf")
    plt.close()
    # plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("preds")
    parser.add_argument("title")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main(parse_arguments())
