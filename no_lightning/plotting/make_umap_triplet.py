import argparse

import numpy as np
from tqdm import tqdm
import matplotlib; from matplotlib import pyplot as plt
import umap

matplotlib.use("pdf") # remove if using plt.show()
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00'])

LABELS = { 0 : "electron", 1 : "gamma", 2 : "muon", 3 : "pion", 4 : "proton" }

def main(args):
    embedding_clf, labels_clf = get_umap_embedding(args.preds_clf)
    embedding_clr, labels_clr = get_umap_embedding(args.preds_clr)
    embedding_dann, labels_dann = get_umap_embedding(args.preds_dann)

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    for label_num, name in LABELS.items():
        embedding_label_clf = embedding_clf[(labels_clf == label_num)]
        ax[0].scatter(embedding_label_clf[:, 0], embedding_label_clf[:, 1], label=name, s=1)
        embedding_label_clr = embedding_clr[(labels_clr == label_num)]
        ax[1].scatter(embedding_label_clr[:, 0], embedding_label_clr[:, 1], label=name, s=1)
        embedding_label_dann = embedding_dann[(labels_dann == label_num)]
        ax[2].scatter(embedding_label_dann[:, 0], embedding_label_dann[:, 1], label=name, s=1)

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    ax[0].set_title("Classifier", fontsize=12)
    ax[1].set_title("CLR", fontsize=12)
    ax[2].set_title("DANN", fontsize=12)
    plt.legend(loc="upper right", fontsize=10, markerscale=3)
    fig.suptitle(args.title, fontsize=14)
    fig.tight_layout()
    plt.savefig("umap_triplet.pdf")
    plt.close()
    # plt.show()

def get_umap_embedding(preds_path):
    data = np.load(preds_path)
    labels, feats = data[:, -1], data[:, :-1]

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(feats)

    return embedding, labels

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("preds_clf")
    parser.add_argument("preds_clr")
    parser.add_argument("preds_dann")
    parser.add_argument("title")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main(parse_arguments())
