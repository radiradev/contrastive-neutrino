import os, argparse

import numpy as np
from tqdm import tqdm
import yaml
import matplotlib; from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

matplotlib.use("pdf") # remove if using plt.show()
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00'])

LABELS = { 0 : "electron", 1 : "gamma", 2 : "muon", 3 : "pion", 4 : "proton" }

def main(args):
    if args.labels == 0:
        labels = { 0 : "electron", 1 : "gamma", 2 : "muon", 3 : "pion", 4 : "proton" }
    elif args.labels == 1:
        labels = { 0 : "electron", 1 : "muon", 2 : "pion", 3 : "proton" }
    else:
        raise ValueError(f"--labels={args.labels} not valid")

    with open(args.preds, "r") as f:
        pred_data = yaml.load(f, Loader=yaml.FullLoader)
    pred_label, true_label = [], []
    for data in pred_data.values():
        pred_label.append(np.argmax(data[0]))
        true_label.append(data[1])

    C = confusion_matrix(true_label, pred_label)

    # group_counts = [ "{0:0.0f}".format(value) for value in C.flatten() ]
    group_percentages = [ "{0:.2%}".format(value) for value in (C / np.sum(C, axis=1)).flatten() ]
    element_labels = group_percentages
    element_labels = np.asarray(element_labels).reshape(len(labels), len(labels))
    sns.set(font_scale=1.4)
    fig, ax = plt.subplots(figsize=(8,7))
    sns.heatmap(C, annot=element_labels, annot_kws={'size': 16}, fmt='', ax=ax, cmap='Blues')
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title(args.title)
    ax.xaxis.set_ticklabels([ label for label in labels.values() ])
    ax.yaxis.set_ticklabels([ label for label in labels.values() ])
    fig.tight_layout()
    plt.savefig("confusion_matrix.pdf")
    plt.close()

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("preds", type=str)
    parser.add_argument("title", type=str)
    parser.add_argument("labels", type=int, help="0|1")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main(parse_arguments())
