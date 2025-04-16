import os

import numpy as np
from tqdm import tqdm
import yaml
import matplotlib; from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score

matplotlib.use("pdf") # remove if using plt.show()
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00'])

CHECKPOINT_DIR = "/home/awilkins/contrastive-neutrino/no_lightning/checkpoints"

CLASSIFIER_XTALK50_MODELNET = (
    "classifier/classifier_segmentedcube_nominal_xtalk50_modelnet_final_moreepoch"
)
CLASSIFIER_XTALK50_CONVNEXT = (
    "classifier/classifier_segmentedcube_nominal_xtalk50_finalfinal"
)
CLASSIFIER_XTALK50_CONVNEXTSMALL = (
    "classifier/classifier_segmentedcube_nominal_xtalk50_convnextsmall_final"
)
CLR_XTALK50_MODELNET = (
    "clr/clr_segmentedcube_nominal_xtalk50_labels_modelnet_final"
)
DATASETS = [
    "xtalk0",
    "xtalk05",
    "xtalk10",
    "xtalk15",
    "xtalk20",
    "xtalk25",
    "xtalk30",
    "xtalk35",
    "xtalk40",
    "xtalk45",
    "xtalk50",
    "xtalk55",
    "xtalk60",
    "xtalk65",
    "xtalk70",
    "xtalk75",
    "xtalk80",
    "xtalk85",
    "xtalk90",
    "xtalk95",
    "xtalk100"
]

def main():
    clf_accs_xt50_modelnet, clf_accs_xt50_convnext, clf_accs_xt50_convnextsmall = [], [], []
    clr_accs_xt50_modelnet = []
    for dataset in tqdm(DATASETS):
        clf_preds_path_xt50_modelnet = os.path.join(
            CHECKPOINT_DIR, CLASSIFIER_XTALK50_MODELNET, "test_results", f"preds_{dataset}.yml"
        )
        clf_accs_xt50_modelnet.append(get_acc(clf_preds_path_xt50_modelnet))
        clr_preds_path_xt50_modelnet = os.path.join(
            CHECKPOINT_DIR, CLR_XTALK50_MODELNET, "test_results", f"preds_{dataset}.yml"
        )
        clr_accs_xt50_modelnet.append(get_acc(clr_preds_path_xt50_modelnet))
        clf_preds_path_xt50_convnext = os.path.join(
            CHECKPOINT_DIR, CLASSIFIER_XTALK50_CONVNEXT, "test_results", f"preds_{dataset}.yml"
        )
        clf_accs_xt50_convnext.append(get_acc(clf_preds_path_xt50_convnext))
        clf_preds_path_xt50_convnextsmall = os.path.join(
            CHECKPOINT_DIR, CLASSIFIER_XTALK50_CONVNEXTSMALL, "test_results", f"preds_{dataset}.yml"
        )
        clf_accs_xt50_convnextsmall.append(get_acc(clf_preds_path_xt50_convnextsmall))

    x = np.linspace(0.0, 1.0, 21)

    # xtalk 50
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.axvline(0.5, c="r", label="_")
    ax.text(0.525, 0.7, "Nominal", rotation=90, c="r", fontsize=13)
    ax.plot(x, clf_accs_xt50_convnext, marker="o", label="ConvNeXT")
    ax.plot(x, clf_accs_xt50_convnextsmall, marker="o", label="ConvNeXT Small")
    ax.plot(x, clf_accs_xt50_modelnet, marker="o", label="ModelNet40")
    ax.set_xlabel("Crosstalk Fraction", fontsize=13)
    ax.set_ylabel("Acc.", fontsize=13)
    ax.grid(axis="both")
    ax.set_ylim(0, 1)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", fontsize=12)
    plt.savefig("segcube_line_chart_xt50_modelnet_convnext.pdf")
    plt.close()

    # xtalk 50 zoomed
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.axvline(0.5, c="r", label="_")
    ax.text(0.525, 0.7, "Nominal", rotation=90, c="r", fontsize=13)
    ax.plot(x, clf_accs_xt50_convnext, marker="o", label="ConvNeXT")
    ax.plot(x, clf_accs_xt50_convnextsmall, marker="o", label="ConvNeXT Small")
    ax.plot(x, clf_accs_xt50_modelnet, marker="o", label="ModelNet40")
    ax.set_xlabel("Crosstalk Fraction", fontsize=13)
    ax.set_ylabel("Acc.", fontsize=13)
    ax.grid(axis="both")
    ax.set_ylim(0.8, 1)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", fontsize=12)
    plt.savefig("segcube_line_chart_xt50_modelnet_convnext_zoomed.pdf")
    plt.close()

    # xtalk 50 clr comparison zoomed
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.axvline(0.5, c="r", label="_")
    ax.text(0.525, 0.7, "Nominal", rotation=90, c="r", fontsize=13)
    ax.plot(x, clf_accs_xt50_modelnet, marker="o", label="ModelNet40 Classifier")
    ax.plot(x, clr_accs_xt50_modelnet, marker="o", label="ModelNet40 CLR")
    ax.set_xlabel("Crosstalk Fraction", fontsize=13)
    ax.set_ylabel("Acc.", fontsize=13)
    ax.grid(axis="both")
    ax.set_ylim(0.8, 1)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", fontsize=12)
    plt.savefig("segcube_line_chart_xt50_modelnet_clr_zoomed.pdf")
    plt.close()

    # xtalk 50 log (too dumb to get this to work, want "reverse" log axis)
    # fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    # ax.axvline(0.5, c="r", label="_")
    # ax.text(0.525, 0.7, "Nominal", rotation=90, c="r", fontsize=13)
    # ax.plot(x, 1 - np.array(clf_accs_xt50_convnext), marker="o", label="ConvNeXT")
    # ax.plot(x, 1 - np.array(clf_accs_xt50_convnextsmall), marker="o", label="ConvNeXT Small")
    # ax.plot(x, 1 - np.array(clf_accs_xt50_modelnet), marker="o", label="ModelNet40")
    # ax.set_xlabel("Crosstalk Fraction", fontsize=13)
    # ax.set_ylabel("Acc.", fontsize=13)
    # ax.grid(axis="both")
    # ax.set_ylim(0.0, 0.9)
    # ax.set_axisbelow(True)
    # ax.legend(loc="lower center", fontsize=12)
    # ax.set_yscale("log")
    # ax.invert_yaxis()
    # ax.set_yticklabels(1 - ax.get_yticks())
    # plt.savefig("segcube_line_chart_xt50_modelnet_convnext_logy.pdf")
    # plt.close()

def get_acc(preds_path):
    with open(preds_path, "r") as f:
        pred_data = yaml.load(f, Loader=yaml.FullLoader)
    pred_label, true_label = [], []
    for data in pred_data.values():
        pred_label.append(np.argmax(data[0]))
        true_label.append(data[1])

    return accuracy_score(pred_label, true_label)

if __name__ == "__main__":
    main()
