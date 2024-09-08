import os

import numpy as np
from tqdm import tqdm
import yaml
import matplotlib; from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score

matplotlib.use("pdf") # remove if using plt.show()
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00'])

CHECKPOINT_DIR = "/home/awilkins/contrastive-neutrino/no_lightning/checkpoints"

CLASSIFIER_XTALK0 = "classifier/classifier_segmentedcube_nominal_xtalk0_finalfinal"
CLASSIFIER_XTALK25 = "classifier/classifier_segmentedcube_nominal_xtalk25_finalfinal"
CLASSIFIER_XTALK50 = "classifier/classifier_segmentedcube_nominal_xtalk50_finalfinal"
CLASSIFIER_XTALK75 = "classifier/classifier_segmentedcube_nominal_xtalk75_finalfinal"
CLASSIFIER_XTALK100 = "classifier/classifier_segmentedcube_nominal_xtalk100_finalfinal"
CLR_XTALK0 = "clr/clr_segmentedcube_nominal_xtalk0_labels_final"
CLR_XTALK25 = "clr/clr_segmentedcube_nominal_xtalk25_labels_final"
CLR_XTALK50 = "clr/clr_segmentedcube_nominal_xtalk50_labels_final"
CLR_XTALK70 = "clr/clr_segmentedcube_nominal_xtalk75_labels_final"
CLR_XTALK100 = "clr/clr_segmentedcube_nominal_xtalk100_labels_final"
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
# TICKLABELS = ["Nominal", "Throw1", "Throw2", "Throw3", "Throw4", "Throw5", "Throw6"]

def main():
    clr_accs_xt0, clr_accs_xt25, clr_accs_xt50, clr_accs_xt75, clr_accs_xt100 = [], [], [], [], []
    clf_accs_xt0, clf_accs_xt25, clf_accs_xt50, clr_accs_xt75, clf_accs_xt100 = [], [], [], [], []
    for dataset in tqdm(DATASETS):
        clf_preds_path_xt0 = os.path.join(
            CHECKPOINT_DIR, CLASSIFIER_XTALK0, "test_results", f"preds_{dataset}.yml"
        )
        clf_accs_xt0.append(get_acc(clf_preds_path_xt0))
        # clf_preds_path_xt25 = os.path.join(
        #     CHECKPOINT_DIR, CLASSIFIER_XTALK25, "test_results", f"preds_{dataset}.yml"
        # )
        clf_accs_xt25.append(get_acc(clf_preds_path_xt25))
        clf_preds_path_xt50 = os.path.join(
            CHECKPOINT_DIR, CLASSIFIER_XTALK50, "test_results", f"preds_{dataset}.yml"
        )
        clf_accs_xt50.append(get_acc(clf_preds_path_xt50))
        # clf_preds_path_xt75 = os.path.join(
        #     CHECKPOINT_DIR, CLASSIFIER_XTALK75, "test_results", f"preds_{dataset}.yml"
        # )
        # clf_accs_xt75.append(get_acc(clf_preds_path_xt75))
        clf_preds_path_xt100 = os.path.join(
            CHECKPOINT_DIR, CLASSIFIER_XTALK100, "test_results", f"preds_{dataset}.yml"
        )
        clf_accs_xt100.append(get_acc(clf_preds_path_xt100))

        clr_preds_path_xt0 = os.path.join(
            CHECKPOINT_DIR, CLR_XTALK0, "test_results", f"preds_{dataset}.yml"
        )
        clr_accs_xt0.append(get_acc(clr_preds_path_xt0))
        clr_preds_path_xt25 = os.path.join(
            CHECKPOINT_DIR, CLR_XTALK25, "test_results", f"preds_{dataset}.yml"
        )
        clr_accs_xt25.append(get_acc(clr_preds_path_xt25))
        clr_preds_path_xt50 = os.path.join(
            CHECKPOINT_DIR, CLR_XTALK50, "test_results", f"preds_{dataset}.yml"
        )
        clr_accs_xt50.append(get_acc(clr_preds_path_xt50))
        clr_preds_path_xt75 = os.path.join(
            CHECKPOINT_DIR, CLR_XTALK75, "test_results", f"preds_{dataset}.yml"
        )
        clr_accs_xt75.append(get_acc(clr_preds_path_xt75))
        clr_preds_path_xt100 = os.path.join(
            CHECKPOINT_DIR, CLR_XTALK100, "test_results", f"preds_{dataset}.yml"
        )
        clr_accs_xt100.append(get_acc(clr_preds_path_xt100))

    x = np.linspace(0.0, 1.0, 21)

    # xtalk 0
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.axvline(0.0, c="r", label="_")
    ax.text(0.025, 0.7, "Nominal", rotation=90, c="r", fontsize=13)
    ax.plot(x, clr_accs_xt0, marker="o", label="Contrastive Pretraining")
    ax.plot(x, clf_accs_xt0, marker="o", label="Classifier")
    ax.set_xticks(x)
    ax.set_xlabel("Crosstalk Fraction", fontsize=13)
    ax.set_ylabel("Acc.", fontsize=13)
    ax.grid(axis="both")
    ax.set_ylim(0, 1)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", fontsize=12)
    plt.savefig("segcube_line_chart_xt0.pdf")
    plt.close()

    # xtalk 25
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.axvline(0.25, c="r", label="_")
    ax.text(0.275, 0.7, "Nominal", rotation=90, c="r", fontsize=13)
    ax.plot(x, clr_accs_xt25, marker="o", label="Contrastive Pretraining")
    # ax.plot(x, clf_accs_xt25, marker="o", label="Classifier")
    ax.set_xticks(x)
    ax.set_xlabel("Crosstalk Fraction", fontsize=13)
    ax.set_ylabel("Acc.", fontsize=13)
    ax.grid(axis="both")
    ax.set_ylim(0, 1)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", fontsize=12)
    plt.savefig("segcube_line_chart_xt25.pdf")
    plt.close()

    # xtalk 50
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.axvline(0.5, c="r", label="_")
    ax.text(0.525, 0.7, "Nominal", rotation=90, c="r", fontsize=13)
    ax.plot(x, clr_accs_xt50, marker="o", label="Contrastive Pretraining")
    ax.plot(x, clf_accs_xt50, marker="o", label="Classifier")
    ax.set_xticks(x)
    ax.set_xlabel("Crosstalk Fraction", fontsize=13)
    ax.set_ylabel("Acc.", fontsize=13)
    ax.grid(axis="both")
    ax.set_ylim(0, 1)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", fontsize=12)
    plt.savefig("segcube_line_chart_xt50.pdf")
    plt.close()

    # xtalk 75
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.axvline(0.75, c="r", label="_")
    ax.text(0.775, 0.7, "Nominal", rotation=90, c="r", fontsize=13)
    ax.plot(x, clr_accs_xt75, marker="o", label="Contrastive Pretraining")
    # ax.plot(x, clf_accs_xt75, marker="o", label="Classifier")
    ax.set_xticks(x)
    ax.set_xlabel("Crosstalk Fraction", fontsize=13)
    ax.set_ylabel("Acc.", fontsize=13)
    ax.grid(axis="both")
    ax.set_ylim(0, 1)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", fontsize=12)
    plt.savefig("segcube_line_chart_xt75.pdf")
    plt.close()

    # xtalk 100
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.axvline(1.0 c="r", label="_")
    ax.text(0.975, 0.7, "Nominal", rotation=90, c="r", fontsize=13)
    ax.plot(x, clr_accs_xt100, marker="o", label="Contrastive Pretraining")
    ax.plot(x, clf_accs_xt100, marker="o", label="Classifier")
    ax.set_xticks(x)
    ax.set_xlabel("Crosstalk Fraction", fontsize=13)
    ax.set_ylabel("Acc.", fontsize=13)
    ax.grid(axis="both")
    ax.set_ylim(0, 1)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", fontsize=12)
    plt.savefig("segcube_line_chart_xt100.pdf")
    plt.close()

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
