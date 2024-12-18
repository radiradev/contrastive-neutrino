import os

import numpy as np
from tqdm import tqdm
import yaml
import matplotlib; from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score

matplotlib.use("pdf") # remove if using plt.show()
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00'])

CHECKPOINT_DIR = "/home/awilkins/contrastive-neutrino/no_lightning/checkpoints"

# CLASSIFIER_NOAUG = "classifier/classifier_nominal_final_standardaugs"
CLASSIFIER = "classifier/classifier_nominal_final_standardaugs_30epoch"
CLRS_UNSUPERVISED = {
    "nominal" : "clr/clr_nominal_final_standardaugs",
    "electhrow1" : "clr/clr_electhrow1_final_standardaugs",
    "electhrow3" : "clr/clr_electhrow3_final_standardaugs",
    "electhrow4" : "clr/clr_electhrow4_final_standardaugs",
    "electhrow5" : "clr/clr_electhrow5_final_standardaugs",
    "electhrow6" : "clr/clr_electhrow6_final_standardaugs",
    "electhrow7" : "clr/clr_electhrow7_final_standardaugs"
}
CLRS_SUPERVISED = {
    "nominal" : "clr/clr_nominal_labels_weight1-0_300epoch",
    "electhrow1" : "clr/clr_nominal_labels_weight1-0_300epoch",
    "electhrow3" : "clr/clr_nominal_labels_weight1-0_300epoch",
    "electhrow4" : "clr/clr_nominal_labels_weight1-0_300epoch",
    "electhrow5" : "clr/clr_nominal_labels_weight1-0_300epoch",
    "electhrow6" : "clr/clr_nominal_labels_weight1-0_300epoch",
    "electhrow7" : "clr/clr_nominal_labels_weight1-0_300epoch"
}
DANNS = {
    "nominal" : "dann/dann_nominal_final",
    "electhrow1" : "dann/dann_electhrow1_final_again",
    "electhrow3" : "dann/dann_electhrow3_final",
    "electhrow4" : "dann/dann_electhrow4_final_again",
    "electhrow5" : "dann/dann_electhrow5_final_again",
    "electhrow6" : "dann/dann_electhrow6_final",
    "electhrow7" : "dann/dann_electhrow7_final"
}
TICKLABELS = ["Nominal", "Throw1", "Throw2", "Throw3", "Throw4", "Throw5", "Throw6"]

def main():
    datasets = list(CLRS_SUPERVISED)
    clf_accs, clr_unsup_accs, clr_sup_accs, dann_accs = [], [], [], []
    for dataset in tqdm(datasets):
        clf_preds_path = os.path.join(
            CHECKPOINT_DIR, CLASSIFIER, "test_results", f"preds_{dataset}.yml"
        )
        clf_accs.append(get_acc(clf_preds_path))
        clr_unsup_preds_path = os.path.join(
            CHECKPOINT_DIR, CLRS_UNSUPERVISED[dataset], "test_results", f"preds_{dataset}.yml"
        )
        clr_unsup_accs.append(get_acc(clr_unsup_preds_path))
        clr_sup_preds_path = os.path.join(
            CHECKPOINT_DIR, CLRS_SUPERVISED[dataset], "test_results", f"preds_{dataset}.yml"
        )
        clr_sup_accs.append(get_acc(clr_sup_preds_path))
        dann_preds_path = os.path.join(
            CHECKPOINT_DIR, DANNS[dataset], "test_results", f"preds_{dataset}.yml"
        )
        dann_accs.append(get_acc(dann_preds_path))

    clf_accs_rel = [ (acc - clf_accs[0]) for acc in clf_accs ]
    clr_unsup_accs_rel = [ (acc - clr_unsup_accs[0]) for acc in clr_unsup_accs ]
    clr_sup_accs_rel = [ (acc - clr_sup_accs[0]) for acc in clr_sup_accs ]
    dann_accs_rel = [ (acc - dann_accs[0]) for acc in dann_accs ]

    x = np.arange(len(datasets))
    width = 0.20
    spacing = 0.0
    fig, ax = plt.subplots(nrows=2, sharex=True, height_ratios=[1, 3], figsize=(12, 7))
    rects1 = ax[1].bar(
        x - (width + spacing) * 1.5, clf_accs, width, label="Classifier w/ augs"
    )
    rects2 = ax[1].bar(
        x - (width + spacing) * 0.5, clr_unsup_accs, width, label="Unsupervised Contrastive"
    )
    rects3 = ax[1].bar(
        x + (width + spacing) * 0.5, clr_sup_accs, width, label="Supervised Contrastive"
    )
    rects4 = ax[1].bar(
        x + (width + spacing) * 1.5, dann_accs, width, label="DANN"
    )
    ax[0].bar(
        x - (width + spacing) * 1.5, clf_accs_rel, width, label="Classifier w/ augs"
    )
    ax[0].bar(
        x - (width + spacing) * 0.5, clr_unsup_accs_rel, width, label="Unsupervised Contrastive"
    )
    ax[0].bar(
        x + (width * spacing) * 0.5, clr_sup_accs_rel, width, label="Supervised Contrastive"
    )
    ax[0].bar(
        x + (width + spacing) * 1.5, dann_accs_rel, width, label="DANN"
    )
    ax[0].set_xticklabels([])
    ax[0].set_ylabel("Acc. - Nominal Acc.", fontsize=13)
    ax[1].set_ylabel("Acc.", fontsize=13)
    ax[0].set_title(
        "Pretrained Contrastive Model Performance for Different 'data' Realisations", fontsize=14
    )
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(TICKLABELS, fontsize=13)
    ax[1].set_ylim(0, 1)
    ax[0].set_ylim(-0.39, 0.2)
    ax[1].set_axisbelow(True)
    ax[1].grid(axis="y")
    ax[0].set_axisbelow(True)
    ax[0].grid(axis="y")
    ax[0].legend(loc="upper center", ncols=4, fontsize=12)
    autolabel(rects1, ax[1])
    autolabel(rects2, ax[1])
    autolabel(rects3, ax[1])
    autolabel(rects4, ax[1])

    fig.subplots_adjust(hspace=0)
    # fig.tight_layout()
    plt.savefig("pretrained_clr_acc.pdf")
    plt.close()
    # plt.show()

def get_acc(preds_path):
    with open(preds_path, "r") as f:
        pred_data = yaml.load(f, Loader=yaml.FullLoader)
    pred_label, true_label = [], []
    for data in pred_data.values():
        pred_label.append(np.argmax(data[0]))
        true_label.append(data[1])

    return accuracy_score(pred_label, true_label)

def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
            weight="bold"
        )

if __name__ == "__main__":
    main()
