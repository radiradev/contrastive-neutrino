import os

import numpy as np
from tqdm import tqdm
import yaml
import matplotlib; from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score

matplotlib.use("pdf") # remove if using plt.show()
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00'])

CHECKPOINT_DIR = "/home/awilkins/contrastive-neutrino/no_lightning/checkpoints"

CLASSIFIER = "classifier/classifier_nominal_final_standardaugs"
NOTRAIN_CLR = "clr/clr_notrain"
CLRS = {
    "nominal" : "clr/clr_nominal_final_standardaugs",
    "electhrow1" : "clr/clr_electhrow1_final_standardaugs",
    "electhrow3" : "clr/clr_electhrow3_final_standardaugs",
    "electhrow4" : "clr/clr_electhrow4_final_standardaugs",
    "electhrow5" : "clr/clr_electhrow5_final_standardaugs",
    "electhrow6" : "clr/clr_electhrow6_final_standardaugs",
    "electhrow7" : "clr/clr_electhrow7_final_standardaugs"
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
# DANNS = {
#     "nominal" : "clr/clr_nominal_labels",
#     "electhrow1" : "clr/clr_nominal_labels",
#     "electhrow3" : "clr/clr_nominal_labels",
#     "electhrow4" : "clr/clr_nominal_labels",
#     "electhrow5" : "clr/clr_nominal_labels",
#     "electhrow6" : "clr/clr_nominal_labels",
#     "electhrow7" : "clr/clr_nominal_labels"
# }
TICKLABELS = ["Nominal", "Throw1", "Throw2", "Throw3", "Throw4", "Throw5", "Throw6"]

def main():
    datasets = list(CLRS)
    nominal_accs, clr_accs, notrain_clr_accs, dann_accs = [], [], [], []
    for dataset in tqdm(datasets):
        # nominal_accs.append(np.random.rand() * 0.85)
        # clr_accs.append(np.random.rand() * 0.85)
        # notrain_clr_accs.append(np.random.rand() * 0.85)
        # dann_accs.append(np.random.rand() * 0.85)
        clf_preds_path = os.path.join(
            CHECKPOINT_DIR, CLASSIFIER, "test_results", f"preds_{dataset}.yml"
        )
        nominal_accs.append(get_acc(clf_preds_path))
        clr_preds_path = os.path.join(
            CHECKPOINT_DIR, CLRS[dataset], "test_results", f"preds_{dataset}.yml"
        )
        clr_accs.append(get_acc(clr_preds_path))
        # notrain_clr_preds_path = os.path.join(
        #     CHECKPOINT_DIR, NOTRAIN_CLR, "test_results", f"preds_{dataset}.yml"
        # )
        # notrain_clr_accs.append(get_acc(notrain_clr_preds_path))
        dann_preds_path = os.path.join(
            CHECKPOINT_DIR, DANNS[dataset], "test_results", f"preds_{dataset}.yml"
        )
        dann_accs.append(get_acc(dann_preds_path))

    nominal_accs_rel = [ (acc - nominal_accs[0]) for acc in nominal_accs ]
    clr_accs_rel = [ (acc - clr_accs[0]) for acc in clr_accs ]
    dann_accs_rel = [ (acc - dann_accs[0]) for acc in dann_accs ]

    x = np.arange(len(datasets))
    width = 0.20
    spacing = 0.0
    fig, ax = plt.subplots(nrows=2, sharex=True, height_ratios=[1, 3], figsize=(12, 7))
    rects1 = ax[1].bar(
        x - (width + spacing), nominal_accs, width, label="Nominal Classifier"
    )
    rects2 = ax[1].bar(
        x, clr_accs, width, label="Contrastive Pretraining"
    )
    rects3 = ax[1].bar(
        x + (width + spacing), dann_accs, width, label="DANN"
    )
    # rects4 = ax[1].bar(x + (width + spacing) * 1.5, notrain_clr_accs, width, label="Random Representation")
    ax[0].bar(
        x - (width + spacing), nominal_accs_rel, width, label="Nominal Classifier"
    )
    ax[0].bar(
        x, clr_accs_rel, width, label="Contrastive Pretraining"
    )
    ax[0].bar(
        x + (width + spacing), dann_accs_rel, width, label="DANN"
    )
    # ax[0].bar(
    #     x + (width + spacing) * 1.5, [0] * len(dann_accs_rel), width, label="Random Representation"
    # )
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
    # autolabel(rects4, ax[1])

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
