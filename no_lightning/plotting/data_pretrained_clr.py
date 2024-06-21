import os

import numpy as np
from tqdm import tqdm
import yaml
import matplotlib; from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score

matplotlib.use("pdf") # remove if using plt.show()
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00'])

CHECKPOINT_DIR = "/home/awilkins/contrastive-neutrino/no_lightning/checkpoints"

CLASSIFIER = "classifier/classifier_nominal_final_tunedaugs"
NOTRAIN_CLR = "clr/clr_notrain"
CLRS = {
    "nominal" : "clr/clr_nominal_final_tunedaugs",
    "electhrow1" : "clr/clr_electhrow1_final_tunedaugs",
    "electhrow3" : "clr/clr_electhrow3_final_tunedaugs",
    "electhrow4" : "clr/clr_electhrow4_final_tunedaugs",
    "electhrow5" : "clr/clr_electhrow5_final_tunedaugs",
    "electhrow6" : "clr/clr_electhrow6_final_tunedaugs",
    "electhrow7" : "clr/clr_electhrow7_final_tunedaugs"
}
DANNS = {
    "nominal" : "dann/dann_nominal_final",
    "electhrow1" : "dann/dann_electhrow1_final",
    "electhrow3" : "dann/dann_electhrow3_final",
    "electhrow4" : "dann/dann_electhrow4_final",
    "electhrow5" : "dann/dann_electhrow5_final",
    "electhrow6" : "dann/dann_electhrow6_final",
    "electhrow7" : "dann/dann_electhrow7_final"
}

def main():
    datasets = list(CLRS)
    nominal_accs, clr_accs, notrain_clr_accs, dann_accs = [], [], [], []
    for dataset in tqdm(datasets):
        # nominal_accs.append(0.8)
        # clr_accs.append(0.8)
        # notrain_clr_accs.append(0.8)
        # dann_accs.append(0.8)
        clf_preds_path = os.path.join(
            CHECKPOINT_DIR, CLASSIFIER, "test_results", f"preds_{dataset}.yml"
        )
        nominal_accs.append(get_acc(clf_preds_path))
        clr_preds_path = os.path.join(
            CHECKPOINT_DIR, CLRS[dataset], "test_results", f"preds_{dataset}.yml"
        )
        clr_accs.append(get_acc(clr_preds_path))
        notrain_clr_preds_path = os.path.join(
            CHECKPOINT_DIR, NOTRAIN_CLR, "test_results", f"preds_{dataset}.yml"
        )
        notrain_clr_accs.append(get_acc(notrain_clr_preds_path))
        dann_preds_path = os.path.join(
            CHECKPOINT_DIR, DANNS[dataset], "test_results", f"preds_{dataset}.yml"
        )
        dann_accs.append(get_acc(dann_preds_path))

    x = np.arange(len(datasets))
    width = 0.15
    spacing = 0.0
    fig, ax = plt.subplots(figsize=(10, 7))
    rects1 = ax.bar(x - (width + spacing) * 1.5, nominal_accs, width, label="nominal classifier")
    rects2 = ax.bar(x - (width + spacing) * 0.5, clr_accs, width, label="pretrained clr")
    rects3 = ax.bar(x + (width + spacing) * 0.5, dann_accs, width, label="DANN")
    rects4 = ax.bar(x + (width + spacing) * 1.5, notrain_clr_accs, width, label="random representation")
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Pretrained CLR performance for different 'data' realisations", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 1)
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    ax.legend(loc="upper right", ncols=4)
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)
    autolabel(rects4, ax)

    fig.tight_layout()
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
