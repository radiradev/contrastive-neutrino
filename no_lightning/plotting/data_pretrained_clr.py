import os

import numpy as np
from tqdm import tqdm
import yaml
import matplotlib # remove if using plt.show()
matplotlib.use("pdf") # remove if using plt.show()
from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score

CHECKPOINT_DIR = "/home/awilkins/contrastive-neutrino/no_lightning/checkpoints"

CLASSIFIER = "classifier/classifier_nominal_aug_batch672"
NOTRAIN_CLR = "clr/clr_notrain"
CLRS = {
    "nominal" : "clr/clr_nominal_batch672",
    "electhrow1" : "clr/clr_electhrow1_batch672",
    "electhrow3" : "clr/clr_electhrow3_batch672",
    "electhrow4" : "clr/clr_electhrow4_batch672"
}
DANNS = {
    "nominal" : "dann/dann_nominal",
    "electhrow1" : "dann/dann_electhrow1",
    "electhrow3" : "dann/dann_electhrow3",
    "electhrow4" : "dann/dann_electhrow4"
}

def main():
    datasets = list(CLRS)
    nominal_accs, clr_accs, notrain_clr_accs, dann_accs = [], [], [], []
    for dataset in tqdm(datasets):
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
    width = 0.2
    spacing = 0.0
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - (width + spacing) * 1.5, nominal_accs, width, label="nominal classifier")
    rects2 = ax.bar(x - (width + spacing) * 0.5, clr_accs, width, label="pretrained clr")
    rects3 = ax.bar(x + (width + spacing) * 0.5, notrain_clr_accs, width, label="DANN")
    rects4 = ax.bar(x + (width + spacing) * 1.5, notrain_clr_accs, width, label="random clr")
    ax.set_ylabel("Accuracy")
    ax.set_title("Pretrained CLR performance for different 'data' realisations")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 1)
    ax.set_axisbelow(True)
    ax.grid(axis='y')
    ax.legend()
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
            ha='center',
            va='bottom',
            fontsize=6
        )

if __name__ == "__main__":
    main()
