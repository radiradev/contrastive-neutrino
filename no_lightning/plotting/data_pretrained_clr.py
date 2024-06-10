import argparse, os

import numpy as np
from tqdm import tqdm
import yaml
from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score

CHECKPOINT_DIR = "/home/awilkins/contrastive-neutrino/no_lightning/checkpoints"

CLASSIFIER = "classifier/classifier_nominal_batch672"
CLRS = {
    "nominal" : "clr/clr_nominal_batch672",
    # "electhrow1" : "clr/clr_electhrow1_batch672",
    "electhrow3" : "clr/clr_electhrow3_batch672"
}

def main():
    models = [ "nominal classifier", "pretrained_clr" ]
    datasets = list(CLRS)
    nominal_accs, clr_accs = [], []
    for dataset in tqdm(datasets):
        clf_preds_path = os.path.join(
            CHECKPOINT_DIR, CLASSIFIER, "test_results", f"preds_{dataset}.yml"
        )
        nominal_accs.append(get_acc(clf_preds_path))
        clr_preds_path = os.path.join(
            CHECKPOINT_DIR, CLRS[dataset], "test_results", f"preds_{dataset}.yml"
        )
        clr_accs.append(get_acc(clr_preds_path))

    x = np.arange(len(datasets))
    width = 0.25
    mul = 0.0
    fig, ax = plt.subplots()
    for model_name, accs in zip(models, [nominal_accs, clr_accs]):
        offset = width * mul
        mul += 1
        rects = ax.bar(x + offset, accs, width, label=model_name)
        ax.bar_label(rects, padding=3)
    ax.set_ylabel("Accuracy")
    ax.set_title("Pretrained CLR performance for different 'data' realisations")
    ax.set_xticks(x + width, datasets)
    ax.legend(loc='upper left', ncols=2)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    plt.savefig("pretrained_clr_acc.pdf")
    # plt.show()

def get_acc(preds_path):
    with open(preds_path, "r") as f:
        pred_data = yaml.load(f, Loader=yaml.FullLoader)
    pred_label, true_label = [], []
    for data in pred_data.values():
        pred_label.append(np.argmax(data[0]))
        true_label.append(data[1])

    return accuracy_score(pred_label, true_label)

# def parse_arguments():
#     parser = argparse.ArgumentParser()

#     args = parser.parse_args()

#     return args

if __name__ == "__main__":
    # args = parse_arguments()
    # main(args)
    main()
