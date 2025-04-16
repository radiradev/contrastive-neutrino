import os

import numpy as np
from tqdm import tqdm
import yaml
import matplotlib; from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score

matplotlib.use("pdf") # remove if using plt.show()
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00'])

CHECKPOINT_DIR = "/home/awilkins/contrastive-neutrino/no_lightning/checkpoints"

# -- Classifier paths
CLF_XTALK0 =   "classifier/classifier_segmentedcube_nominal_xtalk0_truerandomdatafull_final"
CLF_XTALK25 =  "classifier/classifier_segmentedcube_nominal_xtalk25_truerandomdatafull_final"
CLF_XTALK50 =  "classifier/classifier_segmentedcube_nominal_xtalk50_truerandomdatafull_final"
CLF_XTALK75 =  "classifier/classifier_segmentedcube_nominal_xtalk75_truerandomdatafull_final"
CLF_XTALK100 = "classifier/classifier_segmentedcube_nominal_xtalk100_truerandomdatafull_final"

# -- CLR paths
CLR_XTALK0 =   "clr/clr_segmentedcube_nominal_xtalk0_labels_truerandomdatafull_final"
CLR_XTALK25 =  "clr/clr_segmentedcube_nominal_xtalk25_labels_truerandomdatafull_final"
CLR_XTALK50 =  "clr/clr_segmentedcube_nominal_xtalk50_labels_truerandomdatafull_final"
CLR_XTALK75 =  "clr/clr_segmentedcube_nominal_xtalk75_labels_truerandomdatafull_final"
CLR_XTALK100 = "clr/clr_segmentedcube_nominal_xtalk100_labels_truerandomdatafull_final"

# -- CLR nolabel paths
CLR_NOLABEL_XTALK0 =   "clr/clr_segmentedcube_xtalk0_nolabels_truerandomdatafull"
CLR_NOLABEL_XTALK10 =  "clr/clr_segmentedcube_xtalk10_nolabels_truerandomdatafull"
CLR_NOLABEL_XTALK20 =  "clr/clr_segmentedcube_xtalk20_nolabels_truerandomdatafull"
CLR_NOLABEL_XTALK25 =  "clr/clr_segmentedcube_xtalk25_nolabels_truerandomdatafull"
CLR_NOLABEL_XTALK30 =  "clr/clr_segmentedcube_xtalk30_nolabels_truerandomdatafull"
CLR_NOLABEL_XTALK40 =  "clr/clr_segmentedcube_xtalk40_nolabels_truerandomdatafull"
CLR_NOLABEL_XTALK50 =  "clr/clr_segmentedcube_xtalk50_nolabels_truerandomdatafull"
CLR_NOLABEL_XTALK60 =  "clr/clr_segmentedcube_xtalk60_nolabels_truerandomdatafull"
CLR_NOLABEL_XTALK70 =  "clr/clr_segmentedcube_xtalk70_nolabels_truerandomdatafull"
CLR_NOLABEL_XTALK75 =  "clr/clr_segmentedcube_xtalk75_nolabels_truerandomdatafull"
CLR_NOLABEL_XTALK80 =  "clr/clr_segmentedcube_xtalk80_nolabels_truerandomdatafull"
CLR_NOLABEL_XTALK90 =  "clr/clr_segmentedcube_xtalk90_nolabels_truerandomdatafull"
CLR_NOLABEL_XTALK100 = "clr/clr_segmentedcube_xtalk100_nolabels_truerandomdatafull"

LABEL_DATASETS = [
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

NOLABEL_DATASETS = [
    "xtalk0",
    "xtalk10",
    "xtalk20",
    "xtalk25",
    "xtalk30",
    "xtalk40",
    "xtalk50",
    "xtalk60",
    "xtalk70",
    "xtalk75",
    "xtalk80",
    "xtalk90",
    "xtalk100"
]

def main():
    accs_clf_xt0, accs_clf_xt25, accs_clf_xt50, accs_clf_xt75, accs_clf_xt100 = [], [], [], [], []
    accs_clr_xt0, accs_clr_xt25, accs_clr_xt50, accs_clr_xt75, accs_clr_xt100 = [], [], [], [], []
    accs_clr_nolabel_xt0, accs_clr_nolabel_xt25, accs_clr_nolabel_xt50, accs_clr_nolabel_xt75, accs_clr_nolabel_xt100 = [], [], [], [], []
    accs_dann_xt0, accs_dann_xt25, accs_dann_xt50, accs_dann_xt75, accs_dann_xt100 = [], [], [], [], []
    for dataset in LABEL_DATASETS:
        accs_clf_xt0.append(get_acc(CHECKPOINT_DIR, CLF_XTALK0, dataset))
        accs_clf_xt25.append(get_acc(CHECKPOINT_DIR, CLF_XTALK25, dataset))
        accs_clf_xt50.append(get_acc(CHECKPOINT_DIR, CLF_XTALK50, dataset))
        accs_clf_xt75.append(get_acc(CHECKPOINT_DIR, CLF_XTALK75, dataset))
        accs_clf_xt100.append(get_acc(CHECKPOINT_DIR, CLF_XTALK100, dataset))

        accs_clr_xt0.append(get_acc(CHECKPOINT_DIR, CLR_XTALK0, dataset))
        accs_clr_xt25.append(get_acc(CHECKPOINT_DIR, CLR_XTALK25, dataset))
        accs_clr_xt50.append(get_acc(CHECKPOINT_DIR, CLR_XTALK50, dataset))
        accs_clr_xt75.append(get_acc(CHECKPOINT_DIR, CLR_XTALK75, dataset))
        accs_clr_xt100.append(get_acc(CHECKPOINT_DIR, CLR_XTALK100, dataset))

    for dataset in NOLABEL_DATASETS:
        accs_clr_nolabel_xt0.append(get_nolabel_acc(CHECKPOINT_DIR, 0, dataset))
        accs_clr_nolabel_xt25.append(get_nolabel_acc(CHECKPOINT_DIR, 25, dataset))
        accs_clr_nolabel_xt50.append(get_nolabel_acc(CHECKPOINT_DIR, 50, dataset))
        accs_clr_nolabel_xt75.append(get_nolabel_acc(CHECKPOINT_DIR, 75, dataset))
        accs_clr_nolabel_xt100.append(get_nolabel_acc(CHECKPOINT_DIR, 100, dataset))

        accs_dann_xt0.append(get_dann_acc(CHECKPOINT_DIR, 0, dataset))
        accs_dann_xt25.append(get_dann_acc(CHECKPOINT_DIR, 25, dataset))
        accs_dann_xt50.append(get_dann_acc(CHECKPOINT_DIR, 50, dataset))
        accs_dann_xt75.append(get_dann_acc(CHECKPOINT_DIR, 75, dataset))
        accs_dann_xt100.append(get_dann_acc(CHECKPOINT_DIR, 100, dataset))

    np.save("accs_clr_unsup_xt0.npy", accs_clr_nolabel_xt0)
    np.save("accs_clr_unsup_xt25.npy", accs_clr_nolabel_xt25)
    np.save("accs_clr_unsup_xt50.npy", accs_clr_nolabel_xt50)
    np.save("accs_clr_unsup_xt75.npy", accs_clr_nolabel_xt75)
    np.save("accs_clr_unsup_xt100.npy", accs_clr_nolabel_xt100)
    np.save("accs_dann_xt0.npy", accs_dann_xt0)
    np.save("accs_dann_xt25.npy", accs_dann_xt25)
    np.save("accs_dann_xt50.npy", accs_dann_xt50)
    np.save("accs_dann_xt75.npy", accs_dann_xt75)
    np.save("accs_dann_xt100.npy", accs_dann_xt100)

    # -- xtalk 0
    make_plot(
        [
            accs_clr_xt0,
            accs_clf_xt0,
            accs_clr_nolabel_xt0,
            accs_dann_xt0
        ],
        [
            "fine",
            "fine",
            "coarse",
            "coarse"
        ],
        [
            0,
            0,
            0,
            0
        ],
        [
            "Supervised Contrastive",
            "Classifier w/ augs",
            "Unsupervised Contrastive",
            "DANN"
        ],
        0,
        savename="segcube_line_chart_xt0_unsupervised.pdf",
        ylim=(0.3, 0.7),
        colors=["C0", "C1", "C2", "C3"],
        linestyles=["solid", "solid", "solid", "solid"],
        markers=["o", "o", "o", "o"]
    )

    # -- xtalk 25
    make_plot(
        [
            accs_clr_xt25,
            accs_clf_xt25,
            accs_clr_nolabel_xt25,
            accs_dann_xt25
        ],
        [
            "fine",
            "fine",
            "coarse",
            "coarse"
        ],
        [
            5,
            5,
            3,
            3
        ],
        [
            "Supervised Contrastive",
            "Classifier w/ augs",
            "Unsupervised Contrastive",
            "DANN"
        ],
        25,
        savename="segcube_line_chart_xt25_unsupervised.pdf",
        ylim=(0.40, 0.7),
        colors=["C0", "C1", "C2", "C3"],
        linestyles=["solid", "solid", "solid", "solid"],
        markers=["o", "o", "o", "o"]
    )

    # -- xtalk 50
    make_plot(
        [
            accs_clr_xt50,
            accs_clf_xt50,
            accs_clr_nolabel_xt50,
            accs_dann_xt50
        ],
        [
            "fine",
            "fine",
            "coarse",
            "coarse"
        ],
        [
            10,
            10,
            6,
            6
        ],
        [
            "Supervised Contrastive",
            "Classifier w/ augs",
            "Unsupervised Contrastive",
            "DANN"
        ],
        50,
        savename="segcube_line_chart_xt50_unsupervised.pdf",
        ylim=(0.40, 0.7),
        colors=["C0", "C1", "C2", "C3"],
        linestyles=["solid", "solid", "solid", "solid"],
        markers=["o", "o", "o", "o"]
    )
    
    # -- xtalk 75
    make_plot(
        [
            accs_clr_xt75,
            accs_clf_xt75,
            accs_clr_nolabel_xt75,
            accs_dann_xt75
        ],
        [
            "fine",
            "fine",
            "coarse",
            "coarse"
        ],
        [
            15,
            15,
            9,
            9
        ],
        [
            "Supervised Contrastive",
            "Classifier w/ augs",
            "Unsupervised Contrastive",
            "DANN"
        ],
        75,
        savename="segcube_line_chart_xt75_unsupervised.pdf",
        ylim=(0.40, 0.7),
        colors=["C0", "C1", "C2", "C3"],
        linestyles=["solid", "solid", "solid", "solid"],
        markers=["o", "o", "o", "o"]
    )

    # -- xtalk 100
    make_plot(
        [
            accs_clr_xt100,
            accs_clf_xt100,
            accs_clr_nolabel_xt100,
            accs_dann_xt100
        ],
        [
            "fine",
            "fine",
            "coarse",
            "coarse"
        ],
        [
            20,
            20,
            12,
            12
        ],
        [
            "Supervised Contrastive",
            "Classifier w/ augs",
            "Unsupervised Contrastive",
            "DANN"
        ],
        100,
        savename="segcube_line_chart_xt100_unsupervised.pdf",
        ylim=(0.40, 0.7),
        colors=["C0", "C1", "C2", "C3"],
        linestyles=["solid", "solid", "solid", "solid"],
        markers=["o", "o", "o", "o"]
    )

def get_acc(chktpt_dir, exp_rel_path, dataset):
    acc_path = os.path.join(chktpt_dir, exp_rel_path, "test_results", f"preds_{dataset}_acc.yml")
    with open(acc_path, "r") as f:
        acc = yaml.load(f, Loader=yaml.FullLoader)
        acc = acc["acc"]
    return acc

def get_nolabel_acc(chktpt_dir, nom_xtalk, throw_xtalk):
    if throw_xtalk == "xtalk0":
        exp_rel_path = CLR_NOLABEL_XTALK0
    elif throw_xtalk == "xtalk10":
        exp_rel_path = CLR_NOLABEL_XTALK10
    elif throw_xtalk == "xtalk20":
        exp_rel_path = CLR_NOLABEL_XTALK20
    elif throw_xtalk == "xtalk25":
        exp_rel_path = CLR_NOLABEL_XTALK25
    elif throw_xtalk == "xtalk30":
        exp_rel_path = CLR_NOLABEL_XTALK30
    elif throw_xtalk == "xtalk40":
        exp_rel_path = CLR_NOLABEL_XTALK40
    elif throw_xtalk == "xtalk50":
        exp_rel_path = CLR_NOLABEL_XTALK50
    elif throw_xtalk == "xtalk60":
        exp_rel_path = CLR_NOLABEL_XTALK60
    elif throw_xtalk == "xtalk70":
        exp_rel_path = CLR_NOLABEL_XTALK70
    elif throw_xtalk == "xtalk75":
        exp_rel_path = CLR_NOLABEL_XTALK75
    elif throw_xtalk == "xtalk80":
        exp_rel_path = CLR_NOLABEL_XTALK80
    elif throw_xtalk == "xtalk90":
        exp_rel_path = CLR_NOLABEL_XTALK90
    elif throw_xtalk == "xtalk100":
        exp_rel_path = CLR_NOLABEL_XTALK100
    else:
        raise NotImplementedError

    if nom_xtalk == 0:
        preds_name = f"preds_nomxtalk0_{throw_xtalk}_acc.yml"
    elif nom_xtalk == 25:
        preds_name = f"preds_nomxtalk25_{throw_xtalk}_acc.yml"
    elif nom_xtalk == 50:
        preds_name = f"preds_nomxtalk50_{throw_xtalk}_acc.yml"
    elif nom_xtalk == 75:
        preds_name = f"preds_nomxtalk75_{throw_xtalk}_acc.yml"
    elif nom_xtalk == 100:
        preds_name = f"preds_nomxtalk100_{throw_xtalk}_acc.yml"
    else:
        raise NotImplementedError

    acc_path = os.path.join(chktpt_dir, exp_rel_path, "test_results", preds_name)
    with open(acc_path, "r") as f:
        acc = yaml.load(f, Loader=yaml.FullLoader)
        acc = acc["acc"]

    return acc

def get_dann_acc(chktpt_dir, nom_xtalk, throw_xtalk):
    exp_rel_path = f"dann/dann_segmentedcube_target_{throw_xtalk}_source_xtalk{nom_xtalk}_truerandomdatafull"
    preds_name = f"preds_nomxtalk{nom_xtalk}_{throw_xtalk}_acc.yml"

    acc_path = os.path.join(chktpt_dir, exp_rel_path, "test_results", preds_name)
    with open(acc_path, "r") as f:
        acc = yaml.load(f, Loader=yaml.FullLoader)
        acc = acc["acc"]

    return acc

def make_plot(
    datas, resolutions, nom_xtalk_idxs, labels, xtalk,
    savename=None, ylim=(0,1), colors=None, linestyles=None, markers=None
):
    fig, ax = plt.subplots(
        nrows=2, sharex=True, figsize=(12, 7), layout="compressed", height_ratios=[1, 3]
    )

    residuals = [ [ (acc - data[i]) / data[i] for acc in data ] for data, i in zip(datas, nom_xtalk_idxs) ]
    if xtalk == 0:
        ax[1].axvline(0.0, c="r", label="_")
        ax[1].text(0.025, 0.55, "Nominal", rotation=90, c="r", fontsize=13)
        ax[0].axvline(0.0, c="r", label="_")
    elif xtalk == 25:
        ax[1].axvline(0.25, c="r", label="_")
        ax[1].text(0.275, 0.55, "Nominal", rotation=90, c="r", fontsize=13)
        ax[0].axvline(0.25, c="r", label="_")
    elif xtalk == 50:
        ax[1].axvline(0.5, c="r", label="_")
        ax[1].text(0.525, 0.55, "Nominal", rotation=90, c="r", fontsize=13)
        ax[0].axvline(0.5, c="r", label="_")
    elif xtalk == 75:
        ax[1].axvline(0.75, c="r", label="_")
        ax[1].text(0.775, 0.55, "Nominal", rotation=90, c="r", fontsize=13)
        ax[0].axvline(0.75, c="r", label="_")
    elif xtalk == 100:
        ax[1].axvline(1.0, c="r", label="_")
        ax[1].text(0.975, 0.55, "Nominal", rotation=90, c="r", fontsize=13)
        ax[0].axvline(1.0, c="r", label="_")
    else:
        raise ValueError(f"xtalk={xtalk} not valid")

    colors = [ "C" + str(i) for i in range(len(datas)) ] if colors is None else colors
    linestyles = [ "solid" for _ in range(len(datas)) ] if linestyles is None else linestyles
    markers = [ "o" for _ in range(len(datas)) ] if markers is None else markers

    for data, resolution, residual, label, color, linestyle, marker in zip(
        datas, resolutions, residuals, labels, colors, linestyles, markers
    ):
        if resolution == "fine":
            x = np.linspace(0.0, 1.0, 21)
        elif resolution == "coarse":
            x = np.array([0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0])
        else:
            raise NotImplementedError

        ax[1].plot(x, data, marker=marker, label=label, c=color, linestyle=linestyle)
        ax[0].plot(x, residual, marker=marker, label=label, c=color, linestyle=linestyle)

    ax[0].set_ylabel("Frac. Resid.", fontsize=13)
    ax[0].grid(axis="both")
    ax[1].legend(loc="lower center", fontsize=12, ncols=2)
    ax[1].set_ylabel("Acc.", fontsize=13)
    ax[1].set_xlabel("Crosstalk Fraction", fontsize=13)
    ax[1].grid(axis="both")
    ax[1].set_ylim(ylim[0], ylim[1])
    if savename is not None:
        plt.savefig(savename)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    main()
