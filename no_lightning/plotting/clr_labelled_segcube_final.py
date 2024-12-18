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

CLF_NOAUGS_XTALK0 =   "classifier/classifier_segmentedcube_nominal_xtalk0_truerandomdatafull_noaugs_final"
CLF_NOAUGS_XTALK25 =  "classifier/classifier_segmentedcube_nominal_xtalk25_truerandomdatafull_noaugs_final"
CLF_NOAUGS_XTALK50 =  "classifier/classifier_segmentedcube_nominal_xtalk50_truerandomdatafull_noaugs_final"
CLF_NOAUGS_XTALK75 =  "classifier/classifier_segmentedcube_nominal_xtalk75_truerandomdatafull_noaugs_final"
CLF_NOAUGS_XTALK100 = "classifier/classifier_segmentedcube_nominal_xtalk100_truerandomdatafull_noaugs_final"

CLF_MNET_XTALK0 =   "classifier/classifier_segmentedcube_nominal_xtalk0_modelnet_truerandomdatafull_final"
CLF_MNET_XTALK25 =  "classifier/classifier_segmentedcube_nominal_xtalk25_modelnet_truerandomdatafull_final"
CLF_MNET_XTALK50 =  "classifier/classifier_segmentedcube_nominal_xtalk50_modelnet_truerandomdatafull_final"
CLF_MNET_XTALK75 =  "classifier/classifier_segmentedcube_nominal_xtalk75_modelnet_truerandomdatafull_final"
CLF_MNET_XTALK100 = "classifier/classifier_segmentedcube_nominal_xtalk100_modelnet_truerandomdatafull_final"

CLF_MNET_NOAUGS_XTALK0 =   "classifier/classifier_segmentedcube_nominal_xtalk0_modelnet_truerandomdatafull_noaugs_final"
CLF_MNET_NOAUGS_XTALK25 =  "classifier/classifier_segmentedcube_nominal_xtalk25_modelnet_truerandomdatafull_noaugs_final"
CLF_MNET_NOAUGS_XTALK50 =  "classifier/classifier_segmentedcube_nominal_xtalk50_modelnet_truerandomdatafull_noaugs_final"
CLF_MNET_NOAUGS_XTALK75 =  "classifier/classifier_segmentedcube_nominal_xtalk75_modelnet_truerandomdatafull_noaugs_final"
CLF_MNET_NOAUGS_XTALK100 = "classifier/classifier_segmentedcube_nominal_xtalk100_modelnet_truerandomdatafull_noaugs_final"

# -- CLR paths
CLR_XTALK0 =   "clr/clr_segmentedcube_nominal_xtalk0_labels_truerandomdatafull_final"
CLR_XTALK25 =  "clr/clr_segmentedcube_nominal_xtalk25_labels_truerandomdatafull_final"
CLR_XTALK50 =  "clr/clr_segmentedcube_nominal_xtalk50_labels_truerandomdatafull_final"
CLR_XTALK75 =  "clr/clr_segmentedcube_nominal_xtalk75_labels_truerandomdatafull_final"
CLR_XTALK100 = "clr/clr_segmentedcube_nominal_xtalk100_labels_truerandomdatafull_final"

CLR_MNET_XTALK0 =   "clr/clr_segmentedcube_nominal_xtalk0_labels_modelnet_truerandomdatafull_final"
CLR_MNET_XTALK25 =  "clr/clr_segmentedcube_nominal_xtalk25_labels_modelnet_truerandomdatafull_final"
CLR_MNET_XTALK50 =  "clr/clr_segmentedcube_nominal_xtalk50_labels_modelnet_truerandomdatafull_final"
CLR_MNET_XTALK75 =  "clr/clr_segmentedcube_nominal_xtalk75_labels_modelnet_truerandomdatafull_final"
CLR_MNET_XTALK100 = "clr/clr_segmentedcube_nominal_xtalk100_labels_modelnet_truerandomdatafull_final"

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
    accs_clf_xt0, accs_clf_xt25, accs_clf_xt50, accs_clf_xt75, accs_clf_xt100 = [], [], [], [], []
    accs_clf_noaugs_xt0, accs_clf_noaugs_xt25, accs_clf_noaugs_xt50, accs_clf_noaugs_xt75, accs_clf_noaugs_xt100 = [], [], [], [], []
    accs_clf_mnet_xt0, accs_clf_mnet_xt25, accs_clf_mnet_xt50, accs_clf_mnet_xt75, accs_clf_mnet_xt100 = [], [], [], [], []
    accs_clf_mnet_noaugs_xt0, accs_clf_mnet_noaugs_xt25, accs_clf_mnet_noaugs_xt50, accs_clf_mnet_noaugs_xt75, accs_clf_mnet_noaugs_xt100 = [], [], [], [], []
    accs_clr_xt0, accs_clr_xt25, accs_clr_xt50, accs_clr_xt75, accs_clr_xt100 = [], [], [], [], []
    accs_clr_mnet_xt0, accs_clr_mnet_xt25, accs_clr_mnet_xt50, accs_clr_mnet_xt75, accs_clr_mnet_xt100 = [], [], [], [], []
    for dataset in tqdm(DATASETS):
        accs_clf_xt0.append(get_acc(CHECKPOINT_DIR, CLF_XTALK0, dataset))
        accs_clf_xt25.append(get_acc(CHECKPOINT_DIR, CLF_XTALK25, dataset))
        accs_clf_xt50.append(get_acc(CHECKPOINT_DIR, CLF_XTALK50, dataset))
        accs_clf_xt75.append(get_acc(CHECKPOINT_DIR, CLF_XTALK75, dataset))
        accs_clf_xt100.append(get_acc(CHECKPOINT_DIR, CLF_XTALK100, dataset))

        accs_clf_noaugs_xt0.append(get_acc(CHECKPOINT_DIR, CLF_NOAUGS_XTALK0, dataset))
        accs_clf_noaugs_xt25.append(get_acc(CHECKPOINT_DIR, CLF_NOAUGS_XTALK25, dataset))
        accs_clf_noaugs_xt50.append(get_acc(CHECKPOINT_DIR, CLF_NOAUGS_XTALK50, dataset))
        accs_clf_noaugs_xt75.append(get_acc(CHECKPOINT_DIR, CLF_NOAUGS_XTALK75, dataset))
        accs_clf_noaugs_xt100.append(get_acc(CHECKPOINT_DIR, CLF_NOAUGS_XTALK100, dataset))

        accs_clf_mnet_xt0.append(get_acc(CHECKPOINT_DIR, CLF_MNET_XTALK0, dataset))
        accs_clf_mnet_xt25.append(get_acc(CHECKPOINT_DIR, CLF_MNET_XTALK25, dataset))
        accs_clf_mnet_xt50.append(get_acc(CHECKPOINT_DIR, CLF_MNET_XTALK50, dataset))
        accs_clf_mnet_xt75.append(get_acc(CHECKPOINT_DIR, CLF_MNET_XTALK75, dataset))
        accs_clf_mnet_xt100.append(get_acc(CHECKPOINT_DIR, CLF_MNET_XTALK100, dataset))

        accs_clf_mnet_noaugs_xt0.append(get_acc(CHECKPOINT_DIR, CLF_MNET_NOAUGS_XTALK0, dataset))
        accs_clf_mnet_noaugs_xt25.append(get_acc(CHECKPOINT_DIR, CLF_MNET_NOAUGS_XTALK25, dataset))
        accs_clf_mnet_noaugs_xt50.append(get_acc(CHECKPOINT_DIR, CLF_MNET_NOAUGS_XTALK50, dataset))
        accs_clf_mnet_noaugs_xt75.append(get_acc(CHECKPOINT_DIR, CLF_MNET_NOAUGS_XTALK75, dataset))
        accs_clf_mnet_noaugs_xt100.append(get_acc(CHECKPOINT_DIR, CLF_MNET_NOAUGS_XTALK100, dataset))

        accs_clr_xt0.append(get_acc(CHECKPOINT_DIR, CLR_XTALK0, dataset))
        accs_clr_xt25.append(get_acc(CHECKPOINT_DIR, CLR_XTALK25, dataset))
        accs_clr_xt50.append(get_acc(CHECKPOINT_DIR, CLR_XTALK50, dataset))
        accs_clr_xt75.append(get_acc(CHECKPOINT_DIR, CLR_XTALK75, dataset))
        accs_clr_xt100.append(get_acc(CHECKPOINT_DIR, CLR_XTALK100, dataset))

        accs_clr_mnet_xt0.append(get_acc(CHECKPOINT_DIR, CLR_MNET_XTALK0, dataset))
        accs_clr_mnet_xt25.append(get_acc(CHECKPOINT_DIR, CLR_MNET_XTALK25, dataset))
        accs_clr_mnet_xt50.append(get_acc(CHECKPOINT_DIR, CLR_MNET_XTALK50, dataset))
        accs_clr_mnet_xt75.append(get_acc(CHECKPOINT_DIR, CLR_MNET_XTALK75, dataset))
        accs_clr_mnet_xt100.append(get_acc(CHECKPOINT_DIR, CLR_MNET_XTALK100, dataset))

    # -- xtalk 0
    make_plot(
        [
            accs_clr_xt0,
            accs_clf_xt0,
            accs_clf_noaugs_xt0,
            accs_clr_mnet_xt0,
            accs_clf_mnet_xt0,
            accs_clf_mnet_noaugs_xt0
        ],
        [
            "Contrastive ConvNeXt",
            "Classifier ConvNeXt w/ augs",
            "Classifier ConvNeXt",
            "Contrastive ModelNet",
            "Classifier ModelNet w/ augs",
            "Classifier ModelNet"
        ],
        0,
        savename="segcube_line_chart_xt0_everything.pdf",
        ylim=(0.0, 0.8),
        colors=["C0", "C1", "C2", "C0", "C1", "C2"],
        linestyles=["solid", "solid", "solid", "dashed", "dashed", "dashed"],
        markers=["o", "o", "o", "s", "s", "s"]
    )
    make_plot(
        [
            accs_clr_xt0,
            accs_clf_xt0,
            accs_clf_noaugs_xt0,
        ],
        [
            "Supervised Contrastive",
            "Classifier w/ augs",
            "Classifier"
        ],
        0,
        savename="segcube_line_chart_xt0_convnext.pdf",
        ylim=(0.4, 0.7)
    )
    make_plot(
        [
            accs_clr_mnet_xt0,
            accs_clf_mnet_xt0,
            accs_clf_mnet_noaugs_xt0
        ],
        [
            "Contrastive",
            "Classifier w/ augs",
            "Classifier"
        ],
        0,
        savename="segcube_line_chart_xt0_modelnet.pdf",
        ylim=(0.1, 0.8)
    )

    # -- xtalk 25
    make_plot(
        [
            accs_clr_xt25,
            accs_clf_xt25,
            accs_clf_noaugs_xt25,
            accs_clr_mnet_xt25,
            accs_clf_mnet_xt25,
            accs_clf_mnet_noaugs_xt25
        ],
        [
            "Contrastive ConvNeXt",
            "Classifier ConvNeXt w/ augs",
            "Classifier ConvNeXt",
            "Contrastive ModelNet",
            "Classifier ModelNet w/ augs",
            "Classifier ModelNet"
        ],
        25,
        savename="segcube_line_chart_xt25_everything.pdf",
        ylim=(0.4, 0.8),
        colors=["C0", "C1", "C2", "C0", "C1", "C2"],
        linestyles=["solid", "solid", "solid", "dashed", "dashed", "dashed"],
        markers=["o", "o", "o", "s", "s", "s"]
    )
    make_plot(
        [
            accs_clr_xt25,
            accs_clf_xt25,
            accs_clf_noaugs_xt25,
        ],
        [
            "Supervised Contrastive",
            "Classifier w/ augs",
            "Classifier"
        ],
        25,
        savename="segcube_line_chart_xt25_convnext.pdf",
        ylim=(0.4, 0.7)
    )
    make_plot(
        [
            accs_clr_mnet_xt25,
            accs_clf_mnet_xt25,
            accs_clf_mnet_noaugs_xt25
        ],
        [
            "Contrastive",
            "Classifier w/ augs",
            "Classifier"
        ],
        25,
        savename="segcube_line_chart_xt25_modelnet.pdf",
        ylim=(0.4, 0.8)
    )

    # -- xtalk 50
    make_plot(
        [
            accs_clr_xt50,
            accs_clf_xt50,
            accs_clf_noaugs_xt50,
            accs_clr_mnet_xt50,
            accs_clf_mnet_xt50,
            accs_clf_mnet_noaugs_xt50
        ],
        [
            "Contrastive ConvNeXt",
            "Classifier ConvNeXt w/ augs",
            "Classifier ConvNeXt",
            "Contrastive ModelNet",
            "Classifier ModelNet w/ augs",
            "Classifier ModelNet"
        ],
        50,
        savename="segcube_line_chart_xt50_everything.pdf",
        ylim=(0.4, 0.8),
        colors=["C0", "C1", "C2", "C0", "C1", "C2"],
        linestyles=["solid", "solid", "solid", "dashed", "dashed", "dashed"],
        markers=["o", "o", "o", "s", "s", "s"]
    )
    make_plot(
        [
            accs_clr_xt50,
            accs_clf_xt50,
            accs_clf_noaugs_xt50,
        ],
        [
            "Supervised Contrastive",
            "Classifier w/ augs",
            "Classifier"
        ],
        50,
        savename="segcube_line_chart_xt50_convnext.pdf",
        ylim=(0.4, 0.7)
    )
    make_plot(
        [
            accs_clr_mnet_xt50,
            accs_clf_mnet_xt50,
            accs_clf_mnet_noaugs_xt50
        ],
        [
            "Contrastive",
            "Classifier w/ augs",
            "Classifier"
        ],
        50,
        savename="segcube_line_chart_xt50_modelnet.pdf",
        ylim=(0.4, 0.8)
    )

    # -- xtalk 75
    make_plot(
        [
            accs_clr_xt75,
            accs_clf_xt75,
            accs_clf_noaugs_xt75,
            accs_clr_mnet_xt75,
            accs_clf_mnet_xt75,
            accs_clf_mnet_noaugs_xt75
        ],
        [
            "Contrastive ConvNeXt",
            "Classifier ConvNeXt w/ augs",
            "Classifier ConvNeXt",
            "Contrastive ModelNet",
            "Classifier ModelNet w/ augs",
            "Classifier ModelNet"
        ],
        75,
        savename="segcube_line_chart_xt75_everything.pdf",
        ylim=(0.3, 0.8),
        colors=["C0", "C1", "C2", "C0", "C1", "C2"],
        linestyles=["solid", "solid", "solid", "dashed", "dashed", "dashed"],
        markers=["o", "o", "o", "s", "s", "s"]
    )
    make_plot(
        [
            accs_clr_xt75,
            accs_clf_xt75,
            accs_clf_noaugs_xt75,
        ],
        [
            "Supervised Contrastive",
            "Classifier w/ augs",
            "Classifier"
        ],
        75,
        savename="segcube_line_chart_xt75_convnext.pdf",
        ylim=(0.4, 0.7)
    )
    make_plot(
        [
            accs_clr_mnet_xt75,
            accs_clf_mnet_xt75,
            accs_clf_mnet_noaugs_xt75
        ],
        [
            "Contrastive",
            "Classifier w/ augs",
            "Classifier"
        ],
        75,
        savename="segcube_line_chart_xt75_modelnet.pdf",
        ylim=(0.3, 0.8)
    )

    # -- xtalk 100
    make_plot(
        [
            accs_clr_xt100,
            accs_clf_xt100,
            accs_clf_noaugs_xt100,
            accs_clr_mnet_xt100,
            accs_clf_mnet_xt100,
            accs_clf_mnet_noaugs_xt100
        ],
        [
            "Contrastive ConvNeXt",
            "Classifier ConvNeXt w/ augs",
            "Classifier ConvNeXt",
            "Contrastive ModelNet",
            "Classifier ModelNet w/ augs",
            "Classifier ModelNet"
        ],
        100,
        savename="segcube_line_chart_xt100_everything.pdf",
        ylim=(0.3, 0.8),
        colors=["C0", "C1", "C2", "C0", "C1", "C2"],
        linestyles=["solid", "solid", "solid", "dashed", "dashed", "dashed"],
        markers=["o", "o", "o", "s", "s", "s"]
    )
    make_plot(
        [
            accs_clr_xt100,
            accs_clf_xt100,
            accs_clf_noaugs_xt100,
        ],
        [
            "Supervised Contrastive",
            "Classifier w/ augs",
            "Classifier"
        ],
        100,
        savename="segcube_line_chart_xt100_convnext.pdf",
        ylim=(0.4, 0.7)
    )
    make_plot(
        [
            accs_clr_mnet_xt100,
            accs_clf_mnet_xt100,
            accs_clf_mnet_noaugs_xt100
        ],
        [
            "Contrastive",
            "Classifier w/ augs",
            "Classifier"
        ],
        100,
        savename="segcube_line_chart_xt100_modelnet.pdf",
        ylim=(0.3, 0.8)
    )

def get_acc(chktpt_dir, exp_rel_path, dataset):
    acc_path = os.path.join(chktpt_dir, exp_rel_path, "test_results", f"preds_{dataset}_acc.yml")
    with open(acc_path, "r") as f:
        acc = yaml.load(f, Loader=yaml.FullLoader)
        acc = acc["acc"]
    return acc

def make_plot(
    datas, labels, xtalk, savename=None, ylim=(0,1), colors=None, linestyles=None, markers=None
):
    x = np.linspace(0.0, 1.0, 21)

    fig, ax = plt.subplots(
        nrows=2, sharex=True, figsize=(12, 7), layout="compressed", height_ratios=[1, 3]
    )

    if xtalk == 0:
        ax[1].axvline(0.0, c="r", label="_")
        ax[1].text(0.025, 0.55, "Nominal", rotation=90, c="r", fontsize=13)
        ax[0].axvline(0.0, c="r", label="_")
        residuals = [ [ (acc - data[0]) / data[0] for acc in data ] for data in datas ]
    elif xtalk == 25:
        ax[1].axvline(0.25, c="r", label="_")
        ax[1].text(0.275, 0.55, "Nominal", rotation=90, c="r", fontsize=13)
        ax[0].axvline(0.25, c="r", label="_")
        residuals = [ [ (acc - data[5]) / data[5] for acc in data ] for data in datas ]
    elif xtalk == 50:
        ax[1].axvline(0.5, c="r", label="_")
        ax[1].text(0.525, 0.55, "Nominal", rotation=90, c="r", fontsize=13)
        ax[0].axvline(0.5, c="r", label="_")
        residuals = [ [ (acc - data[10]) / data[10] for acc in data ] for data in datas ]
    elif xtalk == 75:
        ax[1].axvline(0.75, c="r", label="_")
        ax[1].text(0.775, 0.55, "Nominal", rotation=90, c="r", fontsize=13)
        ax[0].axvline(0.75, c="r", label="_")
        residuals = [ [ (acc - data[15]) / data[15] for acc in data ] for data in datas ]
    elif xtalk == 100:
        ax[1].axvline(1.0, c="r", label="_")
        ax[1].text(0.975, 0.55, "Nominal", rotation=90, c="r", fontsize=13)
        ax[0].axvline(1.0, c="r", label="_")
        residuals = [ [ (acc - data[20]) / data[20] for acc in data ] for data in datas ]
    else:
        raise ValueError(f"xtalk={xtalk} not valid")

    colors = [ "C" + str(i) for i in range(len(datas)) ] if colors is None else colors
    linestyles = [ "solid" for _ in range(len(datas)) ] if linestyles is None else linestyles
    markers = [ "o" for _ in range(len(datas)) ] if markers is None else markers

    for data, residual, label, color, linestyle, marker in zip(
        datas, residuals, labels, colors, linestyles, markers
    ):
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
