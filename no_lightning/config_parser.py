import os, shutil
from collections import namedtuple

import yaml

from dataset import DataPrepType
from augmentations import get_aug_funcs

defaults = {
    "device" : "cuda:0",
    "max_num_workers" : 4,
    "lr_decay_iter" : 0,
    "save_model" : "never",
    "net_dims" : [96, 192, 384, 768],
    "augs" : ["rotate", "drop", "shift_energy_uniform", "translate", "identity"],
    "n_augs" : 2,
    "data_path_s" : None,
    "alpha" : 1.0,
    "lr" : 1e-4,
    "aug_energy_scale_factor" : 0.1,
    "aug_translate_scale_factor" : 0.3,
    "contrastive_loss_same_label_weight" : 0.5,
    "quantization_size" : 0.38,
    "xtalk" : None
}

mandatory_fields = {
    "data_path",
    "data_prep_type",
    "model",
    "batch_size",
    "epochs",
    "name"
}

def get_config(conf_file, overwrite_dict={}, prep_checkpoint_dir=True):
    print("Reading conf from {}".format(conf_file))

    with open(conf_file) as f:
        conf_dict = yaml.load(f, Loader=yaml.FullLoader)

    for field, val in overwrite_dict.items():
        conf_dict[field] = val

    missing_fields = mandatory_fields - set(conf_dict.keys())
    if missing_fields:
        raise ValueError(
            "Missing mandatory fields {} in conf file at {}".format(missing_fields, conf_file)
        )

    for option in set(defaults.keys()) - set(conf_dict.keys()):
        conf_dict[option] = defaults[option]

    if conf_dict["data_prep_type"] == "contrastive_augmentations":
        conf_dict["data_prep_type"] = DataPrepType.CONTRASTIVE_AUG
    elif conf_dict["data_prep_type"] == "classification":
        conf_dict["data_prep_type"] = DataPrepType.CLASSIFICATION
    elif conf_dict["data_prep_type"] == "classification_augmentations":
        conf_dict["data_prep_type"] = DataPrepType.CLASSIFICATION_AUG
    elif conf_dict["data_prep_type"] == "contrastive_augmentations_labels":
        conf_dict["data_prep_type"] = DataPrepType.CONTRASTIVE_AUG_LABELS
    else:
        raise ValueError("data_prep_type={} not recognised".format(conf_dict["data_prep_type"]))

    if conf_dict["save_model"] not in ["never", "latest", "best", "all", "notrain"]:
        raise ValueError(
            "'save_model': {} invalid, choose 'never', 'latest', 'best', 'all', 'notrain'".format(
                conf_dict["save_model"]
            )
        )

    if conf_dict["model"] == "dann" and conf_dict["data_path_s"] is None:
        raise ValueError("dann models must specifiy source data 'data_path_s'")
  
    if conf_dict["xtalk"] is not None and conf_dict["xtalk"] > 1.0:
        raise ValueError("xtalk must be None or <= 1.0")

    if prep_checkpoint_dir:
        conf_dict["checkpoint_dir"] = os.path.join(conf_dict["checkpoints_dir"], conf_dict["name"])
        if not os.path.exists(conf_dict["checkpoint_dir"]):
            os.makedirs(conf_dict["checkpoint_dir"])
        else:
            print(
                "WARNING: {} already exists, data may be overwritten".format(
                    conf_dict["checkpoint_dir"]
                )
            )
        shutil.copyfile(
            conf_file, os.path.join(conf_dict["checkpoint_dir"], os.path.basename(conf_file))
        )
        if not os.path.exists(os.path.join(conf_dict["checkpoint_dir"], "preds")):
            os.makedirs(os.path.join(conf_dict["checkpoint_dir"], "preds"))

    aug_funcs = get_aug_funcs(
        conf_dict.pop("aug_energy_scale_factor"), conf_dict.pop("aug_translate_scale_factor")
    )
    conf_dict["augs"] = [ aug_funcs[aug] for aug in conf_dict["augs"] ]

    conf_namedtuple = namedtuple("conf", conf_dict)
    conf = conf_namedtuple(**conf_dict)

    return conf

