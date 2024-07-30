import time, argparse, os, glob

import numpy as np

import torch
from torch.utils.data import DataLoader
import MinkowskiEngine as ME

from config_parser import get_config
from simclr import SimCLR
from classifier import Classifier
from dann import DANN
from dataset import ThrowsDataset, DataPrepType
from data_utils import clr_sparse_collate, clr_labels_sparse_collate

def main(args):
    conf = get_config(args.config)

    if conf.model == "sim_clr":
        model = SimCLR(conf)
    elif conf.model == "classifier":
        model = Classifier(conf)
    elif conf.model == "dann":
        model = DANN(conf)
    else:
        raise ValueError("model must be ['sim_clr', 'classifier', 'dann']")

    _, dataloader_train, _, dataloader_val = get_dataloaders(conf.data_path, conf)
    train_n_batches = len(dataloader_train)
    val_n_batches = len(dataloader_val)
    if conf.model == "dann":
        _, dataloader_train_s, _, dataloader_val_s = get_dataloaders(conf.data_path_s, conf)
        train_n_batches = min(train_n_batches, len(dataloader_train_s))
        val_n_batches = min(val_n_batches, len(dataloader_val_s))

    t0 = time.time()
    losses = []
    n_iter = 0
    prev_val_loss = float("inf")

    # Save latest network
    if conf.save_model == "notrain":
        print("Saving latest nets...")
        model.save_network("notrain")
        return

    write_log_str(conf.checkpoint_dir, "Iters per epoch: {}".format(train_n_batches))

    for epoch in range(conf.epochs):
        write_log_str(conf.checkpoint_dir, "==== Epoch {} ====".format(epoch))

        # Train loop
        model.train()
        if conf.model == "dann":
            dataloader_train_iter = zip(dataloader_train_s, dataloader_train)
        else:
            dataloader_train_iter = iter(dataloader_train)
        for n_iter_epoch in range(train_n_batches):
            try:
                data = next(dataloader_train_iter)
            except RuntimeError as e:
                print(f"RunTimeError! {e}")
                n_iter += 1
                continue

            if conf.model == "dann":
                model.set_input(data[0], data[1])
            else:
                model.set_input(data)
            try:
                model.optimize_parameters()
            except RuntimeError as e:
                print("Encountered error: {}\nskipping this iteration...".format(e))
                continue

            losses.append(model.get_current_loss())

            if (
                args.print_iter and
                not isinstance(args.print_iter, str) and
                (n_iter + 1) % args.print_iter == 0
            ):
                t_iter = time.time() - t0
                t0 = time.time()
                loss_str = get_print_str(epoch, losses, n_iter_epoch, n_iter, t_iter)
                write_log_str(conf.checkpoint_dir, loss_str)
                losses = []
            if (
                conf.lr_decay_iter and
                not isinstance(conf.lr_decay_iter, str) and
                (n_iter + 1) % conf.lr_decay_iter == 0
            ):
                model.scheduler_step()
                write_log_str(conf.checkpoint_dir, "LR {}".format(model.scheduler.get_lr()))

            n_iter += 1

        if isinstance(conf.lr_decay_iter, str) and conf.lr_decay_iter == "epoch":
            model.scheduler_step()
            write_log_str(conf.checkpoint_dir, "LR {}".format(model.scheduler.get_lr()))
        if isinstance(args.print_iter, str) and args.print_iter == "epoch":
            t_iter = time.time() - t0
            t0 = time.time()
            loss_str = get_print_str(epoch, losses, n_iter_epoch, n_iter, t_iter)
            write_log_str(conf.checkpoint_dir, loss_str)
            losses = []

        losses_val = []

        # Save latest network
        if conf.save_model in ["latest", "all"]:
            print("Saving latest nets...")
            model.save_network("latest")

        # Validation loop
        model.eval()
        write_log_str(conf.checkpoint_dir, "== Validation Loop ==")
        if conf.model == "dann":
            dataloader_val_iter = zip(dataloader_val_s, dataloader_val)
        else:
            dataloader_val_iter = iter(dataloader_val)
        for n_iter_epoch in range(val_n_batches):
            try:
                data = next(dataloader_val_iter)
            except RuntimeError as e:
                print(f"RunTimeError! {e}")
                continue

            if conf.model == "dann":
                model.set_input(data[0], data[1])
            else:
                model.set_input(data)
            try:
                model.test(compute_loss=True)
            except RuntimeError as e:
                print("Encountered error: {}\nskipping this iteration...".format(e))
                continue

            losses_val.append(model.get_current_loss())

        loss_str = (
            "Validation with {} images:\n".format(len(dataloader_val.dataset)) +
            "Losses: total={:.7f}".format(np.mean(losses_val))
        )
        write_log_str(conf.checkpoint_dir, loss_str)

        if conf.save_model in ["best", "all"]:
            curr_val_loss = np.mean(losses_val)
            if curr_val_loss < prev_val_loss:
                print(
                    "New best loss ({} < {}), saving net...".format(curr_val_loss, prev_val_loss)
                )
                prev_paths = glob.glob(os.path.join(conf.checkpoint_dir, "**best_epoch**.pth"))
                if len(prev_paths) > 2:
                    raise Exception("About to delete {}, really?".format(prev_paths))
                for path in prev_paths:
                    os.remove(path)
                model.save_network("best_epoch{}".format(epoch))
                prev_val_loss = curr_val_loss

def write_log_str(checkpoint_dir, log_str, print_str=True):
    if print_str:
        print(log_str)
    with open(os.path.join(checkpoint_dir, "losses.txt"), 'a') as f:
        f.write(log_str + '\n')

def get_print_str(epoch, losses, n_iter, n_iter_tot, t_iter):
    return (
        "Epoch: {}, Iter: {}, Total Iter: {}, ".format(epoch, n_iter + 1, n_iter_tot + 1) +
        "Time: {:.7f}\n\t".format(t_iter) +
        "Losses: total={:.7f}".format(np.mean(losses))
    )

def get_dataloaders(data_path, conf):
    dataset_train = ThrowsDataset(
        os.path.join(data_path, "train"),
        conf.data_prep_type,
        conf.augs, conf.n_augs,
        conf.quantization_size
    )
    if conf.data_prep_type == DataPrepType.CLASSIFICATION_AUG:
        val_data_prep_type = DataPrepType.CLASSIFICATION
    else:
        val_data_prep_type = conf.data_prep_type
    dataset_val = ThrowsDataset(
        os.path.join(data_path, "val"),
        val_data_prep_type,
        conf.augs, conf.n_augs,
        conf.quantization_size,
        train_mode=False
    )
    if conf.data_prep_type == DataPrepType.CONTRASTIVE_AUG:
        collate_fn = clr_sparse_collate
    elif conf.data_prep_type == DataPrepType.CONTRASTIVE_AUG_LABELS:
        collate_fn = clr_labels_sparse_collate
    else:
        collate_fn = ME.utils.batch_sparse_collate
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=conf.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=min(conf.max_num_workers, conf.batch_size),
        drop_last=True
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=conf.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=min(conf.max_num_workers, conf.batch_size),
        drop_last=True
    )
    return dataset_train, dataloader_train, dataset_val, dataloader_val

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("config")

    parser.add_argument(
        "--print_iter", type=int, default=200, help="zero for never, -1 for every epoch"
    )

    args = parser.parse_args()

    args.print_iter = "epoch" if args.print_iter == -1 else args.print_iter

    return args

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
