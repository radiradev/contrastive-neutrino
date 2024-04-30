import time, argparse, os, glob

import numpy as np

import torch
from torch.utils.data import random_split, DataLoader
import MinkowskiEngine as ME

from config_parser import get_config
from simclr import SimCLR
from classifier import Classifier
from dataset import ThrowsDataset, DataPrepType
from data_utils import clr_sparse_collate

def main(args):
    conf = get_config(args.config)

    if conf.model == "sim_clr":
        model = SimCLR(conf)
    elif conf.mode == "classifier":
        raise NotImplementedError
        # model = Classifier(batch_size=conf.batch_size, device=conf.device)
    else:
        raise ValueError("model must be ['sim_clr', 'classifier']")

    dataset_train = ThrowsDataset(
        os.path.join(conf.data_path, "train"), conf.data_prep_type
    )
    dataset_val = ThrowsDataset(
        os.path.join(conf.data_path, "val"), conf.data_prep_type, train_mode=False
    )
    if conf.data_prep_type == DataPrepType.CLASSIFICATION:
        collate_fn = ME.utils.batch_sparse_collate
    else:
        collate_fn = clr_sparse_collate
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

    t0 = time.time()
    losses = []
    n_iter = 0
    prev_val_loss = float("inf")

    write_log_str(conf.checkpoint_dir, "Iters per epoch: {}".format(len(dataloader_train)))

    for epoch in range(conf.epochs):
        write_log_str(conf.checkpoint_dir, "==== Epoch {} ====".format(epoch))

        # Train loop
        model.train()
        for n_iter_epoch, data in enumerate(dataloader_train):
            model.set_input(data)
            model.optimize_parameters()

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
        for data in dataloader_val:
            model.set_input(data)
            model.test(compute_loss=True)

            losses_val.append(model.get_current_loss())

        loss_str = (
            "Validation with {} images:\n".format(len(dataset_val)) +
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







