# import torch
# from dataset import ThrowsDataset
# from simclr import SimCLR
# from classifier import Classifier
# from torch.utils.data import random_split, DataLoader
# import os
# from data_utils import clr_sparse_collate, load_yaml
# from MinkowskiEngine.utils import batch_sparse_collate
# import torch.distributed as dist

# config = load_yaml('config/config.yaml')


# def dataloaders(batch_size: int, data_path: str, dataset_type: str, num_workers=64, pin_memory=True):
#     # This is a huge mess
#     if dataset_type == 'contrastive' or dataset_type == 'throws_augmented':
#         print("Using dataset with Throws...")
#         data_path = config['data']['data_path']
#     else:
#         data_path = config['data']['nominal_data_path']

#     if dataset_type == 'regression':
#         train_dataset = Regression(os.path.join(data_path, 'train'))
#         val_dataset = Regression(os.path.join(data_path, 'val'))

#     else:
#         # for classification and contrastive
#         dataset_type = 'single_particle_augmented' if dataset_type == 'throws_augmented' else dataset_type
#         train_dataset = ThrowsDataset(dataset_type, os.path.join(data_path, 'train'))

#         val_data_path = data_path
#         if dataset_type != 'contrastive':
#             val_data_path = config['data']['nominal_data_path']
#         val_dataset = ThrowsDataset(dataset_type, os.path.join(val_data_path, 'val'), train_mode=False)

#     collate_fn = batch_sparse_collate if dataset_type == 'single_particle' or dataset_type=='single_particle_augmented' or dataset_type=='regression' else clr_sparse_collate
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, drop_last=True)
#     val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers,drop_last=True)
#     return train_loader, val_dataloader

# # callbacks
# checkpoint_callback = pl.callbacks.ModelCheckpoint(
#     dirpath=config['checkpoint_dirpath'],
#     monitor='val/loss',
#     filename='model-{epoch:02d}-{val_loss:.2f}',
#     save_top_k=2,
#     mode='min',
#     save_last=True,
# )

# num_of_gpus = torch.cuda.device_count()
# assert num_of_gpus > 0, "This code must be run with at least one GPU"


# def set_wandb_vars(tmp_dir=config['wandb_tmp_dir']):
#     environment_variables = ['WANDB_DIR', 'WANDB_CACHE_DIR', 'WANDB_CONFIG_DIR', 'WANDB_DATA_DIR']
#     for variable in environment_variables:
#         os.environ[variable] = tmp_dir


# def train_model(batch_size=256, num_of_gpus=1, num_of_cpus=64, dataset_type='single_particle', model=None, wandb_checkpoint=None, gather_distributed=False, run_name=None):
#     if model == "sim_clr":
#         model = SimCLR(batch_size, num_of_gpus, bool(gather_distributed))
#     elif model == "classifier":
#         model = Classifier(batch_size)
#     elif model == "regressor":
#         model = Regressor(batch_size)
#     else:
#         raise ValueError("Model: sim_clr, classifier or regressor")

#     set_wandb_vars()
#     wandb_logger = WandbLogger(name=run_name, project='contrastive-neutrino', log_model=False)
#     data_path = "" # config['data']['data_path']
#     train_loader, val_dataloader = dataloaders(batch_size, data_path=data_path, dataset_type=dataset_type, num_workers=num_of_cpus)

#     if wandb_checkpoint is not None:
#         checkpoint = get_wandb_ckpt(wandb_checkpoint)
#     else:
#         checkpoint = None

#     limit_train_batches = 1.0 if dataset_type!='throws_augmented' else 0.1
#     print(f"Limiting train batches to {limit_train_batches} for dataset type {dataset_type}")
#     trainer = pl.Trainer(accelerator='gpu', gpus=num_of_gpus, max_epochs=800,
#                          limit_train_batches=limit_train_batches, callbacks=[checkpoint_callback],
#                          logger=wandb_logger, log_every_n_steps=5)
#     trainer.fit(model, train_loader, val_dataloader, ckpt_path=checkpoint)

# if __name__ == '__main__':
#     fire.Fire(train_model)

import time, argparse, os, glob

from collections import defaultdict

import yaml
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
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=min(conf.max_num_workers, conf.batch_size),
        drop_last=True
    )

    t0 = time.time()
    losses = []
    n_iter = 0

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

        # Validation loop
        model.eval()
        print("Validation loop...")
        for data in dataloader_val:
            model.set_input(data)
            model.test(compute_loss=True)

            losses.append(model.get_current_loss())

        loss_str = (
            "Validation with {} images:\n".format(len(dataset_val)) +
            "Losses: total={:.7f}".format(np.mean(losses))
        )
        write_log_str(conf.checkpoint_dir, loss_str)

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







