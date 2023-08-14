import torch
import pytorch_lightning as pl
from tqdm import tqdm
import zipfile

from MinkowskiEngine.utils import sparse_collate

from sim_clr.dataset import CLRDataset
from sim_clr.network import SimCLR
from torch.utils.data import random_split, DataLoader
import os

if "CONDOR_ID" in os.environ:
    workspace = '/scratch/'
    data_path = '/scratch/converted_data/train'
else:
    workspace = '/workspace/'
    data_path = '/mnt/rradev/osf_data_512px/converted_data/train'



def clr_sparse_collate(data, dtype=torch.int32, device=None):
    # Unzip the dataset into separate coordinate and feature tuples for i and j
    x_i, x_j = zip(*data)
    coordinates_i, features_i = zip(*x_i)
    coordinates_j, features_j = zip(*x_j)
    # Collate the coordinate and feature tuples separately
    collated_i = sparse_collate(coords=coordinates_i, feats=features_i, dtype=dtype, device=device)
    collated_j = sparse_collate(coords=coordinates_j, feats=features_j, dtype=dtype, device=device)
    return collated_i, collated_j


# Data
dataset = CLRDataset()
train_dataset, val_dataset = random_split(dataset, lengths=[0.95, 0.05], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=clr_sparse_collate, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=clr_sparse_collate)

# test one batch 
batch = next(iter(train_loader))
xi, xj = batch

model = SimCLR()

# callbacks
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    filename='model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min',
)
checkpoint = '/workspace/lightning_logs/version_21/checkpoints/model-epoch=07-val_loss=0.30.ckpt'

num_of_gpus = torch.cuda.device_count()
assert num_of_gpus > 0, "This code must be run with at least one GPU"

def set_wandb_vars(tmp_dir='/afs/cern.ch/work/r/rradev/public/lightning/wandb_tmp'):
    environment_variables = ['WANDB_DIR', 'WANDB_CACHE_DIR', 'WANDB_CONFIG_DIR', 'WANDB_DATA_DIR']
    for variable in environment_variables:
        os.environ[variable] = tmp_dir

# Trainer
trainer = pl.Trainer(accelerator='gpu', num_nodes=num_of_gpus, max_epochs=30, callbacks=[checkpoint_callback])
trainer.fit(model, train_loader, val_dataloader, ckpt_path=checkpoint)