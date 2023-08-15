import torch
import pytorch_lightning as pl

import os
from train.network_wrapper import VoxelConvNextWrapper
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from MinkowskiEngine.utils import batch_sparse_collate

from pytorch_lightning.loggers import WandbLogger
from sim_clr.dataset import ConvertedDataset
from torch.utils.data import random_split, DataLoader


if "CONDOR_ID" in os.environ:
    workspace = '/scratch/'
    data_path = '/scratch/converted_data/train'
else:
    workspace = '/workspace/'
    data_path = '/mnt/rradev/osf_data_512px/converted_data/train'


# This should be passed from the script that calls this file
dataset = ConvertedDataset(data_path)
train_dataset, val_dataset = random_split(dataset, lengths=[0.95, 0.05], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=batch_sparse_collate, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=batch_sparse_collate)


model = VoxelConvNextWrapper()

# callbacks
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    filename='model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min',
)

checkpoint = os.path.join(workspace, 'lightning_logs/best_run/checkpoints/model-epoch=03-val_loss=0.30.ckpt')

num_of_gpus = torch.cuda.device_count()
assert num_of_gpus > 0, "This code must be run with at least one GPU"

def set_wandb_vars(tmp_dir='/afs/cern.ch/work/r/rradev/public/lightning/wandb_tmp'):
    environment_variables = ['WANDB_DIR', 'WANDB_CACHE_DIR', 'WANDB_CONFIG_DIR', 'WANDB_DATA_DIR']
    for variable in environment_variables:
        os.environ[variable] = tmp_dir


wandb_logger = WandbLogger(project='convnext_classifer', log_model=True)
trainer = pl.Trainer(accelerator='gpu', num_nodes=num_of_gpus, max_epochs=30, callbacks=[checkpoint_callback], logger=wandb_logger)
trainer.fit(model, train_loader, val_dataloader, ckpt_path=checkpoint)