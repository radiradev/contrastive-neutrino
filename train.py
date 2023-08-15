import torch
import pytorch_lightning as pl
from sim_clr.dataset import CLRDataset
from sim_clr.network import SimCLR
from torch.utils.data import random_split, DataLoader
import os
import yaml
from data_utilities.collation import clr_sparse_collate


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)
    
config = load_yaml('config/config.yaml')


# Data
dataset = CLRDataset(config['data']['data_path'])
train_len = int(len(dataset) * 0.8)
lengths = [train_len, len(dataset) - train_len]
train_dataset, val_dataset = random_split(dataset, lengths=lengths, generator=torch.Generator().manual_seed(42))
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

num_of_gpus = torch.cuda.device_count()
assert num_of_gpus > 0, "This code must be run with at least one GPU"

def set_wandb_vars(tmp_dir=config['wandb_tmp_dir']):
    environment_variables = ['WANDB_DIR', 'WANDB_CACHE_DIR', 'WANDB_CONFIG_DIR', 'WANDB_DATA_DIR']
    for variable in environment_variables:
        os.environ[variable] = tmp_dir

# Trainer
checkpoint = None if config['checkpoint'] == 'None' else config['checkpoint']
trainer = pl.Trainer(accelerator='gpu', num_nodes=num_of_gpus, max_epochs=30, callbacks=[checkpoint_callback])
trainer.fit(model, train_loader, val_dataloader, ckpt_path=checkpoint)