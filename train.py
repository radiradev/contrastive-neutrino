import torch
import pytorch_lightning as pl
from sim_clr.dataset import CLRDataset
from sim_clr.network import SimCLR
from torch.utils.data import random_split, DataLoader
import os
import yaml
import wandb
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from data_utilities.collation import clr_sparse_collate
import torch.distributed as dist
import fire

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)
    
config = load_yaml('config/config.yaml')

# Data

def dataloaders(batch_size: int, data_path: str, num_workers=16, pin_memory=True):

    dataset = CLRDataset(config['data']['data_path'])
    train_len = int(len(dataset) * 0.8)
    lengths = [train_len, len(dataset) - train_len]
    train_dataset, val_dataset = random_split(dataset, lengths=lengths, generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=clr_sparse_collate, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=clr_sparse_collate)
    return train_loader, val_dataloader

# callbacks
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    filename='model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min',
)

num_of_gpus = torch.cuda.device_count()
assert num_of_gpus > 0, "This code must be run with at least one GPU"


wandb_logger = pl.loggers.WandbLogger()

# pytorch lightning requires setting:
# devices=number of allocated slurm gpus = number of slurm tasks 
# num_nodes= number of slurm nodes
trainer = pl.Trainer(devices=4,
                        accelerator='gpu', strategy='ddp_find_unused_parameters_false', 
                        logger=wandb_logger)


def train_model(batch_size):
    model = SimCLR()
    
    checkpoint = None if config['checkpoint'] == 'None' else config['checkpoint']
    data_path = config['data']['data_path']
    train_loader, val_dataloader = dataloaders(batch_size, data_path=data_path)
    trainer = pl.Trainer(accelerator='gpu', gpus=num_of_gpus, max_epochs=30, callbacks=[checkpoint_callback], strategy='ddp')
    trainer.fit(model, train_loader, val_dataloader, ckpt_path=checkpoint)
    


if __name__ == '__main__':
    fire.Fire(train_model)




