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

def dataloaders(batch_size: int, data_path: str, dataset_type: str, num_workers=16, pin_memory=True):

    dataset = CLRDataset(dataset_type, config['data']['data_path'])
    train_len = int(len(dataset) * 0.8)
    lengths = [train_len, len(dataset) - train_len]
    train_dataset, val_dataset = random_split(dataset, lengths=lengths, generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=clr_sparse_collate, num_workers=num_workers, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=clr_sparse_collate, drop_last=True)
    return train_loader, val_dataloader

# callbacks
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=config['checkpoint_dirpath'],
    monitor='val_loss',
    filename='model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=2,
    mode='min',
    save_last=True,
)

num_of_gpus = torch.cuda.device_count()
assert num_of_gpus > 0, "This code must be run with at least one GPU"


def set_wandb_vars(tmp_dir=config['wandb_tmp_dir']):
    environment_variables = ['WANDB_DIR', 'WANDB_CACHE_DIR', 'WANDB_CONFIG_DIR', 'WANDB_DATA_DIR']
    for variable in environment_variables:
        os.environ[variable] = tmp_dir


def train_model(batch_size, num_of_gpus, dataset_type, checkpoint=None, gather_distributed=True):
    model = SimCLR(num_of_gpus, bool(gather_distributed))
    set_wandb_vars()
    wandb_logger = pl.loggers.WandbLogger(project='contrastive-neutrino', log_model='all')
    data_path = config['data']['data_path']
    train_loader, val_dataloader = dataloaders(batch_size, data_path=data_path, dataset_type=dataset_type)
    trainer = pl.Trainer(accelerator='gpu', gpus=num_of_gpus, max_epochs=400, callbacks=[checkpoint_callback], strategy='ddp', logger=wandb_logger)
    trainer.fit(model, train_loader, val_dataloader, ckpt_path=checkpoint)
    


if __name__ == '__main__':
    fire.Fire(train_model)




