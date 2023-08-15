import torch
import pytorch_lightning as pl
from sim_clr.dataset import CLRDataset
from sim_clr.network import SimCLR
from torch.utils.data import random_split, DataLoader
import os
import yaml
import wandb
from pytorch_lightning.loggers import WandbLogger
from data_utilities.collation import clr_sparse_collate

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)
    
config = load_yaml('config/config.yaml')

# Data

def dataloaders(wandb_config, data_path: str, num_workers=4, pin_memory=True):

    dataset = CLRDataset(config['data']['data_path'])
    train_len = int(len(dataset) * 0.8)
    lengths = [train_len, len(dataset) - train_len]
    train_dataset, val_dataset = random_split(dataset, lengths=lengths, generator=torch.Generator().manual_seed(42))
    train_batch_size = wandb_config.batch_size
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=clr_sparse_collate, num_workers=2)
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


def set_wandb_vars(tmp_dir=config['wandb_tmp_dir']):
    environment_variables = ['WANDB_DIR', 'WANDB_CACHE_DIR', 'WANDB_CONFIG_DIR', 'WANDB_DATA_DIR']
    for variable in environment_variables:
        os.environ[variable] = tmp_dir


def train_model():
    wandb.init(project="contrastive_neutrino")
    wandb_logger = WandbLogger()
    model = SimCLR()

    for key, val in config.items():
        wandb.config[key] = val

    wandb_logger.watch(model.model)  
    checkpoint = None if config['checkpoint'] == 'None' else config['checkpoint']

    data_path = config['data']['data_path']
    train_loader, val_dataloader = dataloaders(wandb.config, data_path)
    trainer = pl.Trainer(accelerator='gpu', num_nodes=num_of_gpus, max_epochs=30, callbacks=[checkpoint_callback], logger=wandb_logger)
    trainer.fit(model, train_loader, val_dataloader, ckpt_path=checkpoint)


if __name__ == '__main__':
    sweep_config = {
        'method': 'random',
        'name': 'first_sweep',
        'metric': {
            'goal': 'minimize',
            'name': 'val_loss'
        },
        'parameters': {
            'batch_size': {'values': [256, 512, 1024, 2048]},
        }
    }

    sweep_id=wandb.sweep(sweep_config, project="contrastive-neutrino")
    wandb.agent(sweep_id=sweep_id, function=train_model, count=4)