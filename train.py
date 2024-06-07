import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
from pytorch_lightning.loggers import WandbLogger
from utils.data import load_yaml
from utils.data import get_wandb_ckpt
import hydra
from pytorch_lightning import LightningModule

    
config = load_yaml('configs/config.yaml')

def set_wandb_vars(tmp_dir=config['wandb_tmp_dir']):
    environment_variables = ['WANDB_DIR', 'WANDB_CACHE_DIR', 'WANDB_CONFIG_DIR', 'WANDB_DATA_DIR']
    for variable in environment_variables:
        os.environ[variable] = tmp_dir

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=config['checkpoint_dirpath'],
    monitor='val/loss',
    filename='model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=2,
    mode='min',
    save_last=True,
)

def dataloaders(batch_size: int, train_dataset, val_dataset, collate_fn, num_workers=64, pin_memory=True):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers,drop_last=True)
    return train_loader, val_dataloader

@hydra.main(version_base="1.3", config_path="configs", config_name=None)
def train(cfg):
    print(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    train_dataset = hydra.utils.instantiate(cfg.train_dataset)
    val_dataset = hydra.utils.instantiate(cfg.val_dataset)
    collate_fn = hydra.utils.instantiate(cfg.collate_fn)
    train_loader, val_dataloader = dataloaders(cfg.batch_size, train_dataset, val_dataset, collate_fn)

    set_wandb_vars()
    wandb_logger = WandbLogger(name=cfg.run_name, project='contrastive-neutrino2', log_model='all')
    if cfg.wandb_checkpoint != 0:
        checkpoint = get_wandb_ckpt(cfg.wandb_checkpoint)
    else:
        checkpoint = None

    trainer = pl.Trainer(accelerator='gpu', gpus=cfg.num_of_gpus, max_epochs=800, 
                         limit_train_batches=cfg.limit_train_batches, callbacks=[checkpoint_callback], 
                         logger=wandb_logger, log_every_n_steps=5)
    trainer.fit(model, train_loader, val_dataloader, ckpt_path=checkpoint)


if __name__ == '__main__':
    train()



