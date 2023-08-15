import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from pytorch_lightning.loggers import WandbLogger
import wandb
from torch.utils.data import DataLoader,Dataset
import os

import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y
    
class MyDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        N=10000   
        x = torch.unsqueeze(torch.linspace(-1, 1, N), dim=1)                    
        y = x.pow(2) + config.noise*torch.rand(x.size())                             
        self.my_dataset = MyDataset(x,y)
        print(config.noise)

    def train_dataloader(self):
        return DataLoader(self.my_dataset,batch_size=100,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.my_dataset,batch_size=100,shuffle=False)



class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
    def forward(self, x):
        x= F.relu(self.hidden(x))
        x = self.predict(x)
        return x

class MyLightningModule(pl.LightningModule):
    def __init__(self, config): #n_hidden, lr):
        super().__init__()
        self.net = Net(n_feature=1, n_hidden=config.n_hidden, n_output=1)
        self.lr = config.lr
#         self.save_hyperparameters()  # **wandb process fail to finish if this is uncommented**

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        y_hat = self.net(x)
        loss = F.mse_loss(y_hat, x)
        self.log('training_loss',loss,on_step=True, on_epoch=True,prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.net(x)
        loss = F.mse_loss(y_hat, x)
        self.log('validation_loss',loss)
        # return {'val_loss': loss}
        return loss    

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer
    
def train_model():
#     os.environ["WANDB_START_METHOD"] = "thread"'
    wandb.init(project="sweep")
    config=wandb.config
    wandb_logger = WandbLogger()
    data = MyDataModule(config)
    module = MyLightningModule(config)

    wandb_logger.watch(module.net)
    
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=10, 
        default_root_dir="./lightning-example", logger=wandb_logger)
#     wandb.require(experiment="service")
    trainer.fit(module, data)        

    
    
if __name__ == '__main__':
    sweep_config = {
        'method': 'random',
        'name': 'first_sweep',
        'metric': {
            'goal': 'minimize',
            'name': 'validation_loss'
        },
        'parameters': {
            'n_hidden': {'values': [2,3,5,10]},
            'lr': {'max': 1.0, 'min': 0.0001},
            'noise': {'max': 1.0, 'min': 0.}
        }
    }

    sweep_id=wandb.sweep(sweep_config, project="test_sweep")
    wandb.agent(sweep_id=sweep_id, function=train_model, count=5)