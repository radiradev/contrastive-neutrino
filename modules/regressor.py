import torch
from models.voxel_convnext import VoxelConvNeXtRegressor
import pytorch_lightning as pl
from MinkowskiEngine import SparseTensor
import numpy as np
import torchmetrics


class Regressor(pl.LightningModule):
    def __init__(self, batch_size=None, device='cuda'):
        if batch_size is None:
            raise ValueError("batch_size must be specified")
        
        super().__init__()
        self.model = VoxelConvNeXtRegressor(in_chans=1, D=3, num_classes=1).to(device)
        self.loss = torch.nn.L1Loss()

        #metrics
        self.train_r2 = torchmetrics.R2Score()
        self.val_r2 = torchmetrics.R2Score()

        #losses
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()

        #track best loss
        self.val_loss_best = torchmetrics.MinMetric()

        self.batch_size = batch_size

    def on_train_start(self) -> None:
        "Lightning hook called before training, useful to initialize things"
        self.val_loss.reset()
        self.val_loss_best.reset()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, batch_idx):
        coordinates, energy, labels = batch
        labels = labels.unsqueeze(1)
        stensor = SparseTensor(
            features=energy.float(),
            coordinates=coordinates,
            device=self.device
        )

        predictions = self.model(stensor)
        
        loss = self.loss(predictions, labels)
        return loss, predictions, labels
    
    def predict_(self, batch):
        batch = (x.cuda() for x in batch)
        _, predictions, label = self._shared_step(batch, batch_idx=None)
        return predictions, label
    
    def training_step(self, batch, batch_idx):
        loss, predictions, labels = self._shared_step(batch, batch_idx)
        
        #update metrics
        self.train_loss(loss)
        self.train_r2(predictions, labels)

        #log metrics
        self.log('train/loss', self.train_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=self.batch_size)
        self.log('train/acc', self.train_r2, prog_bar=True, on_step=True, on_epoch=True, batch_size=self.batch_size)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, predictions, labels = self._shared_step(batch, batch_idx)

        #update metrics
        self.val_loss(loss)
        self.val_r2(predictions, labels)

        #log metrics
        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, batch_size=self.batch_size)
        self.log('val/acc', self.val_r2, on_step=False, on_epoch=True, batch_size=self.batch_size)
        return loss

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        loss = self.val_loss.compute()  # get current val acc
        self.val_loss_best(loss)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

