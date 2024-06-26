
import torch
import torchmetrics
import pytorch_lightning as pl
from MinkowskiEngine import SparseTensor
from modules.loss import contrastive_loss, NT_Xent, SigLipLoss, nt_xent_loss
from models.voxel_convnext import VoxelConvNeXtCLR
import math
import prodigyopt

class SimCLR(pl.LightningModule):
    def __init__(self, batch_size=None):
        super().__init__()
        self.model = VoxelConvNeXtCLR(in_chans=1, D=3)
        self.criterion = nt_xent_loss
        self.batch_size = batch_size
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.val_loss_best = torchmetrics.MinMetric()

    def forward(self, x):
        return self.model(x)

    def on_train_start(self) -> None:
        "Lightning hook called before training, useful to initialize things"
        self.val_loss.reset()
        self.val_loss_best.reset()
    
    def _create_tensor(self, features, coordinates):
        return SparseTensor(
            features=features.float(),
            coordinates=coordinates,
            device=self.device
        )

    def _shared_step(self, batch, batch_idx):
        xi, xj = batch 
        xi = self._create_tensor(xi[1], xi[0])
        xj = self._create_tensor(xj[1], xj[0])
        xi_out = self.model(xi)
        xj_out = self.model(xj)
        try:
            loss = self.criterion(xi_out, xj_out)
        except Exception as e:
            print(f"An error occurred: {e}")
            loss = None 
        return loss
    
    def training_step(self, batch, batch_idx):
        # Must clear cache at regular interval
        loss = self._shared_step(batch, batch_idx)

        #compute loss and log it
        self.train_loss(loss)
        self.log('train/loss', self.train_loss, prog_bar=True, batch_size=self.batch_size, on_step=True, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        
        #compute loss and log it
        self.val_loss(loss)
        self.log('val/loss', self.val_loss, batch_size=self.batch_size, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        loss = self.val_loss.compute()  # get current val acc
        self.val_loss_best(loss)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)    

    def configure_optimizers(self):
        optimizer = prodigyopt.Prodigy(self.parameters())
        return optimizer