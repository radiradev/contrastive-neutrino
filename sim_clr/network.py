
import torch
import torchmetrics
import pytorch_lightning as pl
from MinkowskiEngine import SparseTensor
from sim_clr.loss import contrastive_loss, NT_Xent
from models.voxel_convnext import VoxelConvNeXtCLR


class SimCLR(pl.LightningModule):
    def __init__(self, num_gpus=1, gather_distributed=True):
        super().__init__()
        self.model = VoxelConvNeXtCLR(in_chans=1, D=3)
        self.criterion = contrastive_loss
        
        # sometimes we might want to use multiple gpus but a smaller efective batch_size
        if num_gpus > 1 and gather_distributed:
            self.gather_distributed = True
        else: 
            self.gather_distributed = False
    def forward(self, x):
        return self.model(x)
    
    def _create_tensor(self, features, coordinates):

        return SparseTensor(
            features=features,
            coordinates=coordinates,
            device=self.device
        )

    def _shared_step(self, batch, batch_idx):
        xi, xj = batch 
        xi = self._create_tensor(xi[1], xi[0])
        xj = self._create_tensor(xj[1], xj[0])
        xi_out = self.model(xi)
        xj_out = self.model(xj)
        loss = self.criterion(xi_out, xj_out, gather_distributed=self.gather_distributed)
        return loss
    
    def training_step(self, batch, batch_idx):
        # Must clear cache at regular interval
        loss = self._shared_step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log('test_accuracy', self.test_accuracy, prog_bar=self.gather_distributed)
        return loss
    
    def on_test_epoch_end(self):
        self.log('test_accuracy_epoch', self.test_accuracy.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer