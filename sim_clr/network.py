
import torch
import torchmetrics
import pytorch_lightning as pl
from MinkowskiEngine import SparseTensor
from sim_clr.loss import contrastive_loss
from models.voxel_convnext import VoxelConvNeXtCLR


class SimCLR(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = VoxelConvNeXtCLR(in_chans=1, D=3)

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
        loss = contrastive_loss(xi_out, xj_out, gather_distributed=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log('test_accuracy', self.test_accuracy, prog_bar=True)
        return loss
    
    def on_test_epoch_end(self):
        self.log('test_accuracy_epoch', self.test_accuracy.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer