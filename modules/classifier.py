import torch
from models.voxel_convnext import VoxelConvNeXtClassifier
import pytorch_lightning as pl
from MinkowskiEngine import SparseTensor
import numpy as np
import torchmetrics
from torchmetrics.functional import accuracy


class SingleParticleModel(pl.LightningModule):
    def __init__(self, batch_size=256):
        super().__init__()
        self.model = VoxelConvNeXtClassifier(in_chans=1, D=3, num_classes=5)
        self.loss = torch.nn.CrossEntropyLoss()
        self.test_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=5)
        self.val_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=5)
        self.batch_size = batch_size

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, batch_idx):
        coordinates, energy, labels = batch
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
        self.log('train_acc', accuracy(predictions, labels), on_step=True, on_epoch=False, prog_bar=True, batch_size=self.batch_size)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False, batch_size=self.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, predictions, labels = self._shared_step(batch, batch_idx)
        self.val_accuracy(predictions, labels)
        self.log('val_acc', self.val_accuracy,on_step=False, on_epoch=True, batch_size=self.batch_size)
        self.log('val_loss', loss, on_step=False, on_epoch=True, batch_size=self.batch_size)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

