import torch
from models.voxel_convnext import VoxelConvNeXtClassifier
import pytorch_lightning as pl
from MinkowskiEngine import SparseTensor
import numpy as np
import torchmetrics


class VoxelConvNextWrapper(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = VoxelConvNeXtClassifier(in_chans=1, D=3, num_classes=5)
        self.loss = torch.nn.CrossEntropyLoss()
        self.test_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=5)
        self.val_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=5)

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
    
    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, predictions, labels = self._shared_step(batch, batch_idx)
        self.val_accuracy.update(predictions, labels)
        self.log('val_accuracy', self.val_accuracy, prog_bar=True)
        self.log('val_loss', loss)
        return loss
    
    def on_validation_epoch_end(self):
        self.log('val_accuracy_epoch', self.val_accuracy.compute())

    def test_step(self, batch, batch_idx):
        loss, predictions, labels = self._shared_step(batch, batch_idx)
        self.test_accuracy.update(predictions, labels)
        self.log('test_loss', loss)
        self.log('test_accuracy', self.test_accuracy, prog_bar=True)
        return loss
    
    def on_test_epoch_end(self):
        self.log('test_accuracy_epoch', self.test_accuracy.compute())



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

