import os

import numpy as np

import torch; import torch.nn as nn
import MinkowskiEngine as ME

from voxel_convnext import VoxelConvNeXtClassifier
from modelnet import MinkowskiFCNNClassifier

class Classifier(nn.Module):
    def __init__(self, conf):
        super().__init__()

        self.device = torch.device(conf.device)
        self.checkpoint_dir = conf.checkpoint_dir

        if conf.net_architecture == "convnext":
            self.net = VoxelConvNeXtClassifier(
                in_chans=1,
                D=3,
                dims=conf.net_dims,
                depths=conf.net_depths,
                num_classes=conf.num_classes
            ).to(self.device)
            self._create_tensor = self._create_sparsetensor
        elif conf.net_architecture == "modelnet40":
            assert isinstance(conf.net_dims, int), "net_dims should be an int for modelnet40"
            self.net = MinkowskiFCNNClassifier(
                in_channel=1, D=3, embedding_channel=conf.net_dims, num_classes=conf.num_classes
            ).to(self.device)
            self._create_tensor = self._create_tensorfield
        else:
            raise ValueError(f"{conf.net_architecture} network architecture not implemented!")

        if conf.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=conf.lr)
        elif conf.optimizer == "AdamW":
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=conf.lr, weight_decay=0.0001)
        else:
            raise ValueError(f"{conf.optimizer} optimizer not implemented!")

        if conf.lr_scheduler == "ExponentialLR":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.95)
        elif conf.lr_scheduler == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, conf.CosineAnnealingLR_T_max
            )
        else:
            raise ValueError(f"{conf.lr_scheduler} lr scheduler not implemented!")

        self.criterion = nn.CrossEntropyLoss()

        self.loss = None

        self.data = None
        self.s_in = None
        self.target_out = None
        self.pred_out = None

    def set_input(self, data):
        self.data = data
        coords, feats, labels = data
        self.target_out = labels.to(self.device)
        self.s_in = self._create_tensor(feats, coords)

    def eval(self):
        self.net.eval()

    def train(self):
        self.net.train()

    def scheduler_step(self):
        self.scheduler.step()

    def save_network(self, suffix):
        torch.save(
            self.net.cpu().state_dict(),
            os.path.join(self.checkpoint_dir, "net_{}.pth".format(suffix))
        )
        self.net.to(self.device)

    def load_network(self, path):
        print("Loading model from {}".format(path))
        state_dict = torch.load(path, map_location=self.device)
        if hasattr(state_dict, "_metadata"):
            del state_dict._metadata
        self.net.load_state_dict(state_dict)

    def get_current_visuals(self):
        return { "s_i" : self.s_in, "target_out" : self.target_out, "pred_out" : self.pred_out }

    def get_current_loss(self):
        return self.loss.item()

    def forward(self):
        self.pred_out = self.net(self.s_in)

    def test(self, compute_loss=True):
        with torch.no_grad():
            self.forward()
            if compute_loss:
                self.loss = self.criterion(self.pred_out, self.target_out)

    def optimize_parameters(self):
        self.forward()

        self.optimizer.zero_grad()
        self.loss = self.criterion(self.pred_out, self.target_out)
        self.loss.backward()
        self.optimizer.step()

    def _create_sparsetensor(self, features, coordinates):
        return ME.SparseTensor(
            features=features.float(), coordinates=coordinates, device=self.device
        )

    def _create_tensorfield(self, features, coordinates):
        return ME.TensorField(
            features=features.float(), coordinates=coordinates, device=self.device
        )
