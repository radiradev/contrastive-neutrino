import os

import numpy as np

import torch; import torch.nn as nn
import MinkowskiEngine as ME

from voxel_convnext import VoxelConvNeXtClassifier

class Classifier(nn.Module):
    num_classes = 5

    def __init__(self, conf):
        super().__init__()

        self.device = torch.device(conf.device)
        self.checkpoint_dir = conf.checkpoint_dir

        self.net = VoxelConvNeXtClassifier(in_chans=1, D=3, num_classes=self.num_classes)
        self.net.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.95)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.loss = None

        self.data = None
        self.s_in = None
        self.target_out = None
        self.pred_out = None

    def set_input(self, data):
        self.data = data
        coords, feats, labels  = data
        self.target_out = labels.to(self.device)
        self.s_in = ME.SparseTensor(coordinates=coords, features=feats.float(), device=self.device)

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
