import os

import torch; import torch.nn as nn
import MinkowskiEngine as ME

from loss import contrastive_loss, contrastive_loss_class_labels
from voxel_convnext import VoxelConvNeXtCLR
from dataset import DataPrepType

class SimCLR(nn.Module):
    def __init__(self, conf):
        super().__init__()

        self.device = torch.device(conf.device)
        self.checkpoint_dir = conf.checkpoint_dir

        self.net = VoxelConvNeXtCLR(in_chans=1, D=3, dims=conf.net_dims).to(self.device)

        if conf.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=conf.lr)
        elif conf.optimizer == "AdamW":
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=conf.lr, weight_decay=0.0001)
        else:
            raise ValueError(f"{conf.optimizer} optimizer not implemented!")

        if conf.lr_scheduler == "ExponentialLR":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.95)
        elif conf.lr_scheduler == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 600)
        else:
            raise ValueError(f"{conf.lr_scheduler} lr scheduler not implemented!")

        if conf.data_prep_type == DataPrepType.CONTRASTIVE_AUG_LABELS:
            self.criterion = contrastive_loss_class_labels
            self.crit_same_label_weight = conf.contrastive_loss_same_label_weight
        else:
            self.criterion = contrastive_loss

        self.loss = None

        self.data = None
        self.s_i = None
        self.s_j = None
        self.s_i_out = None
        self.s_j_out = None
        self.label = None

    def set_input(self, data):
        self.data = data
        xi, xj = data[:2]
        self.s_i = self._create_tensor(xi[1], xi[0])
        self.s_j = self._create_tensor(xj[1], xj[0])
        if len(data) == 3:
            self.label = data[-1]

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
        return {
            "s_i" : self.s_i, "s_j" : self.s_j, "s_i_out" : self.s_i_out, "s_j_out" : self.s_j_out
        }

    def get_current_loss(self):
        return self.loss.item()

    def forward(self):
        self.s_i_out = self.net(self.s_i)
        self.s_j_out = self.net(self.s_j)

    def test(self, compute_loss=True):
        with torch.no_grad():
            self.forward()
            if compute_loss:
                self.loss = self._calc_loss()

    def optimize_parameters(self):
        self.forward()

        self.optimizer.zero_grad()
        self.loss = self._calc_loss()
        self.loss.backward()
        self.optimizer.step()

    def _create_tensor(self, features, coordinates):
        return ME.SparseTensor(
            features=features.float(),
            coordinates=coordinates,
            device=self.device
        )

    def _calc_loss(self):
        if self.label is not None:
            return self.criterion(
                self.s_i_out, self.s_j_out, self.label,
                same_label_weight=self.crit_same_label_weight
            )
        return self.criterion(self.s_i_out, self.s_j_out)
