import torch; import torch.nn as nn
from MinkowskiEngine import SparseTensor
from loss import contrastive_loss, NT_Xent
from voxel_convnext import VoxelConvNeXtCLR

class SimCLR(nn.Module):
    def __init__(self, conf):
        super().__init__()

        self.device = torch.device(conf.device)
        self.checkpoint_dir = conf.checkpoint_dir

        self.net = VoxelConvNeXtCLR(in_chans=1, D=3).to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.95)

        self.criterion = contrastive_loss

        self.loss = None

        self.data = None
        self.s_i = None
        self.s_j = None
        self.s_i_out = None
        self.s_j_out = None

    def set_input(self, data):
        self.data = data
        xi, xj = data
        self.s_i = self._create_tensor(xi[1], xi[0])
        self.s_j = self._create_tensor(xj[1], xj[0])

    def eval(self):
        self.net.eval()

    def train(self):
        self.net.train()

    def scheduler_step(self):
        self.scheduler.step()

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
                self.loss = self.criterion(self.s_i_out, self.s_j_out)

    def optimize_parameters(self):
        self.forward()

        self.optimizer.zero_grad()
        self.loss = self.criterion(self.s_i_out, self.s_j_out)
        self.loss.backward()
        self.optimizer.step()

    def _create_tensor(self, features, coordinates):
        return SparseTensor(
            features=features.float(),
            coordinates=coordinates,
            device=self.device
        )
