import os

import torch; import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiSparseTensor import _get_coordinate_map_key

from voxel_convnext import VoxelConvNeXtClassifier

class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

# class RevGrad(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input_, alpha_):
#         ctx.save_for_backward(input_, alpha_)
#         output = input_.view_as(input_)
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):  # pragma: no cover
#         grad_input = None
#         _, alpha_ = ctx.saved_tensors
#         if ctx.needs_input_grad[0]:
#             grad_input = grad_output.neg() * alpha_
#         return grad_input, None


# revgrad = RevGrad.apply

# class RevGrad(nn.Module):
#     def __init__(self, alpha=1., *args, **kwargs):
#         """
#         A gradient reversal layer.
#         This layer has no parameters, and simply reverses the gradient
#         in the backward pass.
#         """
#         super().__init__(*args, **kwargs)

#         self._alpha = torch.tensor(alpha, requires_grad=False)

#     def forward(self, input_):
#         out_coordinate_map_key = _get_coordinate_map_key(
#             input_, input_.coordinates
#         )
#         outfeat = revgrad(input_.F, self._alpha)
#         return ME.SparseTensor(
#             outfeat,
#             coordinate_map_key=out_coordinate_map_key,
#             coordinate_manager=input_._manager,
#         )


class DANN(nn.Module):
    num_classes = 5

    def __init__(self, conf):
        super().__init__()

        self.device = torch.device(conf.device)
        self.checkpoint_dir = conf.checkpoint_dir

        self.net = VoxelConvNeXtClassifier(
            self.num_classes, in_chans=1, D=3, dims=conf.net_dims
        ).to(self.device)
        self.net.head = ME.MinkowskiGlobalMaxPooling() # Remove MinkowskiLinear

        self.net_label = nn.Linear(conf.net_dims[-1], self.num_classes).to(self.device)
        self.net_domain = nn.Linear(conf.net_dims[-1], 2).to(self.device)

        self.revgrad = ReverseLayerF
        self.alpha = conf.alpha

        self.optimizer = torch.optim.Adam(self.parameters(), lr=conf.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.95)

        self.criterion_label = nn.CrossEntropyLoss()
        self.criterion_domain = nn.CrossEntropyLoss()

        self.loss = None

        self.data_s = None
        self.s_in_s = None
        self.pred_label_s = None
        self.target_label_s = None
        self.pred_domain_s = None
        self.target_domain_s = None
        self.data_t = None
        self.s_in_t = None
        self.pred_domain_t = None
        self.target_domain_t = None

    def set_input(self, data_s, data_t):
        self._reset_data()
        self.data_s, self.data_t = data_s, data_t
        coords, feats, labels = data_s
        self.target_label_s = labels.to(self.device)
        self.target_domain_s = torch.zeros_like(labels, dtype=labels.dtype, device=self.device)
        self.s_in_s= ME.SparseTensor(
            coordinates=coords, features=feats.float(), device=self.device
        )
        coords, feats, labels = data_t
        self.target_domain_t = torch.ones_like(
            labels, dtype=labels.dtype, device=self.device
        )
        self.s_in_t = ME.SparseTensor(
            coordinates=coords, features=feats.float(), device=self.device
        )

    def set_input_test(self, data):
        self._reset_data()
        self.data_s = data
        coords, feats, labels = data
        self.target_label_s = labels.to(self.device)
        self.s_in_s= ME.SparseTensor(
            coordinates=coords, features=feats.float(), device=self.device
        )

    def eval(self):
        self.net.eval()
        self.net_label.eval()
        self.net_domain.eval()

    def train(self):
        self.net.train()
        self.net_label.train()
        self.net_domain.train()

    def scheduler_step(self):
        self.scheduler.step()

    def save_network(self, suffix):
        torch.save(
            {
                "net_state_dict" : self.net.cpu().state_dict(),
                "net_label_state_dict" : self.net_label.cpu().state_dict(),
                "net_domain_state_dict" : self.net_domain.cpu().state_dict(),
            },
            os.path.join(self.checkpoint_dir, "net_{}.pth".format(suffix))
        )
        self.net.to(self.device)
        self.net_label.to(self.device)
        self.net_domain.to(self.device)

    def load_network(self, path):
        print("Loading model from {}".format(path))
        state_dicts = torch.load(path, map_location=self.device)
        for name in state_dicts:
            if hasattr(state_dicts[name], "_metadata"):
                del state_dicts [name]._metadata
        self.net.load_state_dict(state_dicts["net_state_dict"])
        self.net_label.load_state_dict(state_dicts["net_label_state_dict"])
        self.net_domain.load_state_dict(state_dicts["net_domain_state_dict"])

    def get_current_visuals(self):
        return {
            "s_in_s" : self.s_in_s,
            "target_label_s" : self.target_label_s,
            "pred_label_s" : self.pred_label_s,
            "target_domain_s" : self.target_domain_s,
            "pred_domain_s" : self.pred_domain_s,
            "s_in_t" : self.s_in_t,
            "target_domain_t" : self.target_domain_t,
            "pred_domain_t" : self.pred_domain_t
        }

    def get_current_loss(self):
        return self.loss.item()

    def forward(self):
        featvec = self.net(self.s_in_s)
        rev_featvec = self.revgrad.apply(featvec, self.alpha)
        self.pred_label_s = self.net_label(featvec)
        self.pred_domain_s = self.net_domain(rev_featvec)
        featvec = self.net(self.s_in_t)
        rev_featvec = self.revgrad.apply(featvec, self.alpha)
        self.pred_domain_t = self.net_domain(rev_featvec)

    def forward_test(self):
        featvec = self.net(self.s_in_s)
        self.pred_label_s = self.net_label(featvec)

    def test(self, compute_loss=True):
        with torch.no_grad():
            if self.s_in_t is None or self.data_t is None:
                self.forward_test()
                if compute_loss:
                    loss_label_s = self.criterion_label(self.pred_label_s, self.target_label_s)
                    self.loss = loss_label_s
            else:
                self.forward()
                if compute_loss:
                    loss_label_s = self.criterion_label(self.pred_label_s, self.target_label_s)
                    loss_domain_s = self.criterion_domain(self.pred_domain_s, self.target_domain_s)
                    loss_domain_t = self.criterion_domain(self.pred_domain_t, self.target_domain_t)
                    self.loss = loss_label_s + loss_domain_s + loss_domain_t

    def optimize_parameters(self):
        self.forward()

        self.optimizer.zero_grad()
        loss_label_s = self.criterion_label(self.pred_label_s, self.target_label_s)
        loss_domain_s = self.criterion_domain(self.pred_domain_s, self.target_domain_s)
        loss_domain_t = self.criterion_domain(self.pred_domain_t, self.target_domain_t)
        self.loss = loss_label_s + loss_domain_s + loss_domain_t
        self.loss.backward()
        self.optimizer.step()

    def _reset_data(self):
        self.data_s = None
        self.s_in_s = None
        self.pred_label_s = None
        self.target_label_s = None
        self.pred_domain_s = None
        self.target_domain_s = None
        self.data_t = None
        self.s_in_t = None
        self.pred_domain_t = None
        self.target_domain_t = None
