import torch
import torch.nn.functional as F
from torch.distributed import get_world_size, all_gather

import torch
import torch.nn as nn
import torch.distributed as dist

## The implementation of the SigLip loss is from https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/loss.py

def nt_xent_loss(queries, keys, temperature = 0.1):
    b, device = queries.shape[0], queries.device

    n = b * 2
    projs = torch.cat((queries, keys))
    logits = projs @ projs.t()

    mask = torch.eye(n, device=device).bool()
    logits = logits[~mask].reshape(n, n - 1)
    logits /= temperature

    labels = torch.cat(((torch.arange(b, device=device) + b - 1), torch.arange(b, device=device)), dim=0)
    loss = F.cross_entropy(logits, labels, reduction='sum')
    loss /= n
    return loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = dist.P2POp(
        dist.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = dist.P2POp(
        dist.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = dist.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = dist.P2POp(
        dist.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = dist.P2POp(
        dist.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = dist.P2POp(
        dist.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = dist.P2POp(
        dist.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = dist.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels=False,
            rank=0,
            world_size=1,
            bidir=True,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.bidir = bidir

        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logit_scale = torch.exp(logit_scale)
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )

                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left

        return {"contrastive_loss": loss} if output_dict else loss


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out

def contrastive_loss(x_i, x_j, temperature=0.1, gather_distributed=False):
    """
    Contrastive loss function from bmdillon/JetCLR

    Args:
        x_i (torch.Tensor): Input tensor of shape (batch_size, n_features)
        x_j (torch.Tensor): Input tensor of shape after augmentations (batch_size, n_features)
        temperature (float, optional): Temperature parameter. Defaults to 0.1.
    Returns:
        torch.Tensor: Contrastive loss
    """
    if gather_distributed and get_world_size() == 1:
        raise ValueError("gather_distributed=True but number of processes is 1")
    
    xdevice = x_i.get_device()

    if gather_distributed:
        x_i = torch.cat(GatherLayer.apply(x_i), dim=0)
        x_j = torch.cat(GatherLayer.apply(x_j), dim=0)
        
     
    batch_size = x_i.shape[0]
    z_i = F.normalize(x_i, dim=1 )
    z_j = F.normalize(x_j, dim=1 )
    z   = torch.cat( [z_i, z_j], dim=0 )
    similarity_matrix = F.cosine_similarity( z.unsqueeze(1), z.unsqueeze(0), dim=2 )
    sim_ij = torch.diag( similarity_matrix,  batch_size )
    sim_ji = torch.diag( similarity_matrix, -batch_size )
    positives = torch.cat( [sim_ij, sim_ji], dim=0 )
    nominator = torch.exp( positives / temperature )
    negatives_mask = ( ~torch.eye( 2*batch_size, 2*batch_size, dtype=bool ) ).float()
    negatives_mask = negatives_mask.to( xdevice )
    denominator = negatives_mask * torch.exp( similarity_matrix / temperature )
    loss_partial = -torch.log( nominator / torch.sum( denominator, dim=1 ) )
    loss = torch.sum( loss_partial )/( 2*batch_size )
    return loss



class NT_Xent(nn.Module):
    def __init__(self, batch_size, world_size, temperature=0.1):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples         within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size
        z = torch.cat((z_i, z_j), dim=0)
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss