import torch
import torch.nn.functional as F
from torch.distributed import get_world_size, all_gather

import torch
import torch.nn as nn
import torch.distributed as dist

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

def contrastive_loss_class_labels(
    x_i, x_j, labels, temperature=0.1, gather_distributed=False, same_label_weight=0.5
):
    """
    Contrastive loss function from bmdillon/JetCLR

    Uses class labels to define positive pairs.

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
    z = torch.cat( [z_i, z_j], dim=0 )
    similarity_matrix = F.cosine_similarity( z.unsqueeze(1), z.unsqueeze(0), dim=2 )

    # 0.5 for same class pairs, 1.0 for same image pairs
    labels = torch.cat([labels, labels], dim=0)
    positives_mask = (labels[:, None] == labels[None, :]).float() * same_label_weight
    ids = torch.cat([torch.arange(batch_size), torch.arange(batch_size)], dim=0)
    positives_mask += (ids[:, None] == ids[None, :]).float() * (1.0 - same_label_weight)
    positives_mask *= (~torch.eye(2 * batch_size, 2 * batch_size, dtype=bool)).float()
    positives_mask = positives_mask.to(xdevice)
    nominator = positives_mask * torch.exp(similarity_matrix / temperature)

    # NOTE Do I need to include the same 0.5 weighting for the class pairs in the denominator too?
    # If I dont the loss minimum is no longer zero, not sure if that is really going to matter.
    # If I did I would need to tweak the negatives mask to be 0.5 for the class pairs
    negatives_mask = ( ~torch.eye( 2*batch_size, 2*batch_size, dtype=bool ) ).float()
    negatives_mask = negatives_mask.to( xdevice )
    denominator = negatives_mask * torch.exp( similarity_matrix / temperature )

    loss_partial = -torch.log( torch.sum(nominator, dim=1) / torch.sum( denominator, dim=1 ) )
    loss = torch.sum( loss_partial )/( 2*batch_size )

    return loss

def contrastive_loss_class_labels_out(
    x_i, x_j, labels, temperature=0.1, gather_distributed=False, same_label_weight=0.5
):
    """
    Contrastive loss function from bmdillon/JetCLR

    Uses class labels to define positive pairs.

    Formulated in the same way as L^sup_out of eq 2 in arXiv:2004.11362.

    Args:
        x_i (torch.Tensor): Input tensor of shape (batch_size, n_features)
        x_j (torch.Tensor): Input tensor of shape after augmentations (batch_size, n_features)
        temperature (float, optional): Temperature parameter. Defaults to 0.1.
    Returns:
        torch.Tensor: Contrastive loss
    """
    eps = 1e-7

    if gather_distributed and get_world_size() == 1:
        raise ValueError("gather_distributed=True but number of processes is 1")

    xdevice = x_i.get_device()

    if gather_distributed:
        x_i = torch.cat(GatherLayer.apply(x_i), dim=0)
        x_j = torch.cat(GatherLayer.apply(x_j), dim=0)

    batch_size = x_i.shape[0]
    z_i = F.normalize(x_i, dim=1 )
    z_j = F.normalize(x_j, dim=1 )
    z = torch.cat( [z_i, z_j], dim=0 )
    similarity_matrix = F.cosine_similarity( z.unsqueeze(1), z.unsqueeze(0), dim=2 )

    # 0.5 for same class pairs, 1.0 for same image pairs
    labels = torch.cat([labels, labels], dim=0)
    print(labels)
    print()
    positives_mask = (labels[:, None] == labels[None, :]).float() * same_label_weight
    ids = torch.cat([torch.arange(batch_size), torch.arange(batch_size)], dim=0)
    positives_mask += (ids[:, None] == ids[None, :]).float() * (1.0 - same_label_weight)
    positives_mask *= (~torch.eye(2 * batch_size, 2 * batch_size, dtype=bool)).float()
    positives_mask = positives_mask.to(xdevice)
    nominator = positives_mask * torch.exp(similarity_matrix / temperature)

    # NOTE Do I need to include the same 0.5 weighting for the class pairs in the denominator too?
    # If I dont the loss minimum is no longer zero, not sure if that is really going to matter.
    # If I did I would need to tweak the negatives mask to be 0.5 for the class pairs
    negatives_mask = ( ~torch.eye( 2*batch_size, 2*batch_size, dtype=bool ) ).float()
    negatives_mask = negatives_mask.to( xdevice )
    denominator = negatives_mask * torch.exp( similarity_matrix / temperature )

    summand = nominator / torch.sum(denominator, dim=1)
    summand = torch.where(summand < eps, 1.0, summand) # So the log doesnt change the zeros
    summand = torch.log(summand)
    positive_pair_cardinality = torch.count_nonzero(summand, dim=1)
    loss_partial = torch.sum(summand, dim=1)
    loss = torch.sum((-1 / positive_pair_cardinality) * loss_partial) / (2 * batch_size)

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
