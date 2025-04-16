import os
from enum import Enum
from collections import deque

import numpy as np

import torch
import torchvision
from MinkowskiEngine.utils import sparse_quantize

from rotation_conversions import random_rotation

class ThrowsDataset(torchvision.datasets.DatasetFolder):
    def __init__(
        self, dataroot, dataset_type, augs, n_augs, quantization_size, xtalk,
        extensions=None, train_mode=True
    ):
        extensions = self.guess_extensions(dataroot) if extensions is None else extensions
        super().__init__(root=dataroot, extensions=extensions, loader=self.loader)
        self.dataset_type = dataset_type
        self.train_mode = train_mode
        self.augs = augs
        self.n_augs = min(n_augs, len(self.augs))
        self.quantization_size = quantization_size
        self.xtalk = xtalk

        self.index_history = deque(self.__len__() * [0], self.__len__())

    def guess_extensions(self, dataroot):
        for _, _, fnames in os.walk(dataroot):
            if fnames:
                return "." + fnames[0].split(".")[-1]

    def loader(self, path):
        if self.extensions == ".npz":
            return np.load(path)
        else:
            return torch.load(path)

    def contrastive_augmentations(self, xi, xj):
        funcs = self.augs

        coords_i, feat_i = xi
        coords_j, feat_j = xj

        # draw functions and augment i
        funcs_i = np.random.choice(funcs, self.n_augs)
        funcs_j = np.random.choice(funcs, self.n_augs)

        for func in funcs_i:
            coords_i, feat_i = func(coords_i, feat_i)

        for func in funcs_j:
            coords_j, feat_j = func(coords_j, feat_j)

        coords_i, feat_i = self.safe_sparse_quantize(coords_i, feat_i)
        coords_j, feat_j = self.safe_sparse_quantize(coords_j, feat_j)

        return (coords_i, feat_i), (coords_j, feat_j)

    def augment_single(self, coords, feats):
        funcs = self.augs
        funcs = np.random.choice(funcs, self.n_augs)
        for func in funcs:
            coords, feats = func(coords, feats)
        return coords, feats

    def filter_crosstalk(self, hit_mask):
        nb_xtalk = np.sum(hit_mask == False)
        new_nb_xtalk = int(nb_xtalk * self.xtalk)
        xtalk_indices = np.where(hit_mask == False)[0]
        random_indices = np.random.choice(xtalk_indices, new_nb_xtalk, replace=False)
        hit_mask[random_indices] = True

    def randomise_segmentedcube(self, coords, vtx):
        """
        The segmented cube dataset initial vertex and direction is not properly randomised.
        Randomised this as we load the data
        """
        coords -= vtx
        coords = coords @ random_rotation(dtype=coords.dtype, device=coords.device) # new direction
        coords += 2000 * torch.rand(3) - 1000 # new vtx in 1000^3 box
        return coords.numpy()

    def safe_sparse_quantize(self, coords, feats):
        coords, feats = sparse_quantize(coords, feats, quantization_size=self.quantization_size)
        if len(feats.shape) == 1: # Still need feature dimension for single voxel image
            if isinstance(feats, np.ndarray):
                feats = np.expand_dims(feats, axis=0)
            else:
                feats = torch.unsqueeze(feats, 0)
        return coords, feats

    def __getitem__(self, index: int):
        self.index_history.append(index)
        path, label = self.samples[index]
        sample = self.loader(path)

        # Working with segmentedcube data where we vary the crosstalk as a throw
        if self.xtalk is not None:
            reco_hits = sample["reco_hits"]
            hit_mask = sample["Tag_Trk"]
            vtx = sample["TrueIniPos"]
            if self.xtalk < 1.0:
                self.filter_crosstalk(hit_mask)
                reco_hits = reco_hits[hit_mask]
            coords, feats = reco_hits[:, :3], reco_hits[:, 3].reshape(-1, 1)
            coords = self.randomise_segmentedcube(
                torch.tensor(coords, dtype=torch.float), torch.tensor(vtx, dtype=torch.float)
            )
        else: # larnd-sim data
            coords, feats = sample["coordinates"], np.expand_dims(sample["adc"], axis=1)

        if (
            self.dataset_type == DataPrepType.CLASSIFICATION or
            self.dataset_type == DataPrepType.CLASSIFICATION_AUG
        ):
            if self.train_mode and self.dataset_type == DataPrepType.CLASSIFICATION_AUG:
                coords, feats = self.augment_single(
                    torch.tensor(coords, dtype=torch.float), torch.tensor(feats, dtype=torch.float)
                )
            coords, feats = self.safe_sparse_quantize(coords, feats)
            return coords, feats, torch.tensor(label).long().unsqueeze(0)

        elif (
            self.dataset_type == DataPrepType.CONTRASTIVE_AUG or
            self.dataset_type == DataPrepType.CONTRASTIVE_AUG_LABELS
        ):
            xi = (torch.tensor(coords), torch.tensor(feats, dtype=torch.float))
            xj = (torch.tensor(coords), torch.tensor(feats, dtype=torch.float))
            xi, xj = self.contrastive_augmentations(xi, xj)
            if self.dataset_type == DataPrepType.CONTRASTIVE_AUG:
                return xi, xj
            return xi, xj, torch.tensor(label).long().unsqueeze(0)


class DataPrepType(Enum):
    CONTRASTIVE_AUG = 1
    CLASSIFICATION = 2
    CLASSIFICATION_AUG = 3
    CONTRASTIVE_AUG_LABELS = 4
