from enum import Enum
from collections import deque

import numpy as np

import torch
import torchvision

from MinkowskiEngine.utils import sparse_quantize

class ThrowsDataset(torchvision.datasets.DatasetFolder):
    def __init__(
        self, dataroot, dataset_type, augs, n_augs, quantization_size, xtalk,
        extensions='.npz', train_mode=True
    ):
        super().__init__(root=dataroot, extensions=extensions, loader=self.loader)
        self.dataset_type = dataset_type
        self.train_mode = train_mode
        self.augs = augs
        self.n_augs = min(n_augs, len(self.augs))
        self.quantization_size = quantization_size
        self.xtalk = xtalk

        self.index_history = deque(self.__len__() * [0], self.__len__())

    def loader(self, path):
        return np.load(path)

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

        coords_i, feat_i = sparse_quantize(
            coords_i, feat_i, quantization_size=self.quantization_size
        )
        coords_j, feat_j = sparse_quantize(
            coords_j, feat_j, quantization_size=self.quantization_size
        )

        return (coords_i, feat_i), (coords_j, feat_j)

    def augment_single(self, coords, feats):
        funcs = self.augs
        funcs = np.random.choice(funcs, self.n_augs)
        for func in funcs:
            coords, feats = func(coords, feats)
        return coords, feats

    def filter_crosstalk(hit_mask):
        nb_xtalk = np.sum(hit_mask == False)
        new_nb_xtalk = int(nb_xtalk * self.xtalk)
        xtalk_indices = np.where(hit_mask == False)[0]
        random_indices = np.random.choice(xtalk_indices, new_nb_xtalk, replace=False)
        hit_mask[random_indices] = True

    def __getitem__(self, index: int):
        self.index_history.append(index)
        path, label = self.samples[index]
        sample = self.loader(path)

        # Working with segmentedcube data where we vary the crosstalk as a throw
        if self.xtalk is not None:
            reco_hits = sample["reco_hits"]
            hit_mask = sample["Tag_Trk"]
            if self.xtalk < 1.0:
                filter_crosstalk(hit_mask)
                reco_hits = reco_hits[hit_mask]
            coords, feats = reco_hits[:, :3], reco_hits[:, 3].reshape(-1, 1)
        else:
            coords, feats = sample["coordinates"], sample["adc"]

        if (
            self.dataset_type == DataPrepType.CLASSIFICATION or
            self.dataset_type == DataPrepType.CLASSIFICATION_AUG
        ):
            if self.train_mode and self.dataset_type == DataPrepType.CLASSIFICATION_AUG:
                coords, feats = self.augment_single(
                    torch.tensor(coords, dtype=torch.float), torch.tensor(feats, dtype=torch.float)
                )
            coords, feats = sparse_quantize(
                coords, np.expand_dims(feats, axis=1), quantization_size=self.quantization_size
            )
            return coords, feats, torch.tensor(label).long().unsqueeze(0)

        elif (
            self.dataset_type == DataPrepType.CONTRASTIVE_AUG or
            self.dataset_type == DataPrepType.CONTRASTIVE_AUG_LABELS
        ):
            feats = np.expand_dims(feats, axis=1) # dont know why I am doing this
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
