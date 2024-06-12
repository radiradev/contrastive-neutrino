from functools import partial

import torch

from rotation_conversions import random_rotation

# this must be faster to do when batched
def rotate(coords, feats):
    coords = coords @ random_rotation(dtype=coords.dtype, device=coords.device)
    return coords, feats

def drop(coords, feats, p=0.1):
    mask = torch.rand(coords.shape[0]) > p
    if torch.all(mask).all() == coords.shape[0]: # masking everything out will break forward pass
        return coords, feats
    return coords[mask], feats[mask]

def shift_energy_normal(coords, feats, scale_factor=0.1):
    shift = 1 - torch.randn(1, dtype=feats.dtype, device=feats.device) * scale_factor
    return coords, feats * shift

def shift_energy_uniform(coords, feats, scale_factor=0.1):
    shift = 1 - torch.rand(1, dtype=feats.dtype, device=feats.device) * scale_factor
    return coords, feats * shift

def shift_energy_byvoxel(coords, feats, scale_factor=0.1):
    shift = 1 - torch.randn_like(feats, dtype=feats.dtype, device=feats.device) * scale_factor
    return coords, feats * shift

def translate(coords, feats, cube_size=512):
    normalized_shift = torch.rand(3, dtype=coords.dtype, device=coords.device)
    translation = normalized_shift * (cube_size / 10)
    return coords + translation, feats

def translate_byvoxel(coords, feats, scale_factor=0.3):
    shift = torch.randn_like(coords, dtype=coords.dtype, device=coords.device)
    translation = shift * scale_factor
    return coords + translation, feats

def identity(coords, feats):
    return coords, feats

def get_aug_funcs(energy_scale_factor, translate_scale_factor):
    return {
        "rotate" : rotate,
        "drop" : drop,
        "shift_energy_normal" : partial(shift_energy_normal, scale_factor=energy_scale_factor),
        "shift_energy_uniform" : partial(shift_energy_uniform, scale_factor=energy_scale_factor),
        "shift_energy_byvoxel" : partial(shift_energy_byvoxel, scale_factor=energy_scale_factor),
        "translate" : translate,
        "translate_byvoxel" : partial(translate_byvoxel, scale_factor=translate_scale_factor),
        "identity" : identity
    }
