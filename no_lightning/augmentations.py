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

def shift_energy_normal(coords, feats, max_scale_factor=0.1):
    shift = 1 - torch.randn(1, dtype=feats.dtype, device=feats.device) * max_scale_factor
    return coords, feats * shift

def shift_energy_uniform(coords, feats, max_scale_factor=0.1):
    shift = 1 - torch.rand(1, dtype=feats.dtype, device=feats.device) * max_scale_factor
    return coords, feats * shift

def translate(coords, feats, cube_size=512):
    normalized_shift = torch.rand(3, dtype=coords.dtype, device=coords.device)
    translation = normalized_shift * (cube_size / 10)
    return coords + translation, feats

def identity(coords, feats):
    return coords, feats

aug_funcs = {
    "rotate" : rotate,
    "drop" : drop,
    "shift_energy_normal" : shift_energy_normal,
    "shift_energy_uniform" : shift_energy_uniform,
    "translate" : translate,
    "identity" : identity
}
