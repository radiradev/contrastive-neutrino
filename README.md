
# Contrastive Learning for LArTPC
We explore a constrastive learning framework based on [SimCLR](https://arxiv.org/abs/2002.05709), as a method of pretraining and decorrelating from systematic uncertainties and effects related to symmetries within neutrino events. 

This is a two step approach:
- Pretraining phase - we create two augmented versions of events within the batch, then using all $2N$ events, we create a matrix of $(2N)^2$ pairs. Pairs originated from the same event are known as *positive pairs*, while the rest are *negative*. The model tries to get the positive pairs close together, and the negative pairs far apart in the embedding space. 

## Augmentations :recycle:

Currently used: 
- Rotations 
- Translations 
- Energy scale  
- Dropping voxels

*In code:*
```python
def rotate(coords, feats):
    coords = coords @ random_rotation(dtype=coords.dtype, device=coords.device)
    return coords, feats

def drop(coords, feats, p=0.1):
    mask = torch.rand(coords.shape[0]) > p
    return coords[mask], feats[mask]

def shift_energy(coords, feats, max_scale_factor=0.1):
    shift = 1 - torch.rand(1, dtype=feats.dtype, device=feats.device) * max_scale_factor
    return coords, feats * shift

def translate(coords, feats, cube_size=512):
    normalized_shift = torch.rand(3, dtype=coords.dtype, device=coords.device)
    translation = normalized_shift * (cube_size / 10)
    return coords + translation, feats
```

##### Planning to add in the future:

- Local energy scale - shift energy in each voxel independently 
- Cutout (can learn representations that are robust even when particles are close to the edge)
- Smearing the energy 


## Dataset :bar_chart:
- Using the open OSF [dataset](https://osf.io/vruzp/?view_only=). The dataset contains 100k events. [here](https://osf.io/hb437/download)
- Paper describing it [here](https://arxiv.org/pdf/2006.01993.pdf)
- NuTufts [repo](https://github.com/NuTufts/pilarnet_w_larcv1)

We convert the dataset to a single particle per event. It belongs to any of the 5 classes: $e$, $p$, $\gamma$, $\mu$, $\pi$.
#### Converted dataset distribution:


| Particle | Count |
|----------|-------|
| Electron | 291 291|
| Proton   | 212 228|
| Gamma    | 189 227|
| Muon     | 184 699|
| Pion     |  65 248 |

#### Dataset conversion
 - To read in the data we use the [dlp_opendata_api](https://github.com/DeepLearnPhysics/dlp_opendata_api)

## Docker :computer:
Use the `rradev/minkowski:torch1.12_final` image. 
The image uses a custom version of `MinkowskiEngine` with [depthwise convolutions](https://github.com/fededagos/MinkowskiEngine).

Additionally we have to install:

```bash
pip install tensorboardx einops LarpixParser wandb 
pip install pytorch-lightning timm --no-deps
```




## Extra info :books:
- Tutorial on [`lartpc_mlreco3d`](http://stanford.edu/~ldomine/) by Laura
- Tutorial on [ML Reco for DUNE ND](https://indico.fnal.gov/event/50338/)
- `lartpc_mlreco3d` [docs](https://lartpc-mlreco3d.readthedocs.io/)
- Tutorial on [`edep2supera`](https://www.deeplearnphysics.org/edep2supera_tutorials/)


