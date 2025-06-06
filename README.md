
# Contrastive Learning for LArTPC
We explore a constrastive learning framework based on [SimCLR](https://arxiv.org/abs/2002.05709), as a method of pretraining and decorrelating from systematic uncertainties and effects related to symmetries within neutrino events. 

**This repository contains the code used in the paper [arXiv:2502.07724](https://arxiv.org/abs/2502.07724). The code used to produce the paper results is in the `no_lightning/` directory, the top-level code here is from earlier studies on using detector systematic throws as augmentations for the contrastive learning.**

This is a two step approach:
- Pretraining phase - we create two augmented versions of events within the batch, then using all $2N$ events, we create a matrix of $(2N)^2$ pairs. Pairs originated from the same event are known as *positive pairs*, while the rest are *negative*. The model tries to get the positive pairs close together, and the negative pairs far apart in the embedding space. 

## To do 
- [ ]  train contrastive model with augmentations only
- [ ] more detailed comparison between the different throws
- [ ] make a plot of similarity vs a shift in a parameter

## Training 
There two options, training the contrastive learning model or the direct classifier: 

`python3 train.py` by default trains the contrastive learning model

`python3 train.py --dataset_type single_particle --model SingleParticle` trains the direct classifier model

## Docker installation :computer:
Use the `rradev/minkowski:torch1.12_final` image. 
The image uses a custom version of `MinkowskiEngine` with [depthwise convolutions](https://github.com/fededagos/MinkowskiEngine).

Additionally we have to install:

```bash
pip install tensorboardx einops LarpixParser wandb 
pip install pytorch-lightning timm --no-deps
```

## Throws Dataset 
We vary 3 detector systematics parameters taken from the [paper](https://arxiv.org/pdf/2309.04639.pdf) from SLAC. The parameters are the electron lifetime, longitudinal diffusion and electric field strength.

| Parameter | Units      | Nominal Value | Range                 |
|-----------|------------|---------------|-----------------------|
| E         | kV/cm      | 0.5           | [0.45, 0.55]          |
| τ         | µs         | 2200          | [500, 5000]           |
| Dt        | cm²/µs     | 8.8 × 10⁻⁶     | [4 × 10⁻⁶, 14 × 10⁻⁶]    |


We also have 2000 events with fixed values of the throws:
| Throw Number | Description                               |
|------|-------------------------------------------|
| 1    | All positive max                          |
| 2    | All negative max                          |
| 3    | Efield positive max, others nominal       |
| 4    | Trans diffusion positive max, others nominal |
| 5    | Lifetime positive max, others nominal     |
| 6    | Efield negative max, others nominal       |
| 7    | Trans diffusion negative max, others nominal |
| 8    | Lifetime negative max, others nominal     |
| 9    | All 1/4x positive max                     |
| 10   | All 1/4x negative max                     |
| 11   | All 1/2x positive max                     |
| 12   | All 1/2x negative max                     |
| 13   | All 3/4x positive max                     |
| 14   | All 3/4x negative max                     |
| 15   | All nominal                               |
| 16   | All 3x of positive max                    |
| 17   | All 3x of negative max                    |


More info about the generation can be found in my [dune-nd-detector-sim](https://github.com/radiradev/dune-nd-detector-sim) repo.

### Directories
Generation `edep-sim` -> `larnd-sim` -> `.npz` model input

- `edep-sim` - `/wclustre/dune/rradev/larnd-contrast/individual_particles` There are 250 files per particle type in both `.h5` and `root` format. 

- `larnd-sim` - `/wclustre/dune/awilkins/contrastive_learning/larndsim_10throws_5particles_125500eventseven.tar.gz`

On NERSC everything is available in:
```/global/cfs/cdirs/dune/users/rradev/contrastive/individual_particles``` 

with the edeps in `edeps-h5` and `edep-root` and the larnd-sim files in `larndsim-throws`.

The converted `.npz` files are also available on scratch at `larndsim_throws_converted_new`, this should be used for training as IO from scratch would be faster than from CFS.

### Recreate the dataset 
Use the `larnd.convert_data.py` script to convert the files to `.npz`, you may to adjust the input and output filepaths. It will split the data using files up to number 230 as training, 230 < n < 240 -validation and n > 240 for testing. It will also filter out files with 3 voxels or less.


# PiLArNet Method

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


## Extra info :books:
##### Particle Bomb 
DLPGenerator [Tutorial](https://www.deeplearnphysics.org/DLPGenerator/)
ND Simulation [How To](https://hackmd.io/@CuhPVDY3Qregu7G4lr1p7A/H1d1Zj4zi)
DUNE ND LAr Sim [Tutorial](https://github.com/sam-fogarty/simulation-tutorial_DUNE-ND-LAr)
##### Other

- Tutorial on [`lartpc_mlreco3d`](http://stanford.edu/~ldomine/) by Laura
- Tutorial on [ML Reco for DUNE ND](https://indico.fnal.gov/event/50338/)
- `lartpc_mlreco3d` [docs](https://lartpc-mlreco3d.readthedocs.io/)
- Tutorial on [`edep2supera`](https://www.deeplearnphysics.org/edep2supera_tutorials/)


