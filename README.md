# Dataset
- Using the open OSF [dataset](https://osf.io/vruzp/?view_only=). The dataset contains 100k events. [here](https://osf.io/hb437/download)
- Paper describing it [here](https://arxiv.org/pdf/2006.01993.pdf)
- NuTufts [repo](https://github.com/NuTufts/pilarnet_w_larcv1)

## Converted dataset distribution:

Electron: 291291
Gamma: 189227
Muon: 184699
Pion: 65248
Proton: 212228


# Docker
Starting from the docker image [here](https://hub.docker.com/layers/deeplearnphysics/larcv2/ub20.04-cuda11.6-pytorch1.13-larndsim/images/sha256-afe799e39e2000949f3f247ab73fe70039fb411cb301cb3c78678b68c22e37fb?context=explore)

Adding custom packages described below

# Packages
 - To read in the data we use the [dlp_opendata_api](https://github.com/DeepLearnPhysics/dlp_opendata_api)
 - We use `MinkowskiEngine` with depthwise convolutions from this [fork](https://github.com/fededagos/MinkowskiEngine)
 - we also use `timm` and `tensorboardx`


# Extra info 
- Tutorial on [`lartpc_mlreco3d`](http://stanford.edu/~ldomine/) by Laura
- Tutorial on [ML Reco for DUNE ND](https://indico.fnal.gov/event/50338/)
- `lartpc_mlreco3d` [docs](https://lartpc-mlreco3d.readthedocs.io/)
- Tutorial on [`edep2supera`](https://www.deeplearnphysics.org/edep2supera_tutorials/)

# Learning Representations
- Rotation
- Smearing the energy 
- Masking out voxels
- Changing the energy scale - remove a certain percentage of the energy value within the event.