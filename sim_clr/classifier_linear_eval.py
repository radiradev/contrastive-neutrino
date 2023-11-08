## Large parts of this code are taken from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html

from sim_clr.dataset import CLRDataset
from torch import nn
import os
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import torch
from torch.utils import data
from tqdm import tqdm
from copy import deepcopy
from sim_clr.network import SimCLR
from single_particle_classifier.network_wrapper import SingleParticleModel
from MinkowskiEngine.utils import batch_sparse_collate
from MinkowskiEngine import SparseTensor
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from MinkowskiEngine import MinkowskiGlobalMaxPooling

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 12

@torch.no_grad()
def prepare_data_features(sim_clr, dataset, filename):
    features_path = '/global/homes/r/rradev/contrastive-neutrino/sim_clr/clr_features'
    full_filename = os.path.join(features_path, filename)

    if os.path.exists(full_filename):
        print("Found precomputed features, loading...")
        # Load features
        feats, labels = torch.load(full_filename)
        return data.TensorDataset(feats, labels)
    
    # Prepare model
    network = sim_clr.model
    network.head =  nn.Sequential(
            MinkowskiGlobalMaxPooling())
    network.eval()
    network.to(device)

    # Encode all images
    data_loader = data.DataLoader(dataset, batch_size=512, num_workers=NUM_WORKERS, shuffle=False, drop_last=False, collate_fn=batch_sparse_collate)
    feats, labels = [], []
    with torch.inference_mode():
        for batch_coords, batch_feats, batch_labels in tqdm(data_loader):
            batch_coords = batch_coords.to(device)
            batch_feats = batch_feats.to(device)
            stensor = SparseTensor(features=batch_feats.float(), coordinates=batch_coords)
            out = network(stensor)
            feats.append(out.detach().cpu())

            batch_labels = torch.tensor(batch_labels).long()
            labels.append(batch_labels)

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)

    # Sort images by labels
    labels, idxs = labels.sort()
    feats = feats[idxs]
    
    print(f"Features shape {feats.shape}")
    # Save feature
    torch.save((feats, labels), full_filename)

    return data.TensorDataset(feats, labels)


DATA_PATH = '/pscratch/sd/r/rradev/converted_data/'
CKPT_PATH = '/global/homes/r/rradev/contrastive-neutrino/sim_clr/artifacts/model-oujxdjkk_v3_single_particle_not_overfit'
train_dataset = CLRDataset(root=os.path.join(DATA_PATH, 'train'), dataset_type='single_particle_base')
test_dataset = CLRDataset(root=os.path.join(DATA_PATH, 'test'), dataset_type='single_particle_base')

simclr_model = SimCLR.load_from_checkpoint(checkpoint_path=os.path.join(CKPT_PATH, 'model.ckpt'))
train_feats_simclr = prepare_data_features(simclr_model, train_dataset, filename='train_feats_simclr.pt')
test_feats_simclr = prepare_data_features(simclr_model, test_dataset, filename='test_feats_simclr.pt')

from snapml import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.preprocessing import StandardScaler

clf = LogisticRegression(use_gpu=True, verbose=True)
X = train_feats_simclr.tensors[0].numpy()
y = train_feats_simclr.tensors[1].numpy()

# del train_dataset
# del train_feats_simclr
# del simclr_model

scaler = StandardScaler()

from sklearn.utils import shuffle

X, y = shuffle(X, y, random_state=42)  # Shuffling the data
X = scaler.fit_transform(X)
clf.fit(X, y)
y_pred = clf.predict(scaler.transform(test_feats_simclr.tensors[0].numpy()))
print('Accuracy: ', accuracy_score(test_feats_simclr.tensors[1].numpy(), y_pred))
print('Balanced Accuracy: ', balanced_accuracy_score(test_feats_simclr.tensors[1].numpy(), y_pred))