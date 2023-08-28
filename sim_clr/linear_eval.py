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

from MinkowskiEngine.utils import batch_sparse_collate
from MinkowskiEngine import SparseTensor
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 12

from sklearn.metrics import accuracy_score
import numpy as np

import torchmetrics
multi_acc = torchmetrics.Accuracy(task='multiclass', num_classes=5, average=None).to(device)
 



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
    network.mlp = nn.Identity()# Removing projection head g(.)
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
    
    # Save features
    torch.save((feats, labels), full_filename)

    return data.TensorDataset(feats, labels)


DATA_PATH = '/pscratch/sd/r/rradev/converted_data/'
CKPT_PATH =  None #'/global/homes/r/rradev/contrastive-neutrino/sim_clr/artifacts/model-ow2f68hk_v2_single_paritcle_acc84'


train_dataset = CLRDataset(root=os.path.join(DATA_PATH, 'train'), dataset_type='single_particle_base')
test_dataset = CLRDataset(root=os.path.join(DATA_PATH, 'test'), dataset_type='single_particle_base')

if CKPT_PATH is None:
    print('You are not using a checkpoing for this finetune')
    simclr_model = SimCLR()
else:
    simclr_model = SimCLR.load_from_checkpoint(checkpoint_path=os.path.join(CKPT_PATH, 'model.ckpt'))

train_feats_simclr = prepare_data_features(simclr_model, train_dataset, filename='train_feats_simclr.pt')
test_feats_simclr = prepare_data_features(simclr_model, test_dataset, filename='test_feats_simclr.pt')

from snapml import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.preprocessing import StandardScaler



# def get_smaller_dataset(original_dataset, num_imgs_per_label):
#     # NOT SURE IF THIS IS WORKING CORRECTLY
#     # show the dataset shape
#     new_dataset = data.TensorDataset(
#         *[t.unflatten(0, (5, -1))[:,:num_imgs_per_label].flatten(0, 1) for t in original_dataset.tensors]
#     )
#     return new_dataset

# for num_samples in [10, 50, 100, 500, 1000, 5000, 10_000, 100_000]:
#     small_dataset = get_smaller_dataset(train_feats_simclr, num_samples)
#     print('Small dataset shape: ', small_dataset.tensors[0].shape)

    # train svm 
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
#multi_acc.update(test_feats_simclr.tensors[1].numpy(), y_pred)
print('Accuracy: ', accuracy_score(test_feats_simclr.tensors[1].numpy(), y_pred))
print('Balanced Accuracy: ', balanced_accuracy_score(test_feats_simclr.tensors[1].numpy(), y_pred))
np.save('prediction.npy', y_pred[:10000])
np.save('true.npy', test_feats_simclr.tensors[1].numpy()[:10000])