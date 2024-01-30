## Large parts of this code are taken from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html
## mostly copied from sim_clr/linear_eval
import os
import pytorch_lightning as pl
import torch
import joblib

from torch.utils import data
from single_particle_classifier.network_wrapper import SingleParticleModel
from sim_clr.network import SimCLR
from sim_clr.dataset import ThrowsDataset
from torch import nn
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from MinkowskiEngine.utils import batch_sparse_collate
from MinkowskiEngine import SparseTensor
from utils.data import get_wandb_ckpt, load_yaml
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 12 
config = load_yaml('config/config.yaml')

def classifier_predict(loader, model):
    preds, labels = [], []
    with torch.inference_mode():
        for batch in loader:
            pred, label = model.predict_(batch)
            preds.append(pred)
            labels.append(label)
        
        preds = torch.vstack(preds)
        labels = torch.hstack(labels) 
    return preds.cpu().numpy(), labels.cpu().numpy()

def sim_clr_predict(loader, sim_clr):
    # these should not be hardcoded
    sk_model = joblib.load('/global/homes/r/rradev/contrastive-neutrino/bdt_contrastive-model-augmentations-and-throws:v0_no_mlp.pkl')
    scaler = joblib.load('/global/homes/r/rradev/contrastive-neutrino/scaler_contrastive-model-augmentations-and-throws:v0_no_mlp.pkl')
    with torch.inference_mode():
        preds, labels = [], []
        for batch in loader:
            batch_coords, batch_feats, label = batch
            batch_coords = batch_coords.to(device)
            batch_feats = batch_feats.to(device)
            stensor = SparseTensor(features=batch_feats.float(), coordinates=batch_coords)
            features = sim_clr(stensor).cpu().numpy()
            features = scaler.transform(features)
            labels.append(label)

            pred = sk_model.predict_proba(features)
            preds.append(pred)

        labels = np.hstack(labels)
        preds = np.vstack(preds)
    return preds, labels

def latest_wandb_models(model_names=None):
    """Returns a dictionary with the paths of the latest from
    the wandb registry"""
    if model_names is None:
        model_names = [
            'contrastive-model-augmentations-and-throws',
            # 'classifier-augmentations-throws'
            # 'classifier-nominal-only',
            # 'classifier-throws-dataset'
        ]
    print(f'Loading models: {model_names}')
    models = {
        name: get_wandb_ckpt(f"rradev/model-registry/{name}:latest") for name in model_names
    }
    return models

def load_clr_model(ckpt_path):
    clr = SimCLR.load_from_checkpoint(ckpt_path).cuda()
    network = clr.model
    network.mlp = nn.Identity() # Removing projection head g(.)
    network.eval()
    network.to(device)
    return network

def load_models(model_names=None):
    # load all the models
    models_paths = latest_wandb_models(model_names)
    model_names = list(models_paths.keys())
    models = {}
    for name in model_names:
        if 'classifier' in name:
        # Action for the specific model
            models[name] = SingleParticleModel.load_from_checkpoint(models_paths[name]).cuda()
        else:
        # Alternative action for other models
            models[name] = load_clr_model(models_paths[name])
    return models

def evaluate_models(models, throw_type):

    data_path = os.path.join(config['data']['2k_particles'],'larndsim_converted', throw_type)
    dataset = ThrowsDataset(dataset_type='single_particle', root=data_path)
    loader = data.DataLoader(dataset, batch_size=256, collate_fn=batch_sparse_collate, num_workers=12, drop_last=True)    
               
    sample_size = 1792 # batch_size * 7 not all events present in the larnd data
    results = {}
    for model_name in models.keys():
        if 'classifier' in model_name:
            preds, labels = classifier_predict(loader, models[model_name])
        else:
            preds, labels = sim_clr_predict(loader, models[model_name])
        print(labels.shape, preds.shape)
        acc = accuracy_score(labels, preds.argmax(axis=1))
        bacc = balanced_accuracy_score(labels, preds.argmax(axis=1))
        print(f'{throw_type}_{model_name} accuracy: {acc}')
        print(f'{model_name} balanced accuracy: {bacc}')
        results[model_name] = {
            'preds': preds,
            'labels': labels
        }
    return results
    

if __name__ == '__main__':
    models = load_models()
    print('Models loaded', models.keys())
    results = {}
    data_path = os.path.join(config['data']['nominal_data_path'], 'test')
    dataset = ThrowsDataset(dataset_type='single_particle', root=data_path)
    loader = data.DataLoader(dataset, batch_size=256, collate_fn=batch_sparse_collate, num_workers=12, drop_last=True)    
               
    sample_size = 1792 # batch_size * 7 not all events present in the larnd data
    results = {}
    for model_name in models.keys():
        if 'classifier' in model_name:
            preds, labels = classifier_predict(loader, models[model_name])
        else:
            preds, labels = sim_clr_predict(loader, models[model_name])
        print(labels.shape, preds.shape)
        acc = accuracy_score(labels, preds.argmax(axis=1))
        bacc = balanced_accuracy_score(labels, preds.argmax(axis=1))
        print(f'{"Test data throws"}_{model_name} accuracy: {acc}')
        print(f'{model_name} balanced accuracy: {bacc}')
        results[model_name] = {
            'preds': preds,
            'labels': labels
        }

    for throw in config['throws'].values():
        print(f'Processing {throw}')
        results[throw] = evaluate_models(models, throw)
    joblib.dump(results, 'results_mlp.pkl')
    



