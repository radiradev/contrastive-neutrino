## Large parts of this code are taken from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html
## mostly copied from sim_clr/linear_eval
import os
import pytorch_lightning as pl
import torch
import joblib
import datetime

from torch.utils import data
from modules.regressor import Regressor
from modules.simclr import SimCLR
from data.regression import Regression
from torch import nn
from sklearn.metrics import r2_score, mean_absolute_error
from MinkowskiEngine.utils import batch_sparse_collate
from MinkowskiEngine import SparseTensor
from utils.data import get_wandb_ckpt, load_yaml
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 12 
config = load_yaml('config/config.yaml')

def regressor_predict(loader, model):
    preds, labels = [], []
    with torch.inference_mode():
        for batch in loader:
            pred, label = model.predict_(batch)
            preds.append(pred)
            labels.append(label)
        
        preds = torch.hstack(preds)
        labels = torch.hstack(labels) 
    return preds.cpu().numpy(), labels.cpu().numpy()

def sim_clr_predict(loader, clr_collection):
    sim_clr = clr_collection['network']
    scaler = clr_collection['scaler']
    sk_model = clr_collection['bdt']

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
            pred = sk_model.predict(features)
            preds.append(pred)

        labels = np.hstack(labels)
        preds = np.hstack(preds)
    return preds, labels

def wandb_models(model_names=None):
    """Returns a dictionary with the paths of models from
    the wandb registry"""
    if model_names is None:
        model_names = [
            'contrastive-model-augmentations-and-throws:v1',
            'contrastive-augmentations:v3',
            'regressor-energy:v0',
        ]
    print(f'Loading models: {model_names}')
    models = {
        name: get_wandb_ckpt(f"rradev/model-registry/{name}") for name in model_names 
    }
    return models

def load_clr_model(ckpt_path, name):
    clr = SimCLR.load_from_checkpoint(ckpt_path).cuda()
    network = clr.model
    network.mlp = nn.Identity() # Removing projection head g(.)
    network.eval()
    network.to(device)

    #load bdt and scaler 
    features_path = os.path.join(os.environ['PSCRATCH'], 'linear-eval-contrastive')
    load_path = os.path.join(features_path, f"{name}_regression.pkl")

    if not os.path.exists(load_path):
        raise Exception(f'Could not find {load_path}. You must run linear_eval.py first.')

    bdt, scaler = joblib.load(load_path) 
    print(f'loaded model for {name} from {load_path}')
    return {
        'network': network,
        'bdt': bdt,
        'scaler': scaler
    }

def load_models(model_names=None):
    # load all the models
    models_paths = wandb_models(model_names)
    model_names = list(models_paths.keys())
    models = {}
    for name in model_names:
        if 'regressor' in name:
            models[name] = Regressor.load_from_checkpoint(models_paths[name], strict=False).cuda()
        else:
            models[name] = load_clr_model(models_paths[name], name)
    return models

def evaluate_models(models, throw_type):

    data_path = os.path.join(config['data']['throws'],'larndsim_converted', throw_type)
    dataset = Regression(root=data_path, energies='particle_energy_throws.pkl')
    loader = data.DataLoader(dataset, batch_size=256, collate_fn=batch_sparse_collate, num_workers=12, drop_last=True, shuffle=True)    
    
    results = {}
    for model_name in models.keys():
        if 'regressor' in model_name:
            preds, labels = regressor_predict(loader, models[model_name])
        else:
            preds, labels = sim_clr_predict(loader, models[model_name])
        print(labels.shape, preds.shape)
        r2 = r2_score(labels, preds)
        mae = mean_absolute_error(labels, preds)
        print(f'{throw_type}_{model_name} R2 score: {r2}')
        print(f'{model_name} Mean Absolute Error: {mae}')
        results[model_name] = {
            'preds': preds,
            'labels': labels
        }
    return results
    

if __name__ == '__main__':
    models = load_models()
    print('Models loaded', models.keys()) 
    results = {}
    for throw in config['throws'].values():
        print(f'Processing {throw}')
        results[throw] = evaluate_models(models, throw)
    
    # get current date
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    joblib.dump(results, f'/global/homes/r/rradev/contrastive-neutrino/test/regression/results/results_{date}.pkl')
    



