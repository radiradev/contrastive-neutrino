## Large parts of this code are taken from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html
## mostly copied from sim_clr/linear_eval
import os
import pytorch_lightning as pl
import torch
import joblib

from utils.data import load_yaml
from torch.utils import data
from single_particle_classifier.network_wrapper import SingleParticleModel
from sim_clr.network import SimCLR
from sim_clr.dataset import ThrowsDataset
from torch import nn
from torch.utils.data import Subset
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from MinkowskiEngine.utils import batch_sparse_collate
from MinkowskiEngine import SparseTensor
from utils.data import get_wandb_ckpt
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 12 
config = load_yaml('config/config.yaml')

def filter_dataset(dataset, class_name):
    # Find the index of the target class
    target_class_index = dataset.class_to_idx[class_name]

    # Filter out indices of the target class
    target_indices = [i for i, (_, _, label) in enumerate(dataset) if label == target_class_index]
    return Subset(dataset, target_indices)

def predictions(loader, model, target_index):
    preds = []
    for batch in loader:
        pred, label = model.predict_(batch)
        pred = pred.argmax(dim=1)
        preds.append(pred)
    
    preds = torch.hstack(preds)
    labels = torch.ones_like(preds) * target_index #need to revise this once we do multiple classes
    return preds, labels

def sim_clr_predictions(loader, sim_clr, bdt, scaler):
    preds, labels = [], []
    for batch in loader:
        batch_coords, batch_feats, label = batch
        batch_coords = batch_coords.to(device)
        batch_feats = batch_feats.to(device)
        stensor = SparseTensor(features=batch_feats.float(), coordinates=batch_coords)
        features = sim_clr(stensor).cpu().numpy()
        features = scaler.transform(features)
        labels.append(label)

        pred = bdt.predict(features)
        preds.append(pred)

    labels = np.hstack(labels)
    preds = np.hstack(preds)
     #need to revise this once we do multiple classes
    return preds, labels
    
    
def evaluate(wandb_artifact=None, model_type='single_particle', particle_name=None):
    dataset_type = 'single_particle'
    thrown_dataset = ThrowsDataset(dataset_type, config['data']['all_max'])
    nominal_dataset = ThrowsDataset(dataset_type, os.path.join(os.path.dirname(config['data']['data_path']), 'larndsim_throws_converted_nominal', 'test'))
    if particle_name is None:
        print("No particle name provided, using proton")
        particle_name = 'proton'
    particle_label = nominal_dataset.class_to_idx[particle_name]
    nominal_dataset = filter_dataset(nominal_dataset, particle_name)
    thrown_dataset = filter_dataset(thrown_dataset, particle_name)
    
    print("Length of thrown dataset", len(thrown_dataset))
    print("Length of nominal dataset", len(nominal_dataset))
    
    
    collate_fn = batch_sparse_collate
    num_workers = 12
    batch_size = 256
    thrown_loader = data.DataLoader(thrown_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, drop_last=True)
    nominal_loader = data.DataLoader(nominal_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, drop_last=True)
    ckpt_path, artifact_name = get_wandb_ckpt(wandb_artifact, return_name=True)
    
    if model_type=='single_particle':
        model = SingleParticleModel.load_from_checkpoint(ckpt_path).cuda()
        nominal_predictions, nominal_labels = predictions(nominal_loader, model, particle_label)
        thrown_predictions, thrown_labels = predictions(thrown_loader, model, particle_label)
    else: 
        clr_model = SimCLR.load_from_checkpoint(ckpt_path).cuda()
        network = clr_model.model
        network.mlp = nn.Identity() # Removing projection head g(.)
        network.eval()
        network.to(device)

        bdt_path = '/global/homes/r/rradev/contrastive-neutrino/trained_bdt.pkl'
        bdt = joblib.load(bdt_path)
        scaler_path = '/global/homes/r/rradev/contrastive-neutrino/scaler.pkl'
        scaler = joblib.load(scaler_path)
        
        with torch.inference_mode():
            nominal_predictions, nominal_labels = sim_clr_predictions(nominal_loader, network, bdt, scaler)
            thrown_predictions, thrown_labels = sim_clr_predictions(thrown_loader, network, bdt, scaler)
                
        
    sample_size = 1000
    if model_type=='single_particle':
        nominal_labels, nominal_predictions = nominal_labels.cpu().numpy(), nominal_predictions.cpu().numpy()
        thrown_labels, thrown_predictions = thrown_labels.cpu().numpy(), thrown_predictions.cpu().numpy()
    
    nominal_acc = accuracy_score(nominal_labels[:sample_size], nominal_predictions[:sample_size])
    thrown_acc = accuracy_score(thrown_labels[:sample_size], thrown_predictions[:sample_size])

    print(f'Nominal accuracy for {particle_name}', nominal_acc)
    print(f'Shifted Efield accuracy {particle_name}', thrown_acc)

def evaluate_particles(wandb_artifact=None, model_type='single_particle', particle_list=['gamma']):#['proton', 'pion', 'muon', 'electron']):
    for particle in particle_list:
        evaluate(wandb_artifact, model_type, particle)
                       

if __name__ == '__main__':
    import fire
    fire.Fire(evaluate_particles)


