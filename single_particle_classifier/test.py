import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from MinkowskiEngine.utils import batch_sparse_collate
from sim_clr.dataset import CLRDataset
import torchmetrics
from MinkowskiEngine import SparseTensor
from single_particle_classifier.network_wrapper import SingleParticleModel
from tqdm import tqdm
# Initialize metrics
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=5).to(device)
multi_acc = torchmetrics.Accuracy(task='multiclass', num_classes=5, average=None).to(device)
criterion = torch.nn.CrossEntropyLoss()  # Assuming you're using CrossEntropy for your model

# Load the model
ckpt_path = '/global/homes/r/rradev/contrastive-neutrino/sim_clr/artifacts/model-oujxdjkk_v3_single_particle_not_overfit/model.ckpt'
model = SingleParticleModel.load_from_checkpoint(ckpt_path)
model.eval()

# Prepare DataLoader
dataset = CLRDataset(dataset_type='single_particle_base', root='/pscratch/sd/r/rradev/converted_data/test')
dataloader = DataLoader(dataset, batch_size=128, shuffle=True,collate_fn=batch_sparse_collate)

# Initialize device
model.to(device)
test_loss = 0.0

alphas = np.linspace(-0.1, 0.1, 20)

acc_list = []
alpha_list = []
for alpha in alphas:
    for idx, batch in enumerate(tqdm(dataloader)):
        batch_coords, batch_feats, target = batch
        target = target.to(device)
        batch_coords = batch_coords.to(device)
        batch_feats = batch_feats * (1 + alpha)
        batch_feats = batch_feats.to(device)
        stensor = SparseTensor(features=batch_feats.float(), coordinates=batch_coords)

        # Forward pass
        with torch.no_grad():
            output = model(stensor)

        # Compute loss
        loss = criterion(output, target)
        test_loss += loss.item()

        # Compute metrics
        preds = torch.argmax(output, dim=1)
        accuracy.update(preds, target)
        multi_acc.update(preds, target)
        if idx > 30:
            break
    

    acc = accuracy.compute()
    acc_list.append(acc) 
    alpha_list.append(alpha)
    print("Accuracy", acc)
    print("Multi accuracy", multi_acc.compute())
    
    
        
        