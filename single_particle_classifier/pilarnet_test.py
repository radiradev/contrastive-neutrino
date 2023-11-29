import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from MinkowskiEngine.utils import batch_sparse_collate
from sim_clr.dataset import CLRDataset
import torchmetrics
from MinkowskiEngine import SparseTensor
from single_particle_classifier.network_wrapper import SingleParticleModel
from tqdm import tqdm
from utils.data import get_wandb_ckpt, load_yaml
import numpy as np
import fire


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = load_yaml('config/config.yaml')['data']['data_path']

def scale_energy(coords, feats, scale_factor):
    scale_factor = 1.0 + torch.tensor(scale_factor, dtype=feats.dtype, device=feats.device)
    feats = feats * scale_factor
    return coords, feats


def vary_parameter(model, dataset, param_range, augmentation,  num_iters=30, batch_size=128, num_classes=5):
    num_params = len(param_range)
    metrics_dict = {
        'parameter_value': np.zeros(num_params),
        'test_loss': np.zeros(num_params),
        'accuracy': np.zeros(num_params),
        'multi_acc': np.zeros((num_params, num_classes))  # Assuming 5 classes
    }

    accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes).to(device)
    multi_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average=None).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    for i, value in enumerate(param_range):
        test_loss = 0.0
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=batch_sparse_collate, generator=torch.Generator().manual_seed(42))
        iterator = iter(dataloader)
        
        for idx in tqdm(range(num_iters)):
            batch_coords, batch_feats, target = next(iterator)
            target = target.to(device)
            batch_coords = batch_coords.to(device)
            batch_feats = batch_feats.to(device)

            batch_coords, batch_feats = augmentation(batch_coords, batch_feats, value)
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
            
            if idx >= num_iters:
                print("Max iterations reached")
                break

        # Save metrics for this parameter value
        metrics_dict['parameter_value'][i] = value
        metrics_dict['test_loss'][i] = test_loss / num_iters
        metrics_dict['accuracy'][i] = accuracy.compute().item()
        metrics_dict['multi_acc'][i, :] = multi_acc.compute().cpu().numpy()

        # Reset metrics
        accuracy.reset()
        multi_acc.reset()

    return metrics_dict




def vary_parameters(artifact_name):
    ckpt_path = get_wandb_ckpt(artifact_name)
    model = SingleParticleModel.load_from_checkpoint(ckpt_path)
    model.eval()

    # Prepare DataLoader
    dataset = CLRDataset(dataset_type='single_particle_base', root=DATA_PATH)
    model.to(device)

    # Define parameter ranges
    energy_scale_range = np.linspace(-0.3, 0.3, 50)
    metrics = vary_parameter(model, dataset, energy_scale_range, scale_energy)
    np.savez('vary_energy_scale.npz', **metrics)

    
if __name__ == '__main__':
    fire.Fire(vary_parameters)
    
        
        