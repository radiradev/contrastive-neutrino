## Some parts are taken from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html

import os
import torch

from utils.data import load_yaml
from torch.utils import data
from tqdm import tqdm
from modules.simclr import SimCLR
from data.regression import Regression
from torch import nn
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from MinkowskiEngine.utils import batch_sparse_collate
from MinkowskiEngine import SparseTensor
from utils.data import get_wandb_ckpt
# train test split
from torch.utils.data import random_split
import joblib




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 12 
config = load_yaml('config/config.yaml')

@torch.no_grad()
def prepare_data_features(network, dataset, filename):
    features_path = os.path.join(os.environ['PSCRATCH'], 'linear-eval-contrastive')
    full_filename = os.path.join(features_path, filename)

    if os.path.exists(full_filename):
        print("Found precomputed features, loading...")
        # Load features
        feats, labels = torch.load(full_filename)
        return data.TensorDataset(feats, labels)
    

    # Encode all images
    data_loader = data.DataLoader(dataset, batch_size=512, num_workers=NUM_WORKERS, shuffle=True, drop_last=False, collate_fn=batch_sparse_collate)
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


def load_clr_model(wandb_artifact, return_name=False):    
    if wandb_artifact is not None:
        print(f'Loading model from wandb artifact: {wandb_artifact}')
        ckpt_path, artifact_name = get_wandb_ckpt(wandb_artifact,return_name=True)
        sim_clr = SimCLR.load_from_checkpoint(ckpt_path)
    else:
        sim_clr = SimCLR()
        artifact_name = 'random_weights'
    
    network = sim_clr.model
    network.mlp = nn.Identity() # Removing projection head g(.)
    network.eval()
    network.to(device)

    if return_name:
        return network, artifact_name
    
    return network


def save_pipeline(scaler, bdt, identifier):
    pipeline = {
        'scaler': scaler,
        'bdt': bdt
    }
    features_path = os.path.join(os.environ['PSCRATCH'], 'linear-eval-contrastive')
    save_path = os.path.join(features_path, f"{identifier}_pipeline.pkl")
    joblib.dump(pipeline, save_path)
    print(f"Pipeline saved to {save_path}")


def train_linear_model(train_feats_simclr, test_feats_simclr, identifier):
    """
    Trains a logistic regression model on the given training features and evaluates it on the given test features.

    Args:
        train_feats_simclr: A tuple of PyTorch tensors representing the training features and labels.
        test_feats_simclr: A tuple of PyTorch tensors representing the test features and labels.

    Returns:
        A tuple containing the trained logistic regression model, balanced accuracy score, and accuracy score.
    """
    # clf = LogisticRegression(use_gpu=True, verbose=True)
    clf = HistGradientBoostingRegressor(loss='absolute_error')
    # clf = HistGradientBoostingClassifier(max_iter=500, max_depth=100, verbose=1)
    X = train_feats_simclr.tensors[0].numpy()
    y = train_feats_simclr.tensors[1].numpy()

    test_X = test_feats_simclr.tensors[0].numpy()
    test_y = test_feats_simclr.tensors[1].numpy()

    scaler = StandardScaler()
    X, y = shuffle(X, y, random_state=42)  # Shuffling the data
    X = scaler.fit_transform(X)
    clf.fit(X, y)

    # Save pipeline
    save_pipeline(scaler, clf, identifier)

    y_pred = clf.predict(scaler.transform(test_X))
    return clf, mean_absolute_error(test_y, y_pred), r2_score(test_y, y_pred)

def evaluate(wandb_artifact=None, reset_features=True):
    dataset = Regression(root=os.path.join(os.path.dirname(config['data']['data_path']), 'larndsim_throws_converted_nominal', 'train'), energies='particle_energy_train.pkl')
    # TODO Investigate why throw number is present in 2k particles filename
    train_dataset, test_dataset = random_split(dataset, [int(len(dataset)*0.90), len(dataset)-int(len(dataset)*0.90)])
    

    clr_model, artifact_name = load_clr_model(wandb_artifact, return_name=True)

    identifier = f'{artifact_name}'
    print(f'Evaluating {identifier}')
    
    if reset_features:
        print('Resetting features')
        path = os.path.join(os.environ['PSCRATCH'], 'linear-eval-contrastive')
        os.system(f'rm {path}/{identifier}_regression_train.pt')
        os.system(f'rm {path}/{identifier}_regression_test.pt')

    train_feats_simclr = prepare_data_features(clr_model, train_dataset, filename=f'{identifier}_regression_train.pt')
    test_feats_simclr = prepare_data_features(clr_model, test_dataset, filename=f'{identifier}_regression_test.pt')

    clf, mse, r2 = train_linear_model(train_feats_simclr, test_feats_simclr, identifier=identifier)

    print(f'MAE: {mse}')
    print(f'R^2: {r2}')

if __name__ == '__main__':
    import fire
    fire.Fire(evaluate)


