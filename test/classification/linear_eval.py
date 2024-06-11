## Large parts of this code are taken from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html

import os
import torch

from utils.data import load_yaml
from torch.utils import data
from tqdm import tqdm
from modules.simclr import SimCLR
from data.dataset import ClassifierBaseDataset
from torch import nn
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression as skLogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from MinkowskiEngine.utils import batch_sparse_collate
from MinkowskiEngine import SparseTensor
from utils.data import get_wandb_ckpt
import pickle


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 12 
config = load_yaml('configs/config.yaml')

@torch.no_grad()
def prepare_data_features(sim_clr, dataset, filename, drop_mlp=True, reset_features=True):
    features_path = os.path.join(os.environ['PSCRATCH'], 'linear-eval-contrastive')
    full_filename = os.path.join(features_path, filename)

    if os.path.exists(full_filename) and not reset_features:
        print("Found precomputed features, loading...")
        # Load features
        feats, labels = torch.load(full_filename)
        return data.TensorDataset(feats, labels)
    
    # Prepare model
    network = sim_clr.model
    if drop_mlp:
        network.mlp = nn.Identity() # Removing projection head g(.)
    network.eval()
    network.to(device)

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
    clf = skLogisticRegression(verbose=True, solver='saga', max_iter=100, n_jobs=128)
    #clf = HistGradientBoostingClassifier(max_iter=500, max_depth=100, verbose=1)
    X = train_feats_simclr.tensors[0].numpy()
    y = train_feats_simclr.tensors[1].numpy()

    test_X = test_feats_simclr.tensors[0].numpy()
    test_y = test_feats_simclr.tensors[1].numpy()

    scaler = StandardScaler()
    X, y = shuffle(X, y, random_state=42)  # Shuffling the data
    X = scaler.fit_transform(X)
    clf.fit(X, y)

    features_path = os.path.join(os.environ['PSCRATCH'], 'linear-eval-contrastive')
    dump_path = os.path.join(features_path, f"{identifier}.pkl") 
    with open(dump_path, 'wb') as f:
        pickle.dump((clf, scaler), f)

    y_pred = clf.predict(scaler.transform(test_X))
    return clf, balanced_accuracy_score(test_y, y_pred), accuracy_score(test_y, y_pred)

def evaluate(wandb_artifact=None):
    dataset_type = 'single_particle'
    train_dataset = ClassifierBaseDataset(os.path.join(os.path.dirname(config['data']['data_path']), 'larndsim_throws_converted_nominal', 'train'))
    test_dataset = ClassifierBaseDataset(os.path.join(os.path.dirname(config['data']['data_path']), 'larndsim_throws_converted_nominal', 'test'))

    if wandb_artifact is None:
        print('Using a randomly initialized model as baseline')
        simclr_model = SimCLR()
        artifact_name = 'randomly_initialized'
    else:
        ckpt_path, artifact_name = get_wandb_ckpt(wandb_artifact,return_name=True)
        simclr_model = SimCLR.load_from_checkpoint(ckpt_path)


    identifier = f'{artifact_name}'
    print(f'Evaluating {identifier}')
    
    train_feats_simclr = prepare_data_features(simclr_model, train_dataset, filename=f'{identifier}_train_feats.pt')
    test_feats_simclr = prepare_data_features(simclr_model, test_dataset, filename=f'{identifier}_test_feats.pt')

    clf, balanced_acc, acc_score = train_linear_model(train_feats_simclr, test_feats_simclr, identifier=identifier)
    print(f'Balanced accuracy score: {balanced_acc}')
    print(f'Accuracy score: {acc_score}')

if __name__ == '__main__':
    import fire
    fire.Fire(evaluate)


