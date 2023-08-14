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
NUM_WORKERS = 4
CHECKPOINT_PATH = '/workspace/LogisticRegression'

class LogisticRegression(pl.LightningModule):

    def __init__(self, feature_dim, num_classes, lr, weight_decay, max_epochs=20):
        super().__init__()
        self.save_hyperparameters()
        # Mapping from representation h to classes
        self.model = nn.Linear(feature_dim, num_classes)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones=[int(self.hparams.max_epochs*0.6),
                                                                  int(self.hparams.max_epochs*0.8)],
                                                      gamma=0.1)
        return [optimizer], [lr_scheduler]
        

    def _calculate_loss(self, batch, mode='train'):
        feats, labels = batch
        preds = self.model(feats)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(mode + '_loss', loss, prog_bar=True)
        self.log(mode + '_acc', acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode='val')

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode='test')




def prepare_data_features(sim_clr, dataset, filename):
    features_path = '/workspace/sim_clr/clr_features'
    full_filename = os.path.join(features_path, filename)

    if os.path.exists(full_filename):
        print("Found precomputed features, loading...")
        # Load features
        feats, labels = torch.load(full_filename)
        return data.TensorDataset(feats, labels)
    
    # Prepare model
    network = sim_clr.model
    network.mlp = nn.Sequential(
        nn.Linear(768, 768),
        nn.ReLU(),
        nn.Linear(768, 768)
    )  # Removing projection head g(.)
    network.eval()
    network.to(device)

    # Encode all images
    data_loader = data.DataLoader(dataset, batch_size=64, num_workers=NUM_WORKERS, shuffle=False, drop_last=False, collate_fn=batch_sparse_collate)
    feats, labels = [], []
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


def train_logreg(batch_size, train_feats_data, test_feats_data, model_suffix, max_epochs=100, **kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "LogisticRegression"),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode='min', monitor='val_loss'),
                                    LearningRateMonitor("epoch")],
                         enable_progress_bar=True,
                         max_epochs=max_epochs,
                         check_val_every_n_epoch=2)
    trainer.logger._default_hp_metric = None

    # Data loaders
    train_loader = data.DataLoader(train_feats_data, batch_size=batch_size, shuffle=True,
                                   drop_last=False, pin_memory=True, num_workers=0)
    test_loader = data.DataLoader(test_feats_data, batch_size=batch_size, shuffle=False,
                                  drop_last=False, pin_memory=True, num_workers=0)

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"LogisticRegression_{model_suffix}.ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = LogisticRegression.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)  # To be reproducable
        model = LogisticRegression(**kwargs)
        trainer.fit(model, train_loader, test_loader)
        model = LogisticRegression.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    # Test best model on train and validation set
    train_result = trainer.test(model, train_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"train": train_result[0]["test_acc"], "test": test_result[0]["test_acc"]}

    return model, result




DATA_PATH = '/mnt/rradev/osf_data_512px/converted_data/'
CKPT_PATH = '/workspace/lightning_logs/sim_clr/checkpoints/'
train_dataset = CLRDataset(root=os.path.join(DATA_PATH, 'train'))
test_dataset = CLRDataset(root=os.path.join(DATA_PATH, 'test'))

simclr_model = SimCLR.load_from_checkpoint(checkpoint_path=os.path.join(CKPT_PATH, 'model-epoch=04-val_loss=0.33.ckpt'))
train_feats_simclr = prepare_data_features(simclr_model, train_dataset, filename='train_feats_simclr.pt')
test_feats_simclr = prepare_data_features(simclr_model, test_dataset, filename='test_feats_simclr.pt')

from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler



def get_smaller_dataset(original_dataset, num_imgs_per_label):
    # NOT SURE IF THIS IS WORKING CORRECTLY
    # show the dataset shape
    new_dataset = data.TensorDataset(
        *[t.unflatten(0, (5, -1))[:,:num_imgs_per_label].flatten(0, 1) for t in original_dataset.tensors]
    )
    return new_dataset

for num_samples in [10, 50, 100, 500, 1000, 5000, 10_000, 100_000]:
    small_dataset = get_smaller_dataset(train_feats_simclr, num_samples)
    print('Small dataset shape: ', small_dataset.tensors[0].shape)

        # train svm 
    clf = LogisticRegression()
    X = small_dataset.tensors[0].numpy()
    y = small_dataset.tensors[1].numpy()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    clf.fit(X, y)
    y_pred = clf.predict(scaler.transform(test_feats_simclr.tensors[0].numpy()))
    print('Accuracy: ', accuracy_score(test_feats_simclr.tensors[1], y_pred))