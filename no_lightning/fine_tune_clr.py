import argparse, os, pickle

from tqdm import tqdm

import torch; from torch import nn
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.pipeline import make_pipeline
import xgboost as xgb

from config_parser import get_config
from simclr import SimCLR
from dataset import ThrowsDataset, DataPrepType

def main(args):
    print("Setting random seed to 123")
    torch.manual_seed(123)

    conf = get_config(args.config)
    device = torch.device(conf.device)

    print(f"Loading model from {args.clr_weights}")
    model = SimCLR(conf)
    model.load_network(args.clr_weights)
    print("Dropping MLP")
    network = model.net
    network.mlp = nn.Identity()
    network.eval()

    print(f"Finetuning with dataset from {args.finetune_data_path}")
    dataset_train = ThrowsDataset(
        os.path.join(args.finetune_data_path, "train"), DataPrepType.CLASSIFICATION
    )
    dataset_val = ThrowsDataset(
        os.path.join(args.finetune_data_path, "val"), DataPrepType.CLASSIFICATION, train_mode=False
    )
    collate_fn = ME.utils.batch_sparse_collate
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=conf.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=min(conf.max_num_workers, conf.batch_size),
        drop_last=True
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=conf.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=min(conf.max_num_workers, conf.batch_size),
        drop_last=True
    )

    print("Getting simCLR representations of data...")
    feats, labels = [], []
    with torch.inference_mode():
        for batch_coords, batch_feats, batch_labels in tqdm(dataloader_train, desc="train data"):
            batch_coords = batch_coords.to(device)
            batch_feats = batch_feats.to(device)
            stensor = ME.SparseTensor(features=batch_feats.float(), coordinates=batch_coords)
            out = network(stensor)
            feats.append(out.detach().cpu())
            batch_labels = torch.tensor(batch_labels).long()
            labels.append(batch_labels)
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    train_data = torch.utils.data.TensorDataset(feats, labels)
    feats, labels = [], []
    with torch.inference_mode():
        for batch_coords, batch_feats, batch_labels in tqdm(dataloader_val, desc="val data"):
            batch_coords = batch_coords.to(device)
            batch_feats = batch_feats.to(device)
            stensor = ME.SparseTensor(features=batch_feats.float(), coordinates=batch_coords)
            out = network(stensor)
            feats.append(out.detach().cpu())
            batch_labels = torch.tensor(batch_labels).long()
            labels.append(batch_labels)
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    val_data = torch.utils.data.TensorDataset(feats, labels)

    print("Training logistic regression...")
    # clf = xgb.XGBClassifier(verbosity=2, n_jobs=16, early_stopping_rounds=3, n_estimators=150, max_depth=16, learning_rate=0.100, reg_alpha=0.2, gamma=0.8, min_child_weight=0.5)
    clf = LogisticRegression(verbose=True, solver='saga', max_iter=120, n_jobs=16)
    # clf = MLPClassifier(hidden_layer_sizes=[512, 256, 128], learning_rate="adaptive", verbose=True, early_stopping=True, learning_rate_init=0.005, max_iter=100, tol=0.0001)
    x = train_data.tensors[0].numpy()
    y = train_data.tensors[1].numpy()

    test_x = val_data.tensors[0].numpy()
    test_y = val_data.tensors[1].numpy()

    scaler = StandardScaler()
    x, y = shuffle(x, y, random_state=123)  # Shuffling the data
    x = scaler.fit_transform(x)

    # clf.fit(x, y, eval_set=[(test_x, test_y)])
    clf.fit(x, y)
    # print(len(clf.get_booster().get_dump()))

    y_pred = clf.predict(scaler.transform(test_x))
    bal_acc_score = balanced_accuracy_score(test_y, y_pred)
    acc_score = accuracy_score(test_y, y_pred)

    print(f"Balanced accuracy score: {bal_acc_score}")
    print(f"Accuracy score: {acc_score}")

    if args.pickle_model:
        dump_path = os.path.join(conf.checkpoint_dir, "finetune_model_logreg.pkl")
        print(f"Pickling model and transform to {dump_path}")
        with open(dump_path, "wb") as f:
            pickle.dump((clf, scaler), f)

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("config")
    parser.add_argument("finetune_data_path")
    parser.add_argument("clr_weights")

    parser.add_argument("--pickle_model", action="store_true")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
