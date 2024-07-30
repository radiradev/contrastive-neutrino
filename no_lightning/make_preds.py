import argparse, os, pickle

import numpy as np
from tqdm import tqdm
import yaml

import torch; from torch import nn
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from config_parser import get_config
from simclr import SimCLR
from classifier import Classifier
from dann import DANN
from dataset import ThrowsDataset, DataPrepType

def main(args):
    print("Setting random seed to 123")
    torch.manual_seed(123)

    conf = get_config(args.config)
    device = torch.device(conf.device)

    if args.clr:
        print(f"Loading CLR model from {args.weights}")
        model = SimCLR(conf)
        model.load_network(args.weights)
        print("Dropping MLP")
        network = model.net
        network.mlp = nn.Identity()
        network.eval()
        print(f"Loading finetune model from {args.finetune_pickle}")
        with open(args.finetune_pickle, "rb") as f:
            clf, scaler = pickle.load(f)
    elif args.classifier:
        print(f"Loading classifier model from {args.weights}")
        model = Classifier(conf)
        model.load_network(args.weights)
        model.eval()
    else:
        print(f"Loading DANN model from {args.weights}")
        model = DANN(conf)
        model.load_network(args.weights)
        model.eval()

    dataset = ThrowsDataset(
        os.path.join(args.test_data_path, "test"),
        DataPrepType.CLASSIFICATION,
        [], 0,
        conf.quantization_size,
        train_mode=False
    )
    collate_fn = ME.utils.batch_sparse_collate
    dataloader= DataLoader(
        dataset,
        batch_size=conf.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0, # Needs to be single process to access index_history
        drop_last=True
    )

    if args.clr:
        print("Getting simCLR representations of data...")
        feats, labels = [], []
        with torch.inference_mode():
            for batch_coords, batch_feats, batch_labels in tqdm(dataloader):
                batch_coords = batch_coords.to(device)
                batch_feats = batch_feats.to(device)
                stensor = ME.SparseTensor(features=batch_feats.float(), coordinates=batch_coords)
                out = network(stensor)
                feats.append(out.detach().cpu())
                batch_labels = torch.tensor(batch_labels).long()
                labels.append(batch_labels)
        feats = torch.cat(feats, dim=0)
        labels = torch.cat(labels, dim=0)
        finetune_test_data = torch.utils.data.TensorDataset(feats, labels)
        print("Using finetuned model to get label predictions from CLR representations...")
        test_x = finetune_test_data.tensors[0].numpy()
        test_y = finetune_test_data.tensors[1].numpy()
        y_pred = clf.predict_proba(scaler.transform(test_x))
        y_target = test_y
    elif args.classifier:
        print("Getting classifier predictions...")
        y_pred_classifier, y_target_classifier = [], []
        for data in tqdm(dataloader):
            model.set_input(data)
            model.test(compute_loss=False)
            vis = model.get_current_visuals()
            y_pred_classifier.append(vis["pred_out"])
            y_target_classifier.append(vis["target_out"])
        y_pred = torch.cat(y_pred_classifier).detach().cpu().numpy()
        y_target = torch.cat(y_target_classifier).detach().cpu().numpy()
    elif args.dann:
        print("Getting DANN predictions...")
        y_pred_classifier, y_target_classifier = [], []
        for data in tqdm(dataloader):
            model.set_input_test(data)
            model.test(compute_loss=False)
            vis = model.get_current_visuals()
            y_pred_classifier.append(vis["pred_label_s"])
            y_target_classifier.append(vis["target_label_s"])
        y_pred = torch.cat(y_pred_classifier).detach().cpu().numpy()
        y_target = torch.cat(y_target_classifier).detach().cpu().numpy()

    samples, indices = dataloader.dataset.samples, list(dataloader.dataset.index_history)
    preds = {}
    for i, (probs, label) in enumerate(zip(y_pred, y_target)):
        fname = samples[indices[i]][0]
        preds[fname] = [probs.tolist(), int(label)]

    test_dir = os.path.join(conf.checkpoint_dir, "test_results")
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    test_out_path = os.path.join(test_dir, args.preds_fname)
    with open(test_out_path, "w") as f:
        yaml.dump(preds, f)

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("config")
    parser.add_argument("weights")
    parser.add_argument("test_data_path")
    parser.add_argument("preds_fname")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--classifier", action="store_true")
    group.add_argument("--clr", action="store_true")
    group.add_argument("--dann", action="store_true")

    parser.add_argument("--finetune_pickle", type=str, default=None)

    args = parser.parse_args()

    if not args.classifier and not args.clr and not args.dann:
        raise ValueError("Specify --classifier | --clr | --dann")
    if args.clr and args.finetune_pickle is None:
        raise ValueError("Specify --finetune_pickle to make clr predictions with")

    return args

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
