import argparse, os, pickle

import numpy as np
from tqdm import tqdm

import torch; from torch import nn
from torch.utils.data import DataLoader
import MinkowskiEngine as ME

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
    elif args.classifier:
        print(f"Loading classifier model from {args.weights}")
        model = Classifier(conf)
        model.load_network(args.weights)
        model.eval()
        network = model.net
        network.head = ME.MinkowskiGlobalMaxPooling()
        network.eval()
    else:
        print(f"Loading DANN model from {args.weights}")
        model = DANN(conf)
        model.load_network(args.weights)
        model.eval()
        network = model.net
        network.eval()

    dataset = ThrowsDataset(
        os.path.join(args.test_data_path, "test"),
        DataPrepType.CLASSIFICATION,
        [], 0,
        conf.quantization_size,
        conf.xtalk,
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

    print("Getting representations of data...")
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
    y_pred = torch.cat(feats, dim=0).detach().cpu().numpy()
    y_target = torch.cat(labels, dim=0).detach().cpu().numpy()

    y_out = np.concatenate([y_pred, np.expand_dims(y_target, axis=1)], axis=1)

    test_dir = os.path.join(conf.checkpoint_dir, "test_results")
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    test_out_path = os.path.join(test_dir, args.preds_fname)
    np.save(test_out_path, y_out)

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

    args = parser.parse_args()

    if not args.classifier and not args.clr and not args.dann:
        raise ValueError("Specify --classifier | --clr | --dann")

    return args

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
