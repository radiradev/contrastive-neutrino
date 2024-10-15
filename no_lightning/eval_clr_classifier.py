import argparse, os, pickle

import numpy as np
from tqdm import tqdm

import torch; from torch import nn
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

from config_parser import get_config
from simclr import SimCLR
from dann import DANN
from classifier import Classifier
from dataset import ThrowsDataset, DataPrepType

def main(args):
    # print("Setting random seed to 123")
    # torch.manual_seed(123)

    if not args.skip_clr:
        conf = get_config(args.config_clr)
        print(f"Loading CLR model from {args.clr_weights}")
        model_clr = SimCLR(conf)
        model_clr.load_network(args.clr_weights)
        print("Dropping MLP")
        network_clr = model_clr.net
        network_clr.mlp = nn.Identity()
        network_clr.eval()
        print(f"Loading finetune model from {args.finetune_pickle}")
        with open(args.finetune_pickle, "rb") as f:
            clf, scaler = pickle.load(f)

    if not args.skip_clf:
        conf = get_config(args.config_classifier)
        clf_name = "DANN" if args.classifier_is_dann else "classifier"
        print(f"Loading {clf_name} model from {args.classifier_weights}")
        model_classifier = (
            DANN(conf) if args.classifier_is_dann else Classifier(conf)
        )
        model_classifier.load_network(args.classifier_weights)
        model_classifier.eval()

    device = torch.device(conf.device)

    dataset = ThrowsDataset(
        os.path.join(args.test_data_path, "test"),
        DataPrepType.CLASSIFICATION,
        [], 0,
        conf.quantization_size,
        args.xtalk if args.xtalk is not None else conf.xtalk,
        train_mode=False
    )
    collate_fn = ME.utils.batch_sparse_collate
    dataloader= DataLoader(
        dataset,
        batch_size=conf.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=min(conf.max_num_workers, conf.batch_size),
        drop_last=False
    )

    if not args.skip_clr:
        print("Getting simCLR representations of data...")
        feats, labels = [], []
        with torch.inference_mode():
            for i_batch, (batch_coords, batch_feats, batch_labels) in tqdm(
                enumerate(dataloader)
            ):
                batch_coords = batch_coords.to(device)
                batch_feats = batch_feats.to(device)
                stensor = model_clr._create_tensor(batch_feats, batch_coords)
                out = network_clr(stensor)
                feats.append(out.detach().cpu())
                batch_labels = torch.tensor(batch_labels).long()
                labels.append(batch_labels)
                if i_batch >= args.max_test_batches:
                    break
        feats = torch.cat(feats, dim=0)
        labels = torch.cat(labels, dim=0)
        finetune_test_data = torch.utils.data.TensorDataset(feats, labels)
        print("Using finetuned model to get label predictions from CLR representations...")
        test_x = finetune_test_data.tensors[0].numpy()
        test_y = finetune_test_data.tensors[1].numpy()
        y_pred_clr = clf.predict(scaler.transform(test_x))
        y_target_clr = test_y
        acc_score_clr = accuracy_score(y_pred_clr, y_target_clr)

    if not args.skip_clf:
        print(f"Getting {clf_name} predictions...")
        pred_key, target_key = (
            ("pred_label_s", "target_label_s")
            if args.classifier_is_dann
            else ("pred_out", "target_out")
        )
        y_pred_classifier, y_target_classifier = [], []
        for i_batch, data in tqdm(enumerate(dataloader)):
            if args.classifier_is_dann:
                model_classifier.set_input_test(data)
            else:
                model_classifier.set_input(data)
            model_classifier.test(compute_loss=False)
            vis = model_classifier.get_current_visuals()
            y_pred_classifier.append(vis[pred_key].argmax(axis=1))
            y_target_classifier.append(vis[target_key])
            if i_batch >= args.max_test_batches:
                break
        y_pred_classifier = torch.cat(y_pred_classifier).detach().cpu().numpy()
        y_target_classifier = torch.cat(y_target_classifier).detach().cpu().numpy()
        acc_score_classifier = accuracy_score(y_pred_classifier, y_target_classifier)

    if not args.skip_clr:
        print(f"Finetuned CLR accuracy score: {acc_score_clr}")
    if not args.skip_clf:
        print(f"{clf_name} accuracy score: {acc_score_classifier}")

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("config_clr")
    parser.add_argument("config_classifier")
    parser.add_argument("clr_weights")
    parser.add_argument("finetune_pickle")
    parser.add_argument("classifier_weights")
    parser.add_argument("test_data_path")

    parser.add_argument("--classifier_is_dann", action="store_true")
    parser.add_argument("--xtalk", type=float, default=None)
    parser.add_argument("--skip_clr", action="store_true")
    parser.add_argument("--skip_clf", action="store_true")
    parser.add_argument("--max_test_batches", default=None, type=int)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
