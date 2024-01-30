from dataset import ConvertedDataset
import torch
from MinkowskiEngine.utils import batch_sparse_collate

dataset = ConvertedDataset(root='/mnt/rradev/osf_data_512px/converted_data')

print(len(dataset))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=batch_sparse_collate)
batch = next(iter(dataloader))

