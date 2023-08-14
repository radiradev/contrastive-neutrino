from dataset.dataset import ConvertedDataset
from train.network_wrapper import VoxelConvNextWrapper
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from MinkowskiEngine.utils import batch_sparse_collate

dataset = ConvertedDataset('/mnt/rradev/osf_data_512px/converted_data/test')
dataloader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=batch_sparse_collate)
# evaluate 

model = VoxelConvNextWrapper()
# load the model
trainer = pl.Trainer(accelerator='gpu')
ckpt_path = '/workspace/lightning_logs/version_19/checkpoints/epoch=29-step=198630.ckpt'
trainer.test(model, dataloader, ckpt_path=ckpt_path)

