# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from model_utils import (
    LayerNorm,
    MinkowskiLayerNorm,
    MinkowskiGRN,
    MinkowskiDropPath,
    MinkowskiIdentity,
)

from MinkowskiEngine import (
    MinkowskiConvolution,
    MinkowskiDepthwiseConvolution,
    MinkowskiLinear,
    MinkowskiGELU,
    SparseTensor,
    MinkowskiGlobalMaxPooling,
    MinkowskiGlobalAvgPooling,
)
from MinkowskiEngine.MinkowskiOps import (
    to_sparse,
)

class Block(nn.Module):
    """ Sparse ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., D=3, grn_norm=True):
        super().__init__()
        self.dwconv = MinkowskiDepthwiseConvolution(dim, kernel_size=7, bias=True, dimension=D)

        self.norm = MinkowskiLayerNorm(dim, 1e-6)
        self.pwconv1 = MinkowskiLinear(dim, 4 * dim)
        self.act = MinkowskiGELU()
        self.pwconv2 = MinkowskiLinear(4 * dim, dim)
        self.grn = MinkowskiGRN(4  * dim) if grn_norm else MinkowskiIdentity()
        self.drop_path = MinkowskiDropPath(drop_path)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = input + self.drop_path(x)

        return x

class VoxelConvNeXt(nn.Module):
    """ Sparse ConvNeXtV2.

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self,
                 in_chans=1,
                 depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768],
                 drop_path_rate=0.,
                 D=3,
                 grn=None,
                 ):
        super().__init__()
        self.depths = depths
        self.dims = dims
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            MinkowskiConvolution(in_chans, self.dims[0], kernel_size=4, stride=4, dimension=D)
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                MinkowskiLayerNorm(self.dims[i], eps=1e-6),
                MinkowskiConvolution(self.dims[i], self.dims[i+1], kernel_size=2, stride=2, bias=True, dimension=D)
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        if grn is None:
            raise ValueError("contrastive must be set to True or False")
        self.grn = grn
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=self.dims[i], drop_path=dp_rates[cur + j], D=D, grn_norm=self.grn) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # TODO: fix init weights
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, MinkowskiConvolution):
            trunc_normal_(m.kernel, std=.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiDepthwiseConvolution):
            trunc_normal_(m.kernel, std=.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiLinear):
            trunc_normal_(m.linear.weight, std=.02)
            nn.init.constant_(m.linear.bias, 0)

    def forward(self):
        raise NotImplemented

class VoxelConvNeXtRegressor(VoxelConvNeXt):
   def __init__(self, num_classes, pooling='max', *args, **kwargs):
       super().__init__(*args, grn=False, **kwargs)
       pool_layer = MinkowskiGlobalMaxPooling() if pooling == 'max' else MinkowskiGlobalAvgPooling()
       self.head = nn.Sequential(
           pool_layer,
           MinkowskiLinear(768, num_classes)
       )
   def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        x = self.head(x)
        return x.F

class VoxelConvNeXtClassifier(VoxelConvNeXt):
   def __init__(self, num_classes, *args, **kwargs):
       super().__init__(*args, grn=False, **kwargs)
       self.head = nn.Sequential(
           MinkowskiGlobalMaxPooling(),
           MinkowskiLinear(self.dims[-1], num_classes)
       )
   def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        x = self.head(x)
        return x.F

class VoxelConvNeXtCLR(VoxelConvNeXt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, grn=True, **kwargs)
        self.head = nn.Sequential(
            MinkowskiGlobalMaxPooling()) # 768
        self.mlp = nn.Sequential(
            nn.Linear(self.dims[-1], self.dims[-1]),
            nn.ReLU(),
            nn.Linear(self.dims[-1], self.dims[-1]),
            nn.ReLU(),
        )

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        x = self.head(x).F

        x = self.mlp(x)
        return x
