# A rubbish architecture I made up for sanity tests

import torch
import torch.nn as nn
import MinkowskiEngine as ME

class DogShitNet69(nn.Module):
    def __init__(self, mode=1):
        super().__init__()

        if mode == 1:
            self.conv_initial = nn.Sequential(
                ME.MinkowskiConvolution(
                    1,
                    2,
                    kernel_size=3,
                    stride=1,
                    dimension=3,
                ),
                ME.MinkowskiBatchNorm(2),
                ME.MinkowskiReLU()
            )
            self.conv1 = nn.Sequential(
                ME.MinkowskiConvolution(
                    2,
                    4,
                    kernel_size=3,
                    stride=2,
                    dimension=3,
                ),
                ME.MinkowskiBatchNorm(4),
                ME.MinkowskiReLU()
            )
            self.conv2 = nn.Sequential(
                ME.MinkowskiConvolution(
                    4,
                    8,
                    kernel_size=3,
                    stride=2,
                    dimension=3,
                ),
                ME.MinkowskiBatchNorm(8),
                ME.MinkowskiReLU()
            )
            self.conv3 = nn.Sequential(
                ME.MinkowskiConvolution(
                    8,
                    16,
                    kernel_size=3,
                    stride=2,
                    dimension=3,
                ),
                ME.MinkowskiBatchNorm(16),
                ME.MinkowskiReLU()
            )
            self.conv4 = nn.Sequential(
                ME.MinkowskiConvolution(
                    16,
                    32  ,
                    kernel_size=3,
                    stride=2,
                    dimension=3,
                ),
                ME.MinkowskiBatchNorm(32),
                ME.MinkowskiReLU()
            )
            self.conv5 = nn.Sequential(
                ME.MinkowskiConvolution(
                    32,
                    64,
                    kernel_size=3,
                    stride=2,
                    dimension=3,
                ),
                ME.MinkowskiBatchNorm(64),
                ME.MinkowskiReLU()
            )
            self.pool = ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=3)
            self.embedding_channel = 64
        
        elif mode == 2:
            self.conv_initial = nn.Sequential(
                ME.MinkowskiConvolution(
                    1,
                    24,
                    kernel_size=3,
                    stride=1,
                    dimension=3,
                ),
                ME.MinkowskiBatchNorm(24),
                ME.MinkowskiReLU()
            )
            self.conv1 = nn.Sequential(
                ME.MinkowskiConvolution(
                    24,
                    48,
                    kernel_size=3,
                    stride=2,
                    dimension=3,
                ),
                ME.MinkowskiBatchNorm(48),
                ME.MinkowskiReLU()
            )
            self.conv2 = nn.Sequential(
                ME.MinkowskiConvolution(
                    48,
                    96,
                    kernel_size=3,
                    stride=2,
                    dimension=3,
                ),
                ME.MinkowskiBatchNorm(96),
                ME.MinkowskiReLU()
            )
            self.conv3 = nn.Sequential(
                ME.MinkowskiConvolution(
                    96,
                    192,
                    kernel_size=3,
                    stride=2,
                    dimension=3,
                ),
                ME.MinkowskiBatchNorm(192),
                ME.MinkowskiReLU()
            )
            self.conv4 = nn.Sequential(
                ME.MinkowskiConvolution(
                    192,
                    384,
                    kernel_size=3,
                    stride=2,
                    dimension=3,
                ),
                ME.MinkowskiBatchNorm(384),
                ME.MinkowskiReLU()
            )
            self.conv5 = nn.Sequential(
                ME.MinkowskiConvolution(
                    384,
                    768,
                    kernel_size=3,
                    stride=2,
                    dimension=3,
                ),
                ME.MinkowskiBatchNorm(768),
                ME.MinkowskiReLU()
            )
            self.pool = ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=3)
            self.embedding_channel = 768

        elif mode == 3:
            self.conv_initial = nn.Sequential(
                ME.MinkowskiConvolution(
                    1,
                    24,
                    kernel_size=3,
                    stride=1,
                    dimension=3,
                ),
                ME.MinkowskiReLU()
            )
            self.conv1 = nn.Sequential(
                ME.MinkowskiConvolution(
                    24,
                    48,
                    kernel_size=3,
                    stride=2,
                    dimension=3,
                ),
                ME.MinkowskiReLU()
            )
            self.conv2 = nn.Sequential(
                ME.MinkowskiConvolution(
                    48,
                    96,
                    kernel_size=3,
                    stride=2,
                    dimension=3,
                ),
                ME.MinkowskiReLU()
            )
            self.conv3 = nn.Sequential(
                ME.MinkowskiConvolution(
                    96,
                    192,
                    kernel_size=3,
                    stride=2,
                    dimension=3,
                ),
                ME.MinkowskiReLU()
            )
            self.conv4 = nn.Sequential(
                ME.MinkowskiConvolution(
                    192,
                    384,
                    kernel_size=3,
                    stride=2,
                    dimension=3,
                ),
                ME.MinkowskiReLU()
            )
            self.conv5 = nn.Sequential(
                ME.MinkowskiConvolution(
                    384,
                    768,
                    kernel_size=3,
                    stride=2,
                    dimension=3,
                ),
                ME.MinkowskiReLU()
            )
            self.pool = ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=3)
            self.embedding_channel = 768

        elif mode == 4:
            self.conv_initial = nn.Sequential(
                ME.MinkowskiConvolution(
                    1,
                    2,
                    kernel_size=3,
                    stride=1,
                    dimension=3,
                ),
                ME.MinkowskiBatchNorm(2),
                ME.MinkowskiReLU()
            )
            self.conv1 = nn.Sequential(
                ME.MinkowskiConvolution(
                    2,
                    4,
                    kernel_size=3,
                    stride=2,
                    dimension=3,
                ),
                ME.MinkowskiBatchNorm(4),
                ME.MinkowskiReLU()
            )
            self.conv2 = nn.Sequential(
                ME.MinkowskiConvolution(
                    4,
                    8,
                    kernel_size=3,
                    stride=2,
                    dimension=3,
                ),
                ME.MinkowskiBatchNorm(8),
                ME.MinkowskiReLU()
            )
            self.conv3 = nn.Sequential(
                ME.MinkowskiConvolution(
                    8,
                    16,
                    kernel_size=3,
                    stride=2,
                    dimension=3,
                ),
                ME.MinkowskiBatchNorm(16),
                ME.MinkowskiReLU()
            )
            self.conv4 = nn.Identity()
            self.conv5 = nn.Identity()
            self.pool = ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=3)
            self.embedding_channel = 16

        elif mode == 5:
            self.conv_initial = nn.Sequential(
                ME.MinkowskiConvolution(
                    1,
                    2,
                    kernel_size=3,
                    stride=1,
                    dimension=3,
                ),
                ME.MinkowskiBatchNorm(2),
                ME.MinkowskiReLU()
            )
            self.conv1 = nn.Sequential(
                ME.MinkowskiConvolution(
                    2,
                    4,
                    kernel_size=3,
                    stride=2,
                    dimension=3,
                ),
                ME.MinkowskiBatchNorm(4),
                ME.MinkowskiReLU()
            )
            self.conv2 = nn.Sequential(
                ME.MinkowskiConvolution(
                    4,
                    8,
                    kernel_size=3,
                    stride=2,
                    dimension=3,
                ),
                ME.MinkowskiBatchNorm(8),
                ME.MinkowskiReLU()
            )
            self.conv3 = nn.Identity()
            self.conv4 = nn.Identity()
            self.conv5 = nn.Identity()
            self.pool = ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=3)
            self.embedding_channel = 8

        else:
            raise ValueError(f"mode={mode} not a valid dogshitnet mode :(")

    def forward(self, x: ME.SparseTensor):
        x = self.conv_initial(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)
        return x

class DogShitNet69Classifier(DogShitNet69):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Sequential(
            ME.MinkowskiGlobalMaxPooling(),
            ME.MinkowskiLinear(self.embedding_channel, num_classes)
        )

    def forward(self, x):
        x = super().forward(x)

        x = self.head(x).F

        return x
