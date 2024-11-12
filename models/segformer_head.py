# Obtained from: https://github.com/NVlabs/SegFormer
# Modifications: Model construction with loop
# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
# A copy of the license is available at resources/license_segformer

import torch
import torch.nn as nn
from .modules import ConvBNAct

from ops import resize


class MLP(nn.Module):
    """Linear Embedding."""

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        return x


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with
    Transformers
    """

    def __init__(self, num_classes, embed_dim=768, dropout_ratio=0.1, conv_kernel_size=1, in_index=[0, 1, 2, 3], input_channels=[64, 128, 320, 512], normalization=dict(type='BN', requires_grad=True)):
        super(SegFormerHead, self).__init__()

        # decoder_params = kwargs['decoder_params']
        self.embedding_dim = embed_dim
        self.conv_kernel_size = conv_kernel_size
        self.in_index = in_index
        self.in_channels = input_channels
        self.norm = normalization
        self.num_classes = num_classes
        self.dropout =nn.Dropout2d(dropout_ratio)
        self.linear_c = {}
        for i, in_channels in zip(self.in_index, self.in_channels):
            self.linear_c[str(i)] = MLP(
                input_dim=in_channels, embed_dim=self.embedding_dim)
        self.linear_c = nn.ModuleDict(self.linear_c)

        self.linear_fuse = ConvBNAct(
            in_channels=self.embedding_dim * len(self.in_index),
            out_channels=self.embedding_dim,
            kernel_size=self.conv_kernel_size)
            # padding=0 if self.conv_kernel_size == 1 else self.conv_kernel_size // 2,
            # norm_cfg=self.norm)

        self.linear_pred = nn.Conv2d(
            self.embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = inputs
        n, _, h, w = x[-1].shape
        # for f in x:
        #     print(f.shape)

        _c = {}
        for i in self.in_index:
            # mmcv.print_log(f'{i}: {x[i].shape}, {self.linear_c[str(i)]}')
            _c[i] = self.linear_c[str(i)](x[i]).permute(0, 2, 1).contiguous()
            _c[i] = _c[i].reshape(n, -1, x[i].shape[2], x[i].shape[3])
            if i != 0:
                _c[i] = resize(
                    _c[i],
                    size=x[0].size()[2:],
                    mode='bilinear',
                    align_corners=False)

        _c = self.linear_fuse(torch.cat(list(_c.values()), dim=1))

        if self.dropout is not None:
            x = self.dropout(_c)
        else:
            x = _c
        x = self.linear_pred(x)

        return x
