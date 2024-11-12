"""
Paper:      ShelfNet for Fast Semantic Segmentation
Url:        https://arxiv.org/abs/1811.11254
Create by:  zh320
Date:       2023/10/22
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import conv1x1, ConvBNAct, DeConvBNAct, Activation
from .backbone import ResNet
from .mixtransformer import MixVisionTransformer


class ShelfNet2(nn.Module):
    def __init__(self, num_class=1, n_channel=3, backbone_type='mixtransformer',
                 hid_channels=[32, 64, 128, 256], act_type='relu'):
        super(ShelfNet2, self).__init__()
        if 'mixtransformer' in backbone_type:
            # self.backbone = ResNet(backbone_type)
            # channels = [64, 128, 256, 512] if backbone_type in ['resnet18', 'resnet34'] else [256, 512, 1024, 2048]
            self.backbone = MixVisionTransformer.from_type(1)
            # Load the saved weights
            saved_weights_path = './pretrained/mit_b1.pth'
            self.backbone.load_state_dict(torch.load(saved_weights_path))
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            raise NotImplementedError()

        self.conv_A = ConvBNAct(64, hid_channels[0], 1, act_type=act_type)
        self.conv_B = ConvBNAct(128, hid_channels[1], 1, act_type=act_type)
        self.conv_C = ConvBNAct(320, hid_channels[2], 1, act_type=act_type)
        self.conv_D = ConvBNAct(512, hid_channels[3], 1, act_type=act_type)

        self.decoder2 = DecoderBlock(hid_channels, act_type)
        self.encoder3 = EncoderBlock(hid_channels, act_type)
        self.decoder4 = DecoderBlock(hid_channels, act_type)

        self.classifier = conv1x1(hid_channels[0], num_class)

    def forward(self, x):
        size = x.size()[2:]
        x_a, x_b, x_c, x_d = self.backbone(x)

        # Column 1
        x_a = self.conv_A(x_a)
        x_b = self.conv_B(x_b)
        x_c = self.conv_C(x_c)
        x_d = self.conv_D(x_d)

        # Column 2
        x_a, x_b, x_c = self.decoder2(x_a, x_b, x_c, x_d, return_hid_feats=True)

        # Column 3
        x_a, x_b, x_c, x_d = self.encoder3(x_a, x_b, x_c)

        # Column 4
        x = self.decoder4(x_a, x_b, x_c, x_d)

        x = self.classifier(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, channels, act_type):
        super(EncoderBlock, self).__init__()
        self.block_A = SBlock(channels[0], act_type)
        self.down_A = ConvBNAct(channels[0], channels[1], 3, 2, act_type=act_type)

        self.block_B = SBlock(channels[1], act_type)
        self.down_B = ConvBNAct(channels[1], channels[2], 3, 2, act_type=act_type)

        self.block_C = SBlock(channels[2], act_type)
        self.down_C = ConvBNAct(channels[2], channels[3], 3, 2, act_type=act_type)

    def forward(self, x_a, x_b, x_c):
        x_a = self.block_A(x_a)
        x = self.down_A(x_a)

        x_b = self.block_B(x_b, x)
        x = self.down_B(x_b)

        x_c = self.block_C(x_c, x)
        x_d = self.down_C(x_c)

        return x_a, x_b, x_c, x_d


class DecoderBlock(nn.Module):
    def __init__(self, channels, act_type):
        super(DecoderBlock, self).__init__()
        self.block_D = SBlock(channels[3], act_type)
        self.up_D = DeConvBNAct(channels[3], channels[2], act_type=act_type)

        self.block_C = SBlock(channels[2], act_type)
        self.up_C = DeConvBNAct(channels[2], channels[1], act_type=act_type)

        self.block_B = SBlock(channels[1], act_type)
        self.up_B = DeConvBNAct(channels[1], channels[0], act_type=act_type)

        self.block_A = SBlock(channels[0], act_type)

    def forward(self, x_a, x_b, x_c, x_d, return_hid_feats=False):
        x_d = self.block_D(x_d)
        x = self.up_D(x_d)

        x_c = self.block_C(x_c, x)
        x = self.up_C(x_c)

        x_b = self.block_B(x_b, x)
        x = self.up_B(x_b)

        x_a = self.block_A(x_a, x)

        if return_hid_feats:
            return x_a, x_b, x_c
        else:
            return x_a


class SBlock(nn.Module):
    def __init__(self, channels, act_type):
        super(SBlock, self).__init__()
        self.conv1 = ConvBNAct(channels, channels, 3, act_type=act_type)
        self.conv2 = ConvBNAct(channels, channels, 3, act_type='none')
        self.act = Activation(act_type)

    def forward(self, x_l, x_v=0.):
        x = x_l + x_v
        residual = x

        x = self.conv1(x)
        x = self.conv2(x)

        x += residual

        return self.act(x)
