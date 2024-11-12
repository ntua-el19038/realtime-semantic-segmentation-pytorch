# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule


from ops import resize
from .aspp_head import ASPPModule
from .segformer_head import MLP
from .sep_aspp_head import DepthwiseSeparableASPPModule


class ASPPWrapper(nn.Module):

    def __init__(self,
                 in_channels,
                 channels,
                 sep,
                 dilations,
                 pool,
                 norm_cfg,
                 act_cfg,
                 align_corners,
                 context_cfg=None):
        super(ASPPWrapper, self).__init__()
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.align_corners = align_corners
        if pool:
            self.image_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ConvModule(
                    in_channels,
                    channels,
                    1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        else:
            self.image_pool = None
        if context_cfg is not None:
            self.context_layer = build_layer(in_channels, channels,
                                             **context_cfg)
        else:
            self.context_layer = None
        ASPP = {True: DepthwiseSeparableASPPModule, False: ASPPModule}[sep]
        self.aspp_modules = ASPP(
            dilations=dilations,
            in_channels=in_channels,
            channels=channels,
            norm_cfg=norm_cfg,
            conv_cfg=None,
            act_cfg=act_cfg)
        self.bottleneck = ConvModule(
            (len(dilations) + int(pool) + int(bool(context_cfg))) * channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        if self.image_pool is not None:
            aspp_outs.append(
                resize(
                    self.image_pool(x),
                    size=x.size()[2:],
                    mode='bilinear',
                    align_corners=self.align_corners))
        if self.context_layer is not None:
            aspp_outs.append(self.context_layer(x))
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)

        output = self.bottleneck(aspp_outs)
        return output


def build_layer(in_channels, out_channels, type, **kwargs):
    if type == 'id':
        return nn.Identity()
    elif type == 'mlp':
        return MLP(input_dim=in_channels, embed_dim=out_channels)
    elif type == 'sep_conv':
        return DepthwiseSeparableConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=kwargs['kernel_size'] // 2,
            **kwargs)


    elif type == 'conv':
        return ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=kwargs['kernel_size'] // 2,
            **kwargs)

    elif type == 'aspp':
        return ASPPWrapper(
            in_channels=in_channels, channels=out_channels, **kwargs)
    elif type == 'rawconv_and_aspp':
        kernel_size = kwargs.pop('kernel_size')
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2),
            ASPPWrapper(
                in_channels=out_channels, channels=out_channels, **kwargs))
    # elif type == 'isa':
    #     return ISALayer(
    #         in_channels=in_channels, channels=out_channels, **kwargs)
    else:
        raise NotImplementedError(type)



class DAFormerHead(nn.Module):

    def __init__(self, in_channels=[64, 128, 320, 512],
                        in_index=[0, 1, 2, 3],
                        channels=256,
                        dropout_ratio=0.1,
                        num_classes=8,
                        norm_cfg= dict(type='BN', requires_grad=True),
                        align_corners=False,
                        decoder_params=dict(
                            embed_dims=256,
                            embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
                            embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
                            fusion_cfg=dict(
                                type='aspp',
                                sep=True,
                                dilations=(1, 6, 12, 18),
                                pool=False,
                                act_cfg=dict(type='ReLU'),
                                norm_cfg=dict(type='BN', requires_grad=True)))):
        super(DAFormerHead, self).__init__()
        # self.cls_seg = nn.Dropout2d(dropout_ratio)
        # self.cls_seg=ConvBNAct(in_channels=channels, out_channels=1, kernel_size=1)
        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.in_index = in_index
        self.in_channels = in_channels
        self.channels = channels
        self.dropout_ratio = dropout_ratio
        self.num_classes = num_classes
        self.norm_cfg = norm_cfg
        self.align_corners = align_corners
        self.decoder_params = decoder_params
        assert not self.align_corners
        self.embed_dims = decoder_params['embed_dims']
        if isinstance(self.embed_dims, int):
            self.embed_dims = [self.embed_dims] * len(self.in_index)
        self.embed_cfg = self.decoder_params['embed_cfg']
        self.embed_neck_cfg = self.decoder_params['embed_neck_cfg']
        if self.embed_neck_cfg == 'same_as_embed_cfg':
            self.embed_neck_cfg = self.embed_cfg
        self.fusion_cfg = self.decoder_params['fusion_cfg']
        for cfg in [self.embed_cfg, self.embed_neck_cfg, self.fusion_cfg]:
            if cfg is not None and 'aspp' in cfg['type']:
                cfg['align_corners'] = self.align_corners

        self.embed_layers = {}
        for i, in_channels, embed_dim in zip(self.in_index, self.in_channels,
                                             self.embed_dims):
            if i == self.in_index[-1]:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **self.embed_neck_cfg)
            else:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **self.embed_cfg)
        self.embed_layers = nn.ModuleDict(self.embed_layers)

        self.fuse_layer = build_layer(
            sum(self.embed_dims), self.channels, **self.fusion_cfg)

    def forward(self, inputs):
        x = inputs
        # print(inputs)
        n, _, h, w = x[-1].shape
        # for f in x:
        #     mmcv.print_log(f'{f.shape}', 'mmseg')

        os_size = x[0].size()[2:]
        _c = {}
        for i in self.in_index:
            # mmcv.print_log(f'{i}: {x[i].shape}', 'mmseg')
            _c[i] = self.embed_layers[str(i)](x[i])
            if _c[i].dim() == 3:
                _c[i] = _c[i].permute(0, 2, 1).contiguous()\
                    .reshape(n, -1, x[i].shape[2], x[i].shape[3])
            # mmcv.print_log(f'_c{i}: {_c[i].shape}', 'mmseg')
            if _c[i].size()[2:] != os_size:
                # mmcv.print_log(f'resize {i}', 'mmseg')
                _c[i] = resize(
                    _c[i],
                    size=os_size,
                    mode='bilinear',
                    align_corners=self.align_corners)

        x = self.fuse_layer(torch.cat(list(_c.values()), dim=1))
        # print(x.shape)
        x = self.cls_seg(x)
        # seg_logit = resize(
        #     input=x,
        #     size=h,
        #     mode='bilinear',
        #     align_corners=self.align_corners)

        # x = F.interpolate(x, 288, mode='bilinear', align_corners=True)

        return x
