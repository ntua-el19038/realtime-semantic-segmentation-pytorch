import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from .mixtransformer import MixVisionTransformer, OverlapPatchEmbed
# from .mixtransformer1 import MixVisionTransformer1
from .modules import conv1x1, DWConvBNAct, ConvBNAct


class FarSeeNet4(nn.Module):
    def __init__(self, num_class=1, n_channel=3, backbone_type='mixtransformer', act_type='relu'):
        super(FarSeeNet4, self).__init__()
        if 'mixtransformer' in backbone_type and n_channel == 3:
            self.frontend_network = MixVisionTransformer.from_type(5)
            # Load the saved weights
            saved_weights_path = './pretrained/mit_b5.pth'
            self.frontend_network.load_state_dict(torch.load(saved_weights_path))
        elif backbone_type == 'mixtransformer' and n_channel == 1:
            self.frontend_network = MixVisionTransformer.from_type(5)
            # Load the saved weights
            saved_weights_path = './pretrained/mit_b5.pth'
            self.frontend_network.load_state_dict(torch.load(saved_weights_path))
            self.frontend_network.patch_embed1 = OverlapPatchEmbed(img_size=224, patch_size=7, stride=4, in_chans=n_channel,
                                              embed_dim=64)
        else:
            raise NotImplementedError()

        # Modify backend network to handle all 4 feature outputs
        self.backend_network = FASPP(high_channels=512, low_channels=320, mid_channels2=64, num_class=num_class, act_type=act_type)

    def forward(self, x):

        size = x.size()[2:]
        # Incorporate all 4 outputs from the frontend_network
        x1, x2, x3, x4 = self.frontend_network(x)  # 64, 128, 320, 512 channels, respectively

        x = self.backend_network(x4, x3, x2, x1)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return x



class FASPP(nn.Module):
    def __init__(self, high_channels, low_channels, mid_channels2, num_class, act_type,
                 dilations=[6, 12, 18], hid_channels=256):
        super(FASPP, self).__init__()
        # High level convolutions
        # self.dropout = nn.Dropout2d(0.1)
        dilation=1
        self.conv_high = nn.ModuleList([
            ConvModule(
                high_channels,
                hid_channels,
                1 if dilation == 1 else 3,
                dilation=dilation,
                padding=0 if dilation == 1 else dilation)
        ])
        for dt in dilations:
            self.conv_high.append(
                nn.Sequential(
                    ConvModule(
                        high_channels,
                        hid_channels,
                        1 ,
                        dilation=dt,
                        padding=0 if dilation == 1 else dilation),
                    DWConvBNAct(hid_channels, hid_channels, 3, dilation=dt, act_type=act_type)
                )
            )

        self.sub_pixel_high = nn.Sequential(
            conv1x1(hid_channels * 4, hid_channels * 2 * (2 ** 2)),
            nn.PixelShuffle(2)
        )

        # Low level convolutions
        self.conv_low_init =  ConvModule(
                        low_channels,
                        48,
                        1 ,
                        dilation=dilation,
                        padding=0 if dilation == 1 else dilation)
        self.conv_low = nn.ModuleList([
            ConvModule(
                hid_channels * 2 + 48,
                hid_channels // 2,
                1,
                dilation=dilation,
                padding=0 if dilation == 1 else dilation)])
        for dt in dilations:
            self.conv_low.append(
                nn.Sequential(
                    ConvModule(
                        hid_channels * 2 + 48,
                        hid_channels // 2,
                        1,
                        dilation=dilation,
                        padding=0 if dilation == 1 else dilation),
                    DWConvBNAct(hid_channels // 2, hid_channels // 2, 3, dilation=dt, act_type=act_type)
                )
            )

        self.conv_low_last = nn.Sequential(
            ConvModule(
                hid_channels // 2 * 4,
                hid_channels * 2,
                1,
                dilation=dilation,
                padding=0 if dilation == 1 else dilation),
            ConvModule(
                hid_channels * 2,
                hid_channels * 2,
                1,
                dilation=dilation,
                padding=0 if dilation == 1 else dilation)
        )

        self.sub_pixel_low = nn.Sequential(
            conv1x1(hid_channels * 2, num_class * (4 ** 2)),
            nn.PixelShuffle(4) #2
        )
        # Mid2 level convolutions
        self.conv_mid2_init = ConvModule(
                mid_channels2,
               48,
                1,
                dilation=dilation,
                padding=0 if dilation == 1 else dilation)
        self.conv_mid2 = nn.ModuleList([
            ConvModule(
                48 + num_class, #50 for greyscale, 56 for uavid input:1024, 67 for cityscapes input: 512
                hid_channels // 2,
                1,
                dilation=dilation,
                padding=0 if dilation == 1 else dilation)
    ])
        for dt in dilations[:-1]:
            self.conv_mid2.append(
                nn.Sequential(
                    ConvModule(
                        48 + num_class, #50 for greyscale, 56 for uavid input:1024, 67 for cityscapes input: 512
                        hid_channels // 2,
                        1,
                        dilation=dilation,
                        padding=0 if dilation == 1 else dilation),
                    DWConvBNAct(hid_channels // 2, hid_channels // 2, 3, dilation=dt, act_type=act_type)
                )
            )

        self.conv_mid2_last = nn.Sequential(
            ConvModule(
                hid_channels // 2 * 3,
                hid_channels * 2,
                1,
                dilation=dilation,
                padding=0 if dilation == 1 else dilation),
        ConvModule(
            hid_channels * 2,
            hid_channels * 2,
            1,
            dilation=dilation,
            padding=0 if dilation == 1 else dilation)
        )

        self.sub_pixel_mid2 = nn.Sequential(
            conv1x1(hid_channels * 2, num_class * (4 ** 2)),
            nn.PixelShuffle(4)
        )

    def forward(self, x_high, x_low, xmid1, xmid2):
        high_feats = []
        for conv_high in self.conv_high:
            high_feats.append(conv_high(x_high))
        x = torch.cat(high_feats, dim=1)
        x = self.sub_pixel_high(x)

        # Low level features
        x_low = self.conv_low_init(x_low)
        x = torch.cat([x, x_low], dim=1)

        low_feats = []
        for conv_low in self.conv_low:
            low_feats.append(conv_low(x))

        x = torch.cat(low_feats, dim=1)
        x = self.conv_low_last(x)
        x = self.sub_pixel_low(x)



        # Mid2 level features
        xmid2 = self.conv_mid2_init(xmid2)
        x = torch.cat([x, xmid2], dim=1)

        mid2_feats = []
        for conv_mid2 in self.conv_mid2:
            mid2_feats.append(conv_mid2(x))

        x = torch.cat(mid2_feats, dim=1)
        x = self.conv_mid2_last(x)
        x = self.sub_pixel_mid2(x)
        # x = self.dropout(x)

        return x


