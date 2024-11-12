import torch
import torch.nn as nn
import torch.nn.functional as F

from .segformer_head import SegFormerHead
from .mixtransformer import MixVisionTransformer, OverlapPatchEmbed


class SegFormer(nn.Module):
    def __init__(self, num_class=3, n_channel=3, backbone_type='mixtransformer', act_type='relu'):
        super(SegFormer, self).__init__()
        if 'mixtransformer' in backbone_type and n_channel == 3:
            self.backbone_network = MixVisionTransformer.from_type(5)
            # Load the saved weights
            saved_weights_path = './pretrained/mit_b5.pth'
            self.backbone_network.load_state_dict(torch.load(saved_weights_path))
        elif 'mixtransformer' in backbone_type and n_channel == 1:
            self.backbone_network  = MixVisionTransformer.from_type(5)
            # Load the saved weights
            saved_weights_path = './pretrained/mit_b5.pth'
            self.backbone_network .load_state_dict(torch.load(saved_weights_path))
            self.backbone_network .patch_embed1 = OverlapPatchEmbed(img_size=224, patch_size=7, stride=4,
                                                                   in_chans=n_channel,
                                                                   embed_dim=64)
        else:
            raise NotImplementedError()

        self.decoder = SegFormerHead(num_classes=num_class)

    def forward(self, x):
        size = x.size()[2:]

        x1, x2, x3, x4 = self.backbone_network(x)

        x = self.decoder([x1, x2, x3, x4])

        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        return x