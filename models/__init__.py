import os, torch
import segmentation_models_pytorch as smp
from dask.dataframe.io.tests.test_csv import comment_footer

from .adscnet import ADSCNet
from .aglnet import AGLNet
from .bisenetv1 import BiSeNetv1
from .bisenetv2 import BiSeNetv2
from .canet import CANet
from .cfpnet import CFPNet
from .cgnet import CGNet
from .contextnet import ContextNet
from .dabnet import DABNet
from .ddrnet import DDRNet
from .dfanet import DFANet
from .edanet import EDANet
from .enet import ENet
from .erfnet import ERFNet
from .esnet import ESNet
from .espnet import ESPNet
from .espnetv2 import ESPNetv2
from .fanet import FANet
from .farseenet import FarSeeNet
from .farseenet4 import FarSeeNet4
from .fastscnn import FastSCNN
from .fddwnet import FDDWNet
from .fpenet import FPENet
from .fssnet import FSSNet
from .icnet import ICNet
from .lednet import LEDNet
from .linknet import LinkNet
from .lite_hrnet import LiteHRNet
from .liteseg import LiteSeg
from .mininet import MiniNet
from .mininetv2 import MiniNetv2
from .pp_liteseg import PPLiteSeg
from .regseg import RegSeg
from .segnet import SegNet
from .shelfnet import ShelfNet
from .sqnet import SQNet
from .stdc import STDC, LaplacianConv
from .swiftnet import SwiftNet
from .mlpmixer import MLP_Mixer
from .farseenet2 import FarSeeNet2
from .farseenet3 import FarSeeNet3
from .shelfnet2 import ShelfNet2
from  .farseenet1 import FarSeeNet1
from .daformer import DaFormer
from .segformer import SegFormer


decoder_hub = {'deeplabv3':smp.DeepLabV3, 'deeplabv3p':smp.DeepLabV3Plus, 'fpn':smp.FPN,
               'linknet':smp.Linknet, 'manet':smp.MAnet, 'pan':smp.PAN, 'pspnet':smp.PSPNet,
               'unet':smp.Unet, 'unetpp':smp.UnetPlusPlus}

model_hub = {'adscnet':ADSCNet, 'aglnet':AGLNet, 'bisenetv1':BiSeNetv1, 
                'bisenetv2':BiSeNetv2, 'canet':CANet, 'cfpnet':CFPNet, 
                'cgnet':CGNet, 'contextnet':ContextNet, 'dabnet':DABNet, 
                'ddrnet':DDRNet, 'dfanet':DFANet, 'edanet':EDANet, 
                'enet':ENet, 'erfnet':ERFNet, 'esnet':ESNet, 
                'espnet':ESPNet, 'espnetv2':ESPNetv2, 'fanet':FANet, 'farseenet':FarSeeNet,
                'farseenet2':FarSeeNet2, 'farseenet3':FarSeeNet3, 'farseenet4': FarSeeNet4,
                'fastscnn':FastSCNN, 'fddwnet':FDDWNet, 'fpenet':FPENet, 
                'fssnet':FSSNet, 'icnet':ICNet, 'lednet':LEDNet,
                'linknet':LinkNet, 'lite_hrnet':LiteHRNet, 'liteseg':LiteSeg, 'mininet':MiniNet, 
                'mininetv2':MiniNetv2, 'ppliteseg':PPLiteSeg, 'regseg':RegSeg,
                'segnet':SegNet, 'shelfnet':ShelfNet, 'sqnet':SQNet, 
                'stdc':STDC, 'swiftnet':SwiftNet, 'mlpmixer': MLP_Mixer, 'shelfnet2': ShelfNet2,
                "daformer":DaFormer, 'segformer':SegFormer}
def get_model(config):
    mmodel_hub = {'adscnet':ADSCNet, 'aglnet':AGLNet, 'bisenetv1':BiSeNetv1,
                'bisenetv2':BiSeNetv2, 'canet':CANet, 'cfpnet':CFPNet,
                'cgnet':CGNet, 'contextnet':ContextNet, 'dabnet':DABNet,
                'ddrnet':DDRNet, 'dfanet':DFANet, 'edanet':EDANet,
                'enet':ENet, 'erfnet':ERFNet, 'esnet':ESNet,
                'espnet':ESPNet, 'espnetv2':ESPNetv2, 'fanet':FANet, 'farseenet':FarSeeNet,
                'farseenet2':FarSeeNet2, 'farseenet3':FarSeeNet3, 'farseenet4': FarSeeNet4,
                'fastscnn':FastSCNN, 'fddwnet':FDDWNet, 'fpenet':FPENet,
                'fssnet':FSSNet, 'icnet':ICNet, 'lednet':LEDNet,
                'linknet':LinkNet, 'lite_hrnet':LiteHRNet, 'liteseg':LiteSeg, 'mininet':MiniNet,
                'mininetv2':MiniNetv2, 'ppliteseg':PPLiteSeg, 'regseg':RegSeg,
                'segnet':SegNet, 'shelfnet':ShelfNet, 'sqnet':SQNet,
                'stdc':STDC, 'swiftnet':SwiftNet, 'mlpmixer': MLP_Mixer, 'shelfnet2': ShelfNet2,
                "daformer":DaFormer, 'segformer':SegFormer}

    # The following models currently support auxiliary heads
    aux_models = ['bisenetv2', 'ddrnet', 'icnet']

    # The following models currently support detail heads
    detail_head_models = ['stdc']
    
    if config.model == 'smp':   # Use segmentation models pytorch
        if config.decoder not in decoder_hub:
            raise ValueError(f"Unsupported decoder type: {config.decoder}")

        model = decoder_hub[config.decoder](encoder_name=config.encoder, 
                                            encoder_weights=config.encoder_weights, 
                                            in_channels=3, classes=config.num_class)

    elif config.model in model_hub.keys():
        if config.model in aux_models:
            model = model_hub[config.model](num_class=config.num_class, use_aux=config.use_aux)
        elif config.model in detail_head_models:
            model = model_hub[config.model](num_class=config.num_class, use_detail_head=config.use_detail_head, use_aux=config.use_aux)
        else:
            if config.use_aux:
                raise ValueError(f'Model {config.model} does not support auxiliary heads.\n')
            if config.use_detail_head:
                raise ValueError(f'Model {config.model} does not support detail heads.\n')
            if config.dataset == 'prostate' or config.dataset == 'cardiac':
                model = model_hub[config.model](num_class=config.num_class, n_channel=1)
            else:
                model = model_hub[config.model](num_class=config.num_class)

    else:
        raise NotImplementedError(f"Unsupport model type: {config.model}")

    return model


def get_teacher_model(config, device):
    if config.kd_training:
        if not os.path.isfile(config.teacher_ckpt):
            raise ValueError(f'Could not find teacher checkpoint at path {config.teacher_ckpt}.')

        # if config.teacher_decoder not in decoder_hub.keys():
        #     raise ValueError(f"Unsupported teacher decoder type: {config.teacher_decoder}")      

        # model = decoder_hub[config.teacher_decoder](encoder_name=config.teacher_encoder, 
        #                     encoder_weights=None, in_channels=3, classes=config.num_class)        
        if config.model in model_hub.keys():
            model = model_hub[config.teacher_model](num_class=config.num_class)

        teacher_ckpt = torch.load(config.teacher_ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(teacher_ckpt['state_dict'])
        del teacher_ckpt

        model = model.to(device)
        model.eval()
    else:
        model = None

    return model