import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from configs import MyConfig, load_parser
from models import get_model


def cal_model_params(config, imgw=1024, imgh=512):
    model = get_model(config)
    print(f'\nModel: {config.model}\nEncoder: {config.encoder}\nDecoder: {config.decoder}')

    try:
        from ptflops import get_model_complexity_info
        model.eval()
        '''
        Notice that ptflops doesn't take into account torch.nn.functional.* operations.
        If you want to get correct macs result, you need to modify the modules like 
        torch.nn.functional.interpolate to torch.nn.Upsample.
        '''
        _, params = get_model_complexity_info(model, (3, imgh, imgw), as_strings=True, 
                                                print_per_layer_stat=False, verbose=False)
        print(f'Number of parameters: {params}\n')
    except:
        import numpy as np
        params = np.sum([p.numel() for p in model.parameters()])
        print(f'Number of parameters: {params / 1e6:.2f}M\n')
        param_size = 0
        param_count = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
            param_count += param.nelement()
        buffer_size = 0
        buffer_count = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            buffer_count += buffer.nelement()

        size_all_mb = (param_size + buffer_size) / 1024**2
        param_size = param_size/1024**2
        buffer_size = buffer_size /1024**2
        print(f'Model Size: {size_all_mb:.3f} MB')
        print(f'Parameter Count: {param_count}, {param_size:.3f}MB')
        print(f'Buffer Count: {buffer_count}, {buffer_size:.3f}MB')


if __name__ == '__main__':
    config = MyConfig()
    config = load_parser(config)
    
    cal_model_params(config)