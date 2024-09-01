from .base_config import BaseConfig


class MyConfig(BaseConfig):
    def __init__(self,):
        super(MyConfig, self).__init__()
        # Dataset
        self.dataset = 'prostate'
        self.data_root = './datasets/prostate/'
        self.num_class = 3
        
        # Model
        self.model = 'farseenet3'
        
        # Training
        self.total_epoch = 50
        self.train_bs = 1
        self.loss_type = 'ce'
        self.optimizer_type = 'adamw'
        self.logger_name = 'seg_trainer'
        self.use_aux = False

        # Validating
        self.val_bs = 3
        
        # Testing
        self.is_testing = False
        self.test_bs = 1
        self.test_data_folder = './datasets/cityscapes/leftImg8bit/test'
        self.load_ckpt_path =  'save/farseenet_mitb5_train_all-prostate/last.pth'
        self.save_mask = True
        
        # Training setting
        self.use_ema = False
        
        # Augmentation
        self.crop_size = 512
        self.randscale = [-0.5, 1.0]
        self.scale = 1.0
        self.brightness = 0.5
        self.contrast = 0.5
        self.saturation = 0.5
        self.h_flip = 0.5
        
        # Knowledge Distillation
        self.kd_training = False
        self.teacher_ckpt = 'save/farseenet_mitb5_train_all-prostate/last.pth'
        self.teacher_model = 'farseenet3'
        self.teacher_encoder = None
        self.teacher_decoder = None

        #Explainability
        self.explainability = False
        self.path_to_sample = "./sample/pic1.png"
        
        #GROKFAST
        self.grokfast = False