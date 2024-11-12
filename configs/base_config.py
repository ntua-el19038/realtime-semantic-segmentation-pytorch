class BaseConfig:
    def __init__(self,):
        # Dataset
        self.dataset = 'gta'
        # self.data_root = './datasets/vol/biomedic3/cc215/data/ACDC/preprocessed'
        self.data_root = './datasets/gta'
        self.num_class = 19
        self.ignore_index = 255

        # Model
        self.model = 'farseenet4'
        self.encoder = None
        self.decoder = None
        self.encoder_weights = 'imagenet'

        # Detail Head (For STDC)
        self.use_detail_head = False
        self.detail_thrs = 0.1
        self.detail_loss_coef = 1.0
        self.dice_loss_coef = 1.0
        self.bce_loss_coef = 1.0

        # Training
        self.total_epoch = 50
        self.base_lr = 0.01
        self.train_bs = 1 # For each GPU
        self.accumulate_grad_batches = 8
        self.use_aux = False
        self.aux_coef = None

        # Validating
        self.val_bs = 3      # For each GPU
        self.begin_val_epoch = 0    # Epoch to start validation
        self.val_interval = 1   # Epoch interval between validation

        # Testing
        self.is_testing = False
        self.test_bs = 1
        self.test_data_folder = './datasets/gta/test'
        # self.test_data_folder = './datasets/vol/biomedic3/cc215/data/ACDC/ACDC_artefacted/RandomBias'
        self.colormap = 'gta'
        self.save_mask = True
        self.blend_prediction = True
        self.blend_alpha = 0.3

        # Loss
        self.loss_type = 'ce'
        self.class_weights = None
        self.ohem_thrs = 0.7

        # Scheduler
        self.lr_policy = 'cos_warmup'
        self.warmup_epochs = 3

        # Optimizer
        self.optimizer_type = 'adamw'
        self.momentum = 0.9         # For SGD
        self.weight_decay = 1e-4    # For SGD

        # Monitoring
        self.save_ckpt = True
        self.save_dir = './save/farseenet4_bs1_ga8_gta'
        self.use_tb = True          # tensorboard
        self.tb_log_dir = None
        self.ckpt_name = None

        # Training setting
        self.amp_training = False
        self.resume_training = True
        self.load_ckpt = True
        self.load_ckpt_path = './save/farseenet4_bs1_ga8_gta/last.pth'
        self.base_workers = 5
        self.random_seed = 1
        self.use_ema = False

        # Augmentation
        self.crop_size = 512
        self.crop_h = None
        self.crop_w = None
        self.scale = 1.0
        self.randscale = 0.0
        self.brightness = 0.0
        self.contrast = 0.0
        self.saturation = 0.0
        self.h_flip = 0.0
        self.v_flip = 0.0

        # DDP
        self.synBN = True

        # Knowledge Distillation
        self.kd_training = False
        self.teacher_ckpt = './save/sqnet_mitb0-gta/last.pth'
        self.teacher_model = 'sqnet'
        self.teacher_encoder = None
        self.teacher_decoder = None
        self.kd_loss_type = 'kl_div'
        self.kd_loss_coefficient = 1.0
        self.kd_temperature = 4.0

        #Explainability
        self.explainability = False
        self.path_to_sample = "./sample/pic1.png"

        #GROKFAST
        self.grokfast = False

    def init_dependent_config(self):
        if self.load_ckpt_path is None and not self.is_testing:
            self.load_ckpt_path = f'{self.save_dir}/last.pth'

        if self.tb_log_dir is None:
            self.tb_log_dir = f'{self.save_dir}/tb_logs/'
            
        if self.crop_h is None:
            self.crop_h = self.crop_size
            
        if self.crop_w is None:
            self.crop_w = self.crop_size