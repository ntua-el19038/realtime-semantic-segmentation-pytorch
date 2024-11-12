import os
from collections import namedtuple

import albumentations as AT
import numpy as np
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from utils import transforms


class Gta(Dataset):
    # Codes are based on https://github.com/mcordts/cityscapesScripts

    # --------------------------------------------------------- #
    # a label and all meta information
    Label = namedtuple('Label', [

        'name',  # The identifier of this label, e.g. 'car', 'person', ... .
        # We use them to uniquely name a class

        'id',  # An integer ID that is associated with this label.
        # The IDs are used to represent the label in ground truth images
        # An ID of -1 means that this label does not have an ID and thus
        # is ignored when creating ground truth images (e.g. license plate).
        # Do not modify these IDs, since exactly these IDs are expected by the
        # evaluation server.

        'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
        # ground truth images with train IDs, using the tools provided in the
        # 'preparation' folder. However, make sure to validate or submit results
        # to our evaluation server using the regular IDs above!
        # For trainIds, multiple labels might have the same ID. Then, these labels
        # are mapped to the same class in the ground truth images. For the inverse
        # mapping, we use the label that is defined first in the list below.
        # For example, mapping all void-type classes to the same ID in training,
        # might make sense for some approaches.
        # Max value is 255!

        'category',  # The name of the category that this label belongs to

        'categoryId',  # The ID of this category. Used to create ground truth images
        # on category level.

        'hasInstances',  # Whether this label distinguishes between single instances or not

        'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
        # during evaluations or not

        'color',  # The color of this label
    ])

    # --------------------------------------------------------------------------------
    # A list of all labels
    # --------------------------------------------------------------------------------

    # Please adapt the train IDs as appropriate for your approach.
    # Note that you might want to ignore labels with ID 255 during training.
    # Further note that the current train IDs are only a suggestion. You can use whatever you like.
    # Make sure to provide your results using the original IDs and not the training IDs.
    # Note that many IDs are ignored in evaluation and thus you never need to predict these!

    labels = [
        #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
        Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        Label('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        Label('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        Label('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        Label('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        Label('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        Label('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        Label('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        Label('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        Label('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        Label('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        Label('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        Label('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        Label('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        Label('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        Label('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        Label('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        Label('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        Label('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        Label('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        Label('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        Label('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        Label('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        Label('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        Label('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        Label('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        Label('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        Label('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        Label('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        Label('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        Label('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        Label('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        Label('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        Label('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        Label('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    id_to_train_id = np.array([label.trainId for label in labels])

    def __init__(self, config, mode='train'):
        self.mode = mode
        data_root = os.path.expanduser(config.data_root)
        if mode == 'train':
            img_dir = os.path.join(data_root, 'images')
            msk_dir = os.path.join(data_root, 'labels')
        elif mode == 'val' or mode == 'test':
            # img_dir = os.path.join(data_root, 'images')
            # msk_dir = os.path.join(data_root, 'labels')
            img_dir = os.path.join('./datasets/cityscapes/', 'leftImg8bit', mode)
            msk_dir = os.path.join('./datasets/cityscapes/', 'gtFine', mode)
        if not os.path.isdir(img_dir):
            raise RuntimeError(f'Image directory: {img_dir} does not exist.')

        if not os.path.isdir(msk_dir):
            raise RuntimeError(f'Mask directory: {msk_dir} does not exist.')
        self.images = []
        self.masks = []
        if mode == 'train':
            self.transform = AT.Compose([
                AT.Resize(height=config.scale, width=config.scale),  # Replace Scale with Resize
                AT.RandomScale(scale_limit=config.randscale),
                AT.PadIfNeeded(min_height=config.crop_h, min_width=config.crop_w, value=(114, 114, 114),
                               mask_value=(0, 0, 0)),
                AT.RandomCrop(height=config.crop_h, width=config.crop_w),
                AT.ColorJitter(brightness=config.brightness, contrast=config.contrast, saturation=config.saturation),
                AT.HorizontalFlip(p=config.h_flip),
                AT.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))
                base_name, extension = os.path.splitext(file_name)
                new_filename = f'{base_name}_labelTrainIds{extension}'
                self.masks.append(os.path.join(msk_dir, new_filename))



        elif mode == 'val' or mode == 'test':
            self.transform = AT.Compose([
                AT.Resize(height=config.scale, width=config.scale),
                # Replace Scale with Resize                transforms.Scale(scale=config.scale),
                AT.PadIfNeeded(min_height=config.crop_h, min_width=config.crop_w, value=(114, 114, 114),
                               mask_value=(0, 0, 0)),  # remove for cityscapes evaluation
                AT.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

            for city in os.listdir(img_dir):
                city_img_dir = os.path.join(img_dir, city)
                city_mask_dir = os.path.join(msk_dir, city)

                for file_name in os.listdir(city_img_dir):
                    self.images.append(os.path.join(city_img_dir, file_name))
                    mask_name = f"{file_name.split('_leftImg8bit')[0]}_gtFine_labelIds.png"
                    self.masks.append(os.path.join(city_mask_dir, mask_name))

                    # else:
        # for file_name in os.listdir(img_dir):
        #     self.images.append(os.path.join(img_dir, file_name))
        #     base_name, extension = os.path.splitext(file_name)
        #     new_filename = f'{base_name}_labelTrainIds{extension}'
        #     self.masks.append(os.path.join(msk_dir, new_filename))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.asarray(Image.open(self.images[index]).convert('RGB'))
        mask = np.asarray(Image.open(self.masks[index]).convert('L'))

        # Perform augmentation and normalization
        augmented = self.transform(image=image, mask=mask)
        image, mask = augmented['image'], augmented['mask']

        # Encode mask using trainId
        if (self.mode == 'val'):
            mask = self.encode_target(mask)
        return image, mask

    @classmethod
    def encode_target(cls, mask):
        return cls.id_to_train_id[np.array(mask)]    