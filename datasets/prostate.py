import os
import random

import albumentations as albu
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .common_utils.basic_operations import check_dir, save_dict, load_dict
from .common_utils.dataset_utils import *

DATASET_NAME = 'Prostate'
IDX2CLASS_DICT = {
    0: 'BG',
    1: 'PZ',
    2: 'CZ',
}
IMAGE_FORMAT_NAME = '{pid}/t2_img_clipped.nii.gz'
LABEL_FORMAT_NAME = '{pid}/label_clipped.nii.gz'
IMAGE_SIZE = (320, 320, 1)
LABEL_SIZE = (320, 320)


def get_training_transform():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.Rotate(limit=20, p=0.5),
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        albu.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        albu.ElasticTransform(alpha=1, sigma=50, alpha_affine=None, p=0.3),  # Corrected alpha_affine
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    # crop_aug = SmartCropV1(crop_size=768, max_ratio=0.75, ignore_index=255, nopad=False)
    # img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    img = np.squeeze(img, axis=0)
    assert img.shape == mask.shape, f"Image and mask shapes do not match: {image.shape} vs {mask.shape}"
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    img = np.expand_dims(img, axis=0)
    return img, mask


def get_val_transform():
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform)


def val_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    img = np.squeeze(img, axis=0)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    img = np.expand_dims(img, axis=0)
    return img, mask


class ProstateDataset(Dataset):
    def __init__(self, config,  mode='val', use_cache=False,
                 data_setting_name='all', cval=0, new_spacing=None, image_format_name = IMAGE_FORMAT_NAME, label_format_name = LABEL_FORMAT_NAME,
                 ignore_black_slice=True, debug=False):
        # self.data_root = config.data_root
        self.split = mode
        self.root_dir='./datasets/prostate_dataset/reorganized/G-MedicalDecathlon'
        self.root_name_dir='/datasets/prostate_dataset/reorganized/G-MedicalDecathlon'
        self.data_setting_name=data_setting_name
        self.cval=cval
        self.image_format_name = image_format_name
        self.label_format_name = label_format_name
        self.new_spacing = new_spacing
        self.ignore_black_slice=ignore_black_slice
        if mode == 'train':
            use_cache = True
        self.use_cache = use_cache
        self.datasize, self.patient_id_list, self.index2pid_dict, self.index2slice_dict,self.pid2spacing_dict =self.scan_dataset(use_cache)
        self.index = 0  # index for selecting which slices
        self.pid = self.patient_id_list[0]  # current pid
        self.patient_number = len(self.patient_id_list)
        self.slice_id = self.index2slice_dict[0]
        self.dataset_name = DATASET_NAME + '_{}_{}'.format(str(data_setting_name), self.split)
        self.debug=debug
        if self.split == 'train':
            self.dataset_name += str(cval)

        print('load {},  containing {}, found {} slices'.format(
            self.root_dir, len(self.patient_id_list), self.datasize))


    def scan_dataset(self, use_cache=False):
        '''
        given the data setting names and split, cross validation id
        :return: dataset size, a list of pids for training/testing/validation,
         and a dict for retrieving patient id and slice id.
        '''
        cache_dir = './datasets/log/cache'
        check_dir(cache_dir, create=True)
        cache_file_name = self.root_name_dir.replace('/', '_') + self.image_format_name.replace('/',
                                                                                           '_') + self.label_format_name.replace(
            '/', '_') + str(self.data_setting_name) + str(self.cval) + self.split + '.pkl'
        cache_file_path = os.path.join(cache_dir, cache_file_name)
        self.cache_file_path = cache_file_path
        if use_cache and os.path.exists(cache_file_path):
            print('load basic information from cache:', cache_file_path)
            cache_dict = load_dict(cache_file_path)
            datasize = cache_dict['datasize']
            patient_id_list = cache_dict['patient_id_list']
            index2slice_dict = cache_dict['index2slice_dict']
            index2pid_dict = cache_dict['index2patientid']
            index2spacing_dict = cache_dict['index2spacing']
        else:
            patient_id_list = self.get_pid_list(identifier=self.data_setting_name, cval=self.cval)[self.split]

            # print ('{} set has {} patients'.format(self.split,len(patient_id_list)))
            index2pid_dict = {}
            index2slice_dict = {}
            index2spacing_dict = {}
            cur_ind = 0
            for pid in patient_id_list:
                img_path = os.path.join(self.root_dir, self.image_format_name.format(pid=pid))
                label_path = os.path.join(self.root_dir, self.label_format_name.format(pid=pid))
                try:
                    ndarray, label_arr, sitkImage, sitkLabel = self.load_img_label_from_path(img_path, label_path)
                    if self.new_spacing is not None:
                        spacing = list(self.new_spacing)
                    else:
                        spacing = list(sitkImage.GetSpacing())
                    num_slices = ndarray.shape[0]
                    if num_slices != label_arr.shape[0]:
                        print(
                            'image and label slice number not match, found {} slices in image, {} slices in label'.format(
                                num_slices, label_arr.shape[0]))
                        continue
                    for cnt in range(num_slices):
                        if self.ignore_black_slice:
                            img_slice_data = ndarray[cnt, :, :]
                            img_slice_data -= np.mean(img_slice_data)
                            if np.sum(abs(img_slice_data) - 0) > 1e-4:
                                index2pid_dict[cur_ind] = pid
                                index2slice_dict[cur_ind] = cnt
                                cur_ind += 1
                        else:
                            index2pid_dict[cur_ind] = pid
                            index2slice_dict[cur_ind] = cnt
                            index2spacing_dict[cur_ind] = spacing
                            cur_ind += 1
                except IOError:
                    print(f'error in loading image and label for pid:{pid},{img_path}')
            datasize = cur_ind

            cache_dict = {
                'datasize': datasize,
                'patient_id_list': patient_id_list,
                'index2slice_dict': index2slice_dict,
                'index2patientid': index2pid_dict,
                'index2spacing': index2spacing_dict
            }
            save_dict(cache_dict, file_path=cache_file_path)
            self.cache_dict  =cache_dict

        return datasize, patient_id_list, index2pid_dict, index2slice_dict, index2spacing_dict

    def get_pid_list(self, identifier, cval):
        assert cval >= 0, 'cval must be >0'
        all_p_id_list = sorted(os.listdir(self.root_dir))
        test_ids = ['patient_17', 'patient_7', 'patient_12', 'patient_22', 'patient_0',
                    'patient_24', 'patient_5']
        train_val_ids = list(set(all_p_id_list) - set(test_ids))
        train_ids, val_ids = train_test_split(train_val_ids, test_size=0.1, random_state=cval)
        size = len(train_val_ids)
        labelled_ids = train_ids[:(size // 2)]
        unlabelled_ids = train_ids[(size // 2):]
        if identifier == 'all':
            # use all training data as labelled data
            labelled_ids_split = train_ids
            unlabelled_ids = []
        elif identifier == 'three_shot':
            labelled_ids_split, _ = train_test_split(labelled_ids, train_size=3, random_state=cval)
        elif identifier == 'three_shot_upperbound':
            labelled_ids_split, _ = train_test_split(labelled_ids, train_size=3, random_state=cval)
            labelled_ids_split = labelled_ids_split + unlabelled_ids
            unlabelled_ids = []
        elif identifier == 'full':
            labelled_ids_split = labelled_ids
        elif isinstance(float(identifier), float):
            identifier = float(identifier)
            if 0 < identifier < 1:
                labelled_ids_split, _ = train_test_split(labelled_ids, train_size=identifier, random_state=cval)
            elif identifier > 1:
                identifier = int(identifier)
                if 0 < identifier < len(labelled_ids):
                    labelled_ids_split, _ = train_test_split(labelled_ids, train_size=identifier, random_state=cval)
                elif abs(identifier + 1) < 1e-6:
                    labelled_ids_split = labelled_ids
                else:
                    raise ValueError
            else:
                raise NotImplementedError
        else:
            print('use all training subjects')
            labelled_ids_split = labelled_ids
        return {
            'name': str(identifier) + '_cv_' + str(cval),
            'train': labelled_ids_split,
            'val': val_ids,
            'test': test_ids,
            'test+unlabelled': test_ids + unlabelled_ids,
            'unlabelled': unlabelled_ids,
        }




    def set_id(self, index):
        '''
        set the current id with semantic information (e.g. patient id)
        :return:
        '''
        return self.index

    def __getitem__(self, index):
        img_dict = self.load_data(index)
        image = img_dict['image']
        label = img_dict['label']
        pid = img_dict['pid']
        image = np.transpose(image, (2, 0, 1))
        if(self.split == "train"):
            image, label = train_aug(image, label)
        elif (self.split == "val"):
            image, label = val_aug(image, label)
        else:
            return image, label

        return image, label

    def __len__(self):
        return self.datasize
    def find_pid_slice_id(self, index):
        '''
        given an index, find the patient id and slice id
        return the current id
        :return:
        '''
        self.pid = self.index2pid_dict[index]
        self.slice_id = self.index2slice_dict[index]
        return self.pid, self.slice_id
    def load_data(self, index):
        '''
        give a index to fetch a data package for one patient
        :return:
        data from a patient.
        class dict: {
        'image': ndarray,H*W*CH, CH =1, for gray images
        'label': ndaray, H*W
        '''
        assert len(self.patient_id_list) > 0, "no data found in the disk at {}".format(self.root_dir)
        patient_id, slice_id = self.find_pid_slice_id(index)
        self.pid = patient_id
        self.slice_id = slice_id
        if self.debug:
            print(patient_id)
        image_3d, label_3d, sitkImage, sitkLabel = self.load_patientImage_from_nrrd(
            patient_id, new_spacing=self.new_spacing)
        max_id = image_3d.shape[0]
        id_list = list(np.arange(max_id))

        image = image_3d[slice_id]
        label = label_3d[slice_id]
        # remove slice w.o objects
        if self.ignore_black_slice:
            while True:
                if abs(np.sum(label) - 0) > 1e-4:
                    break
                else:
                    id_list.remove(slice_id)
                    random.shuffle(id_list)
                    slice_id = id_list[0]
                image = image_3d[slice_id]
                label = label_3d[slice_id]

        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
        if self.debug:
            print(image.shape)
            print(label.shape)
        self.pid = patient_id
        cur_data_dict = {'image': image,
                         'label': label,
                         'pid': patient_id,
                         'new_spacing': self.new_spacing}
        del image_3d, label_3d, sitkImage, sitkLabel
        self.temp_data_dict = cur_data_dict
        return cur_data_dict

    def load_patientImage_from_nrrd(self, patient_id, new_spacing=None, normalize=False):
        if "pid" in self.image_format_name:
            img_name = self.image_format_name.format(pid=patient_id)
            label_name = self.label_format_name.format(pid=patient_id)
            img_path = os.path.join(self.root_dir, img_name)
            label_path = os.path.join(self.root_dir, label_name)
        elif "p_id" in self.image_format_name:
            ## for historical reasons, we use p_id to represent patient id
            img_name = self.image_format_name.format(p_id=patient_id)
            label_name = self.label_format_name.format(p_id=patient_id)
            img_path = os.path.join(self.root_dir, img_name)
            label_path = os.path.join(self.root_dir, label_name)
        else:
            raise ValueError("image_format_name should contain pid or p_id")
        # load data
        img_arr, label_arr, sitkImage, sitkLabel = self.load_img_label_from_path(
            img_path, label_path, new_spacing=new_spacing, normalize=normalize)
        return img_arr, label_arr, sitkImage, sitkLabel
    def load_img_label_from_path(self, img_path, label_path=None, new_spacing=None, normalize=False, keep_z_spacing=True,
                                 z_score=False):
        '''
        given two strings of image and label path
        return a tuple of 'image' ndarray, 'label' ndarray and sitk image and label.
        '''
        sitkImage = sitk.ReadImage(img_path)
        sitkImage = sitk.Cast(sitkImage, sitk.sitkFloat32)
        if not label_path is None:
            sitkLabel = sitk.ReadImage(label_path)
        if new_spacing is not None:
            new_spacing = list(new_spacing)
            if keep_z_spacing is True or new_spacing[2] <= 0:
                new_spacing[2] = list(sitkImage.GetSpacing())[2]
            sitkImage = resample_by_spacing(sitkImage, new_spacing=new_spacing,
                                            interpolator=sitk.sitkLinear, keep_z_spacing=keep_z_spacing)
            if not label_path is None:
                sitkLabel = resample_by_spacing(sitkLabel, new_spacing=new_spacing,
                                                interpolator=sitk.sitkNearestNeighbor, keep_z_spacing=keep_z_spacing)
        ndarray = sitk.GetArrayFromImage(sitkImage)
        if normalize:
            if not z_score:
                ndarray = normalize_minmax_data(ndarray)
            else:
                ndarray = normalize_minmax_data(ndarray)


        if not label_path is None:
            label_ndarray = sitk.GetArrayFromImage(sitkLabel)
        if not label_path is None:
            return ndarray, label_ndarray, sitkImage, sitkLabel
        else:
            return ndarray, sitkImage
if __name__ == '__main__':
    # from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    pit = ProstateDataset(config="l", mode='train')
    x1, x2, x3, x4, x5 = pit.scan_dataset()
    # print(x1)
    # print(x2)
    # print(x3)
    # print(x4)
    # print(x5)
    image, mask=pit.__getitem__(0)
    image = np.transpose(image, (2, 0, 1))
    print(image.shape)
    print(mask.shape)

    image_nd, label_nd, image, label_image = pit.load_img_label_from_path("./prostate_dataset/reorganized/G-MedicalDecathlon/patient_31/t2_img_clipped.nii.gz", label_path = "./prostate_dataset/reorganized/G-MedicalDecathlon/patient_31/label_clipped.nii.gz")

    print(image_nd.shape)
    # print("imagenow")
    print(label_nd.shape)


    # Convert the SimpleITK image to a NumPy array
    image_array = sitk.GetArrayViewFromImage(image)
    label_array = sitk.GetArrayViewFromImage(label_image)
    # Number of slices in the 3D image
    num_slices = image_array.shape[0]

    # Create a figure to display all slices
    fig, axes = plt.subplots(2, num_slices, figsize=(num_slices * 2, 2))

    # Loop through each slice and display it
    for i in range(num_slices):
        axes[0][i].imshow(image_array[i], cmap='gray')
        axes[0][i].axis('off')
        axes[0][i].set_title(f'Slice {i + 1}')
        axes[1][i].imshow(label_array[i], cmap='gray')
        axes[1][i].axis('off')
        axes[1][i].set_title(f'Slice {i + 1}')

    # Adjust layout
    plt.tight_layout()
    plt.show()
