# for file level performance only
import inspect
import os
import random
import sys

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, currentdir)
from utils.utils import transform_custom
from data.channel_wise_aug import augmentations
from preprocessing.preprocess import random_idxs, middle_idxs


class OLIVES_ds(Dataset):
    """This data loader return b-scans uniformly selected based on teh stored indexes."""

    def __init__(self, annotation_path,
                 loader_type='random',
                 augment=False,
                 augmentation_list=None,
                 image_size=256,
                 categories=["dr", "dme"],
                 model_type='ViViT',
                 gray_scale=True,
                 logger=None,
                 n_frames=20):

        self.samples = []
        self.loader_type = loader_type
        self.image_size = image_size
        self.augment = augment
        self.annotation_path = annotation_path
        self.augmentation_list = augmentation_list
        self.categories = categories
        self.model_type = model_type
        self.gray_scale = gray_scale
        self.n_frames = n_frames
        self.labels_dict = {self.categories[x]: x for x in range(len(self.categories))}
        self.samples = pd.read_csv(self.annotation_path)
        self.length = []
        self.logger = logger

    def __len__(self):
        return len(self.samples)

    def cal_cls_weight(self, weight_type=None):
        from sklearn.utils import class_weight
        target = self.samples['label']
        class_weights = class_weight.compute_class_weight(class_weight=weight_type, classes=np.unique(target), y=target)
        class_weights = torch.tensor(class_weights, dtype=torch.float)

        return class_weights

    def __getitem__(self, idx):
        sample = self.samples.iloc[idx]
        sample_path = sample['sample_path']
        imgs_file = pydicom.dcmread(sample_path)

        if 'SamplesPerPixel' not in imgs_file:
            imgs_file.SamplesPerPixel = 1
        imgs_all = imgs_file.pixel_array
        """fixed, random, otherwise return the whole volume"""
        if self.loader_type == 'fixed':
            str2arr = (lambda x: [int(x) for x in x.replace("[", "").replace("]", "").split(sep=',')])
            indexes = str2arr(sample['indexes'])
            imgs = [imgs_all[idx] for idx in indexes]
        elif self.loader_type == 'random':
            indexes = random_idxs(n_selection=self.n_frames, array_len=len(imgs_all))
            imgs = [imgs_all[idx] for idx in indexes]
        elif self.loader_type == 'random_middle':
            indexes = middle_idxs(n_selection=self.n_frames, len_arr=len(imgs_all))
            imgs = [imgs_all[idx] for idx in indexes]
        elif self.loader_type == 'central':
            center = (len(imgs_all) / 2) - 1 if len(imgs_all) % 2 == 0 else (len(imgs_all) // 2)
            imgs = [imgs_all[center]]
        else:
            imgs = imgs_all

        # read using cv2 to avoid problems with uint16 images
        imgs = [np.array(cv2.normalize(pa, None, 0, 255, cv2.NORM_MINMAX), dtype=np.uint8) for pa in imgs]
        #  Data Augmentation
        if self.augment:  # you only apply one augmentation at a time
            function = self.augmentation_list[random.randint(0, len(self.augmentation_list) - 1)]
            imgs = augmentations[function](imgs)

        imgs = np.array(imgs)
        imgs = np.expand_dims(imgs, axis=-1)  # expand dimension for gray scale
        if not self.gray_scale:  # instead of reading image as rgb, we repeat the image for three times
            imgs = np.repeat(imgs, 3, axis=-1)
        imgs = transform_custom(imgs, self.image_size, self.model_type, gray_scale=self.gray_scale)
        # label
        _label = self.labels_dict[sample['label']]

        return imgs, _label


    def statistics(self):
        tmp_result = self.samples['label'].value_counts()
        print('Current Set Info:\n'
              '[INFO] Number of samples: {}; '
              'Number of categories: {}; '
              'Samples Distribution in Each Category:\n{}'.format(self.__len__(), len(self.categories), tmp_result))
