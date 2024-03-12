# -*- coding: utf8 -*-
import os.path
import torch
import torch.utils.data as data
from data.image_folder import make_dataset
from PIL import Image
import random
import nibabel as nib
from abc import ABC, abstractmethod
import numpy as np
from util.myutils import normalizationminmax1
from util.myutils import normalizationclinicalminmax1
from util.myutils import normalizationmicrominmax1
from util.myutils import crop_nifti_2D
from util.myutils import crop_nifti_withpos_2D
from data.base_dataset import BaseDataset, get_params, get_transform
from scipy.ndimage.interpolation import zoom
import re 
import glob
import logging
import pdb

import monai
from monai.transforms import (
    AddChanneld,
    AsDiscreted,
    CastToTyped,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
    RandSpatialCropd,
    RandSpatialCropSamplesd,
)

from monai.transforms.transform import Transform, MapTransform
from monai.config import KeysCollection
from typing import Mapping, Dict, Any, Hashable

class medical2Dmonaidataset(BaseDataset):
    def __init__(self, opt):
        self.opt = opt
        # data_folder = "/homes/tzheng/code/TestMONAI/Toydataset/Train"
        data_folder = "/homes/tzheng/code/COVID_challenge/COVID-19-20_v2/Train"
        self.images = sorted(glob.glob(os.path.join(data_folder, "*_ct.nii.gz")))
        logging.info(f"training: image/label ({len(self.images)}) folder: {data_folder}")

        amp = True  # auto. mixed precision
        # keys = ("image", "label")
        keys = ("micro",)
       
        train_files = [{keys[0]: img} for img in zip(self.images)]
        
        # create a training data loader
        batch_size = 2
        logging.info(f"batch size {batch_size}")
        train_transforms = self.get_xforms("train", keys)
        self.train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms)
        
    def __getitem__(self, index):
        dim1 = index // len(self.train_ds[0]) # 病例的序号
        dim2 = index % len(self.train_ds[0]) # 此patch的序号
        # print('case {}'.format(dim1))
        # print('slice {}'.format(dim2))
        micro_slice = self.train_ds[dim1][dim2]["micro"][:,:,:,0] # 原寸大小的clinical CT image
        clinical_slice = zoom(micro_slice, (1,0.25,0.25))
        # add gaussian noise
        
        if random.random() <= 0.1:
            noise = np.random.normal(0, 0.05, clinical_slice.shape) 
            clinical_slice += noise

        # clinical_slice = nib.Nifti1Image(clinical_slice, np.eye(4))
        # nib.save(clinical_slice, '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/clinical_patch.nii.gz')
        # micro_slice = nib.Nifti1Image(micro_slice, np.eye(4))
        # nib.save(micro_slice, '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/micro_patch.nii.gz')
        # os._exit(0)
             
        clinical_slice = torch.from_numpy(clinical_slice)
        micro_slice = torch.from_numpy(micro_slice)
        return {'clinical': clinical_slice, 'micro': micro_slice, 'small_clinical': clinical_slice.clone(), 'big_clinical': micro_slice.clone(), 'clinical_paths': "", 'micro_paths': ""}
   
    def __len__(self):
        length = len(self.images) * self.opt.num_samples
        # print(length)
        # return 40000
        return length

    class downsample(MapTransform):
        def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
            """
            Args:
                keys: keys of the corresponding items to be transformed.
                    See also: :py:class:`monai.transforms.compose.MapTransform`
                allow_missing_keys: don't raise exception if key is missing.
            """
            super().__init__(keys, allow_missing_keys)
        def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
            d = dict(data)
            LR = zoom(d["micro"], (1, 0.25,0.25,1))
            d["clinical"] = LR
            
            # img1 = d["micro"][0,:,:,:]
            # img2 = d["clinical"][0,:,:,:]
            # img1 = nib.Nifti1Image(img1, np.eye(4))
            # img2 = nib.Nifti1Image(img2, np.eye(4))

            # nib.save(img1, '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/clinical.nii.gz')
            # nib.save(img2, '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/micro.nii.gz')
            # os._exit(0)
            return d

    class dimenreduct(MapTransform):
        def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
            """
            Args:
                keys: keys of the corresponding items to be transformed.
                    See also: :py:class:`monai.transforms.compose.MapTransform`
                allow_missing_keys: don't raise exception if key is missing.
            """
            super().__init__(keys, allow_missing_keys)
        def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
            d = dict(data)
            for key in self.key_iterator(d):
                d[key] = d[key][:,:,:,0]
            
            return d

    def get_xforms(self, mode, keys):
        """returns a composed transform for train/val/infer."""

        xforms = [
            LoadImaged(keys),
            AddChanneld(keys),
            Orientationd(keys, axcodes="LPS"),
            Spacingd(keys, pixdim=(1.25, 1.25, 5.0), mode=("bilinear", "bilinear")[: len(keys)]),
            ScaleIntensityRanged(keys, a_min=-1000.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
        ]
        if mode == "train":
            xforms.extend(
                [
                    SpatialPadd(keys, spatial_size=(192, 192, -1), mode="reflect"),  # ensure at least 192x192
                    RandAffined(
                        keys,
                        prob=0.15,
                        rotate_range=(0.05, 0.05, None),  # 3 parameters control the transform on 3 dimensions
                        scale_range=(0.1, 0.1, None), 
                        mode=("bilinear"),
                        as_tensor_output=False,
                    ),
                    # RandCropByPosNegLabeld(keys, label_key=keys[1], spatial_size=(192, 192, 16), num_samples=1), # 这里的num_samples决定了从每个case取patch的数量 192patch大小
                    # RandSpatialCropd(keys, roi_size=(192, 192, 1), random_center=True, random_size=False),
                    RandSpatialCropSamplesd(keys, roi_size=(192, 192, 1), random_center=True, random_size=False, num_samples=self.opt.num_samples),
                    RandFlipd(keys, spatial_axis=0, prob=0.5),
                    RandFlipd(keys, spatial_axis=1, prob=0.5),
                    RandFlipd(keys, spatial_axis=2, prob=0.5),
                    # self.downsample(keys),
                    # RandGaussianNoised(keys, prob=0.15, std=0.01),
                ]
            )
            # keys = ("micro", "clinical")
            # xforms.extend([RandGaussianNoised(keys[1], prob=0.15, std=0.01), self.dimenreduct(keys)]) # add noise to LR images
            
            dtype = (np.float32)
        if mode == "val":
            dtype = (np.float32)
        if mode == "infer":
            dtype = (np.float32,)
        # keys = ("micro", "clinical")
        # xforms.extend([CastToTyped(keys, dtype=dtype), ToTensord(keys)])
        xforms.extend([CastToTyped(keys, dtype=dtype),])
        return monai.transforms.Compose(xforms)