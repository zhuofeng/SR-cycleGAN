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
from data.base_dataset import BaseDataset, get_params, get_transform

#我觉得应该在这个地方就生成10000个patch

class medical2Dtestdataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """
    def __init__(self, opt):
        self.opt = opt
        self.dir_clinical = opt.clinical_folder
        self.dir_micro = opt.micro_folder

        self.all_clinical_patchs = np.zeros((1, 512, 512), dtype='float32')
        self.all_micro_patchs = np.zeros((1, 4096, 4096), dtype='float32')

        test_clinical_array = nib.load(opt.all_clinical_paths[0]).get_fdata()
        test_clinical_array = np.float32(test_clinical_array)
        test_clinical_array = normalizationclinicalminmax1(test_clinical_array)

        test_micro_array = nib.load(opt.all_micro_paths[0]).get_fdata()
        test_micro_array = np.float32(test_micro_array)
        test_micro_array = normalizationmicrominmax1(test_micro_array)

        self.all_clinical_patchs[0,:,:] = test_clinical_array[:,:,200]

    def __getitem__(self, index):
        clinical_slice = self.all_clinical_patchs[0,:,:]
        micro_slice = self.all_micro_patchs[0,:,:]
        clinical_slcie = torch.from_numpy(clinical_slice)
        micro_slice = torch.from_numpy(micro_slice)

        '''
        clinical_slice = self.all_clinical_patchs[int(index % self.all_clinical_patchs.shape[0]) : int(index % self.all_clinical_patchs.shape[0] + 1), :, :]
        micro_slice = self.all_micro_patchs[int(index % self.all_micro_patchs.shape[0]) : int(index % self.all_micro_patchs.shape[0] + 1), :, :]
        
        clinical_slice = torch.from_numpy(clinical_slice)
        micro_slice = torch.from_numpy(micro_slice)
        '''
        return {'clinical': clinical_slice, 'micro': micro_slice, 'clinical_paths':self.dir_clinical, 'micro_paths':self.dir_micro}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return len(self.opt.all_clinical_paths) * self.opt.batch_num
