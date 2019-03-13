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
from scipy.misc import imresize
from scipy.ndimage.interpolation import zoom

class medical2Ddataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        #8倍downsampling
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        #文件夹的路径
        self.dir_clinical = opt.clinical_folder
        self.dir_micro = opt.micro_folder

        #self.all_clinical_patchs = np.zeros((len(opt.all_clinical_paths)*opt.batch_num, opt.clinical_patch_size, opt.clinical_patch_size), dtype='float32')
        self.all_down_patchs = np.zeros((len(opt.all_micro_paths)*opt.batch_num, opt.clinical_patch_size, opt.clinical_patch_size), dtype='float32')
        self.all_micro_patchs = np.zeros((len(opt.all_micro_paths)*opt.batch_num, opt.micro_patch_size, opt.micro_patch_size), dtype='float32')
        #最好保证二者相同组织灰度在同一值上
        '''
        #make clinical patchs
        for i, path in enumerate(opt.all_clinical_paths):
            this_clinical_array = nib.load(path).get_fdata()
            this_clinical_array = np.float32(this_clinical_array)
            this_clinical_array = normalizationclinicalminmax1(this_clinical_array)
            #this_clinical_slices = this_clinical_array.shape[2]
            
            for j in range(0, opt.batch_num):
                this_patch = crop_nifti_2D(this_clinical_array, opt.clinical_patch_size, is_random=True)
                self.all_clinical_patchs[i*opt.batch_num + j,:,:] = this_patch
            del this_clinical_array
        '''

        #make micro patchs
        for i, path in enumerate(opt.all_micro_paths):
            this_micro_array = nib.load(path).get_fdata()
            this_micro_array_slices = this_micro_array.shape[2]
            this_micro_array = this_micro_array[150:854, 150:854, int(this_micro_array_slices*0.1/8)*8:int(this_micro_array_slices*0.9/8)*8]
            this_micro_array = np.float32(this_micro_array)
            this_micro_array = normalizationmicrominmax1(this_micro_array)
            for j in range(0, opt.batch_num):
                this_patch = crop_nifti_2D(this_micro_array, opt.micro_patch_size, is_random=True)
                this_down_patch = zoom(this_patch, (0.125, 0.125))
                self.all_micro_patchs[i*opt.batch_num + j,:,:] = this_patch
                self.all_down_patchs[i*opt.batch_num + j,:,:] = this_down_patch
            del this_micro_array

    def __getitem__(self, index):
        #clinical_slice = self.all_clinical_patchs[index % self.all_clinical_patchs.shape[0] : index % self.all_clinical_patchs.shape[0] + 1, :, :]
        micro_slice = self.all_micro_patchs[index % self.all_micro_patchs.shape[0] : index % self.all_micro_patchs.shape[0] + 1, :, :]
        down_slice = self.all_down_patchs[index % self.all_down_patchs.shape[0] : index % self.all_down_patchs.shape[0] + 1, :, :]
        #clinical_slice = torch.from_numpy(clinical_slice)
        down_slice = torch.from_numpy(down_slice)
        micro_slice = torch.from_numpy(micro_slice)
        return {'clinical': down_slice, 'micro': micro_slice, 'clinical_paths':self.dir_clinical, 'micro_paths':self.dir_micro}
        
    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return len(self.opt.all_clinical_paths) * self.opt.batch_num
