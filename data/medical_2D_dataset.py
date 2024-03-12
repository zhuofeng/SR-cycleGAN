# -*- coding: utf8 -*-
import os.path
import torch
import torch.utils.data as data
import random
import nibabel as nib
import numpy as np
from util.myutils import normalizationclinicalminmax1
from util.myutils import normalizationmicrominmax1
from util.myutils import crop_nifti_2D
from util.myutils import crop_nifti_withpos_2D
from data.base_dataset import BaseDataset
from scipy.ndimage.interpolation import zoom
import re
import pdb

class medical2Ddataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.
    I use a very special clinical-micro CT dataset. micro CT images are for guising the SR of clinical CT images.
    """

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        print('initialize the medical_2D_dataset')
        self.opt = opt
        
        # paths to clinical and micro CT images
        self.dir_clinical = opt.clinical_folder
        self.dir_micro = opt.micro_folder

        self.all_clinical_patchs = np.zeros((len(opt.all_clinical_paths)*opt.batch_num, opt.clinical_patch_size, opt.clinical_patch_size), dtype='float32')
        self.all_micro_patchs = np.zeros((len(opt.all_micro_paths)*opt.batch_num, opt.micro_patch_size, opt.micro_patch_size), dtype='float32')
        self.down_micro_patchs = np.zeros((len(opt.all_micro_paths)*opt.batch_num, opt.clinical_patch_size, opt.clinical_patch_size), dtype='float32')
        
        self.big_clinical_patchs = np.zeros((len(opt.all_full_clinical_paths)*opt.batch_num, opt.micro_patch_size, opt.micro_patch_size), dtype='float32')
        self.small_clinical_patchs = np.zeros((len(opt.all_full_clinical_paths)*opt.batch_num, opt.clinical_patch_size, opt.clinical_patch_size), dtype='float32')
        
        
        # initialze the clinical and micro CT patches
        for i, path in enumerate(opt.all_clinical_paths):
            this_clinical_array = nib.load(path).get_fdata()
            filename = re.search('nulung\d*', path).group()
            print(path+'/'+filename)
            filename = filename + 'closedmask.nii.gz'
            maskpath = os.path.join(opt.maskdatafolder, filename)
            this_mask = nib.load(maskpath).get_fdata()
            this_mask = this_mask.astype(np.float32)

            this_clinical_array = np.float32(this_clinical_array)
            j = 0
            while j < opt.batch_num:
                this_patch, pos = crop_nifti_withpos_2D(this_clinical_array, opt.clinical_patch_size, is_random=True)
                if 0 in (this_mask[pos[0]:pos[0]+opt.clinical_patch_size, pos[1]:pos[1]+opt.clinical_patch_size, pos[2]]): #不在mask里面
                    continue
                #print("in the mask")
                self.all_clinical_patchs[i*opt.batch_num + j,:,:] = this_patch
                j = j + 1
            del this_clinical_array

        self.all_clinical_patchs = normalizationclinicalminmax1(self.all_clinical_patchs)
        
        #make micro patchs
        for i, path in enumerate(opt.all_micro_paths):
            print(path)
            this_micro_array = nib.load(path).get_fdata()
            this_micro_array = this_micro_array.astype(np.float32)
            this_micro_array = normalizationmicrominmax1(this_micro_array)

            for j in range(0, opt.batch_num):
                this_patch = crop_nifti_2D(this_micro_array, opt.micro_patch_size, is_random=True)
                this_down_patch = zoom(this_patch, (0.125, 0.125))
                self.all_micro_patchs[i*opt.batch_num + j,:,:] = this_patch
                self.down_micro_patchs[i*opt.batch_num + j,:,:] = this_down_patch
            del this_micro_array        
            
    def __getitem__(self, index):
        if index % 4 == 0:
            clinical_slice = self.down_micro_patchs[index % self.down_micro_patchs.shape[0] : index % self.down_micro_patchs.shape[0] + 1, :, :]
            micro_slice = self.all_micro_patchs[index % self.all_micro_patchs.shape[0] : index % self.all_micro_patchs.shape[0] + 1, :, :]
        else:
            clinical_slice = self.all_clinical_patchs[index % self.all_clinical_patchs.shape[0] : index % self.all_clinical_patchs.shape[0] + 1, :, :]
            micro_slice = self.all_micro_patchs[index % self.all_micro_patchs.shape[0] : index % self.all_micro_patchs.shape[0] + 1, :, :]
            
        clinical_slice = torch.from_numpy(clinical_slice)
        micro_slice = torch.from_numpy(micro_slice)

        return {'clinical': clinical_slice, 'micro': micro_slice, 'clinical_paths':self.dir_clinical, 'micro_paths':self.dir_micro}
        
    def __len__(self):
        return len(self.opt.all_clinical_paths) * self.opt.batch_num