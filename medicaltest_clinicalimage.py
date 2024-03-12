from models import networks
import argparse
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import nibabel as nib
import numpy as np
import torch
from util.myutils import normalizationclinicalminmax1
from scipy.ndimage.interpolation import zoom
from PIL import Image
from util.myutils import normalizationmicrominmax1
# import cv2
from collections import OrderedDict
from util.myutils import crop_nifti_2D
import torch.nn.functional as F
from util.myutils import my_ssim, my_psnr
import pdb
import torch
import torch.nn as nn
from ignite.contrib.handlers import ProgressBar

import logging
import glob
import time
import monai
from monai.handlers import CheckpointSaver, MeanDice, StatsHandler, ValidationHandler
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
)

def normalizationmin0max1(data):
    print('min/max data: {}/{}'.format(np.min(data),np.max(data)))
    data = data.astype(np.float32)
    max = np.max(data)
    min = np.min(data)
    data = data - min
    newmax = np.max(data)
    data = data / newmax
    return data

# 原来的patch大小是192 需要写一个滑窗inference的程序

def testclinical_wholevolume():
    opt = TestOptions().parse()
    opt.num_threads = 1
    opt.batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    model = create_model(opt)      # create a model given opt.model and other options 目前是medical_cycle_gan
    print("successfully created the model!")
    
    print("successfully loaded the network!")
    if opt.eval:
        model.eval()

    model.setup(opt)               # regular setup: load and print networks; create schedulers #把网络打印出来.参见base_mode    
    
    # rootpath = '/homes/tzheng/code/TestMONAI/Toydataset/Test'
    rootpath = '/homes/tzheng/code/COVID_challenge/COVID-19-20_v2/Validation'
    filenames = []
    for curDir, dirs, files in os.walk('/homes/tzheng/code/COVID_challenge/COVID-19-20_v2/Validation'):
        filenames = files

    savepath = '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/tmp/SRtrainwholedata/epoch62'
    
    for name in filenames:
        clinical_path = os.path.join(rootpath, name)
        print('Processig {}'.format(clinical_path))
        clinical_array = nib.load(clinical_path).get_fdata()
        clinical_array = clinical_array.astype(np.float32)
        clinical_array[clinical_array<-1000] = -1000 
        clinical_array[clinical_array>500] = 500
        clinical_array = normalizationmin0max1(clinical_array)
        save_clinical_array = clinical_array.copy()
        print('successfuly normalizated clinical_array')
        test_start = 0
        test_end = save_clinical_array.shape[2]

        patch_size = [512, 512] # 确保patch是正方形的
        down_patch_size = [int(a * 0.25) for a in patch_size]
        
        overlap_rate = 0.5 # 在inference的时候 patch的重合率
        
        fake_micro = np.zeros((192,192,1), dtype='float32')
        fake_micro = fake_micro.transpose(2,0,1)
        fake_micro = fake_micro[np.newaxis, :]
        fake_micro = torch.from_numpy(fake_micro)

        # padding patch to n times of 192?
        pddw = (clinical_array.shape[0] // patch_size[0] + 1) * patch_size[0]
        pddh = (clinical_array.shape[1] // patch_size[1] + 1) * patch_size[1]
        pad_array = np.zeros((pddw,pddh,clinical_array.shape[2]), dtype=np.float32)
        
        for num in range(0,clinical_array.shape[2]):
            pad_array[:,:,num] = np.pad(clinical_array[:,:,num], (int((pddw-clinical_array.shape[0]) / 2),), mode='reflect')
        downsample_arr = zoom(pad_array, (0.25,0.25,1.0)) 
        result_array = np.zeros_like(pad_array)
        slidewindow_num = int((pad_array.shape[0] - patch_size[0]) // (patch_size[0]*(1-overlap_rate)) + 1)
        
        # 进行滑窗inference
        for k in range(test_start, test_end):
            for i in range(0, slidewindow_num):
                for j in range(0, slidewindow_num):
                    x_startpos = int(i * patch_size[0]*(1-overlap_rate) / 4)
                    y_startpos = int(j * patch_size[0]*(1-overlap_rate) / 4)
                    this_slice = downsample_arr[x_startpos:x_startpos+down_patch_size[0],y_startpos:y_startpos+down_patch_size[1],k]
                    this_slice = this_slice[np.newaxis, np.newaxis, :]
                    this_slice = torch.from_numpy(this_slice)
                    data = {'clinical': this_slice, 'micro': fake_micro, 'small_clinical': this_slice, 'big_clinical': fake_micro, 'clinical_paths':'guagua', 'micro_paths':'guagua'}
                    model.set_input(data)
                    model.test()
                    visuals = model.get_current_visuals()
                    for label, im_data in visuals.items():
                        if label == 'fake_B':
                            thismicro_patch = im_data.cpu().numpy()
                            thismicro_patch = thismicro_patch[0,0,:,:]
                            largex_startpos = int(x_startpos*4)
                            largey_startpos = int(y_startpos*4)
                            # overlap operation
                            
                            if i == 0 and j == 0: # 矩阵的左边
                                result_array[largex_startpos:largex_startpos+patch_size[0],largey_startpos:largey_startpos+patch_size[1],k] = thismicro_patch
                            
                            elif i == 0 and j != 0:
                                result_array[largex_startpos:largex_startpos+patch_size[0],largey_startpos:largey_startpos+int(patch_size[1]*overlap_rate),k] = \
                                    (thismicro_patch[0:patch_size[0],0:int(patch_size[1]*overlap_rate)] + result_array[largex_startpos:largex_startpos+patch_size[0],largey_startpos:largey_startpos+int(patch_size[1]*overlap_rate),k]) / 2
                            
                                result_array[largex_startpos:largex_startpos+patch_size[0],largey_startpos+int(patch_size[1]*overlap_rate):largey_startpos+patch_size[1],k] = \
                                    thismicro_patch[0:patch_size[0],int(patch_size[1]*overlap_rate):patch_size[1]]
                            
                            elif i !=0 and j == 0:
                                result_array[largex_startpos:largex_startpos+int(patch_size[0]*overlap_rate),largey_startpos:largey_startpos+patch_size[1],k] = \
                                    (thismicro_patch[0:int(patch_size[0]*overlap_rate),0:patch_size[1]] + result_array[largex_startpos:largex_startpos+int(patch_size[0]*overlap_rate),largey_startpos:largey_startpos+patch_size[1],k]) / 2
                                
                                result_array[largex_startpos+int(patch_size[0]*overlap_rate):largex_startpos+patch_size[0],largey_startpos:largey_startpos+patch_size[1],k] = \
                                    thismicro_patch[int(patch_size[0]*overlap_rate):patch_size[0],0:patch_size[1]]
                            
                            else: # 最多的情况 需要考虑一个patch分成4个部分
                                result_array[largex_startpos:largex_startpos+int(patch_size[0]*overlap_rate),largey_startpos:largey_startpos+int(patch_size[1]*overlap_rate),k] = \
                                    (thismicro_patch[0:int(patch_size[0]*overlap_rate),0:int(patch_size[1]*overlap_rate)] + result_array[largex_startpos:largex_startpos+int(patch_size[0]*overlap_rate),largey_startpos:largey_startpos+int(patch_size[1]*overlap_rate),k]) / 2

                                result_array[largex_startpos:largex_startpos+int(patch_size[0]*overlap_rate),largey_startpos+int(patch_size[1]*overlap_rate):largey_startpos+patch_size[1],k] = \
                                    (thismicro_patch[0:int(patch_size[0]*overlap_rate),int(patch_size[1]*overlap_rate):patch_size[1]] + result_array[largex_startpos:largex_startpos+int(patch_size[0]*overlap_rate),largey_startpos+int(patch_size[1]*overlap_rate):largey_startpos+patch_size[1],k]) / 2

                                result_array[largex_startpos+int(patch_size[0]*overlap_rate):largex_startpos+patch_size[0],largey_startpos:largey_startpos+int(patch_size[1]*overlap_rate),k] = \
                                    (thismicro_patch[int(patch_size[0]*overlap_rate):patch_size[0],0:int(patch_size[1]*overlap_rate)] + result_array[largex_startpos+int(patch_size[0]*overlap_rate):largex_startpos+patch_size[0],largey_startpos:largey_startpos+int(patch_size[1]*overlap_rate),k]) / 2

                                result_array[largex_startpos+int(patch_size[0]*overlap_rate):largex_startpos+patch_size[0],largey_startpos+int(patch_size[1]*overlap_rate):largey_startpos+patch_size[1],k] = \
                                    thismicro_patch[int(patch_size[0]*overlap_rate):patch_size[0],int(patch_size[1]*overlap_rate):patch_size[1]]

            print('one slice!')
        cropstart = int((result_array.shape[0] - clinical_array.shape[0]) / 2)
        result_array = result_array[cropstart:cropstart+clinical_array.shape[0],cropstart:cropstart+clinical_array.shape[1],:]
        result_array = nib.Nifti1Image(result_array, np.eye(4))
        nib.save(result_array, os.path.join(savepath, 'SR'+name))
        clinical_array = nib.Nifti1Image(clinical_array, np.eye(4))
        nib.save(clinical_array, os.path.join(savepath, name))
        print('finished {}'.format(name))

# for meeting with Dr.Nakamura
def testclinical_lungcancervolume():
    opt = TestOptions().parse()
    opt.num_threads = 1
    opt.batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    model = create_model(opt)      # create a model given opt.model and other options 目前是medical_cycle_gan
    print("successfully created the model!")
    
    print("successfully loaded the network!")
    if opt.eval:
        model.eval()

    model.setup(opt)               # regular setup: load and print networks; create schedulers #把网络打印出来.参见base_mode    
    
    filenames = [
        # '/homes/tzheng/CTdata/CTMicroNUrespsurg/cct/nulung014-4.nii.gz',
        '/homes/tzheng/CTdata/CTMicroNUrespsurg/cct/nulung015-5.nii.gz',
        # '/homes/tzheng/CTdata/CTMicroNUrespsurg/cct/nulung034.512.nii.gz',
         
    ]
    
    savepath = '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/lungcancerSR'
    casenum = 2
    for name in filenames:
        clinical_array = nib.load(name).get_fdata()
        clinical_array = clinical_array.astype(np.float32)
        clinical_array = clinical_array - 2000
        clinical_array[clinical_array<-1000] = -1000 
        clinical_array[clinical_array>500] = 500
        clinical_array = normalizationmin0max1(clinical_array)
        clinical_array = clinical_array[:,:,270:300]

        ### save clinical array
        clinical_array = nib.Nifti1Image(clinical_array, np.eye(4))
        nib.save(clinical_array, '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/lungcancerSR/clinical015nii.gz')
        ###

        save_clinical_array = clinical_array.copy()
        print('successfuly normalizated clinical_array')
        test_start = 0
        test_end = save_clinical_array.shape[2]

        patch_size = [128, 128] # 确保patch是正方形的
        # down_patch_size = [int(a * 0.25) for a in patch_size]
        overlap_rate = 0.5 # 在inference的时候 patch的重合率
        
        fake_micro = np.zeros((192,192,1), dtype='float32')
        fake_micro = fake_micro.transpose(2,0,1)
        fake_micro = fake_micro[np.newaxis, :]
        fake_micro = torch.from_numpy(fake_micro)

        # padding patch to n times of patch size
        pddw = (clinical_array.shape[0] // patch_size[0] + 1) * patch_size[0]
        pddh = (clinical_array.shape[1] // patch_size[1] + 1) * patch_size[1]
        pad_array = np.zeros((pddw,pddh,clinical_array.shape[2]), dtype=np.float32)
        
        for num in range(0,clinical_array.shape[2]):
            pad_array[:,:,num] = np.pad(clinical_array[:,:,num], (int((pddw-clinical_array.shape[0]) / 2),), mode='reflect')
        # downsample_arr = zoom(pad_array, (0.25,0.25,1.0)) 
        
        SR_scale = 4
        result_array = np.zeros((int(pad_array.shape[0]*SR_scale), int(pad_array.shape[1]*SR_scale), int(pad_array.shape[2])), dtype=np.float32)
        slidewindow_num = int((pad_array.shape[0] - patch_size[0]) // (patch_size[0]*(1-overlap_rate)) + 1)
        
        large_patch_size = [int(a * 4) for a in patch_size]
        # 进行滑窗inference
        for k in range(test_start, test_end):
            for i in range(0, slidewindow_num):
                for j in range(0, slidewindow_num):
                    x_startpos = int(i * patch_size[0]*(1-overlap_rate))
                    y_startpos = int(j * patch_size[0]*(1-overlap_rate))
                    this_slice = pad_array[x_startpos:x_startpos+patch_size[0],y_startpos:y_startpos+patch_size[1],k]
                    
                    this_slice = this_slice[np.newaxis, np.newaxis, :]
                    this_slice = torch.from_numpy(this_slice)
                    data = {'clinical': this_slice, 'micro': fake_micro, 'small_clinical': this_slice, 'big_clinical': fake_micro, 'clinical_paths':'guagua', 'micro_paths':'guagua'}
                    model.set_input(data)
                    model.test()
                    visuals = model.get_current_visuals()
                    for label, im_data in visuals.items():
                        if label == 'fake_B':
                            thismicro_patch = im_data.cpu().numpy()
                            thismicro_patch = thismicro_patch[0,0,:,:]
                            largex_startpos = int(x_startpos*4)
                            largey_startpos = int(y_startpos*4)
                            # overlap operation
                            if i == 0 and j == 0: # 矩阵的左边
                                result_array[largex_startpos:largex_startpos+large_patch_size[0],largey_startpos:largey_startpos+large_patch_size[1],k] = thismicro_patch
                            
                            elif i == 0 and j != 0:
                                result_array[largex_startpos:largex_startpos+large_patch_size[0],largey_startpos:largey_startpos+int(large_patch_size[1]*overlap_rate),k] = \
                                    (thismicro_patch[0:large_patch_size[0],0:int(large_patch_size[1]*overlap_rate)] + result_array[largex_startpos:largex_startpos+large_patch_size[0],largey_startpos:largey_startpos+int(large_patch_size[1]*overlap_rate),k]) / 2
                            
                                result_array[largex_startpos:largex_startpos+large_patch_size[0],largey_startpos+int(large_patch_size[1]*overlap_rate):largey_startpos+large_patch_size[1],k] = \
                                    thismicro_patch[0:large_patch_size[0],int(large_patch_size[1]*overlap_rate):large_patch_size[1]]
                            
                            elif i !=0 and j == 0:
                                result_array[largex_startpos:largex_startpos+int(large_patch_size[0]*overlap_rate),largey_startpos:largey_startpos+large_patch_size[1],k] = \
                                    (thismicro_patch[0:int(large_patch_size[0]*overlap_rate),0:large_patch_size[1]] + result_array[largex_startpos:largex_startpos+int(large_patch_size[0]*overlap_rate),largey_startpos:largey_startpos+large_patch_size[1],k]) / 2
                                
                                result_array[largex_startpos+int(large_patch_size[0]*overlap_rate):largex_startpos+large_patch_size[0],largey_startpos:largey_startpos+large_patch_size[1],k] = \
                                    thismicro_patch[int(large_patch_size[0]*overlap_rate):large_patch_size[0],0:large_patch_size[1]]
                            
                            else: # 最多的情况 需要考虑一个patch分成4个部分
                                result_array[largex_startpos:largex_startpos+int(large_patch_size[0]*overlap_rate),largey_startpos:largey_startpos+int(large_patch_size[1]*overlap_rate),k] = \
                                    (thismicro_patch[0:int(large_patch_size[0]*overlap_rate),0:int(large_patch_size[1]*overlap_rate)] + result_array[largex_startpos:largex_startpos+int(large_patch_size[0]*overlap_rate),largey_startpos:largey_startpos+int(large_patch_size[1]*overlap_rate),k]) / 2

                                result_array[largex_startpos:largex_startpos+int(large_patch_size[0]*overlap_rate),largey_startpos+int(large_patch_size[1]*overlap_rate):largey_startpos+large_patch_size[1],k] = \
                                    (thismicro_patch[0:int(large_patch_size[0]*overlap_rate),int(large_patch_size[1]*overlap_rate):large_patch_size[1]] + result_array[largex_startpos:largex_startpos+int(large_patch_size[0]*overlap_rate),largey_startpos+int(large_patch_size[1]*overlap_rate):largey_startpos+large_patch_size[1],k]) / 2

                                result_array[largex_startpos+int(large_patch_size[0]*overlap_rate):largex_startpos+large_patch_size[0],largey_startpos:largey_startpos+int(large_patch_size[1]*overlap_rate),k] = \
                                    (thismicro_patch[int(large_patch_size[0]*overlap_rate):large_patch_size[0],0:int(large_patch_size[1]*overlap_rate)] + result_array[largex_startpos+int(large_patch_size[0]*overlap_rate):largex_startpos+large_patch_size[0],largey_startpos:largey_startpos+int(large_patch_size[1]*overlap_rate),k]) / 2

                                result_array[largex_startpos+int(large_patch_size[0]*overlap_rate):largex_startpos+large_patch_size[0],largey_startpos+int(large_patch_size[1]*overlap_rate):largey_startpos+large_patch_size[1],k] = \
                                    thismicro_patch[int(large_patch_size[0]*overlap_rate):large_patch_size[0],int(large_patch_size[1]*overlap_rate):large_patch_size[1]]

            print('one slice!')
        cropstart = int((result_array.shape[0] - clinical_array.shape[0]) / 2)
        result_array = nib.Nifti1Image(result_array, np.eye(4))
        print('Save file to: '+ os.path.join(savepath, 'SR'+str(casenum))+'.nii.gz')
        nib.save(result_array, os.path.join(savepath, 'SR'+str(casenum))+'.nii.gz')
        
def evaluation_PSNR_SSIM_folder():

    # SRfolder = '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/tmp/SRtrainwholedata/epoch55'
    SRfolder = '/homes/tzheng/code/pseudo-sr/test/COVID/tmptest/35820'
    Originalfolder = '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/tmp/HR'

    for curDir, dirs, SRfiles in os.walk(SRfolder):
        SRfilenames = SRfiles

    for curDir, dirs, Originalfiles in os.walk(Originalfolder):
        Originalfilenames = Originalfiles

    allPSNR = 0
    allSSIM = 0
    for i in range(len(SRfilenames)):
        SR_path = os.path.join(SRfolder, SRfilenames[i])
        Original_filename = SRfilenames[i]
        Original_filename = Original_filename[2:]
        Original_path = os.path.join(Originalfolder, Original_filename)
        print(Original_path)
        print(SR_path)
        
        SRarr = nib.load(SR_path).get_data()
        Originalarr = nib.load(Original_path).get_fdata()
        
        total_SSIM = 0
        total_PSNR = 0
        
        for i in range(SRarr.shape[2]):
            SSIM = my_ssim(SRarr[:,:,i], Originalarr[:,:,i])
            PSNR = my_psnr(SRarr[:,:,i], Originalarr[:,:,i])
            total_SSIM += SSIM
            total_PSNR += PSNR
            print('evaluated one slice')        

        total_SSIM /= SRarr.shape[2]
        total_PSNR /= SRarr.shape[2]
        
        print(total_SSIM)
        print(total_PSNR)
        
        allPSNR += total_PSNR
        allSSIM += total_SSIM

    allPSNR /= len(SRfilenames)
    allSSIM /= len(SRfilenames)

    print('average psnr is {}'.format(allPSNR))
    print('average ssim is {}'.format(allSSIM))
    

def evaluation_PSNR_SSIM():
    
    SRimage = '/homes/tzheng/code/pseudo-sr/test/COVID/SRvolume-covid19-A-0089_ct.nii.gz'
    Originalimage = '/homes/tzheng/code/pseudo-sr/test/COVID/volume-covid19-A-0089_ct.nii.gz'
    
    SRarr = nib.load(SRimage).get_fdata()
    Originalarr = nib.load(Originalimage).get_fdata()
    
    total_SSIM = 0
    total_PSNR = 0
    
    for i in range(SRarr.shape[2]):
        SSIM = my_ssim(SRarr[:,:,i], Originalarr[:,:,i])
        PSNR = my_psnr(SRarr[:,:,i], Originalarr[:,:,i])
        total_SSIM += SSIM
        total_PSNR += PSNR
        print('evaluated one slice')

    total_SSIM /= SRarr.shape[2]
    total_PSNR /= SRarr.shape[2]
    
    print(total_SSIM)  # 
    print(total_PSNR)  # 

if __name__ == '__main__':
    # testclinical_wholevolume()
    # testclinical_lungcancervolume()
    # evaluation_PSNR_SSIM()
    evaluation_PSNR_SSIM_folder()
    ''' # 目前来看是patch越大 定量精度越高
    patch 192: ssim 0.9558  PSNR: 21.5231
    patch 256: ssim 0.9743  PSNR: 23.3723
    patch 384: ssim 0.9855  PSNR: 25.3573
    patch 480: ssim 0.9874  PSNR: 25.9935
    patch 512: ssim 0.9880  PSNR: 26.1860 
    '''