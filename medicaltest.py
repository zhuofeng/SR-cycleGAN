"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
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
import time


def patchtest(): 
    opt = TestOptions().parse()
    opt.num_threads = 1
    opt.batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    #create dataset

    opt.dataset_mode = 'medical_2D_test'
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options 目前是medical_cycle_gan
    model.setup(opt)               # regular setup: load and print networks; create schedulers #把网络打印出来.参见base_mode

    if opt.eval:
        model.eval()
    opt.num_test = int(512*512/32/32)

    real_clinical = np.zeros((512,512,1), dtype='float32')
    fake_micro = np.zeros((4096,4096,1), dtype='float32')
    #real_micro = np.zeros((1024,1024,1), dtype='float32')
    #fake_clinical = np.zeros((128,128,1), dtype='float32')

    for i, data in enumerate(dataset):
        i = int(i)
        if i>= opt.num_test:
            break
        model.set_input(data)
        #print(data['clinical'].shape)
        #print(type(data)) #<class 'dict'>
        #print(len(data)) #4
        model.test()
        visuals = model.get_current_visuals()
        for label, im_data in visuals.items():
            #print(label) #real_A fake_B rec_A real_B fake_A rec_B
            if i < 256:
                if label == 'real_A':
                    real_clinical[int((i//16)*32):int(((i//16)+1)*32),int((i%16)*32):int((i%16+1)*32),0] = im_data
                '''
                if label == 'fake_B':
                    fake_micro[int((i//16)*256):int(((i//16)+1)*256),int((i%16)*256):int((i%16+1)*256),0] = im_data
                if i < 16:
                    if label == 'real_B':
                        real_micro[int((i//4)*256):int(((i//4)+1)*256),int((i%4)*256):int((i%4+1)*256),0] = im_data
                    if label == 'fake_A':
                        fake_clinical[int((i//4)*32):int(((i//4)+1)*32),int((i%4)*32):int((i%4+1)*32),0] = im_data
                '''
            else:
                break

    real_clinical = real_clinical[0:256, 0:256, :]
    fake_micro = fake_micro[0:2048, 0:2048, :]

    real_clinical = real_clinical.transpose(2,0,1)
    fake_micro = fake_micro.transpose(2,0,1)
    real_clinical = real_clinical[np.newaxis, :]
    fake_micro = fake_micro[np.newaxis, :]

    real_clinical = torch.from_numpy(real_clinical)
    fake_micro = torch.from_numpy(fake_micro)
    print(real_clinical.shape)

    data = {'clinical': real_clinical, 'micro': fake_micro, 'clinical_paths':'haha', 'micro_paths':'haha'}
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    
    finalmicro = np.zeros((4096,4096,1), dtype='float32')

    for label, im_data in visuals.items():
        if label == 'fake_B':
            finalmicro = im_data.cpu().numpy()
            finalmicro = finalmicro[0,:,:,:]
            print(finalmicro.shape)
    finalmicro = nib.Nifti1Image(finalmicro, np.eye(4))
    nib.save(finalmicro, './CTdata/superoluteclinical.nii.gz')

def smallpatchtest():
    opt = TestOptions().parse()
    opt.num_threads = 1
    opt.batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    model = create_model(opt)      # create a model given opt.model and other options 目前是medical_cycle_gan
    print("successfully created the model!")
    model.setup(opt)               # regular setup: load and print networks; create schedulers #把网络打印出来.参见base_mode
    print("successfully loaded the network!")
    if opt.eval:
        model.eval()
    
    #create dataset
    test_start = 395
    test_end = 405

    # patch size for inputting into the network
    patch_size = (192, 256)
    fake_micro = np.zeros((1536,2048,1), dtype='float32')
    fake_micro = fake_micro.transpose(2,0,1)
    fake_micro = fake_micro[np.newaxis, :]
    fake_micro = torch.from_numpy(fake_micro)
    
    final_micro = np.zeros((1536,2048,test_end-test_start), dtype='float32')
    clinical_path = '/homes/tzheng/CTdata/CTMicroNUrespsurg/cct/nulung032.512.nii.gz'
    clinical_array = nib.load(clinical_path).get_fdata()
    clinical_array = clinical_array[78:270, 145:401, :]
    clinical_array = np.float32(clinical_array)
    clinical_array = normalizationclinicalminmax1(clinical_array)
    print(clinical_array.shape)
    print('successfuly normalizated clinical_array')

    save_clinical_array = nib.Nifti1Image(clinical_array[:,:,test_start:test_end], np.eye(4))
    nib.save(save_clinical_array, './results/clinical.nii.gz')
    print('successfuly saved clinical CT')
    
    for k in range(test_start, test_end):
        for i in range(0, int(clinical_array.shape[0] / patch_size[0])):
            for j in range(0, int(clinical_array.shape[1] / patch_size[1])):
                clinical_slice = clinical_array[i*patch_size[0]:(i+1)*patch_size[0], j*patch_size[1]:(j+1)*patch_size[1], k:k+1]
                #print('this is shape of clinial_slice')
                #print(clinical_slice.shape)
                clinical_slice = clinical_slice.transpose(2,0,1)
                clinical_slice = clinical_slice[np.newaxis, :]
                clinical_slice = torch.from_numpy(clinical_slice)
                data = {'clinical': clinical_slice, 'micro': fake_micro, 'clinical_paths':'guagua', 'micro_paths':'guagua'}
                model.set_input(data)
                model.test()
                visuals = model.get_current_visuals()
                for label, im_data in visuals.items():
                    if label == 'fake_B':
                        thismicro_patch = im_data.cpu().numpy()
                        thismicro_patch = thismicro_patch[0,0,:,:]
                        final_micro[8*i*patch_size[0]:8*(i+1)*patch_size[0], 8*j*patch_size[1]:8*(j+1)*patch_size[1], k-test_start] = thismicro_patch
        print('one slice!')
    final_micro = nib.Nifti1Image(final_micro, np.eye(4))
    nib.save(final_micro, './results/SR.nii.gz')

def microCTsmallpatchtest(): 
    opt = TestOptions().parse()
    opt.num_threads = 1
    opt.batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    model = create_model(opt)      # create a model given opt.model and other options 目前是medical_cycle_gan
    model.setup(opt)               # regular setup: load and print networks; create schedulers #把网络打印出来.参见base_mode
    if opt.eval:
        model.eval()
    
    #create dataset
    patch_size = (256, 256)
    fake_micro = np.zeros((512,512,1), dtype='float32') 
    fake_micro = fake_micro.transpose(2,0,1)
    fake_micro = fake_micro[np.newaxis, :]
    fake_micro = torch.from_numpy(fake_micro)

    micro_path = '/homes/tzheng/CTdata/CTMicroNUrespsurg/nii/nulung030/DICOM_nulung030_cb_004_zf_ringRem_med3_cropped.nii.gz'
    micro_array = nib.load(micro_path).get_fdata()
    micro_array = np.float32(micro_array)
    micro_array = normalizationmicrominmax1(micro_array)
    print(micro_array.shape) 
    micro_array = micro_array[0:512,0:512,:]

    save_micro_array = nib.Nifti1Image(micro_array[:], np.eye(4))
    nib.save(save_micro_array, './results/micro_array.nii.gz')
    final_micro = np.zeros((micro_array.shape[0],micro_array.shape[1],micro_array.shape[2]), dtype='float32')

    downsample_micro_array = zoom(micro_array, (0.125,0.125,1.0))
    save_downsample_micro_array = nib.Nifti1Image(downsample_micro_array[:], np.eye(4))
    nib.save(save_downsample_micro_array, './results/save_downsample_micro_array_07_20.nii.gz')
    
    print('successfuly normalizated and downsampled micro_array')
    
    for k in range(0, micro_array.shape[2]):
        for i in range(0, int(micro_array.shape[0] / patch_size[0])):
            for j in range(0, int(micro_array.shape[1] / patch_size[1])):
                clinical_slice = downsample_micro_array[int(i*patch_size[0]/8):int((i+1)*patch_size[0]/8), int(j*patch_size[1]/8):int((j+1)*patch_size[1]/8), k:k+1]
                clinical_slice = clinical_slice.transpose(2,0,1)
                clinical_slice = clinical_slice[np.newaxis, :]
                clinical_slice = torch.from_numpy(clinical_slice)
                data = {'clinical': clinical_slice, 'micro': fake_micro, 'clinical_paths':'guagua', 'micro_paths':'guagua'}
                model.set_input(data)
                model.test()
                visuals = model.get_current_visuals()
                for label, im_data in visuals.items():
                    if label == 'fake_B':
                        thismicro_patch = im_data.cpu().numpy()
                        thismicro_patch = thismicro_patch[0,0,:,:]
                        final_micro[i*patch_size[0]:(i+1)*patch_size[0], j*patch_size[1]:(j+1)*patch_size[1], k] = thismicro_patch

        print("finished one slice!")
    final_micro = nib.Nifti1Image(final_micro, np.eye(4))
    nib.save(final_micro, './results/SuperoluteMicro_07_20_patch256.nii.gz')

def SR_mask_lung():
    lungpath_1 = '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/FOR_JMI/lung032/UNITclinical_01.nii.gz'
    lungpath_2 = '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/FOR_JMI/lung032/UNITclinical_02.nii.gz'
    lungmask = '/homes/tzheng/CTdata/CTMicroNUrespsurg/Mask/nulung032closedmask.nii.gz'
    clinical_path = '/homes/tzheng/CTdata/CTMicroNUrespsurg/cct/nulung032.512.nii.gz'
    test_start = 395
    test_end = 405
    clinical_array = nib.load(clinical_path).get_fdata()
    clinical_array = clinical_array[:,:,test_start:test_end]
    clinical_array = normalizationclinicalminmax1(clinical_array)

    big_clinical_array = np.zeros((4096, 4096, 10), dtype='float32')
    for i in range(clinical_array.shape[2]):
        big_clinical_array[:,:,i] = cv2.resize(clinical_array[:,:,i], None, fx=8, fy=8,interpolation=cv2.INTER_NEAREST)
    
    save_clinical_array = nib.Nifti1Image(big_clinical_array, np.eye(4))
    #nib.save(save_clinical_array, '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/FOR_JMI/lung032/clinicalnearest.nii.gz')
    
    clinicalmask = nib.load(lungmask).get_fdata()
    SRlung_1 = nib.load(lungpath_1).get_fdata()
    SRlung_2 = nib.load(lungpath_2).get_fdata()

    
    clinicalmask = clinicalmask[:,:,test_start:test_end]
    
    big_clinicalmask = np.zeros((4096, 4096, 10), dtype='float32')
    for j in range(clinicalmask.shape[2]):
        big_clinicalmask[:,:,j] = cv2.resize(clinicalmask[:,:,j], None, fx=8, fy=8,interpolation=cv2.INTER_NEAREST)

    big_clinicalmask = big_clinicalmask.astype("bool")
    big_reverse_clinialmask = ~big_clinicalmask
    big_reverse_clinialmask = big_reverse_clinialmask.astype("float32")
    big_clinical_array = big_clinical_array * big_reverse_clinialmask
    big_clinicalmask = big_clinicalmask.astype("float32")
    #clinical_array = clinical_array[278:438, 146:402, :] #之前的一组数据
    #clinical_array = clinical_array[78:270, 145:401, :] #之前的一组数据
    SRlung_1 = SRlung_1
    SRlung_2 = SRlung_2
    SRlung_1 = SRlung_1 * big_clinicalmask[2224:3504, 1168:3216, :]
    SRlung_2 = SRlung_2 * big_clinicalmask[624:2160, 1160:3208, :]
    big_clinical_array[2224:3504, 1168:3216, :] = big_clinical_array[2224:3504, 1168:3216, :] + SRlung_1
    big_clinical_array[624:2160, 1160:3208, :] = big_clinical_array[624:2160, 1160:3208, :] + SRlung_2
    save_SR_clinical_array = nib.Nifti1Image(big_clinical_array, np.eye(4))
    nib.save(save_SR_clinical_array, '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/FOR_JMI/lung032/UNIT_SRwholeclinical.nii.gz')

def testclinical(): 
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
    #create dataset
    test_start = 0
    test_end = 52
    #控制输入网络的patch大小
    patch_size = (128, 128)
    fake_micro = np.zeros((512,512,1), dtype='float32')
    fake_micro = fake_micro.transpose(2,0,1)
    fake_micro = fake_micro[np.newaxis, :]
    fake_micro = torch.from_numpy(fake_micro)
    final_micro = np.zeros((512,512,test_end-test_start), dtype='float32')

    clinical_path = '/homes/tzheng/code/COVID_challenge/COVID-19-20_v2/Validation/volume-covid19-A-0064_ct.nii.gz'
    clinical_array = nib.load(clinical_path).get_fdata()
    clinical_array = clinical_array.astype(np.float32)
    clinical_array[clinical_array<-1000] = -1000
    clinical_array[clinical_array>500] = 500
    clinical_array = normalizationmin0max1(clinical_array)
    save_clinical_array = clinical_array.copy()
    clinical_array = zoom(clinical_array, (0.25,0.25,1.0)) 
    print('successfuly normalizated clinical_array')

    for k in range(test_start, test_end):
        for i in range(0, int(clinical_array.shape[0] / patch_size[0])):
            for j in range(0, int(clinical_array.shape[1] / patch_size[1])):
                clinical_slice = clinical_array[i*patch_size[0]:(i+1)*patch_size[0], j*patch_size[1]:(j+1)*patch_size[1], k:k+1]
                #print('this is shape of clinial_slice')
                #print(clinical_slice.shape)
                clinical_slice = clinical_slice.transpose(2,0,1)
                clinical_slice = clinical_slice[np.newaxis, :]
                clinical_slice = torch.from_numpy(clinical_slice)
                data = {'clinical': clinical_slice, 'micro': fake_micro, 'small_clinical': clinical_slice, 'big_clinical': fake_micro, 'clinical_paths':'guagua', 'micro_paths':'guagua'}
                model.set_input(data)
                model.test()
                visuals = model.get_current_visuals()
                for label, im_data in visuals.items():
                    if label == 'fake_B':
                        thismicro_patch = im_data.cpu().numpy()
                        #print('this is the shape of fake_B')
                        #print(thismicro_patch.shape)
                        #save_micro_slice = nib.Nifti1Image(thismicro_patch[0,:,:,0], np.eye(4))
                        #nib.save(save_micro_slice, './CTdata/savemicropatch.nii.gz')
                        thismicro_patch = thismicro_patch[0,0,:,:]
                        final_micro[8*i*patch_size[0]:8*(i+1)*patch_size[0], 8*j*patch_size[1]:8*(j+1)*patch_size[1], k-test_start] = thismicro_patch
        print('one slice!')
    final_micro = nib.Nifti1Image(final_micro, np.eye(4))
    nib.save(final_micro, '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/SR.nii.gz')
    clinical_array = nib.Nifti1Image(clinical_array, np.eye(4))
    nib.save(clinical_array, '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/LR.nii.gz')
    save_clinical_array = nib.Nifti1Image(save_clinical_array, np.eye(4))
    nib.save(save_clinical_array, '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/HR.nii.gz')
     
if __name__ == '__main__':
    smallpatchtest()
    # testclinical()
    # microCTsmallpatchtest()