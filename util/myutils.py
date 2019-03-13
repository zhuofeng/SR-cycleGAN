# -*- coding: utf8 -*-
import nibabel as nib
import os
import random
import math
from skimage.measure import block_reduce

import scipy
from scipy.ndimage.interpolation import zoom 
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import cv2
#import path
win_min=3000
win_max=12000

def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='RGB')

def crop_sub_imgs_fn(x, size, is_random=False):
    x = crop(x, wrg=64, hrg=64, is_random=is_random)
    return x

def crop_sub_imgs_fn3D(img, cropsize, is_random=False):
    imgshape = img.shape
    
    if is_random:
        x = random.randint(0, imgshape[0]-cropsize)
        y = random.randint(0, imgshape[1]-cropsize)
        z = random.randint(0, imgshape[2]-cropsize)
    else:
        x = math.ceil((imgshape[0] - cropsize)/2)
        y = math.ceil((imgshape[1] - cropsize)/2)
        z = math.ceil((imgshape[2] - cropsize)/2)
    
    img = img[x:x+cropsize, y:y+cropsize, z:z+cropsize]
    return img

def train_crop_sub_imgs_fn_andsmall3D(img, batchsize, cropsize, small_size, is_random=False):
    imgshape = img.shape
    imgbig = np.arange(batchsize*cropsize*cropsize*cropsize, dtype = 'float32').reshape(batchsize, cropsize, cropsize, cropsize, 1)
    imgsmall= np.arange(batchsize*small_size*small_size*small_size, dtype = 'float32').reshape(batchsize, small_size, small_size, small_size, 1)
    if is_random:
        for i in range(0, batchsize):
            x = random.randint(0, imgshape[0]-cropsize)
            y = random.randint(0, imgshape[1]-cropsize)
            z = random.randint(0, imgshape[2]-cropsize)
            imgbig[i,:,:,:,0] = img[x:x+cropsize, y:y+cropsize, z:z+cropsize]
            
    else:
        for i in range(0, batchsize):
            x = math.ceil((imgshape[0] - cropsize)/2)
            y = math.ceil((imgshape[1] - cropsize)/2)
            z = math.ceil((imgshape[2] - cropsize)/2)
            imgbig[i,:,:,:,0] = img[x:x+cropsize, y:y+cropsize, z:z+cropsize]
    imgsmall = block_reduce(imgbig, block_size = (1,8,8,8,1), func=np.mean)
    imgsmall = zoom(imgsmall, (1,8.,8.,8.,1))
    
    return imgbig, imgsmall

def train_crop_both_imgs_fn_andsmall3D(imgbig, imgsmall, cropsize, is_random=False):
    imgshape = imgbig.shape
    if is_random:
        x = random.randint(0, imgshape[0]-cropsize)
        y = random.randint(0, imgshape[1]-cropsize)
        z = random.randint(0, imgshape[2]-cropsize)
        imgpatchbig = imgbig[x:x+cropsize, y:y+cropsize, z:z+cropsize]
        imgpatchsmall = imgsmall[x:x+cropsize, y:y+cropsize, z:z+cropsize]
    else:
        x = math.ceil((imgshape[0] - cropsize)/2)
        y = math.ceil((imgshape[1] - cropsize)/2)
        z = math.ceil((imgshape[2] - cropsize)/2)
        imgpatchbig = imgbig[x:x+cropsize, y:y+cropsize, z:z+cropsize]
        imgpatchsmall = imgsmall[x:x+cropsize, y:y+cropsize, z:z+cropsize]
    
    return imgpatchbig, imgpatchsmall

def train_crop_both_imgs_fn_andsmall(imgbig, imgsmall, cropsize, is_random=False):
    imgshape = imgbig.shape
    if is_random:
        x = random.randint(0, imgshape[0]-cropsize)
        y = random.randint(0, imgshape[1]-cropsize)
        z = random.randint(0, imgshape[2]-cropsize)
        imgpatchbig = imgbig[x:x+cropsize, y:y+cropsize, z]
        imgpatchsmall = imgsmall[x:x+cropsize, y:y+cropsize, z]
    else:
        x = math.ceil((imgshape[0] - cropsize)/2)
        y = math.ceil((imgshape[1] - cropsize)/2)
        z = math.ceil((imgshape[2] - cropsize)/2)
        imgpatchbig = imgbig[x:x+cropsize, y:y+cropsize, z]
        imgpatchsmall = imgsmall[x:x+cropsize, y:y+cropsize, z]
    
    return imgpatchbig, imgpatchsmall

def valid_crop_sub_imgs_fn_andsmall3D(img, xaxis, yaxis, zaxis, batchsize, cropsize, small_size, is_random=False):
    imgshape = img.shape #(1024, 1024, 64)
    imgbig = np.arange(batchsize*cropsize*cropsize*cropsize, dtype = 'float32').reshape(batchsize, cropsize, cropsize, cropsize, 1)
    imgsmall= np.arange(batchsize*small_size*small_size*small_size, dtype = 'float32').reshape(batchsize, small_size, small_size, small_size, 1)
    if is_random:
        for i in range(0, batchsize):
            x = random.randint(0, imgshape[0]-cropsize)
            y = random.randint(0, imgshape[1]-cropsize)
            z = random.randint(0, imgshape[2]-cropsize)
            imgbig[i,:,:,:,0] = img[x:x+cropsize, y:y+cropsize, z:z+cropsize]
            
    else:
        for i in range(0, batchsize):
            x = xaxis
            y = yaxis
            z = zaxis
            imgbig[i,:,:,:,0] = img[x:x+cropsize, y:y+cropsize, z:z+cropsize]
    imgsmall = block_reduce(imgbig, block_size = (1,8,8,8,1), func=np.mean)
    imgsmall = zoom(imgsmall, (1,8.,8.,8.,1))
    
    return imgsmall

def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    #print("before downsample:")
    #print(x.shape)
    #x = imresize(x, size=[96, 96], interp='bicubic', mode=None)
    #print(x.shape)
    #gaussian blurring
    #x = gaussian_filter(x, 2, order=0, output=None, mode='reflect')
    x = zoom(x, (0.125,0.125,1.0))  #8timesdownsampling
    #print(x.shape) #(96,96,3)
    return x

def downsample_zoom_fn(x):
    x = block_reduce(x, block_size = (8, 8, 1), func=np.mean)
    x = zoom(x, (8, 8, 1))
    return x
def downsample_fn2(x):
    x = zoom(x, (1,0.25,0.25))
    return x

def normalizationminmax1threhold(data):
    print('min/max data: {}/{} => {}/{}'.format(np.min(data),np.max(data),win_min,win_max))
    data = np.float32(data)
    data[data<win_min] = win_min
    data[data>win_max] = win_max
    data = data-np.min(data) 
    max = np.max(data)
    data = data - (max / 2.)
    data = data / max
    return data

def normalizationminmax1(data):
    print('min/max data: {}/{}'.format(np.min(data),np.max(data)))
    data = np.float32(data)
    max = np.max(data)
    min = np.min(data)
    data = data-min
    newmax = np.max(data)
    data = (data-(newmax/2)) / (newmax/2.)
    #print('this is the minmax of normalization')
    #print(np.max(data))
    #print(np.min(data))
    return data
def normalizationclinicalminmax1(data):
    print('min/max data: {}/{}'.format(np.min(data),np.max(data)))
    data = np.float32(data)
    max = np.max(data)
    min = np.min(data)
    data = data-min
    newmax = np.max(data)
    data = (data-(newmax/2)) / (newmax/2.)
    #print('this is the minmax of normalization')
    #print(np.max(data))
    #print(np.min(data))
    return data

def normalizationmicrominmax1(data):
    print('min/max data: {}/{}'.format(np.min(data),np.max(data)))
    data = np.float32(data)
    data[data<0.] = 0.
    data[data>15000.] = 15000.
    max = np.max(data)
    min = np.min(data)
    data = data-min
    newmax = np.max(data)
    data = (data-(newmax/2)) / (newmax/2.)
    #print('this is the minmax of normalization')
    #print(np.max(data))
    #print(np.min(data))
    return data

def normalizationmin0max1(data):
    print('min/max data: {}/{}'.format(np.min(data),np.max(data)))
    data = np.float32(data)
    data[data<0.] = 0.
    data[data>12000] = 12000
    #data[data<2000] = 2000
    #data = (data-(newmax/2)) / (newmax/2.)
    data = data / 12000.
    print('this is the minmax of normalization')
    print(np.max(data))
    print(np.min(data))
    return data

def normalizationtominmax(data):
    data[data<win_min] = win_min
    data[data>win_max] = win_max
    data = data-np.min(data)
    return data
def normalizationtoimg(data):
    print('min/max data: {}/{} => {}/{}'.format(np.min(data),np.max(data),win_min,win_max))
    data = data-np.min(data)
    data = data * (255.0/np.max(data))
    return data
def my_psnr(im1,im2):
    mse = ((im1 - im2) ** 2.).mean(axis=None)
    rmse = np.sqrt(mse)
    psnr = 20.*np.log10(1./rmse)
    return psnr

def my_ssim(im1,im2):
    mu1 = np.mean(im1)
    mu2 = np.mean(im2)
    c1 = 1e-4
    c2 = 1e-4
    sigma1 = np.std(im1)
    sigma2 = np.std(im2)
    
    im1 = im1 - mu1
    im2 = im2 - mu2
    cov12 = np.mean(np.multiply(im1,im2))
    
    
    ssim = (2*mu1*mu2+c1) * (2*cov12+c2) / (mu1**2+mu2**2+c1) / (sigma1**2 + sigma2**2 + c2)
    return ssim
def readnii(path):
    dpath = path
    img = nib.load(dpath)
    #print("this is the shape of img:{}".format(img.shape))
    #print(type(img)) #<class 'nibabel.nifti1.Nifti1Image'>
    #print("this is the shape of img.affine.shape:{}")
    #print("this is the header of img{}".format(img.header))
    data = img.get_fdata()
    #print(data.shape) #1024*1024*549
    #print(type(data))  #<class 'numpy.ndarray'>
    return data, img.header

def backtoitensity(path):
    
    #get the header
    correspondingimg = nib.load('/homes/tzheng/CTdata/CTMicroNUrespsurg/converted/DICOM_nulung026_cb_003_zf_ringRem.nii.gz')
    correspondingheader = correspondingimg.header
    empty_header = nib.Nifti1Header()
    empty_header = correspondingheader
    #print(empty_header)
    #print(correspondingimg.affine)
    #正规化导致neuves不能正常渲染
    
    thisimg = correspondingimg.get_fdata()
    valid_hr_slices = thisimg.shape[2]
    
    dpath = path
    img = nib.load(dpath)
    data = img.get_fdata()
    data = data * 12000.
    
    thisimg[160:810,160:810,int(valid_hr_slices*0.1/8)*8+10:int(valid_hr_slices*0.1/8)*8+410] = data[10:660,10:660,10:410]
    
    #saveimg = nib.Nifti1Image(data, correspondingimg.affine, empty_header)
    saveimg = nib.Nifti1Image(thisimg, correspondingimg.affine, empty_header)
    nib.save(saveimg, '/homes/tzheng/Mypythonfiles/densunetdiscirminator/samples/medicaltest/SRbacktoitensity.nii.gz')
    
def mean_squared_error3d(output, target, is_mean=False, name="mean_squared_error"):
    if output.get_shape().ndims == 5:  # [batch_size, l, w, h, c]
        if is_mean:
            mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), [1, 2, 3]), name=name)
        else:
            mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), [1, 2, 3]), name=name)
    else:
        raise Exception("Unknow dimension")
    return mse

def fz(a):
    return a[::-1]
def FZ(mat):
    return np.array(fz(list(map(fz, mat))))

def rotatenii_180():
    img = nib.load(config.VALID.Clinicalmedical_path)
    data = img.get_fdata()
    empty_header = nib.Nifti1Header()
    empty_header = img.header
    for i in range(0, data.shape[2]):
        tempimg = FZ(data[:,:,i])
        data[:,:,i] = tempimg
    data = nib.Nifti1Image(data, img.affine, empty_header) 
    nib.save(data, '/homes/tzheng/Mypythonfiles/densunetdiscirminator/samples/medicaltest3D/rotatedclinical.nii.gz')

def crop_nifti_2D(img, cropsize, is_random=False):
    imgshape = img.shape
    if is_random:
        x = random.randint(0, imgshape[0]-cropsize)
        y = random.randint(0, imgshape[1]-cropsize)
        z = random.randint(0, imgshape[2]-1)
        imgpatch = img[x:x+cropsize, y:y+cropsize, z]
    else:
        x = math.ceil((imgshape[0] - cropsize)/2)
        y = math.ceil((imgshape[1] - cropsize)/2)
        z = math.ceil(imgshape[2] / 2)
        imgpatch = img[x:x+cropsize, y:y+cropsize, z]
    
    return imgpatch

def crop_nifti_withpos_2D(img, cropsize, is_random=False):
    imgshape = img.shape
    if is_random:
        x = random.randint(0, imgshape[0]-cropsize)
        y = random.randint(0, imgshape[1]-cropsize)
        z = random.randint(0, imgshape[2]-1)
        imgpatch = img[x:x+cropsize, y:y+cropsize, z]
    else:
        x = math.ceil((imgshape[0] - cropsize)/2)
        y = math.ceil((imgshape[1] - cropsize)/2)
        z = math.ceil(imgshape[2] / 2)
        imgpatch = img[x:x+cropsize, y:y+cropsize, z]
    
    return imgpatch, (x,y,z)

def dilatenifitimask():
    mask = nib.load('/homes/tzheng/CTdata/CTMicroNUrespsurg/Mask/nulung036mask.nii.gz')
    maskdata = mask.get_fdata()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
    
    for i in range(maskdata.shape[2]):
        maskdata[:,:,i] = cv2.dilate(maskdata[:,:,i], kernel)

    dilated = nib.Nifti1Image(maskdata, mask.affine, mask.header) 
    nib.save(dilated, '/homes/tzheng/CTdata/CTMicroNUrespsurg/Mask/nulung036diatedmask.nii.gz')

if __name__ == '__main__':
    #接下来要做的实验：提取Clinical CT的肺部领域，再正规化，与正规化的microCT进行比较
    clinical_path = '/homes/tzheng/CTdata/CTMicroNUrespsurg/cct/nulung030.nii.gz'
    micro_path = '/homes/tzheng/CTdata/CTMicroNUrespsurg/nii/nulung050/nulung050_053_000.nii.gz'
    clinical_mask_path = '/homes/tzheng/CTdata/CTMicroNUrespsurg/Mask/nulung030diatedmask.nii.gz'

    clinical = nib.load(clinical_path)
    micro = nib.load(micro_path)
    clinical_mask = nib.load(clinical_mask_path)

    clinical_data = clinical.get_fdata()
    micro_data = micro.get_fdata()
    clinical_mask_data = clinical_mask.get_fdata()

    maxdata = np.max(clinical_data[clinical_mask_data>0])
    mindata = np.min(clinical_data[clinical_mask_data>0])

    print(maxdata)
    print(mindata)

    