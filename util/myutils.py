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
# import cv2
from scipy import ndimage
import torch
import re
from PIL import Image

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

def my_psnr(im1,im2): #计算PSNR
    mse = ((im1 - im2) ** 2.).mean(axis=None)
    rmse = np.sqrt(mse)
    psnr = 20.*np.log10(1./rmse)
    return psnr
#计算SSIM

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

# crop 2D patches from multi-axis
def crop_nifti_2D_3axis(img, cropsize, is_random=False):
    imgshape = img.shape
    from_whichaxis = np.random.randint(1,4) # 1 ~ 3
    if is_random:
        if from_whichaxis == 1:
            x = random.randint(0, imgshape[0]-1)
            y = random.randint(0, imgshape[1]-cropsize)
            z = random.randint(0, imgshape[2]-cropsize)
            imgpatch = img[x, y:y+cropsize, z:z+cropsize]
        elif from_whichaxis == 2:
            x = random.randint(0, imgshape[0]-cropsize)
            y = random.randint(0, imgshape[1]-1)
            z = random.randint(0, imgshape[2]-cropsize)
            imgpatch = img[x:x+cropsize, y, z:z+cropsize]
        elif from_whichaxis == 3:
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

# crop 2D patches from multi-axis
def crop_nifti_withpos_2D_3axis(img, cropsize, is_random=False):
    imgshape = img.shape
    from_whichaxis = np.random.randint(1,4) # 1 ~ 3
    if is_random:
        if from_whichaxis == 1:
            x = random.randint(0, imgshape[0]-1)
            y = random.randint(0, imgshape[1]-cropsize)
            z = random.randint(0, imgshape[2]-cropsize)
            imgpatch = img[x, y:y+cropsize, z:z+cropsize]
        elif from_whichaxis == 2:
            x = random.randint(0, imgshape[0]-cropsize)
            y = random.randint(0, imgshape[1]-1)
            z = random.randint(0, imgshape[2]-cropsize)
            imgpatch = img[x:x+cropsize, y, z:z+cropsize]
        elif from_whichaxis == 3:
            x = random.randint(0, imgshape[0]-cropsize)
            y = random.randint(0, imgshape[1]-cropsize)
            z = random.randint(0, imgshape[2]-1)
            imgpatch = img[x:x+cropsize, y:y+cropsize, z]
    else:
        x = math.ceil((imgshape[0] - cropsize)/2)
        y = math.ceil((imgshape[1] - cropsize)/2)
        z = math.ceil(imgshape[2] / 2)
        imgpatch = img[x:x+cropsize, y:y+cropsize, z]
    
    return imgpatch, (x,y,z), from_whichaxis

def crop_nifti_withpos_3D(img, cropsize, is_random=False):
    imgshape = img.shape
    if is_random :
        x = random.randint(0, imgshape[0]-cropsize)
        y = random.randint(0, imgshape[1]-cropsize)
        z = random.randint(0, imgshape[2]-cropsize)
        imgpatch = img[x:x+cropsize, y:y+cropsize, z:z+cropsize]
    else:
        x = math.ceil((imgshape[0] - cropsize)/2)
        y = math.ceil((imgshape[1] - cropsize)/2)
        z = math.ceil((imgshape[2] - cropsize)/2)
        imgpatch = img[x:x+cropsize, y:y+cropsize, z:z+cropsize]
    return imgpatch, (x,y,z)

# def dilatenifitimask(): #图像膨胀
#     mask = nib.load('/homes/tzheng/CTdata/CTMicroNUrespsurg/Air_Mask/nulung032.512_AIR_MASK.nii.gz')
#     maskdata = mask.get_fdata()
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
    
#     for i in range(maskdata.shape[2]):
#         maskdata[:,:,i] = cv2.dilate(maskdata[:,:,i], kernel)

#     dilated = nib.Nifti1Image(maskdata, mask.affine, mask.header) 
#     nib.save(dilated, '/homes/tzheng/CTdata/CTMicroNUrespsurg/Air_Mask/nulung032.512_AIR_MASK_dilated.nii.gz')

# def dilate_erode_niftimask(): #先膨胀再腐蚀 这种做法比较符合实际情况
#     mask = nib.load('/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/AftermodifingGenerator/SRclinical_epoch60_BINARIZATION.nii.gz')
#     maskdata = mask.get_fdata()
#     kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
#     kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
#     for i in range(maskdata.shape[2]):
#         maskdata[:,:,i] = cv2.dilate(maskdata[:,:,i], kernel1)
#         maskdata[:,:,i] = cv2.erode(maskdata[:,:,i], kernel2)

#     print(maskdata.shape)
#     newmask = nib.Nifti1Image(maskdata, mask.affine, mask.header) 
#     nib.save(newmask, '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/AftermodifingGenerator/SRclinical_epoch60_BINARIZATION_eroded.nii.gz')

# def erode_dilate_niftimask(): #先腐蚀再膨胀 实践证明不需要这个操作
#     mask = nib.load('/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/AftermodifingGenerator/SRclinical_epoch60_BINARIZATION_eroded.nii.gz')
#     maskdata = mask.get_fdata()
#     kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
#     kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
#     for i in range(maskdata.shape[2]):
#         maskdata[:,:,i] = cv2.erode(maskdata[:,:,i], kernel1)
#         maskdata[:,:,i] = cv2.dilate(maskdata[:,:,i], kernel2)

#     print(maskdata.shape)
#     newmask = nib.Nifti1Image(maskdata, mask.affine, mask.header) 
#     nib.save(newmask, '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/AftermodifingGenerator/SRclinical_epoch60_BINARIZATION_multi_opre.nii.gz')

# def closeimage(): #图像的闭运算
#     mask = nib.load('/homes/tzheng/CTdata/CTMicroNUrespsurg/Mask/nulung016diatedmask.nii.gz')
#     maskdata = mask.get_fdata()
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))

#     for i in range(maskdata.shape[2]):
#         maskdata[:,:,i] = cv2.morphologyEx(maskdata[:,:,i], cv2.MORPH_CLOSE, kernel)
    
#     closed = nib.Nifti1Image(maskdata, mask.affine, mask.header) 
#     nib.save(closed, '/homes/tzheng/CTdata/CTMicroNUrespsurg/Mask/nulung016closedmask.nii.gz')
    
def block_mean(ar, fact):
    assert isinstance(fact, int), type(fact)
    sx, sy = ar.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy/fact * (X/fact) + Y/fact
    res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
    res.shape = (int(sx/fact), int(sy/fact))
    return res

def sobelfilter_2d(src, kernel):
    m, _ = kernel.shape
    d = int((m-1)/2)
    h, w = src.shape[0], src.shape[1]

    dst = np.zeros((h, w))

    for y in range(d, h - d):
        for x in range(d, w - d):
            dst[y][x] = np.sum(src[y-d:y+d+1, x-d:x+d+1]*kernel)
    
    return dst

def sobelfilter_torchtesor_2d(src, kernel):
    #src尺寸[1,1,hight, width]
    #torch.sum()
    #torch tensor可以用同样的乘法
    m = kernel.size(0)
    d = int((m-1)/2)
    h, w = src.size(2), src.size(3)
    dst = torch.zeros(1,1,h,w)

    for y in range(d, h - d):
        for x in range(d, w - d):
            dst[y][x] = torch.sum(src[0, 0, y-d:y+d+1, x-d:x+d+1]*kernel)
    return dst

# def bicubicopencv_2D(src, times):
#     up_clinical_array = np.zeros((src.shape[0]*times, src.shape[1]*times, src.shape[2]), dtype = 'float32')
#     for i in range(src.shape[2]):
#         up_clinical_array[:,:,i] = cv2.resize(src[:,:,i], None, fx = times, fy = times, interpolation = cv2.INTER_CUBIC)
#     return up_clinical_array

# def bicubicdownsample_opencv_2D(src, times):
#     up_clinical_array = np.zeros((int(src.shape[0]/times), int(src.shape[1]/times), src.shape[2]), dtype = 'float32')
#     for i in range(src.shape[2]):
#         up_clinical_array[:,:,i] = cv2.resize(src[:,:,i], (int(src.shape[0]/times), int(src.shape[1]/times)), interpolation = cv2.INTER_CUBIC)
    
#     return up_clinical_array

def resize_nearest(src, h, w):

    # 出力画像用の配列生成（要素は全て空）
    dst = np.empty((h,w))

    # 元画像のサイズを取得
    hi, wi = src.shape[0], src.shape[1]

    # 拡大率を計算
    ax = w / float(wi)
    ay = h / float(hi)

    # 最近傍補間
    for y in range(0, h):
        for x in range(0, w):
            xi, yi = int(round(x/ax)), int(round(y/ay))
            # 存在しない座標の処理
            if xi > wi -1: xi = wi -1
            if yi > hi -1: yi = hi -1

            dst[y][x] = src[yi][xi]
    return dst

# def generate_airnoise():
#     airnoise_path1 = "/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/Clinical_LUNG_SR/nulung032.512/Clinical_LUNG_SR_200.nii.gz"
#     airnoise_path2 = "/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/Clinical_LUNG_SR/nulung032.512/Clinical_LUNG_SR_300.nii.gz"

#     airmask_path = "/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/AIR_MASK/nulung032.512_AIR_MASK_half_rotated.nii.gz"

#     airarray1 = nib.load(airnoise_path1).get_fdata()
#     airarray2 = nib.load(airnoise_path2).get_fdata()
#     maskarray = nib.load(airmask_path).get_fdata()
#     thisnum1 = re.findall(r"SR_\d+",airnoise_path1)
#     thisnum2 = re.findall(r"SR_\d+",airnoise_path1)
#     thisnum1 = re.findall(r"\d+",thisnum1[0])
#     thisnum2 = re.findall(r"\d+",thisnum2[0])


#     for k in range(0, airarray1.shape[2]):
#         this_image = airarray1[0:4096,2048:4096,k:k+1]

#         this_mask1 = maskarray[:,:,int(thisnum1[0]) + k]

#         '''
#         print(this_mask1.shape)
#         print(this_mask1.shape[0])
#         #this_mask1 = cv2.resize(this_mask1,(this_mask1.shape[0]*8,this_mask1.shape[1]*8),interpolation=cv2.INTER_NEAREST)
#         this_mask1 = cv2.resize(this_mask1, None, fx=8, fy=8,interpolation=cv2.INTER_NEAREST)
#         print(this_mask1.shape)
#         this_mask1 = this_mask1[:, :, np.newaxis] 
#         print(this_mask1.shape)
#         print(this_image.shape)
#         this_image = nib.Nifti1Image(this_image, np.eye(4))
#         this_mask1 = nib.Nifti1Image(this_mask1, np.eye(4))
#         nib.save(this_image, '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/Cropped_Airnoise_Sample/this_image.nii.gz')
#         nib.save(this_mask1, '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/Cropped_Airnoise_Sample/this_mask1.nii.gz')
#         '''

#         this_mask1 = cv2.resize(this_mask1, None, fx=8, fy=8,interpolation=cv2.INTER_NEAREST)
#         print("dtype of mask array")
#         print(this_mask1.dtype)
#         #random地取patch
#         j = 0
#         while j < 10:
#             x = random.randint(0, this_image.shape[0]-256)
#             y = random.randint(0, this_image.shape[1]-256)
#             if 0 in this_mask1[x:x+256, y:y+256]:
#                 continue
#             else:
#                 save_noise1 = nib.Nifti1Image(this_image[x:x+256, y:y+256, :], np.eye(4))
#                 print(k*10+j)
#                 nib.save(save_noise1, '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/Cropped_Airnoise_Sample/Apos_{}.nii.gz'.format(k*10+j))
#                 j = j + 1

#     for a in range(0, airarray2.shape[2]):
#         this_image = airarray2[0:4096,2048:4096,a:a+1]

#         this_mask2 = maskarray[:,:,int(thisnum2[0]) + a]

#         this_mask2 = cv2.resize(this_mask2, None, fx=8, fy=8,interpolation=cv2.INTER_NEAREST)

#         print("dtype of mask array")
#         print(this_mask2.dtype)
#         #random地取patch
#         b = 0
#         while b < 10:
#             x = random.randint(0, this_image.shape[0]-256)
#             y = random.randint(0, this_image.shape[1]-256)
#             if 0 in this_mask2[x:x+256, y:y+256]:
#                 continue
#             else:
#                 save_noise1 = nib.Nifti1Image(this_image[x:x+256, y:y+256, :], np.eye(4))
#                 print(a*10+b)
#                 nib.save(save_noise1, '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/Cropped_Airnoise_Sample/Bpos_{}.nii.gz'.format(a*10+b))
#                 b = b + 1

def generate_mask_forimportanttissue(): #形成重要区域的mask(比如重要的血管部分) 
    imagepath = '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/AftermodifingGenerator/clinical_bicubic.nii.gz'
    imagearray = nib.load(imagepath).get_fdata()
    # imagemean = np.median(imagearray) # median不好用
    imagemean = np.mean(imagearray)
    imagearray[imagearray < imagemean] = -1
    imagearray[imagearray > imagemean] = 1
    imagearray = nib.Nifti1Image(imagearray, np.eye(4))
    nib.save(imagearray, '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/AftermodifingGenerator/clinical_bicubic_binariation.nii.gz')

def nifti2jpg():
    imgpath = '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/SRclinical.nii.gz'
    imgarray = nib.load(imgpath).get_fdata()
    print(np.max(imgarray))
    print(np.min(imgarray))
    imgarray = imgarray - np.min(imgarray)
    imgarray = imgarray * (255 / np.max(imgarray))
    imgarray = np.around(imgarray)
    print(np.min(imgarray))
    print(np.max(imgarray))
    print(imgarray.shape)
    imgarray = imgarray[:,:,0]
    scipy.misc.imsave('outfile.jpg', imgarray)

def normlizeSRresult():
    CTpath = '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/For_JAMIT2020/clinicalSR_3D.nii.gz'
    CTaxis1path = '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/For_JAMIT2020/axis1_clinicalSR_3D.nii.gz'
    OriginalCTpath = '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/For_JAMIT2020/clinical_3d_20slices.nii.gz'
    
    CTdata = nib.load(CTpath).get_data()
    CTdata = CTdata - np.min(CTdata)
    CTdata = CTdata * (1200 / (np.max(CTdata) - np.min(CTdata)))
    CTdata = np.ceil(CTdata)
    CTdata = nib.Nifti1Image(CTdata, np.eye(4))

    CTaxis1data = nib.load(CTaxis1path).get_data()
    CTaxis1data = CTaxis1data - np.min(CTaxis1data)
    CTaxis1data = CTaxis1data * (1200 / (np.max(CTaxis1data) - np.min(CTaxis1data)))
    CTaxis1data = np.ceil(CTaxis1data)
    CTaxis1data = nib.Nifti1Image(CTaxis1data, np.eye(4))

    OriginalCTdata = nib.load(OriginalCTpath).get_data()
    OriginalCTdata = OriginalCTdata - np.min(OriginalCTdata)
    OriginalCTdata = OriginalCTdata * (1200 / (np.max(OriginalCTdata) - np.min(OriginalCTdata)))
    OriginalCTdata = np.ceil(OriginalCTdata)
    OriginalCTdata = nib.Nifti1Image(OriginalCTdata, np.eye(4))


    nib.save(CTdata, '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/For_JAMIT2020/clinicalSR_3D_normalized.nii.gz')
    nib.save(CTaxis1data, '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/For_JAMIT2020/clinicalSRaxis1_3D_normalized.nii.gz')
    nib.save(OriginalCTdata, '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/For_JAMIT2020/clinical_3d_20slices_normalized.nii.gz')

if __name__ == '__main__':
    '''
    original_micro_path = '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/FOR_CARS/original.nii.gz'
    old_micro_path = '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/FOR_CARS/recons_micro_old.nii.gz'
    modified_micro_path = '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/FOR_CARS/recons_micro.nii.gz'

    original = nib.load(original_micro_path).get_fdata()
    old = nib.load(old_micro_path).get_fdata() 
    modified = nib.load(modified_micro_path).get_fdata()

    ssimold = my_ssim(original,old)
    ssimnew = my_ssim(original,modified)
    psnrold = my_psnr(original,old)
    psnrnew = my_psnr(original,modified)
    print(ssimold) # 0.6829764367578338
    print(ssimnew) # 0.73690633957392
    print(psnrold) # 11.946611436354889
    print(psnrnew) # 12.73143684247188
    '''
    # nifti2jpg()
    normlizeSRresult()