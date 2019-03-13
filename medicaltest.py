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
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import nibabel as nib
import numpy as np
import torch

def oldtest():
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options 目前是medical_cycle_gan
    model.setup(opt)               # regular setup: load and print networks; create schedulers #把网络打印出来.参见base_model
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory opt.epoch=latest,加载最后一个model
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  # save the HTML

def test():
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
    real_micro = np.zeros((1024,1024,1), dtype='float32')
    fake_clinical = np.zeros((128,128,1), dtype='float32')

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
                if label == 'fake_B':
                    fake_micro[int((i//16)*256):int(((i//16)+1)*256),int((i%16)*256):int((i%16+1)*256),0] = im_data
                if i < 16:
                    if label == 'real_B':
                        real_micro[int((i//4)*256):int(((i//4)+1)*256),int((i%4)*256):int((i%4+1)*256),0] = im_data
                    if label == 'fake_A':
                        fake_clinical[int((i//4)*32):int(((i//4)+1)*32),int((i%4)*32):int((i%4+1)*32),0] = im_data
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
    nib.save(finalmicro, './CTdata/finalmicro.nii.gz')
    

def downsamplemicroCTtest():
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

    downsampled_micro = np.zeros((128,128,1), dtype='float32')
    super_micro = np.zeros((1024, 1024, 1), dtype='float32')
    real_micro = np.zeros((1024, 1024, 1), dtype='float32')
    fake_down_micro = np.zeros((128,128,1), dtype='float32')

    for i, data in enumerate(dataset):
        i = int(i)
        if i>= opt.num_test:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        for label, im_data in visuals.items():
            #print(label) #real_A fake_B rec_A real_B fake_A rec_B
            if i < 16:
                if label == 'real_A':
                    downsampled_micro[int((i//4)*32):int(((i//4)+1)*32),int((i%4)*32):int((i%4+1)*32),0] = im_data
                if label == 'fake_B':
                    super_micro[int((i//4)*256):int(((i//4)+1)*256),int((i%4)*256):int((i%4+1)*256),0] = im_data
                if label == 'real_B':
                    real_micro[int((i//4)*256):int(((i//4)+1)*256),int((i%4)*256):int((i%4+1)*256),0] = im_data
                if label == 'fake_A':
                    fake_down_micro[int((i//4)*32):int(((i//4)+1)*32),int((i%4)*32):int((i%4+1)*32),0] = im_data
            else:
                break

    downsampled_micro = nib.Nifti1Image(downsampled_micro, np.eye(4))
    super_micro = nib.Nifti1Image(super_micro, np.eye(4))
    real_micro = nib.Nifti1Image(real_micro, np.eye(4))
    fake_down_micro = nib.Nifti1Image(fake_down_micro, np.eye(4))   

    nib.save(downsampled_micro, './CTdata/downsampled_micro.nii.gz')
    nib.save(super_micro, './CTdata/super_micro.nii.gz')
    nib.save(real_micro, './CTdata/real_micro.nii.gz')
    nib.save(fake_down_micro, './CTdata/fake_down_micro.nii.gz')

def downsamplemicroCTpixeltest():
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
    opt.num_test = int((128-opt.clinical_patch_size+1) **2)

    downsampled_micro = np.zeros((128,128,1), dtype='float32')
    super_micro = np.zeros((1024, 1024, 1), dtype='float32')
    real_micro = np.zeros((1024, 1024, 1), dtype='float32')
    fake_down_micro = np.zeros((128,128,1), dtype='float32')

    for i, data in enumerate(dataset):
        i = int(i)
        if i>= opt.num_test:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        for label, im_data in visuals.items():
            #print(label) #real_A fake_B rec_A real_B fake_A rec_B
            if i < int((128-opt.clinical_patch_size+1) **2):
                if label == 'real_A':
                    continue
                    #downsampled_micro[int((i//4)*32):int(((i//4)+1)*32),int((i%4)*32):int((i%4+1)*32),0] = im_data
                if label == 'fake_B':
                    continue
                    #super_micro[int((i//4)*256):int(((i//4)+1)*256),int((i%4)*256):int((i%4+1)*256),0] = im_data
                    super_micro[int(i//(128-opt.clinical_patch_size+1))*8+124:int(i//(128-opt.clinical_patch_size+1))*8+132, 
                    int(i%(128-opt.clinical_patch_size+1))*8+124:int(i%(128-opt.clinical_patch_size+1))*8+132, 0] = im_data[124:132, 124:132, :]
                if label == 'real_B':
                    continue
                    #real_micro[int((i//4)*256):int(((i//4)+1)*256),int((i%4)*256):int((i%4+1)*256),0] = im_data
                if label == 'fake_A':
                    fake_down_micro[int((i//4)*32):int(((i//4)+1)*32),int((i%4)*32):int((i%4+1)*32),0] = im_data
                    continue
            else:
                break

    downsampled_micro = nib.Nifti1Image(downsampled_micro, np.eye(4))
    super_micro = nib.Nifti1Image(super_micro, np.eye(4))
    real_micro = nib.Nifti1Image(real_micro, np.eye(4))
    fake_down_micro = nib.Nifti1Image(fake_down_micro, np.eye(4))   

    nib.save(downsampled_micro, './CTdata/downsampled_micro.nii.gz')
    nib.save(super_micro, './CTdata/super_micro.nii.gz')
    nib.save(real_micro, './CTdata/real_micro.nii.gz')
    nib.save(fake_down_micro, './CTdata/fake_down_micro.nii.gz')

if __name__ == '__main__':
    downsamplemicroCTpixeltest()