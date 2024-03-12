from scipy.ndimage import gaussian_filter
import nibabel as nib
import numpy as np
import pdb

def gaussianblur():
    paths = [
        '/homes/tzheng/code/ClinicalSpecimen_registration/CTdata/clinical_skyscanrigis/036/registrated/skyscan.nii.gz',
        '/homes/tzheng/code/ClinicalSpecimen_registration/CTdata/clinical_skyscanrigis/044/registrated/044_skyscan.nii.gz', 
        '/homes/tzheng/code/ClinicalSpecimen_registration/CTdata/clinical_skyscanrigis/046/registrated/046_skyscan_denoised.nii.gz'
    ]
    casenum = 0
    for path in paths:
        arr = nib.load(path).get_data().astype(np.float32)
        Gaussian_arr = gaussian_filter(arr, sigma=0.6, mode='reflect')
        Gaussian_arr = nib.Nifti1Image(Gaussian_arr, np.eye(4))
        print('/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/tmp/gaussian_filter/'+str(casenum)+'.nii.gz')
        nib.save(Gaussian_arr, '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/tmp/gaussian_filter/'+str(casenum)+'.nii.gz')
        casenum += 1

if __name__ == '__main__':
    gaussianblur()