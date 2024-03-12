"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data

from PIL import Image
import os
import os.path
import nibabel as nib
NIFTI_EXTENSIONS = [
    '.nii.gz', '.nii',
]


def is_nifti_file(filename):
    return any(filename.endswith(extension) for extension in NIFTI_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float("inf")):
    niftis = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_nifti_file(fname):
                path = os.path.join(root, fname)
                niftis.append(path)
    return niftis[:min(max_dataset_size, len(niftis))]

def default_loader(path):
    img = nib.load(path)
    return img.get_fdata(), img.header()

class NiftiFolder(data.Dataset):
    def __init__(self, root, normalization=True, return_paths=False,
                 loader=default_loader):
        niftis = make_dataset(root)
        if len(niftis) == 0:
            raise(RuntimeError("Found 0 niftis in: " + root + "\n"))
        
        self.root = root
        self.niftis = niftis
        self.normalization = normalization
        self.return_paths = return_paths
        self.loader = loader
    
    def __getitem__(self, index):
        path = self.niftis[index]
        nifti = self.loader(path)
        if self.normalization is not None:
            nifti = self.normalization(nifti)
        if self.return_paths:
            return nifti, path
        else:
            return nifti

    def __len__(self):
        return len(self.nifti)