# -*- coding: utf8 -*-
"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import pdb
import os
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset

def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset" #这个是定义引用模型路径。目前是unaligned_dataset
    datasetlib = importlib.import_module(dataset_filename) #根据输入的dataset_filename来引用模型。比如说dataset_filename如果是aligned，那么就引用aligned_dataset.py
    
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():  #__dict__是abc里面的方法，是路径。 items()是： dict.items()によって返されるオブジェクトは、ビューオブジェクトです
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset): #因为自定义的dataset类都继承于BaseDataset类。这里是判断是不是dataset的子类. lower()是小文字变换
            dataset = cls
    
    #print(name) #Image
    #print(type(name)) <class 'str'>
    #print(dataset.__mro__) #多重继承，继承自class abc.ABC和torch dataset.
    #os._exit(0)

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    #print("created dataloader!")
    dataset = data_loader.load_data() #实验证明，这一步只是返回一个初始化完成的CustomDatasetDataLoader对象，不是读数据
    #print("created dataset!")
    return dataset

class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""
    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)  #dataset_mode default='unaligned' 是一个torch dataset对象。目前是UnalignedDataset。opt指定一些属性。find_dataset_using_name这个方法返回一个ビューオブジェクト
        self.dataset = dataset_class(opt) #将ビューオブジェクト实例化。 在这里是实例化UnalignedDataset
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,  #这个dataset继承于torch dataset class。 目前是unaligned_dataset类对象
            batch_size=opt.batch_size, #自动取batch_size大小的数据
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self
    
    def __len__(self):
        """Return the number of data in the dataset"""
        #print(type(self.dataset)) #<class 'data.unaligned_dataset.UnalignedDataset'>
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self): #这个对应于enumerate，当对这个类调用enumerate方法的时候进行什么样的操作
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            #print("iter in CustomDatasetDataLoader")
            if i * self.opt.batch_size >= self.opt.max_dataset_size: #已经读到了人为设定的batch上限
                break
            yield data
