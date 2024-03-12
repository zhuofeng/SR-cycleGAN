from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))

        # my options
        parser.add_argument('--clinical_folder', type=str, default='', help='floder of clinicalCT')
        parser.add_argument('--micro_folder', type=str, default='', help='folder of microCT')
        parser.add_argument('--all_clinical_paths', type=str, default=['/homes/tzheng/CTdata/CTMicroNUrespsurg/cct/nulung031.512.nii.gz',
                                                                       '/homes/tzheng/CTdata/CTMicroNUrespsurg/cct/nulung050.nii.gz', 
                                                                       '/homes/tzheng/CTdata/CTMicroNUrespsurg/cct/nulung030.nii.gz'], help='all medical paths')
        parser.add_argument('--all_micro_paths', type=str, default=['/homes/tzheng/CTdata/CTMicroNUrespsurg/nii/nulung050/nulung050_053_000.nii.gz',
                                                                    '/homes/tzheng/CTdata/CTMicroNUrespsurg/converted/DICOM_nulung030_cb_004_zf_ringRem_med3.nii.gz'], help='all clinical paths')
        parser.add_argument('--batch_num', type=int, default=2000, help='the batch num')
        parser.add_argument('--clinical_patch_size', type=int, default=32, help='patch size of clinicalCT')
        parser.add_argument('--micro_patch_size', type=int, default=256, help='patch size of microCT')
        parser.add_argument('--sampling_times', type=int, default=3, help='2^n times of upsampling from clinicalCT to microCT')
        parser.add_argument('--maskdatafolder', type=str, default='/homes/tzheng/CTdata/CTMicroNUrespsurg/Mask')
        
        self.isTrain = False
        return parser
