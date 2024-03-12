from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.00001, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations') #现在的lr是 0.1
        parser.add_argument('--A_patchs_num', type=int, default=10000, help='patch number of data A')
        parser.add_argument('--B_patchs_num', type=int, default=10000, help='patch number of data B')
        parser.add_argument('--A_patch_size', type=int, default=32, help='patch size of data A')
        parser.add_argument('--B_patch_size', type=int, default=256, help='patch size of data B')
        # my opintion
        parser.add_argument('--clinical_folder', type=str, default='', help='floder of clinicalCT')
        parser.add_argument('--micro_folder', type=str, default='', help='folder of microCT')
        parser.add_argument('--all_clinical_paths', type=str, default=[ '/homes/tzheng/CTdata/CTMicroNUrespsurg/HE_matched/nulung014-4.nii.gz', 
                                                                        '/homes/tzheng/CTdata/CTMicroNUrespsurg/HE_matched/nulung015-5.nii.gz',
                                                                        '/homes/tzheng/CTdata/CTMicroNUrespsurg/HE_matched/nulung030.nii.gz',
                                                                        '/homes/tzheng/CTdata/CTMicroNUrespsurg/HE_matched/nulung031.nii.gz', 
                                                                        '/homes/tzheng/CTdata/CTMicroNUrespsurg/HE_matched/nulung050.nii.gz'
                                                                        ], help='all clinical paths')
        
        parser.add_argument('--all_micro_paths', type=str, default=['/homes/tzheng/CTdata/CTMicroNUrespsurg/nii/nulung014/uCT/nulung014_027_000_cropped.nii.gz',
                                                                    '/homes/tzheng/CTdata/CTMicroNUrespsurg/nii/nulung015/uCT/nulung015_034_001_cropped.nii.gz',
                                                                    '/homes/tzheng/CTdata/CTMicroNUrespsurg/nii/nulung030/DICOM_nulung030_cb_004_zf_ringRem_med3_cropped.nii.gz',
                                                                    '/homes/tzheng/CTdata/CTMicroNUrespsurg/nii/nulung031/uCT/DICOM_nulung031_cb_003_zf_ringRem_med3_cropped.nii.gz',
                                                                    '/homes/tzheng/CTdata/CTMicroNUrespsurg/nii/nulung050/nulung050_053_000_cropped.nii.gz'
                                                                    ], help='all micro paths')
        
        parser.add_argument('--all_full_clinical_paths', type=str, default=['/homes/tzheng/CTdata/CTMicroNUrespsurg/cct/nulung015-5.nii.gz',
                                                                            ], help='all micro paths')
        '''
        '/homes/tzheng/CTdata/CTMicroNUrespsurg/cct/nulung030.nii.gz',
                                                                            '/homes/tzheng/CTdata/CTMicroNUrespsurg/cct/nulung031.512.nii.gz',
                                                                            '/homes/tzheng/CTdata/CTMicroNUrespsurg/cct/nulung014-4.nii.gz',
                                                                            '/homes/tzheng/CTdata/CTMicroNUrespsurg/cct/nulung050.nii.gz'
        '''
        parser.add_argument('--all_registrated_clinical_paths', type=str, default=['/homes/tzheng/CTdata/Hatakoshi_registration_forAlphaGAN/CT_image_normalized_cropped_fit_with_uCT_reso.nii.gz',
                                                                                ], help='all registrated clinical paths')

        parser.add_argument('--all_registrated_micro_paths', type=str, default=['/homes/tzheng/CTdata/Hatakoshi_registration_forAlphaGAN/outputResult.nii.gz',
                                                                                ], help='all registrated micro paths')
        parser.add_argument('--batch_num', type=int, default=2000, help='the batch num')
        # parser.add_argument('--clinical_patch_size', type=int, default=32, help='patch size of clinicalCT')
        # parser.add_argument('--micro_patch_size', type=int, default=256, help='patch size of microCT')
        # JMI rebuttal: adjust patch size -- and changing dataset size is
        parser.add_argument('--clinical_patch_size', type=int, default=32, help='patch size of clinicalCT')
        parser.add_argument('--micro_patch_size', type=int, default=256, help='patch size of microCT')
        parser.add_argument('--sampling_times', type=int, default=3, help='2^n times of upsampling from clinicalCT to microCT') # 决定了是几倍的SR 比如说3就是2^3=8倍
        parser.add_argument('--maskdatafolder', type=str, default='/homes/tzheng/CTdata/CTMicroNUrespsurg/Mask')
        parser.add_argument('--code_channel', type=int, default=128)
        parser.add_argument('--encoder_path', type=str, default='/homes/tzheng/Mypythonfiles/alphaGAN-master/alphagan-results/e-epoch-100.pth', help='path of the alpha-GAN encoder')
        parser.add_argument('--num_samples', type=int, default=200, help='number of samples take from each clinical CT case')

        self.isTrain = True
        return parser
