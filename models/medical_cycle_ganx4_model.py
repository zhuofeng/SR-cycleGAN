import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torch.nn import AvgPool2d
from torch.nn import UpsamplingNearest2d, UpsamplingBilinear2d
import pytorch_ssim
import numpy as np
import os
import cv2
import pdb

class MedicalCycleGANx4Model(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=1.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=1.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.2, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_downsample_loss', type=float, default=0.7, help='downsample_loss')
            parser.add_argument('--downsample_loss', type=float, default=1., help='downsample_loss')
            parser.add_argument('--lambda_upsample_loss', type=float, default=0.3, help='upsample_loss')
            parser.add_argument('--upsample_loss', type=float, default=1., help='upsample_loss')
            parser.add_argument('--lambda_clinical_ssim', type=float, default=1.0, help='weight for ssim between downsampled fake microCT and clinicalCT')
            parser.add_argument('--clinical_ssim_loss', type=float, default=1.0, help='clinical_ssim_loss')
            parser.add_argument('--lambda_micro_ssim', type=float, default=1.0, help='weight for ')
            parser.add_argument('--micro_ssim_loss', type=float, default=1.0, help='micro_ssim_loss')
            parser.add_argument('--random_mesh_ssim', type=float, default=-1.0, help='inspired by the paper https://openreview.net/pdf?id=BktMD6isM')
            parser.add_argument('--lambda_random_mesh_ssim', type=float, default=0.5, help='')
            parser.add_argument('--random_mesh_size', type=int, default=30, help='size of random mesh') #原来是20
            parser.add_argument('--random_mesh_num', type=int, default=20, help='num of random meshes') #原来是10
            parser.add_argument('--random_mesh_average', type=int, default=-1, help='use or not use mesh average')
            parser.add_argument('--lambda_random_mesh_average', type=float, default=0.2, help='num of random meshes')
            parser.add_argument('--sobel_loss', type=int, default=0, help='use sobel felter or not') #sobel filter 突出图像纹理，然后用突出纹理的图像来计算差分
            parser.add_argument('--structure_loss', type=float, default=-1., help='use structure loss or not') #在CT图像中明亮的部分通常为重要的部分。 比较超过平均值的部分
            parser.add_argument('--lambda_structure_loss', type=float, default=0.8, help='the lambda(weight) of structure loss')
            parser.add_argument('--lambda_G_A', type=float, default=1.0, help='weight of G_A')
            parser.add_argument('--lambda_G_B', type=float, default=1.0, help='weight of G_B')
            parser.add_argument('--clinical_inter', type=float, default=-1.0, help='use clinical inter loss or not')
            parser.add_argument('--lambda_clinical_inter', type=float, default=5.0, help='weight of clinical inter loss')
            parser.add_argument('--fillhole', type=float, default=-1.0, help='use fill hole loss or not')
            parser.add_argument('--lambda_fillhole', type=float, default=0.1, help='weight of fill hole loss')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        print('Init the MedicalCycleGANModelx4 model')
        
        BaseModel.__init__(self, opt)
        
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'downsample', 'upsample', 'clinical_ssim', 'micro_ssim', 'mesh_ssim', 'mesh_average_sum', 'structure_ssim']
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'downsample', 'upsample', 'clinical_ssim', 'micro_ssim']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        
        #print("this is the lambda_identity")
        #print(self.opt.lambda_identity) #-1.0
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize G_B(A) ad G_A(B)
            visual_names_A.append('idt_A')
            #目前不使用idt_B
            visual_names_B.append('idt_B')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        opt.sampling_times = 2
        self.netG_A = networks.define_G(opt.medical_input_nc, opt.medical_output_nc, opt.ngf, opt.clinical2micronetG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.sampling_times)
        
        self.netG_B = networks.define_G(opt.medical_output_nc, opt.medical_input_nc, opt.ngf, opt.micro2clinicalnetG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, opt.sampling_times)
        
        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.medical_output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            
            self.netD_B = networks.define_D(opt.medical_input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            
        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB' #true or false
        self.real_A = input['clinical' if AtoB else 'micro'].to(self.device)
        self.real_B = input['micro' if AtoB else 'clinical'].to(self.device)

        # self.small_clinical = input['small_clinical'].to(self.device)
        # self.big_clinical = input['big_clinical'].to(self.device)

        self.image_paths = input['clinical_paths' if AtoB else 'micro_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
        
    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self, epoch):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # Identity loss 看变换之后和原来差多少(举例：用生成浮世绘的generator计算浮世绘，看生成图像和原图差多少)
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            m = AvgPool2d(4, stride=4)
            self.idt_A = m(self.idt_A)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            
            #将B downsampling之后只有4个像素，就算upsampling也没有意义
            self.idt_B = self.netG_B(self.real_A)
            n = UpsamplingNearest2d(scale_factor=4)
            # n = UpsamplingBilinear2d(scale_factor=4)
            self.idt_B = n(self.idt_B)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt * 0.1
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
        
        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True) * self.opt.lambda_G_A#之所以设成True，是因为要减小这个loss，使generator生成真实的图像
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True) * self.opt.lambda_G_B
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        #downsample fake micro to clinical scale and calculate loss
        if self.opt.downsample_loss > 0:
            #print('use the downsampleloss')
            m = AvgPool2d(4, stride=4)
            self.downsample_fake_B = m(self.fake_B)
            self.loss_downsample = self.criterionCycle(self.downsample_fake_B, self.real_A) * self.opt.lambda_downsample_loss
            
        else:
            self.loss_downsample = 0

        #upsample fake clinical to micro scale and calculate loss
        if self.opt.upsample_loss > 0:
            #print('use the upsampleloss')
            n = UpsamplingNearest2d(scale_factor=4)
            # n = UpsamplingBilinear2d(scale_factor=4)
            self.upsample_fake_A = n(self.fake_A)
            self.loss_upsample = self.criterionCycle(self.upsample_fake_A, self.real_B) * self.opt.lambda_upsample_loss
        else:
            self.loss_upsample = 0

        if self.opt.clinical_ssim_loss > 0:
            
            #print('use the clinical ssim loss')
            m = AvgPool2d(4, stride=4)
            self.downsample_fake_B = m(self.fake_B)
            ssim_loss = pytorch_ssim.SSIM()
            #self.loss_clinical_ssim = -ssim_loss(self.downsample_fake_B, self.real_A) * self.opt.lambda_clinical_ssim
            self.loss_clinical_ssim = (1 - ssim_loss(self.downsample_fake_B, self.real_A)) * self.opt.lambda_clinical_ssim
        else:
            self.loss_clinical_ssim  = 0

        if self.opt.micro_ssim_loss > 0:
            #print('use the micro ssim loss')
            n = UpsamplingNearest2d(scale_factor=4)
            # n = UpsamplingBilinear2d(scale_factor=4)
            self.upsample_fake_A = n(self.fake_A)
            ssim_loss = pytorch_ssim.SSIM()
            #self.loss_micro_ssim = -ssim_loss(self.upsample_fake_A, self.real_B) * self.opt.lambda_micro_ssim
            self.loss_micro_ssim = (1 - ssim_loss(self.upsample_fake_A, self.real_B)) * self.opt.lambda_micro_ssim
        else:
            self.loss_micro_ssim = 0

        #如果算像素之间的差分就没有意义了。算某个随机位置的ssim
        if self.opt.random_mesh_ssim > 0:
            #print('use_mesh_ssim')
            self.random_mesh_size = self.opt.random_mesh_size
            n = UpsamplingNearest2d(scale_factor=4)
            # n = UpsamplingBilinear2d(scale_factor=4)
            self.upsample_real_A = n(self.real_A)
            ssim_loss = pytorch_ssim.SSIM()
            mesh_ssim = 0.
            random_x = 0
            random_y = 0
            for i in range(0, self.opt.random_mesh_num): 
                random_x = np.random.randint(0, self.opt.micro_patch_size - self.opt.random_mesh_size)
                random_y = np.random.randint(0, self.opt.micro_patch_size - self.opt.random_mesh_size)
                #print('this is random_X')
                #print(random_x)
                mesh_upsample_real_A = self.upsample_real_A[:, :, random_x:random_x+self.opt.random_mesh_size, random_y:random_y+self.opt.random_mesh_size]
                mesh_fake_B = self.fake_B[:, :, random_x:random_x+self.opt.random_mesh_size, random_y:random_y+self.opt.random_mesh_size]
                mesh_ssim += (ssim_loss(mesh_upsample_real_A, mesh_fake_B) ** 2)
            
            #self.loss_mesh_ssim = -(mesh_ssim / self.opt.random_mesh_num) * self.opt.lambda_random_mesh_ssim
            self.loss_mesh_ssim = (1 / (mesh_ssim / self.opt.random_mesh_num)) * self.opt.lambda_random_mesh_ssim
        else:
            self.loss_mesh_ssim = 0

        #随机的mesh平均差分
        if self.opt.random_mesh_average > 0:
            #print('use_mesh_average')
            self.random_mesh_size = self.opt.random_mesh_size
            # n = UpsamplingNearest2d(scale_factor=4)
            n = UpsamplingBilinear2d(scale_factor=4)
            self.upsample_real_A = n(self.real_A)
            mesh_average_sum = 0.
            for i in range(0, self.opt.random_mesh_num):
                random_x = np.random.randint(0, self.opt.micro_patch_size - self.opt.random_mesh_size)
                random_y = np.random.randint(0, self.opt.micro_patch_size - self.opt.random_mesh_size)
                mesh_upsample_real_A = self.upsample_real_A[:, :, random_x:random_x+self.opt.random_mesh_size, random_y:random_y+self.opt.random_mesh_size]
                mesh_fake_B = self.fake_B[:, :, random_x:random_x+self.opt.random_mesh_size, random_y:random_y+self.opt.random_mesh_size]
                mesh_average_sum += self.criterionCycle(mesh_upsample_real_A, mesh_fake_B.detach())
            self.loss_mesh_average_sum = mesh_average_sum / self.opt.random_mesh_num * self.opt.lambda_random_mesh_average
        else:
            self.loss_mesh_average_sum = 0

        # if self.opt.clinical_inter > 0:
        #     self.loss_clinical_inter = self.criterionCycle(self.big_clinical, self.fake_big_clinical) * self.opt.lambda_clinical_inter
        # else:
        #     self.loss_clinical_inter = 0

        if self.opt.fillhole > 0 and epoch > 15:
            temp_fake_B = self.fake_B.clone()
            mean_fake_B = torch.mean(temp_fake_B)
            temp_fake_B[temp_fake_B < mean_fake_B] = -1
            temp_fake_B[temp_fake_B > mean_fake_B] = 1
            temp_fake_B = temp_fake_B.cpu().detach().numpy()
            fillhole_fake_B = np.copy(temp_fake_B)

            kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
            kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
            for i in range(temp_fake_B.shape[0]):
                fillhole_fake_B[i,0,:,:] = cv2.dilate(fillhole_fake_B[i,0,:,:], kernel1)
                fillhole_fake_B[i,0,:,:] = cv2.erode(fillhole_fake_B[i,0,:,:], kernel2)

            Residualmatrix = fillhole_fake_B - temp_fake_B
            Residualmatrix[Residualmatrix==-1] = 0
            totalresidual = np.sum(Residualmatrix) / (Residualmatrix.size)
            self.loss_fillhole = totalresidual * self.opt.lambda_fillhole
        else:
            self.loss_fillhole = 0
        
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_downsample + self.loss_upsample + self.loss_clinical_ssim + self.loss_micro_ssim + self.loss_fillhole
        self.loss_G.backward()

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G(epoch)             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights