# SR-CycleGAN: Unsupervised Super-Resolution of Lung CT Images across Modalities

Super-resolution of clinical lung CT images using lung micro-CT images without supervision. The method has been tested on publicly available datasets, ensuring its generality.

## Paper Link and Program Path

- [Paper Link](https://www.spiedigitallibrary.org/journals/journal-of-medical-imaging/volume-9/issue-02/024003/SR-CycleGAN--super-resolution-of-clinical-CT-to-micro/10.1117/1.JMI.9.2.024003.full)

## Introduction

This method is an unsupervised super-resolution approach based on the assumption that it is possible to enhance the resolution of clinical CT images using micro-CT images. The approach started from the observation of the CycleGAN method gaining attention in image transformation in 2019. Hence, we repurposed CycleGAN for image super-resolution by:
1. Proposing new loss functions
2. Proposing a new network architecture
3. Conducting experiments on a new dataset.

## Environment Setup

- Important Packages:
cudnn=7.6.5
cuda=10.1_0
torch=2.0.0
torchvision=0.15.1


*Note:* For other package versions, refer to `environment.yml`. The code is executable after setting up an Anaconda environment.

## Training and Testing

- **Quick Train:**
Run the script inside the folder:
bash run.sh

This trains an 8x super-resolution model for clinical CT images by default.

- **Quick Test:**
bash test.sh

This performs 8x super-resolution on clinical lung CT images using the trained SR-CycleGAN model. View the clinical CT images in `./results/clinical.nii.gz` with ITK-SNAP.

Super-resolution output example: `./results/SR.nii.gz`. View using ITK-SNAP.

*Note:* Numerous CT images are stored in this folder for use as inputs to the trained model. However, caution is advised as the method specializes in lung imaging; super-resolving regions other than the lungs may result in anomalous images.

## Important Parameters for Training and Testing

- Read these files thoroughly:
- `./Options/base_options.py`: Shared settings for training and inference
- `./Options/train_options.py`: Training settings
- `./Options/test_options.py`: Inference settings

## Loss Functions

The loss functions are defined in `./models/medical_cycle_gan_model.py`. Our proposed loss consists of four parts, ensuring consistency of images before and after SR (super-resolution). Refer to the paper for details.

## Quantitative Evaluation

As mentioned in the paper, since the clinical CT and micro-CT used in this study are not registered, obtaining corresponding micro-CT images for clinical CTs is not feasible. Hence, quantitative evaluation is not performed. We adopt a compromise: testing if the trained model can reconstruct HR (high-resolution) micro-CT from LR (low-resolution) micro-CT.

To test:
python medicaltest.py --method microCTsmallpatchtest


## Additional Notes

- To grasp the overall landscape of SR, start by reading this survey paper: [Deep Learning for Image Super-Resolution: A Survey](https://ieeexplore.ieee.org/document/9044873).
- Unsupervised or supervised SR has been extensively researched in recent years. Finding new approaches is necessary to publish new papers.
