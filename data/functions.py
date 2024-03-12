        # 将一部分的clinical CT作为输入 
        for i, path in enumerate(opt.all_full_clinical_paths):
            print(path)
            this_full_clinical_array = nib.load(path).get_fdata()
            this_full_clinical_array = this_full_clinical_array.astype(np.float32)
            this_full_clinical_array = normalizationclinicalminmax1(this_full_clinical_array)
            print(np.min(this_full_clinical_array))
            print(np.max(this_full_clinical_array))
            this_big_clinical_patch = crop_nifti_2D(this_full_clinical_array, opt.micro_patch_size, is_random=True)
            this_small_clinical_patch = zoom(this_big_clinical_patch, (0.125, 0.125))
            
            save_this_big_clinical_patch = nib.Nifti1Image(this_big_clinical_patch, np.eye(4))
            save_this_small_clinical_patch = nib.Nifti1Image(this_small_clinical_patch, np.eye(4))
            nib.save(save_this_big_clinical_patch, '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/savedataset/bigclinicalpatch.nii.gz')
            nib.save(save_this_small_clinical_patch, '/homes/tzheng/Mypythonfiles/pytorch-CycleGAN-and-pix2pix-master/CycleGANnewloss/CTdata/savedataset/smallclinicalpatch.nii.gz')
            self.big_clinical_patchs[i*opt.batch_num + j,:,:] = this_big_clinical_patch
            self.small_clinical_patchs[i*opt.batch_num + j,:,:] = this_small_clinical_patch
