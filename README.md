# PGMR
This is official Pytorch implementation of "[Plug-and-Play General Image Registration for Misaligned Multi-Modal Image Fusion](https://ieeexplore.ieee.org/document/11005625)".

## Framework
![The overall framework of the proposed PGMR.](https://github.com/stwts/PGMR/blob/main/figure/framework.jpg)

## Recommended Environment

 - [ ] torch  2.2.0 
 - [ ] torchvision 0.17.0 
 - [ ] kornia 0.7.1
 - [ ] opencv  4.7.0 
 - [ ] pillow  10.2.0

## Dataset
Please download the following datasets:
*   [RGB-IR](https://github.com/Linfeng-Tang/MSRS)
*   [RGB-SAR](https://github.com/AmberHen/WHU-OPT-SAR-dataset)
*   [PET-MRI](https://www.med.harvard.edu/AANLIB/home.html)
*   [SPECT-MRI](https://www.med.harvard.edu/AANLIB/home.html)

## Pretrained Models
1. Pretrained models of registration are as follows:
*   [regisration_model]() (code: wyi0) or [regisration_model]() (Google Link)


2. Pretrained models of detail enhancement are as follows:
*   [detail_enhancement_model]() (code: nju6) or [detail_enhancement_model]() (Google Link)

  
## To Test
Please place the pre-trained model weights in ‘/checkpoint’. Please place the data to be registered in ‘/test_dataset/orign’, and the reference data in ‘/test_dataset/target’.
#### RGB-IR
    python test.py --modal_orign=ir --modal_target==vi
#### RGB-SAR
    python test.py --modal_orign=sar --modal_target==vi
#### PET-MRI
    python test.py --modal_orign=MRI --modal_target==PET
#### SPECT-MRI
    python test.py --modal_orign=MRI --modal_target==SPECT

## To Train
Please place the datasets in ‘/train_dataset’.

    python train.py
    
## If this work is helpful to you, please cite it as：
```
@ARTICLE{11005625,
  author={Zheng, Tianheng and Dong, Guanglu and Zhang, Pingping and He, Xiaohai and Ren, Chao},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Plug-and-Play General Image Registration for Misaligned Multi-Modal Image Fusion}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Image registration;Image fusion;Circuits and systems;Training;Data models;Computational modeling;Integrated circuit modeling;Information processing;Deformation;Data mining;Image registration;image fusion;prompt learning;multi-modal image;details enhancement},
  doi={10.1109/TCSVT.2025.3570530}}
```
