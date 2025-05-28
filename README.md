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
