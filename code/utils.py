import torch
import kornia
import kornia.geometry.transform as KGT
import kornia.utils as KU
import kornia.filters as KF
import numpy as np
from time import time
import os
from tensorboardX import SummaryWriter
import torchvision


def randflow(img,angle=7,trans=0.07,ratio=1,sigma=15,base=500):
    h,w=img.shape[2],img.shape[3]
    # affine
    if not base is None:
        base_scale = base/torch.FloatTensor([w,h]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        angle = max(w,h)/base*angle
    else:
        base_scale = 1
    rand_angle = (torch.rand(1)*2-1)*angle
    rand_trans = (torch.rand(1,2)*2-1)*trans
    M = KGT.get_affine_matrix2d(translations=rand_trans,center=torch.zeros(1,2),scale=torch.ones(1,2),angle=rand_angle)
    M = M.inverse()
    grid = KU.create_meshgrid(h,w).to(img.device)
    warp_grid = kornia.geometry.linalg.transform_points(M,grid)
    #elastic
    disp = torch.rand([1,2,h,w])*2-1
    for i in range(5):
        disp = KF.gaussian_blur2d(disp,kernel_size=((3*sigma)//2*2+1,(3*sigma)//2*2+1),sigma=(sigma,sigma))
    disp = KF.gaussian_blur2d(disp,kernel_size=((3*sigma)//2*2+1,(3*sigma)//2*2+1),sigma=(sigma,sigma)).permute(0,2,3,1)*ratio

    disp = (disp+warp_grid-grid)*base_scale
    trans_grid = grid+disp
    
    mask = trans_grid<-1
    mask = torch.logical_or(trans_grid>1,mask)

    return trans_grid,trans_grid-grid,mask



def randrot(img):
    mode = np.random.randint(0,4)
    return rot(img,mode)

def randfilp(img):
    mode = np.random.randint(0,3)
    return flip(img,mode)

def rot(img, rot_mode):
    if rot_mode == 0:
        img = img.transpose(-2, -1)
        img = img.flip(-2)
    elif rot_mode == 1:
        img = img.flip(-2)
        img = img.flip(-1)
    elif rot_mode == 2:
        img = img.flip(-2)
        img = img.transpose(-2, -1)
    return img

def flip(img, flip_mode):
    if flip_mode == 0:
        img = img.flip(-2)
    elif flip_mode == 1:
        img = img.flip(-1)
    return img


class Saver():
    def __init__(self, opts):
        self.display_dir = os.path.join(opts.display_dir, opts.name)
        self.model_dir = os.path.join(opts.result_dir, opts.name)
        self.image_dir = os.path.join(self.model_dir, 'images')
        self.display_freq = opts.display_freq
        self.img_save_freq = opts.img_save_freq
        self.model_save_freq = opts.model_save_freq

        # make directory
        if not os.path.exists(self.display_dir):
            os.makedirs(self.display_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        # create tensorboard writer
        self.writer = SummaryWriter(logdir=self.display_dir)

    # write losses and images to tensorboard
    def write_display(self, total_it, model):
        if (total_it + 1) % self.display_freq == 0:
            # write loss
            members = [attr for attr in dir(model) if not callable(
                getattr(model, attr)) and not attr.startswith("__") and 'loss' in attr]
            for m in members:
                self.writer.add_scalar(m, getattr(model, m), total_it)
            # write img
            image_dis = torchvision.utils.make_grid(
                model.image_display, nrow=model.image_display.size(0)//2)
            self.writer.add_image('Image', image_dis, total_it)

    # save result images
    def write_img(self, ep, model, data_pair, stage='RF'):
        if ep % self.img_save_freq == 0:
            if stage == 'FS':
                assembled_images = model.assemble_outputs1()
            else:
                assembled_images = model.assemble_outputs()
            img_filename = '%s/gen_%05d_%s.jpg' % (self.image_dir, ep, data_pair)
            torchvision.utils.save_image(
                assembled_images, img_filename, nrow=1)
        elif ep == -1:
            assembled_images = model.assemble_outputs()
            img_filename = '%s/gen_last.jpg' % (self.image_dir, ep)
            torchvision.utils.save_image(
                assembled_images, img_filename, nrow=1)

    # save model
    def write_model(self, ep, total_it, model):
        print('--- save the model @ ep %d ---' % (ep))
        mode_save_path = '%s/%s.pth' % (self.model_dir, 'Reg')
        model.save(mode_save_path, ep, total_it)


def RGB2YCrCb(rgb_image):
    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = Y.clamp(0.0,1.0)
    Cr = Cr.clamp(0.0,1.0).detach()
    Cb = Cb.clamp(0.0,1.0).detach()
    return Y, Cb, Cr


def YCbCr2RGB(Y, Cb, Cr):
    ycrcb = torch.cat([Y, Cr, Cb], dim=1)
    B, C, W, H = ycrcb.shape
    im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    start = time()
    temp = (im_flat + bias).mm(mat)
    end = time()
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
    out = out.clamp(0,1.0)
    return out