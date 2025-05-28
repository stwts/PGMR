
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
import os


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Up(nn.Module):
    def __init__(self,in_channels,out_channels,bilinear=True):

        super(Up, self).__init__()
        if bilinear:
            
            self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
            self.conv = doubleConv(in_channels,out_channels,in_channels//2) 
        else:
           
            self.up = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2)
            self.conv = doubleConv(in_channels,out_channels)

    def forward(self,x1,x2):
        x1 = self.up(x1)
        x = torch.cat([x1,x2],dim=1)
        x = self.conv(x)
        return x


def doubleConv(in_channels,out_channels,mid_channels=None):
    if mid_channels is None:
        mid_channels = out_channels
    layer = []
    layer.append(nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1,bias=False))
    layer.append(nn.BatchNorm2d(mid_channels))
    layer.append(nn.ReLU(inplace=True))
    layer.append(nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=1,bias=False))
    layer.append(nn.BatchNorm2d(out_channels))
    layer.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layer)


def down(in_channels,out_channels):
    layer = []
    layer.append(nn.MaxPool2d(2,stride=2))
    layer.append(doubleConv(in_channels,out_channels))
    return nn.Sequential(*layer)

class U_net(nn.Module):
    def __init__(self,in_channels,out_channels,bilinear=False,base_channel=64):
        super(U_net, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.in_conv = doubleConv(self.in_channels,base_channel)

        self.down1 = down(base_channel,base_channel*2) 
        self.down2 = down(base_channel*2,base_channel*4)
        self.down3 = down(base_channel*4,base_channel*8)

        factor = 2  if self.bilinear else 1
        self.down4 = down(base_channel*8,base_channel*16 // factor) 

        self.up1 = Up(base_channel*16 ,base_channel*8 // factor,self.bilinear) 
        self.up2 = Up(base_channel*8 ,base_channel*4 // factor,self.bilinear)
        self.up3 = Up(base_channel*4 ,base_channel*2 // factor,self.bilinear)
        self.up4 = Up(base_channel*2 ,base_channel,self.bilinear)
        
        self.out = nn.Conv2d(in_channels=base_channel,out_channels=self.out_channels,kernel_size=1)

    def forward(self,x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.out(x)

        return out
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass


def get_scheduler(optimizer, opts, cur_ep=-1):
    if opts.lr_policy == 'lambda':
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - opts.n_ep_decay) / \
                float(opts.n_ep - opts.n_ep_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
    elif opts.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opts.n_ep_decay, gamma=0.1, last_epoch=cur_ep)
    else:
        return NotImplementedError('no such learn rate policy')
    return scheduler

