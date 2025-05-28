import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.utils as KU
from code.modules import U_net, gaussian_weights_init, get_scheduler
import kornia.filters as KF
from code.utils import RGB2YCrCb, YCbCr2RGB
from code.loss import *
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


##########################################################################
##---------- Prompt Gen Module -----------------------
class PromptGenBlock(nn.Module):
    def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192):
        super(PromptGenBlock,self).__init__()
        #将prompt当作可训练的参数
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        #全连接层将输入图像的通道数转换为prompt的个数
        self.linear_layer = nn.Linear(lin_dim,prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)
        

    def forward(self,x):
        B,C,H,W = x.shape
        #利用输入特征图获得prompt的权重
        emb = x.mean(dim=(-2,-1))
        prompt_weights = F.softmax(self.linear_layer(emb),dim=1)
        #对prompt加权
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        #将所有维度的prompt相加
        prompt = torch.sum(prompt,dim=1)
        #将prompt插值到与特征图同一尺寸
        prompt = F.interpolate(prompt,(H,W),mode="bilinear")
        #3*3卷积
        prompt = self.conv3x3(prompt)
        return prompt

class Conv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride=1, padding=0, dilation=1, norm=None, act=nn.LeakyReLU,bias=False):
        super(Conv2d, self).__init__()
        model = []
        model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size,
                            stride=stride, padding=padding, bias=bias, dilation=dilation)]
        if not norm is None:
            model += [norm(n_out, affine=False)]
        if act is nn.LeakyReLU:
            model += [act(negative_slope=0.1,inplace=True)]
        elif act is None:
            model +=[]
        else:
            model +=[act()]
        self.model = nn.Sequential(*model)
        # elif == 'Group'

    def forward(self, x):
        return self.model(x)

class ResConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, dilation=1, norm=None,):
        super(ResConv2d, self).__init__()
        model = []
        model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size,
                            stride=stride, padding=padding, bias=False, dilation=dilation)]
        if not norm is None:
            model += [norm(n_out, affine=False)]
        model += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        # elif == 'Group'

    def forward(self, x):
        return self.model(x)+x



class SpatialTransformer(nn.Module):
    def __init__(self, h,w, gpu_use, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        grid = KU.create_meshgrid(h,w)
        grid = grid.type(torch.FloatTensor).cuda() if gpu_use else grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, src, disp):
        if disp.shape[1]==2:
            disp = disp.permute(0,2,3,1)
        if disp.shape[1] != self.grid.shape[1] or disp.shape[2] != self.grid.shape[2]:
            self.grid = KU.create_meshgrid(disp.shape[1],disp.shape[2]).to(disp.device)
        flow = self.grid + disp
        return F.grid_sample(src, flow, mode=self.mode, padding_mode='zeros', align_corners=False)



class Feature_extractor_unshare(nn.Module):
    def __init__(self,depth,base_ic,base_oc,base_dilation,norm):
        super(Feature_extractor_unshare,self).__init__()
        feature_extractor = nn.ModuleList([])
        ic = base_ic
        oc = base_oc
        dilation = base_dilation
        for i in range(depth):
            if i%2==1:
                dilation *= 2
            if ic == oc:
                feature_extractor.append(ResConv2d(ic,oc,kernel_size=3,stride=1,padding=dilation,dilation=dilation, norm=norm))
            else:
                feature_extractor.append(Conv2d(ic,oc,kernel_size=3,stride=1,padding=dilation,dilation=dilation, norm=norm))
            ic = oc
            if i%2==1 and i<depth-1:
                oc *= 2
        self.ic = ic
        self.oc = oc
        self.dilation = dilation
        self.layers = feature_extractor

    def forward(self,x):
        for i,layer in enumerate(self.layers):
            x = layer(x)
        return x



class DispRefiner(nn.Module):
    def __init__(self,channel,dilation=1,depth=4):
        super(DispRefiner,self).__init__()
        self.preprocessor = nn.Sequential(Conv2d(channel,channel,3,dilation=dilation,padding=dilation,norm=None,act=None))
        self.featcompressor = nn.Sequential(Conv2d(channel*2,channel*2,3,padding=1),
        Conv2d(channel*2,channel,3,padding=1,norm=None,act=None))
        oc = channel
        ic = channel+2
        dilation = 1
        estimator = nn.ModuleList([])
        for i in range(depth-1):
            oc = oc//2
            estimator.append(Conv2d(ic,oc,kernel_size=3,stride=1,padding=dilation,dilation=dilation, norm=nn.BatchNorm2d))
            ic = oc
            dilation *= 2
        estimator.append(Conv2d(oc,2,kernel_size=3,padding=1,dilation=1,act=None,norm=None))
        #estimator.append(nn.Tanh())
        self.estimator = nn.Sequential(*estimator)
    def forward(self,feat1,feat2,disp):
        
        b=feat1.shape[0]
        feat = torch.cat([feat1,feat2])
        feat = self.preprocessor(feat)
        feat = self.featcompressor(torch.cat([feat[:b],feat[b:]],dim=1))
        corr = torch.cat([feat,disp],dim=1)
        delta_disp = self.estimator(corr)
        disp = disp+delta_disp
        return disp 



class DispEstimator(nn.Module):
    def __init__(self,channel,depth=4,norm=nn.BatchNorm2d,dilation=1):
        super(DispEstimator,self).__init__()
        estimator = nn.ModuleList([])
        self.corrks = 7
        self.preprocessor = Conv2d(channel,channel,3,act=None,norm=None,dilation=dilation,padding=dilation)
        self.featcompressor = nn.Sequential(Conv2d(channel*2,channel*2,3,padding=1),
        Conv2d(channel*2,channel,3,padding=1,act=None))
        #self.localcorrpropcessor = nn.Sequential(Conv2d(self.corrks**2,32,3,padding=1,bias=True,norm=None),
        #                                         Conv2d(32,2,3,padding=1,bias=True,norm=None),)
        oc = channel
        ic = channel+self.corrks**2
        dilation = 1
        for i in range(depth-1):
            oc = oc//2
            estimator.append(Conv2d(ic,oc,kernel_size=3,stride=1,padding=dilation,dilation=dilation, norm=norm))
            ic = oc
            dilation *= 2
        estimator.append(Conv2d(oc,2,kernel_size=3,padding=1,dilation=1,act=None,norm=None))
        #estimator.append(nn.Tanh())
        self.layers = estimator
        self.scale = torch.FloatTensor([256,256]).cuda().unsqueeze(-1).unsqueeze(-1).unsqueeze(0)-1

    def localcorr(self,feat1,feat2):
        feat = self.featcompressor(torch.cat([feat1,feat2],dim=1))
        b,c,h,w = feat2.shape
        feat1_smooth = KF.gaussian_blur2d(feat1,(13,13),(3,3),border_type='constant')
        feat1_loc_blk = F.unfold(feat1_smooth,kernel_size=self.corrks,dilation=4,padding=2*(self.corrks-1),stride=1).reshape(b,c,-1,h,w)
        localcorr = (feat2.unsqueeze(2)-feat1_loc_blk).pow(2).mean(dim=1)
        corr = torch.cat([feat,localcorr],dim=1)
        return corr

    def forward(self,feat1,feat2):
        b,c,h,w = feat1.shape
        feat = torch.cat([feat1,feat2])
        feat = self.preprocessor(feat)
        feat1 = feat[:b]
        feat2 = feat[b:]
        if self.scale[0,1,0,0] != w-1 or self.scale[0,0,0,0] != h-1:
            self.scale = torch.FloatTensor([w,h]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)-1
            self.scale = self.scale.to(feat1.device)
        corr = self.localcorr(feat1,feat2)
        for i,layer in enumerate(self.layers):
            corr = layer(corr)
        corr = KF.gaussian_blur2d(corr,(13,13),(3,3),border_type='replicate')
        disp = corr.clamp(min=-300,max=300)
        # print(disp.shape)
        # print(feat1.shape)
        return disp/self.scale


def save_high_low_frequencies_from_multichannel_image(image, output_path, grid_cols=8):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    # 检查输入是否为四维，并提取实际特征图
    if len(image.shape) != 4 or image.shape[0] != 1:
        raise ValueError("输入图像必须是四维数组，形状为 (1, H, W, C)。")

    image = image[0]  # 忽略第一维，提取 (H, W, C)
    h, w, c = image.shape
    
    # 对所有通道进行平均池化，得到单通道特征图
    averaged_image = np.mean(image, axis=-1)  # 计算每个像素位置的通道平均值，得到 (H, W)
    
    # 对平均池化后的单通道特征图进行傅里叶变换
    fft = np.fft.fft2(averaged_image)
    fft_shifted = np.fft.fftshift(fft)

    # 提取低频信息（频谱中心部分）
    low_freq = np.zeros_like(fft_shifted)
    crow, ccol = h // 2, w // 2  # 中心位置
    radius = min(h, w) // 8  # 半径为图像最小尺寸的1/8
    low_freq[crow - radius:crow + radius, ccol - radius:ccol + radius] = fft_shifted[
        crow - radius:crow + radius, ccol - radius:ccol + radius
    ]

    # 提取高频信息（其余部分）
    high_freq = fft_shifted - low_freq

    # 傅里叶逆变换，得到低频和高频图像
    low_freq_img = np.abs(np.fft.ifft2(np.fft.ifftshift(low_freq)))
    high_freq_img = np.abs(np.fft.ifft2(np.fft.ifftshift(high_freq)))

    # 创建绘图网格
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # 可视化原始通道平均池化后的特征图、低频图像和高频图像
    axs[0].imshow(averaged_image, cmap="gray")
    axs[0].set_title("Averaged Feature Map (Single Channel)")
    axs[0].axis("off")

    axs[1].imshow(low_freq_img, cmap="gray")
    axs[1].set_title("Low Frequency")
    axs[1].axis("off")

    axs[2].imshow(high_freq_img, cmap="gray")
    axs[2].set_title("High Frequency")
    axs[2].axis("off")

    # 调整布局
    plt.tight_layout()

    # 保存结果图像
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"高低频特征图已保存至: {output_path}")


#配准器
class DenseMatcher(nn.Module):
    def __init__(self,model_orign,model_target,unshare_depth=4,matcher_depth=4,num_pyramids=2):
        super(DenseMatcher,self).__init__()
        self.num_pyramids=num_pyramids
        #这个可以不要了，不改变尺寸，维度放大
        self.feature_extractor_unshare1 = Feature_extractor_unshare(depth=unshare_depth,base_ic=3,base_oc=8,base_dilation=1,norm=nn.InstanceNorm2d)
        self.feature_extractor_unshare2 = Feature_extractor_unshare(depth=unshare_depth,base_ic=3,base_oc=8,base_dilation=1,norm=nn.InstanceNorm2d)
        #self.feature_extractor_unshare2 = self.feature_extractor_unshare1
        base_ic = self.feature_extractor_unshare1.ic
        base_oc = self.feature_extractor_unshare1.oc
        base_dilation = self.feature_extractor_unshare1.dilation
        #修改这里，这里是编码器
        self.feature_extractor_share1 = nn.Sequential(Conv2d(base_oc,base_oc*2,kernel_size=3,stride=1,padding=1,dilation=1, norm=nn.InstanceNorm2d),
        Conv2d(base_oc*2,base_oc*2,kernel_size=3,stride=2,padding=1,dilation=1, norm=nn.InstanceNorm2d))
        self.feature_extractor_share2 = nn.Sequential(Conv2d(base_oc*2,base_oc*4,kernel_size=3,stride=1,padding=2,dilation=2, norm=nn.InstanceNorm2d),
        Conv2d(base_oc*4,base_oc*4,kernel_size=3,stride=2,padding=2,dilation=2, norm=nn.InstanceNorm2d))
        self.feature_extractor_share3 = nn.Sequential(Conv2d(base_oc*4,base_oc*8,kernel_size=3,stride=1,padding=4,dilation=4, norm=nn.InstanceNorm2d),
        Conv2d(base_oc*8,base_oc*8,kernel_size=3,stride=2,padding=4,dilation=4, norm=nn.InstanceNorm2d))
        ###########################
        self.matcher1 = DispEstimator(base_oc*4,matcher_depth,dilation=4)
        self.matcher2 = DispEstimator(base_oc*8,matcher_depth,dilation=2)
        self.refiner = DispRefiner(base_oc*2,1)
        self.grid_down = KU.create_meshgrid(64,64).cuda()
        self.grid_full = KU.create_meshgrid(128,128).cuda()
        self.scale = torch.FloatTensor([256,256]).cuda().unsqueeze(-1).unsqueeze(-1).unsqueeze(0)-1

        ##prompt相关#####################################################################################
        self.model_orign = model_orign
        self.model_target = model_target
        #vis-ir,1_ir,2_vis
        self.feat31_PGB = PromptGenBlock(prompt_dim=base_oc*8,prompt_len=5,prompt_size = 96,lin_dim = base_oc*8).to(device)
        self.feat32_PGB = PromptGenBlock(prompt_dim=base_oc*8,prompt_len=5,prompt_size = 96,lin_dim = base_oc*8).to(device)
        self.feat21_PGB = PromptGenBlock(prompt_dim=base_oc*4,prompt_len=5,prompt_size = 96,lin_dim = base_oc*4).to(device)
        self.feat22_PGB = PromptGenBlock(prompt_dim=base_oc*4,prompt_len=5,prompt_size = 96,lin_dim = base_oc*4).to(device)
        self.feat11_PGB = PromptGenBlock(prompt_dim=base_oc*2,prompt_len=5,prompt_size = 96,lin_dim = base_oc*2).to(device)
        self.feat12_PGB = PromptGenBlock(prompt_dim=base_oc*2,prompt_len=5,prompt_size = 96,lin_dim = base_oc*2).to(device)
        #SPECT-MRI,1_MRI,2_SPECT
        self.feat31_PGB_SPECT_MRI = PromptGenBlock(prompt_dim=base_oc*8,prompt_len=5,prompt_size = 96,lin_dim = base_oc*8).to(device)
        self.feat32_PGB_SPECT_MRI = PromptGenBlock(prompt_dim=base_oc*8,prompt_len=5,prompt_size = 96,lin_dim = base_oc*8).to(device)
        self.feat21_PGB_SPECT_MRI = PromptGenBlock(prompt_dim=base_oc*4,prompt_len=5,prompt_size = 96,lin_dim = base_oc*4).to(device)
        self.feat22_PGB_SPECT_MRI = PromptGenBlock(prompt_dim=base_oc*4,prompt_len=5,prompt_size = 96,lin_dim = base_oc*4).to(device)
        self.feat11_PGB_SPECT_MRI = PromptGenBlock(prompt_dim=base_oc*2,prompt_len=5,prompt_size = 96,lin_dim = base_oc*2).to(device)
        self.feat12_PGB_SPECT_MRI = PromptGenBlock(prompt_dim=base_oc*2,prompt_len=5,prompt_size = 96,lin_dim = base_oc*2).to(device)
        #CT-MRI,2_PET
        self.feat32_PGB_PET = PromptGenBlock(prompt_dim=base_oc*8,prompt_len=5,prompt_size = 96,lin_dim = base_oc*8).to(device)
        self.feat22_PGB_PET = PromptGenBlock(prompt_dim=base_oc*4,prompt_len=5,prompt_size = 96,lin_dim = base_oc*4).to(device)
        self.feat12_PGB_PET = PromptGenBlock(prompt_dim=base_oc*2,prompt_len=5,prompt_size = 96,lin_dim = base_oc*2).to(device)
        #VI-SAR,1_SAR
        self.feat31_PGB_SAR = PromptGenBlock(prompt_dim=base_oc*8,prompt_len=5,prompt_size = 96,lin_dim = base_oc*8).to(device)
        self.feat21_PGB_SAR = PromptGenBlock(prompt_dim=base_oc*4,prompt_len=5,prompt_size = 96,lin_dim = base_oc*4).to(device)
        self.feat11_PGB_SAR = PromptGenBlock(prompt_dim=base_oc*2,prompt_len=5,prompt_size = 96,lin_dim = base_oc*2).to(device)
        #特征参考融合器
        self.feat_prompt_fusion3 = U_net(base_oc*8*2,base_oc*8,bilinear=False,base_channel=128).to(device)
        #self.feat_prompt_fusion32 = U_net(base_oc*8*2,base_oc*8,bilinear=False,base_channel=128).to(device)
        self.feat_prompt_fusion2 = U_net(base_oc*4*2,base_oc*4,bilinear=False,base_channel=128).to(device)
        #self.feat_prompt_fusion22 = U_net(base_oc*4*2,base_oc*4,bilinear=False,base_channel=128).to(device)
        self.feat_prompt_fusion1 = U_net(base_oc*2*2,base_oc*2,bilinear=False,base_channel=128).to(device)
        #self.feat_prompt_fusion12 = U_net(base_oc*2*2,base_oc*2,bilinear=False,base_channel=128).to(device)
        ######################################################################################################

    def match(self,feat11,feat12,feat21,feat22,feat31,feat32):
        #compute scale (w,h)
        if self.scale[0,1,0,0]*2 != feat11.shape[2]-1 or self.scale[0,0,0,0]*2 != feat11.shape[3]-1:
            self.h,self.w = feat11.shape[2],feat11.shape[3]
            self.scale = torch.FloatTensor([self.w,self.h]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)-1 
            self.scale = self.scale.to(feat11.device)

        #嵌入prompt#######################
        if self.model_orign == 'ir':
            feat31_Prompt = self.feat31_PGB(feat31)
            feat32_Prompt = self.feat32_PGB(feat32)
            #print('ir-vi')

        elif self.model_orign == 'sar':
            feat31_Prompt = self.feat31_PGB_SAR(feat31)
            feat32_Prompt = self.feat32_PGB(feat32)
            #print('sar-vi')

        elif self.model_orign == 'vi':
            feat31_Prompt = self.feat32_PGB(feat31)
            if self.model_target == 'sar':
                feat32_Prompt = self.feat31_PGB_SAR(feat32)
                #print('vi-sar')
            if self.model_target == 'ir':
                feat32_Prompt = self.feat31_PGB(feat32)
                #print('vi-ir')

        elif self.model_orign == 'MRI':
            feat31_Prompt = self.feat31_PGB_SPECT_MRI(feat31)
            if self.model_target == 'SPECT':
                feat32_Prompt = self.feat32_PGB_SPECT_MRI(feat32)
                #print('MRI-SPECT')
            if self.model_target == 'PET':
                feat32_Prompt = self.feat32_PGB_PET(feat32)
                #print('MRI-PET')
        
        elif self.model_orign == 'SPECT':
            feat31_Prompt = self.feat32_PGB_SPECT_MRI(feat31)
            feat32_Prompt = self.feat31_PGB_SPECT_MRI(feat32)
            #print('SPECT-MRI')

        elif self.model_orign == 'PET':
            feat31_Prompt = self.feat32_PGB_PET(feat31)
            feat32_Prompt = self.feat31_PGB_SPECT_MRI(feat32)
            #print('PET-MRI')

     
        feat31_and_P = torch.cat([feat31,feat31_Prompt],dim=1)
        feat32_and_P = torch.cat([feat32,feat32_Prompt],dim=1)
        feat31 = self.feat_prompt_fusion3(feat31_and_P)
        feat32 = self.feat_prompt_fusion3(feat32_and_P)


        ##############################################

        #estimate disp src(feat1) to tgt(feat2) in low resolution
        disp2_raw = self.matcher2(feat31,feat32) 
        
        #upsample disp and grid
        disp2 = F.interpolate(disp2_raw,[feat21.shape[2],feat21.shape[3]],mode='bilinear')
        if disp2.shape[2] != self.grid_down.shape[1] or disp2.shape[3] != self.grid_down.shape[2]:
            self.grid_down = KU.create_meshgrid(feat21.shape[2],feat21.shape[3]).cuda()

        #嵌入prompt#######################
        if self.model_orign == 'ir':
            feat21_Prompt = self.feat21_PGB(feat21)
            feat22_Prompt = self.feat22_PGB(feat22)
            #print('ir-vi')

        elif self.model_orign == 'sar':
            feat21_Prompt = self.feat21_PGB_SAR(feat21)
            feat22_Prompt = self.feat22_PGB(feat22)
            #print('sar-vi')

        elif self.model_orign == 'vi':
            feat21_Prompt = self.feat22_PGB(feat21)
            if self.model_target == 'sar':
                feat22_Prompt = self.feat21_PGB_SAR(feat22)
                #print('vi-sar')
            if self.model_target == 'ir':
                feat22_Prompt = self.feat21_PGB(feat22)
                #print('vi-ir')

        elif self.model_orign == 'MRI':
            feat21_Prompt = self.feat21_PGB_SPECT_MRI(feat21)
            if self.model_target == 'SPECT':
                feat22_Prompt = self.feat22_PGB_SPECT_MRI(feat22)
                #print('MRI-SPECT')
            if self.model_target == 'PET':
                feat22_Prompt = self.feat22_PGB_PET(feat22)
                #print('MRI-PET')
        
        elif self.model_orign == 'SPECT':
            feat21_Prompt = self.feat22_PGB_SPECT_MRI(feat21)
            feat22_Prompt = self.feat21_PGB_SPECT_MRI(feat22)
            #print('SPECT-MRI')

        elif self.model_orign == 'PET':
            feat21_Prompt = self.feat22_PGB_PET(feat21)
            feat22_Prompt = self.feat21_PGB_SPECT_MRI(feat22)
            #print('PET-MRI')

        

        feat21_and_P = torch.cat([feat21,feat21_Prompt],dim=1)
        feat22_and_P = torch.cat([feat22,feat22_Prompt],dim=1)
        feat21 = self.feat_prompt_fusion2(feat21_and_P)
        feat22 = self.feat_prompt_fusion2(feat22_and_P)
        ##############################################

        #warp the last src(fea1) to tgt(feat2) with disp2
        feat21 = F.grid_sample(feat21,self.grid_down+disp2.permute(0,2,3,1))

        #estimate disp src(feat1) to tgt(feat2) in low resolution
        disp1_raw = self.matcher1(feat21,feat22)

        #upsample
        disp1 = F.interpolate(disp1_raw,[feat11.shape[2],feat11.shape[3]],mode='bilinear')
        disp2 = F.interpolate(disp2,[feat11.shape[2],feat11.shape[3]],mode='bilinear')
        if disp1.shape[2] != self.grid_full.shape[1] or disp1.shape[3] != self.grid_full.shape[2]:
            self.grid_full = KU.create_meshgrid(feat11.shape[2],feat11.shape[3]).cuda()

        #嵌入prompt#######################
        if self.model_orign == 'ir':
            feat11_Prompt = self.feat11_PGB(feat11)
            feat12_Prompt = self.feat12_PGB(feat12)
            #print('ir-vi')

        elif self.model_orign == 'sar':
            feat11_Prompt = self.feat11_PGB_SAR(feat11)
            feat12_Prompt = self.feat12_PGB(feat12)
            #print('sar-vi')

        elif self.model_orign == 'vi':
            feat11_Prompt = self.feat12_PGB(feat11)
            if self.model_target == 'sar':
                feat12_Prompt = self.feat11_PGB_SAR(feat12)
                #print('vi-sar')
            if self.model_target == 'ir':
                feat12_Prompt = self.feat11_PGB(feat12)
                #print('vi-ir')

        elif self.model_orign == 'MRI':
            feat11_Prompt = self.feat11_PGB_SPECT_MRI(feat11)
            if self.model_target == 'SPECT':
                feat12_Prompt = self.feat12_PGB_SPECT_MRI(feat12)
                #print('MRI-SPECT')
            if self.model_target == 'PET':
                feat12_Prompt = self.feat12_PGB_PET(feat12)
                #print('MRI-PET')
        
        elif self.model_orign == 'SPECT':
            feat11_Prompt = self.feat12_PGB_SPECT_MRI(feat11)
            feat12_Prompt = self.feat11_PGB_SPECT_MRI(feat12)
            #print('SPECT-MRI')

        elif self.model_orign == 'PET':
            feat11_Prompt = self.feat12_PGB_PET(feat11)
            feat12_Prompt = self.feat11_PGB_SPECT_MRI(feat12)
            #print('PET-MRI')

        

        feat11_and_P = torch.cat([feat11,feat11_Prompt],dim=1)
        feat12_and_P = torch.cat([feat12,feat12_Prompt],dim=1)
        feat11 = self.feat_prompt_fusion1(feat11_and_P)
        feat12 = self.feat_prompt_fusion1(feat12_and_P)

        

        

        
        #####################################################################

        #warp
        feat11 = F.grid_sample(feat11,self.grid_full+(disp1+disp2).permute(0,2,3,1))

        #finetune
        disp_scaleup = (disp1+disp2)*self.scale
        disp = self.refiner(feat11,feat12,disp_scaleup)
        disp = KF.gaussian_blur2d(disp,(17,17),(5,5),border_type='replicate')/self.scale
        if self.training:
            return disp,disp_scaleup/self.scale,disp2
        return disp,None,None    
        
    def forward(self,src,tgt,type='ir2vis'):
        b,c,h,w = tgt.shape
        feat01 = self.feature_extractor_unshare1(src)
        feat02 = self.feature_extractor_unshare2(tgt)
        feat0 = torch.cat([feat01,feat02])
        feat1 = self.feature_extractor_share1(feat0)
        feat2 = self.feature_extractor_share2(feat1)
        feat3 = self.feature_extractor_share3(feat2)
        feat11,feat12 = feat1[0:b],feat1[b:]
        feat21,feat22 = feat2[0:b],feat2[b:]
        feat31,feat32 = feat3[0:b],feat3[b:]
        disp_12 = None
        disp_21 = None
        if type == 'bi':
            disp_12,disp_12_down4,disp_12_down8 = self.match(feat11,feat12,feat21,feat22,feat31,feat32)
            #暂时互换
            a = self.model_orign
            self.model_orign = self.model_target
            self.model_target = a
            disp_21,disp_21_down4,disp_21_down8 = self.match(feat12,feat11,feat22,feat21,feat32,feat31)
            #换回来
            a = self.model_orign
            self.model_orign = self.model_target
            self.model_target = a
            t = torch.cat([disp_12,disp_21,disp_12_down4,disp_21_down4,disp_12_down8,disp_21_down8])
            t = F.interpolate(t,[h,w],mode='bilinear')
            down2,down4,donw8 = torch.split(t,2*b,dim=0)
            disp_12_,disp_21_ = torch.split(down2,b,dim=0)
        elif type == 'ir2vis':
            disp_12,_,_= self.match(feat11,feat12,feat21,feat22,feat31,feat32)
            disp_12 = F.interpolate(disp_12,[h,w],mode='bilinear')
        elif type =='vis2ir':
            disp_21,_,_ = self.match(feat12,feat11,feat22,feat21,feat32,feat31)
            disp_21 = F.interpolate(disp_21,[h,w],mode='bilinear')
        if self.training:
            return {'ir2vis':disp_12_,'vis2ir':disp_21_,
            'down2':down2,
            'down4':down4,
            'down8':donw8}    
        return {'ir2vis':disp_12,'vis2ir':disp_21}

    def change(self, orign, target):
        self.model_orign = orign
        self.model_target = target





class SuperFusion(nn.Module):
    def __init__(self, opts=None):
        super(SuperFusion, self).__init__()

        # parameters
        lr = 0.0001
        # encoders
        self.model_orign = opts.model_orign
        self.model_target = opts.model_target
        self.DM = DenseMatcher(model_orign=self.model_orign, model_target=self.model_target)#配准
        self.resume_flag = False
        self.ST = SpatialTransformer(256, 256, True)
        #self.FN = FusionNet()#融合

        # optimizers
        self.DM_opt = torch.optim.Adam(
            self.DM.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.00001)
        # self.FN_opt = torch.optim.Adam(
        #     self.FN.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.00001)
        self.gradientloss = gradientloss()
        #self.ncc_loss = ncc_loss()
        #self.ssim_loss = ssimloss
        #self.weights_sim = [1, 1, 0.2]
        #self.weights_ssim1 = [0.3, 0.7]
        #self.weights_ssim2 = [0.7, 0.3]

        self.deformation_1 = {}
        self.deformation_2 = {}
        self.border_mask = torch.zeros([1, 1, 256, 256])
        self.border_mask[:, :, 10:-10, 10:-10] = 1
        self.AP = nn.AvgPool2d(5, stride=1, padding=2)
        self.initialize()


    def initialize(self):
        self.DM.apply(gaussian_weights_init)
        # self.FN.apply(gaussian_weights_init)

    def set_scheduler(self, opts, last_ep=0):
        self.DM_sch = get_scheduler(self.DM_opt, opts, last_ep)
        #self.FN_sch = get_scheduler(self.FN_opt, opts, last_ep)

    def setgpu(self, gpu):
        self.gpu = gpu
        self.DM.cuda(self.gpu)
        #self.FN.cuda(self.gpu)

    # def test_forward(self, image_ir, image_vi):
    #     deformation = self.DM(image_ir, image_vi)
    #     image_ir_Reg = self.ST(image_ir, deformation['ir2vis'])
    #     image_fusion = self.FN(image_ir_Reg, image_vi)
    #     return image_fusion

    def generate_mask(self):
        flow = self.ST.grid + self.disp
        goodmask = torch.logical_and(flow >= -1, flow <= 1)
        if self.border_mask.device != goodmask.device:
            self.border_mask = self.border_mask.to(goodmask.device)
        self.goodmask = torch.logical_and(goodmask[..., 0], goodmask[..., 1]).unsqueeze(1) * 1.0
        for i in range(2):
            self.goodmask = (self.AP(self.goodmask) > 0.3).float()

        flow = self.ST.grid - self.disp
        goodmask = F.grid_sample(self.goodmask, flow)
        self.goodmask_inverse = goodmask

    def forward(self, ir, vi):
        disp = self.DM(ir, vi)['ir2vis']
        ir_reg = self.ST(ir, disp)
        # vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(vi)
        # fu = self.FN(ir_reg[:, 0:1], vi_Y)
        # fu = YCbCr2RGB(fu, vi_Cb, vi_Cr)
        return ir_reg

    def registration_forward(self, ir, vi):
        disp = self.DM(ir, vi)['ir2vis']
        ir_reg = self.ST(ir, disp)
        return ir_reg, disp

    # def fusion_forward(self, ir, vi):
    #     vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(vi)
    #     fu = self.FN(ir[:, 0:1], vi_Y)
    #     fu = YCbCr2RGB(fu, vi_Cb, vi_Cr)
    #     return fu

    def train_forward_RF(self):
        #batch_size
        b = self.image_ir_warp_RGB.shape[0]
        #将形变图像与原图像concat在b维上,b维在经过网络时不会互相影响
        ir_stack = torch.cat([self.image_ir_warp_RGB, self.image_ir_RGB])
        vi_stack = torch.cat([self.image_vi_RGB, self.image_vi_warp_RGB])
        #经过DM模块，得到形变场
        deformation = self.DM(ir_stack, vi_stack, type='bi')

        self.down2 = deformation['down2']
        self.down4 = deformation['down4']
        self.down8 = deformation['down8']
        self.deformation_1['vis2ir'], self.deformation_2['vis2ir'] = deformation['vis2ir'][0:b, ...], deformation[
                                                                                                          'vis2ir'][b:,
                                                                                                      ...]
        self.deformation_1['ir2vis'], self.deformation_2['ir2vis'] = deformation['ir2vis'][0:b, ...], deformation[
                                                                                                          'ir2vis'][b:,
                                                                                                      ...]
        img_stack = torch.cat([ir_stack, vi_stack])
        disp_stack = torch.cat([deformation['ir2vis'], deformation['vis2ir']])
        img_warp_stack = self.ST(img_stack, disp_stack)

        self.image_ir_Reg_RGB, self.image_ir_warp_fake_RGB, self.image_vi_warp_fake_RGB, self.image_vi_Reg_RGB = torch.split(
            img_warp_stack, b, dim=0)

        self.image_vi_Y, self.image_vi_Cb, self.image_vi_Cr = RGB2YCrCb(self.image_vi_RGB)
        self.image_vi_Reg_Y, self.image_vi_Reg_Cb, self.image_vi_Reg_Cr = RGB2YCrCb(self.image_vi_Reg_RGB)
        self.image_ir_Y = self.image_ir_RGB[:, 0:1, ...]
        self.image_ir_Reg_Y = self.image_ir_Reg_RGB[:, 0:1, ...]

        ir_stack_reg_Y = torch.cat([self.image_ir_Y, self.image_ir_Reg_Y])
        vi_stack_reg_Y = torch.cat([self.image_vi_Reg_Y, self.image_vi_Y])

        # fusion_img = self.FN(ir_stack_reg_Y, vi_stack_reg_Y)
        # self.image_fusion_1, self.image_fusion_2 = torch.split(fusion_img, b, dim=0)

        self.generate_mask()
        self.image_display = torch.cat((self.image_ir_RGB[0:1, 0:1], self.image_ir_warp_RGB[0:1, 0:1],
                                        self.image_ir_Reg_RGB[0:1, 0:1],
                                        (self.image_vi_RGB - self.image_vi_warp_RGB)[0:1].abs().mean(dim=1,
                                                                                                     keepdim=True),
                                        self.image_vi_Y[0:1], RGB2YCrCb(self.image_vi_warp_RGB[0:1])[0],
                                        self.image_vi_Reg_Y[0:1],
                                        (self.image_vi_RGB - self.image_vi_Reg_RGB)[0:1].abs().mean(dim=1,
                                                                                                    keepdim=True)), dim=0).detach()

    # def train_forward_FS(self):
    #     self.image_vi_Y, self.image_vi_Cb, self.image_vi_Cr = RGB2YCrCb(self.image_vi_RGB)
    #     self.image_ir_Y = self.image_ir_RGB[:, 0:1, ...]

    #     fusion_img = self.FN(self.image_ir_Y, self.image_vi_Y)
    #     self.image_fusion = fusion_img
    #     self.fused_image_RGB = YCbCr2RGB(self.image_fusion, self.image_vi_Cb, self.image_vi_Cr)

    # def update_FS(self, image_ir, image_vi, seg_model, label=None, dataset_name='MSRS'):
    #     self.image_ir_RGB = image_ir
    #     self.image_vi_RGB = image_vi
    #     self.seg_model = seg_model
    #     if dataset_name == 'MSRS':
    #         self.label = label
    #     else:
    #         self.label = self.image_ir_RGB[:, 0:1, :, :]
    #     self.FN_opt.zero_grad()
    #     self.train_forward_FS()
    #     # update DM, FM
    #     self.backward_FS(seg_flag=False, dataset_name=dataset_name)
    #     nn.utils.clip_grad_norm_(self.FN.parameters(), 5)
    #     self.FN_opt.step()

    def update_RF(self, image_ir, image_vi, image_ir_warp, image_vi_warp, disp, orign, target):
        self.DM.change(orign, target)
        self.image_ir_RGB = image_ir
        self.image_vi_RGB = image_vi
        self.image_ir_warp_RGB = image_ir_warp
        self.image_vi_warp_RGB = image_vi_warp
        self.disp = disp
        #self.FN_opt.zero_grad()
        self.DM_opt.zero_grad()#DM模块梯度归零
        self.train_forward_RF()
        self.backward_RF()
        nn.utils.clip_grad_norm_(self.DM.parameters(), 5)
        #nn.utils.clip_grad_norm_(self.FN.parameters(), 5)
        self.DM_opt.step()

        
        #self.FN_opt.step()

    def imgloss(self, src, tgt, mask=1, weights=[0.1, 0.9]):
        return weights[0] * (l1loss(src, tgt, mask) + l2loss(src, tgt, mask)) + weights[1] * self.gradientloss(src, tgt,
                                                                                                               mask)

    def weightfiledloss(self, ref, tgt, disp, disp_gt):
        ref = (ref - ref.mean(dim=[-1, -2], keepdim=True)) / (ref.std(dim=[-1, -2], keepdim=True) + 1e-5)
        tgt = (tgt - tgt.mean(dim=[-1, -2], keepdim=True)) / (tgt.std(dim=[-1, -2], keepdim=True) + 1e-5)
        g_ref = KF.spatial_gradient(ref, order=2).mean(dim=1).abs().sum(dim=1).detach().unsqueeze(1)
        g_tgt = KF.spatial_gradient(tgt, order=2).mean(dim=1).abs().sum(dim=1).detach().unsqueeze(1)
        w = (((g_ref + g_tgt)) * 2 + 1) * self.border_mask
        return (w * (1000 * (disp - disp_gt).abs().clamp(min=1e-2).pow(2))).mean()

    def border_suppression(self, img, mask):
        return (img * (1 - mask)).mean()

    # def fusloss(self, ir, vi, fu, weights=[1, 0, 0.5, 0]):
    #     grad_ir = KF.spatial_gradient(ir, order=2).abs().sum(dim=[1, 2])
    #     grad_vi = KF.spatial_gradient(vi, order=2).abs().sum(dim=[1, 2])
    #     grad_fus = KF.spatial_gradient(fu, order=2).abs().sum(dim=[1, 2])
    #     loss_grad = 0.5 * F.l1_loss(grad_fus, grad_ir) + 0.5 * F.l1_loss(grad_fus, grad_vi)
    #     loss_ssim = 0.5 * self.ssim_loss(ir, fu) + 0.5 * self.ssim_loss(vi, fu)
    #     loss_intensity = 0.5 * F.l1_loss(fu, ir) + 0.5 * F.l1_loss(fu, vi)
    #     loss_total = weights[0] * loss_grad + weights[1] * loss_ssim + weights[2] * loss_intensity
    #     return loss_intensity, loss_ssim, loss_grad, loss_total

    # def fusloss_forRF(self, ir, vi, fu, weights=[0.6, 0.3, 0.1], mask=1):
    #     mask_ = (torch.logical_and(ir > 0, vi > 0) * mask).detach()
    #     if (fu > 2.0 / 255).sum() < 100:
    #         mask_ = 1
    #     ir = ir.detach()
    #     vi = vi.detach()
    #     fu = fu
    #     grad_ir = KF.spatial_gradient(ir, order=2).abs().sum(dim=[1, 2])
    #     grad_vi = KF.spatial_gradient(vi, order=2).abs().sum(dim=[1, 2])
    #     grad_fus = KF.spatial_gradient(fu, order=2).abs().sum(dim=[1, 2])
    #     grad_joint = torch.max(grad_ir, grad_vi)
    #     loss_grad = (((grad_joint - grad_fus).abs().clamp(min=1e-9)) * mask_).mean()
    #     loss_ssim = (self.ssim_loss(ir, fu) + self.ssim_loss(vi, fu))
    #     # print(loss_ssim)
    #     intensity_joint = torch.max(vi, ir) * mask_
    #     Loss_intensity = F.l1_loss(fu * mask_, intensity_joint)
    #     return weights[0] * loss_grad + weights[1] * loss_ssim + weights[2] * Loss_intensity

    # def Seg_loss(self, fused_image, label, seg_model):
    #     '''
    #     利用预训练好的分割网络,计算在融合结果上的分割结果与真实标签之间的语义损失
    #     :param fused_image:
    #     :param label:
    #     :param seg_model: 分割模型在主函数中提前加载好,避免每次充分load分割模型
    #     :return seg_loss:
    #     fused_image 在输入Seg_loss函数之前需要由YCbCr色彩空间转换至RGB色彩空间
    #     '''
    #     # 计算语义损失

    #     lb = torch.squeeze(label, 1)
    #     out, mid = seg_model(fused_image)
    #     out = F.softmax(out, 1)
    #     mid = F.softmax(mid, 1)
    #     seg_results = torch.argmax(out, dim=1, keepdim=True)
    #     lossp = lovasz_softmax(out, lb)
    #     loss2 = lovasz_softmax(mid, lb)
    #     seg_loss = lossp + 0.25 * loss2
    #     return seg_loss, seg_results

    def backward_RF(self):
        # Similarity loss for deformation
        # loss_reg_img = self.imgloss(self.image_ir_warp,self.image_ir_warp_fake)+self.imgloss(self.image_ir_Reg,self.image_ir)+\
        #     self.imgloss(self.image_vi_warp,self.image_vi_warp_fake)+self.imgloss(self.image_vi_Reg,self.image_vi)
        loss_reg_img = self.imgloss(self.image_ir_warp_RGB, self.image_ir_warp_fake_RGB, self.goodmask) + self.imgloss(
            self.image_ir_Reg_RGB, self.image_ir_RGB, self.goodmask * self.goodmask_inverse) + \
                       self.imgloss(self.image_vi_warp_RGB, self.image_vi_warp_fake_RGB, self.goodmask) + self.imgloss(
            self.image_vi_Reg_RGB, self.image_vi_RGB, self.goodmask * self.goodmask_inverse)
        loss_reg_field = self.weightfiledloss(self.image_ir_warp_RGB, self.image_vi_warp_fake_RGB,
                                              self.deformation_1['vis2ir'], self.disp.permute(0, 3, 1, 2)) + \
                         self.weightfiledloss(self.image_vi_warp_RGB, self.image_ir_warp_fake_RGB,
                                              self.deformation_2['ir2vis'], self.disp.permute(0, 3, 1, 2))
        loss_smooth = smoothloss(self.deformation_1['vis2ir'])+smoothloss(self.deformation_1['ir2vis'])+\
            smoothloss(self.deformation_2['vis2ir'])+smoothloss(self.deformation_2['ir2vis'])
        loss_smooth_down2 = smoothloss(self.down2)
        loss_smooth_down4 = smoothloss(self.down4)
        loss_smooth_down8 = smoothloss(self.down8)
        loss_smooth = loss_smooth_down2 + loss_smooth_down4 + loss_smooth_down8
        loss_border_re = 0.1 * self.border_suppression(self.image_ir_Reg_RGB,
                                                       self.goodmask_inverse) + 0.1 * self.border_suppression(
            self.image_vi_Reg_RGB, self.goodmask_inverse) + \
                         self.border_suppression(self.image_ir_warp_fake_RGB, self.goodmask) + self.border_suppression(
            self.image_vi_warp_fake_RGB, self.goodmask)

        # loss_fus = self.fusloss_forRF(self.image_ir_Reg_Y, self.image_vi_Y, self.image_fusion_2,
        #                               mask=self.goodmask * self.goodmask_inverse) + \
        #            self.fusloss_forRF(self.image_ir_Y, self.image_vi_Reg_Y, self.image_fusion_1,
        #                               mask=self.goodmask * self.goodmask_inverse)

        mask_ = torch.logical_and(self.image_ir_Y > 1e-5, self.image_vi_Y > 1e-5)
        mask_ = torch.logical_and(self.image_ir_Reg_Y > 1e-5, mask_)
        mask_ = torch.logical_and(self.image_vi_Reg_Y > 1e-5, mask_)
        mask_ = mask_ * self.goodmask * self.goodmask_inverse
        # loss_ncc = self.imgloss(self.image_fusion_1, self.image_fusion_2, mask_)
        assert not loss_reg_img is None, 'loss_reg_img is None'
        assert not loss_reg_field is None, 'loss_reg_filed is None'
        assert not loss_smooth is None, 'loss_smooth is None'
        #loss_total = loss_reg_img * 10 + loss_reg_field + loss_smooth + 10 * loss_fus + loss_ncc + loss_border_re
        loss_total = loss_reg_img * 12 + loss_reg_field +   loss_smooth + loss_border_re
        # loss_MF = loss_fus*10
        (loss_total).backward()

        self.loss_reg_img = loss_reg_img
        self.loss_reg_field = loss_reg_field
        #self.loss_fus = loss_fus
        self.loss_smooth = loss_smooth
        #self.loss_ncc = loss_ncc
        self.loss_total = loss_total

    # def backward_FS(self, seg_flag=False, dataset_name=None):
    #     loss_intensity, loss_ssim, loss_grad, loss_fus = self.fusloss(self.image_ir_Y, self.image_vi_Y,
    #                                                                   self.image_fusion)
    #     if dataset_name == 'MSRS':
    #         loss_seg, seg_results = self.Seg_loss(self.fused_image_RGB, self.label, self.seg_model)
    #     else:
    #         loss_seg = loss_fus
    #         seg_results = self.image_ir_Y
    #     # 
    #     if seg_flag:
    #         self.image_display = torch.cat(
    #             (self.image_ir_Y[0:1], self.image_vi_Y[0:1], self.image_fusion[0:1], seg_results[0:1], self.label[0:1]),
    #             dim=0).detach()
    #     else:
    #         self.image_display = torch.cat((self.image_ir_Y[0:1], self.image_vi_Y[0:1], self.image_fusion[0:1]),
    #                                        dim=0).detach()

    #     assert not torch.isnan(loss_fus), 'loss_fus is NaN'
    #     # assert not torch.isnan(loss_ncc), 'loss_ncc is NaN'
    #     if dataset_name == 'MSRS':
    #         loss_total = loss_fus * 10 + 0.0 * loss_seg
    #     else:
    #         loss_total = loss_fus * 10 + 0 * loss_seg
    #     # loss_MF = loss_fus*10
    #     loss_total.backward()

    #     self.loss_intensity = loss_intensity
    #     self.loss_ssim = loss_ssim
    #     self.loss_fus = loss_fus
    #     self.loss_grad = loss_grad
    #     self.loss_seg = loss_seg
    #     self.loss_total = loss_total

    def update_lr(self):
        self.DM_sch.step()
        #self.FN_sch.step()

    #恢复训练
    def resume(self, model_dir, train=True):
        self.resume_flag = True
        checkpoint = torch.load(model_dir)
        # weight
        try:
            self.DM.load_state_dict({k: v for k, v in checkpoint['DM'].items() if k in self.DM.state_dict()})
        except:
            pass
        # try:
        #     self.FN.load_state_dict({k: v for k, v in checkpoint['FN'].items() if k in self.FN.state_dict()})
        # except:
        #     pass
        # optimizer
        if train:
            self.DM_opt.param_groups[0]['initial_lr'] = 0.0001
            # self.FN_opt.param_groups[0]['initial_lr'] = 0.001
        return checkpoint['ep'], checkpoint['total_it']

    def save(self, filename, ep, total_it):
        state = {
            'DM': self.DM.state_dict(),
            #'FN': self.FN.state_dict(),
            'DM_opt': self.DM_opt.state_dict(),
            #'FN_opt': self.FN_opt.state_dict(),
            'ep': ep,
            'total_it': total_it
        }
        torch.save(state, filename)
        return

    def assemble_outputs1(self):
        images_ir = self.normalize_image(self.image_ir_RGB).detach()
        images_vi = self.normalize_image(self.image_vi_RGB).detach()
        images_fusion = self.normalize_image(self.image_fusion).detach()
        row = torch.cat((images_ir[0:1, ::], images_vi[0:1, ::], images_fusion[0:1, ::]), 3)
        return row

    def assemble_outputs(self):
        images_ir = self.normalize_image(self.image_ir_RGB).detach()
        images_vi = self.normalize_image(self.image_vi_RGB).detach()
        images_ir_warp = self.normalize_image(self.image_ir_warp_RGB).detach()
        images_vi_warp = self.normalize_image(self.image_vi_warp_RGB).detach()
        images_ir_Reg = self.normalize_image(self.image_ir_Reg_RGB).detach()
        images_vi_Reg = self.normalize_image(self.image_vi_Reg_RGB).detach()
        # images_fusion_1 = self.normalize_image(self.image_fusion_1).detach()
        # images_fusion_2 = self.normalize_image(self.image_fusion_2).detach()
        # row1 = torch.cat(
        #     (images_ir[0:1, ::], images_ir_warp[0:1, ::], images_ir_Reg[0:1, ::], images_fusion_1[0:1, ::]), 3)
        # row2 = torch.cat(
        #     (images_vi[0:1, ::], images_vi_warp[0:1, ::], images_vi_Reg[0:1, ::], images_fusion_2[0:1, ::]), 3)
        row1 = torch.cat(
            (images_ir[0:1, ::], images_ir_warp[0:1, ::], images_ir_Reg[0:1, ::]), 3)
        row2 = torch.cat(
            (images_vi[0:1, ::], images_vi_warp[0:1, ::], images_vi_Reg[0:1, ::]), 3)
        return torch.cat((row1, row2), 2)

        self.image_display = torch.cat(
            (self.real_A_encoded[0:1].detach().cpu(), self.fake_B_encoded[0:1].detach().cpu(),
             self.fake_B_random[0:1].detach().cpu(), self.fake_AA_encoded[0:1].detach().cpu(),
             self.fake_A_recon[0:1].detach().cpu(), self.real_B_encoded[0:1].detach().cpu(),
             self.fake_A_encoded[0:1].detach().cpu(), self.fake_A_random[0:1].detach().cpu(),
             self.fake_BB_encoded[0:1].detach().cpu(), self.fake_B_recon[0:1].detach().cpu()), dim=0)

    def normalize_image(self, x):
        return x[:, 0:1, :, :]