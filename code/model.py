import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.utils as KU
from code.modules import U_net, gaussian_weights_init, get_scheduler
import kornia.filters as KF
from code.utils import RGB2YCrCb
from code.loss import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


##---------- Prompt Gen Module -----------------------
class PromptGenBlock(nn.Module):
    def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192):
        super(PromptGenBlock,self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        self.linear_layer = nn.Linear(lin_dim,prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)
        
    def forward(self,x):
        B,C,H,W = x.shape
        emb = x.mean(dim=(-2,-1))
        prompt_weights = F.softmax(self.linear_layer(emb),dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        prompt = torch.sum(prompt,dim=1)
        prompt = F.interpolate(prompt,(H,W),mode="bilinear")
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
        return disp/self.scale



class regis_model(nn.Module):
    def __init__(self,modal_orign,modal_target,unshare_depth=4,matcher_depth=4,num_pyramids=2):
        super(regis_model,self).__init__()
        self.num_pyramids=num_pyramids
       
        self.feature_extractor_unshare1 = Feature_extractor_unshare(depth=unshare_depth,base_ic=3,base_oc=8,base_dilation=1,norm=nn.InstanceNorm2d)
        self.feature_extractor_unshare2 = Feature_extractor_unshare(depth=unshare_depth,base_ic=3,base_oc=8,base_dilation=1,norm=nn.InstanceNorm2d)
        
        base_ic = self.feature_extractor_unshare1.ic
        base_oc = self.feature_extractor_unshare1.oc
        base_dilation = self.feature_extractor_unshare1.dilation
        
        self.feature_extractor_share1 = nn.Sequential(Conv2d(base_oc,base_oc*2,kernel_size=3,stride=1,padding=1,dilation=1, norm=nn.InstanceNorm2d),
        Conv2d(base_oc*2,base_oc*2,kernel_size=3,stride=2,padding=1,dilation=1, norm=nn.InstanceNorm2d))
        self.feature_extractor_share2 = nn.Sequential(Conv2d(base_oc*2,base_oc*4,kernel_size=3,stride=1,padding=2,dilation=2, norm=nn.InstanceNorm2d),
        Conv2d(base_oc*4,base_oc*4,kernel_size=3,stride=2,padding=2,dilation=2, norm=nn.InstanceNorm2d))
        self.feature_extractor_share3 = nn.Sequential(Conv2d(base_oc*4,base_oc*8,kernel_size=3,stride=1,padding=4,dilation=4, norm=nn.InstanceNorm2d),
        Conv2d(base_oc*8,base_oc*8,kernel_size=3,stride=2,padding=4,dilation=4, norm=nn.InstanceNorm2d))
        
        self.matcher1 = DispEstimator(base_oc*4,matcher_depth,dilation=4)
        self.matcher2 = DispEstimator(base_oc*8,matcher_depth,dilation=2)
        self.refiner = DispRefiner(base_oc*2,1)
        self.grid_down = KU.create_meshgrid(64,64).cuda()
        self.grid_full = KU.create_meshgrid(128,128).cuda()
        self.scale = torch.FloatTensor([256,256]).cuda().unsqueeze(-1).unsqueeze(-1).unsqueeze(0)-1

        ##prompt#####################################################################################
        self.modal_orign = modal_orign
        self.modal_target = modal_target
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
    
        self.feat_prompt_fusion3 = U_net(base_oc*8*2,base_oc*8,bilinear=False,base_channel=128).to(device)
        self.feat_prompt_fusion2 = U_net(base_oc*4*2,base_oc*4,bilinear=False,base_channel=128).to(device)
        self.feat_prompt_fusion1 = U_net(base_oc*2*2,base_oc*2,bilinear=False,base_channel=128).to(device)
        ######################################################################################################

    def match(self,feat11,feat12,feat21,feat22,feat31,feat32):
        #compute scale (w,h)
        if self.scale[0,1,0,0]*2 != feat11.shape[2]-1 or self.scale[0,0,0,0]*2 != feat11.shape[3]-1:
            self.h,self.w = feat11.shape[2],feat11.shape[3]
            self.scale = torch.FloatTensor([self.w,self.h]).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)-1 
            self.scale = self.scale.to(feat11.device)

        #prompt#######################
        if self.modal_orign == 'ir':
            feat31_Prompt = self.feat31_PGB(feat31)
            feat32_Prompt = self.feat32_PGB(feat32)
            #print('ir-vi')

        elif self.modal_orign == 'sar':
            feat31_Prompt = self.feat31_PGB_SAR(feat31)
            feat32_Prompt = self.feat32_PGB(feat32)
            #print('sar-vi')

        elif self.modal_orign == 'vi':
            feat31_Prompt = self.feat32_PGB(feat31)
            if self.modal_target == 'sar':
                feat32_Prompt = self.feat31_PGB_SAR(feat32)
                #print('vi-sar')
            if self.modal_target == 'ir':
                feat32_Prompt = self.feat31_PGB(feat32)
                #print('vi-ir')

        elif self.modal_orign == 'MRI':
            feat31_Prompt = self.feat31_PGB_SPECT_MRI(feat31)
            if self.modal_target == 'SPECT':
                feat32_Prompt = self.feat32_PGB_SPECT_MRI(feat32)
                #print('MRI-SPECT')
            if self.modal_target == 'PET':
                feat32_Prompt = self.feat32_PGB_PET(feat32)
                #print('MRI-PET')
        
        elif self.modal_orign == 'SPECT':
            feat31_Prompt = self.feat32_PGB_SPECT_MRI(feat31)
            feat32_Prompt = self.feat31_PGB_SPECT_MRI(feat32)
            #print('SPECT-MRI')

        elif self.modal_orign == 'PET':
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

        #prompt#######################
        if self.modal_orign == 'ir':
            feat21_Prompt = self.feat21_PGB(feat21)
            feat22_Prompt = self.feat22_PGB(feat22)
            #print('ir-vi')

        elif self.modal_orign == 'sar':
            feat21_Prompt = self.feat21_PGB_SAR(feat21)
            feat22_Prompt = self.feat22_PGB(feat22)
            #print('sar-vi')

        elif self.modal_orign == 'vi':
            feat21_Prompt = self.feat22_PGB(feat21)
            if self.modal_target == 'sar':
                feat22_Prompt = self.feat21_PGB_SAR(feat22)
                #print('vi-sar')
            if self.modal_target == 'ir':
                feat22_Prompt = self.feat21_PGB(feat22)
                #print('vi-ir')

        elif self.modal_orign == 'MRI':
            feat21_Prompt = self.feat21_PGB_SPECT_MRI(feat21)
            if self.modal_target == 'SPECT':
                feat22_Prompt = self.feat22_PGB_SPECT_MRI(feat22)
                #print('MRI-SPECT')
            if self.modal_target == 'PET':
                feat22_Prompt = self.feat22_PGB_PET(feat22)
                #print('MRI-PET')
        
        elif self.modal_orign == 'SPECT':
            feat21_Prompt = self.feat22_PGB_SPECT_MRI(feat21)
            feat22_Prompt = self.feat21_PGB_SPECT_MRI(feat22)
            #print('SPECT-MRI')

        elif self.modal_orign == 'PET':
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

        #prompt#######################
        if self.modal_orign == 'ir':
            feat11_Prompt = self.feat11_PGB(feat11)
            feat12_Prompt = self.feat12_PGB(feat12)
            #print('ir-vi')

        elif self.modal_orign == 'sar':
            feat11_Prompt = self.feat11_PGB_SAR(feat11)
            feat12_Prompt = self.feat12_PGB(feat12)
            #print('sar-vi')

        elif self.modal_orign == 'vi':
            feat11_Prompt = self.feat12_PGB(feat11)
            if self.modal_target == 'sar':
                feat12_Prompt = self.feat11_PGB_SAR(feat12)
                #print('vi-sar')
            if self.modal_target == 'ir':
                feat12_Prompt = self.feat11_PGB(feat12)
                #print('vi-ir')

        elif self.modal_orign == 'MRI':
            feat11_Prompt = self.feat11_PGB_SPECT_MRI(feat11)
            if self.modal_target == 'SPECT':
                feat12_Prompt = self.feat12_PGB_SPECT_MRI(feat12)
                #print('MRI-SPECT')
            if self.modal_target == 'PET':
                feat12_Prompt = self.feat12_PGB_PET(feat12)
                #print('MRI-PET')
        
        elif self.modal_orign == 'SPECT':
            feat11_Prompt = self.feat12_PGB_SPECT_MRI(feat11)
            feat12_Prompt = self.feat11_PGB_SPECT_MRI(feat12)
            #print('SPECT-MRI')

        elif self.modal_orign == 'PET':
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
            
            a = self.modal_orign
            self.modal_orign = self.modal_target
            self.modal_target = a
            disp_21,disp_21_down4,disp_21_down8 = self.match(feat12,feat11,feat22,feat21,feat32,feat31)
            
            a = self.modal_orign
            self.modal_orign = self.modal_target
            self.modal_target = a
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
        self.modal_orign = orign
        self.modal_target = target



class PGMR(nn.Module):
    def __init__(self, opts=None):
        super(PGMR, self).__init__()

        # parameters
        lr = 0.0001
        # encoders
        self.modal_orign = opts.modal_orign
        self.modal_target = opts.modal_target
        self.RM = regis_model(modal_orign=self.modal_orign, modal_target=self.modal_target)
        self.resume_flag = False
        self.ST = SpatialTransformer(256, 256, True)

        # optimizers
        self.RM_opt = torch.optim.Adam(
            self.RM.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.00001)
        self.gradientloss = gradientloss()

        self.deformation_1 = {}
        self.deformation_2 = {}
        self.border_mask = torch.zeros([1, 1, 256, 256])
        self.border_mask[:, :, 10:-10, 10:-10] = 1
        self.AP = nn.AvgPool2d(5, stride=1, padding=2)
        self.initialize()


    def initialize(self):
        self.RM.apply(gaussian_weights_init)
        

    def set_scheduler(self, opts, last_ep=0):
        self.RM_sch = get_scheduler(self.RM_opt, opts, last_ep)
       

    def setgpu(self, gpu):
        self.gpu = gpu
        self.RM.cuda(self.gpu)
        

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
        disp = self.RM(ir, vi)['ir2vis']
        ir_reg = self.ST(ir, disp)
        return ir_reg


    def registration_forward(self, ir, vi):
        disp = self.RM(ir, vi)['ir2vis']
        ir_reg = self.ST(ir, disp)
        return ir_reg, disp


    def train_forward_RF(self):
        #batch_size
        b = self.image_ir_warp_RGB.shape[0]

        ir_stack = torch.cat([self.image_ir_warp_RGB, self.image_ir_RGB])
        vi_stack = torch.cat([self.image_vi_RGB, self.image_vi_warp_RGB])
        
        deformation = self.RM(ir_stack, vi_stack, type='bi')

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

        self.generate_mask()
        self.image_display = torch.cat((self.image_ir_RGB[0:1, 0:1], self.image_ir_warp_RGB[0:1, 0:1],
                                        self.image_ir_Reg_RGB[0:1, 0:1],
                                        (self.image_vi_RGB - self.image_vi_warp_RGB)[0:1].abs().mean(dim=1,
                                                                                                     keepdim=True),
                                        self.image_vi_Y[0:1], RGB2YCrCb(self.image_vi_warp_RGB[0:1])[0],
                                        self.image_vi_Reg_Y[0:1],
                                        (self.image_vi_RGB - self.image_vi_Reg_RGB)[0:1].abs().mean(dim=1,
                                                                                                    keepdim=True)), dim=0).detach()


    def update_RF(self, image_ir, image_vi, image_ir_warp, image_vi_warp, disp, orign, target):
        self.RM.change(orign, target)
        self.image_ir_RGB = image_ir
        self.image_vi_RGB = image_vi
        self.image_ir_warp_RGB = image_ir_warp
        self.image_vi_warp_RGB = image_vi_warp
        self.disp = disp
        self.RM_opt.zero_grad()
        self.train_forward_RF()
        self.backward_RF()
        nn.utils.clip_grad_norm_(self.RM.parameters(), 5)
        self.RM_opt.step()


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


    def backward_RF(self):
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


        mask_ = torch.logical_and(self.image_ir_Y > 1e-5, self.image_vi_Y > 1e-5)
        mask_ = torch.logical_and(self.image_ir_Reg_Y > 1e-5, mask_)
        mask_ = torch.logical_and(self.image_vi_Reg_Y > 1e-5, mask_)
        mask_ = mask_ * self.goodmask * self.goodmask_inverse

        assert not loss_reg_img is None, 'loss_reg_img is None'
        assert not loss_reg_field is None, 'loss_reg_filed is None'
        assert not loss_smooth is None, 'loss_smooth is None'
        
        loss_total = loss_reg_img * 12 + loss_reg_field +   loss_smooth + loss_border_re
        
        (loss_total).backward()

        self.loss_reg_img = loss_reg_img
        self.loss_reg_field = loss_reg_field
        self.loss_smooth = loss_smooth
        self.loss_total = loss_total


    def update_lr(self):
        self.RM_sch.step()
        

    def resume(self, model_dir, train=True):
        self.resume_flag = True
        checkpoint = torch.load(model_dir)
        # weight
        try:
            self.RM.load_state_dict({k: v for k, v in checkpoint['RM'].items() if k in self.RM.state_dict()})
        except:
            pass
        # optimizer
        if train:
            self.RM_opt.param_groups[0]['initial_lr'] = 0.0001
        return checkpoint['ep'], checkpoint['total_it']

    def save(self, filename, ep, total_it):
        state = {
            'RM': self.RM.state_dict(),
            'RM_opt': self.RM_opt.state_dict(),
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
        row1 = torch.cat(
            (images_ir[0:1, ::], images_ir_warp[0:1, ::], images_ir_Reg[0:1, ::]), 3)
        row2 = torch.cat(
            (images_vi[0:1, ::], images_vi_warp[0:1, ::], images_vi_Reg[0:1, ::]), 3)
        return torch.cat((row1, row2), 2)


    def normalize_image(self, x):
        return x[:, 0:1, :, :]