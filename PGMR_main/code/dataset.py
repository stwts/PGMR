import torch
import torchvision
import os
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import kornia.utils as KU
import cv2
from code.utils import randflow, randrot, randfilp
import torch.nn.functional as F

def imsave(img, filename):
    img = img.squeeze().cpu()
    img = KU.tensor_to_image(img) * 255.
    cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


class RegData_MSRS(torch.utils.data.Dataset):
    def __init__(self, opts):
        super(RegData_MSRS, self).__init__()
        self.vi_folder = os.path.join(opts.dataroot_MSRS, 'vi')
        self.ir_folder = os.path.join(opts.dataroot_MSRS, "ir")
        self.crop = torchvision.transforms.RandomCrop(256)
        
        self.vi_list = sorted(os.listdir(self.vi_folder))
        self.ir_list = sorted(os.listdir(self.ir_folder))
        print("len of vi:",len(self.vi_list))
        print("len of ir:",len(self.ir_list))

    def __getitem__(self, index):
        # gain image path
        vi_path = os.path.join(self.vi_folder, self.vi_list[index])
        ir_path = os.path.join(self.ir_folder, self.ir_list[index])

        assert os.path.basename(vi_path) == os.path.basename(ir_path), f"Mismatch ir:{os.path.basename(ir_path)} vi:{os.path.basename(vi_path)}."

        # read image as type Tensor
        vi = self.imread(path=vi_path, flags=cv2.IMREAD_GRAYSCALE)
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE)

        resize = transforms.Resize([ir.shape[-2], ir.shape[-1]])
        vi = resize(vi)

        vi_ir = torch.cat([vi,ir],dim=1)
        if vi_ir.shape[-1]<=256 or vi_ir.shape[-2]<=256:
            vi_ir=TF.resize(vi_ir,256)
        vi_ir = randfilp(vi_ir)
        vi_ir = randrot(vi_ir)

        flow,disp,_ = randflow(vi_ir,10,0.1)
        vi_ir_warped = F.grid_sample(vi_ir, flow, align_corners=False, mode='bilinear')

        patch = torch.cat([vi_ir,vi_ir_warped,disp.permute(0,3,1,2)], dim=1)
        patch = self.crop(patch)

        vi, ir, vi_warped, ir_warped, disp = torch.split(patch, [3,3,3,3,2], dim=1)
        h,w = vi_ir.shape[2],vi_ir.shape[3]
        scale = (torch.FloatTensor([w,h]).unsqueeze(0).unsqueeze(0)-1)/(self.crop.size[0]*1.0-1)
        disp = disp.permute(0,2,3,1)*scale

        return ir, vi, ir_warped, vi_warped, disp

    def __len__(self):
        return len(self.vi_list)

    @staticmethod
    def imread(path, flags=cv2.IMREAD_GRAYSCALE):
        img = Image.open(path).convert('RGB')
        im_ts = TF.to_tensor(img).unsqueeze(0)
        return im_ts



class RegData_SAR(torch.utils.data.Dataset):
    def __init__(self, opts, crop=lambda x: x):
        super(RegData_SAR, self).__init__()
        self.vi_folder = os.path.join(opts.dataroot_VI_SAR, 'vi')
        self.sar_folder = os.path.join(opts.dataroot_VI_SAR, "sar")
        self.crop = torchvision.transforms.RandomCrop(256)
        
        self.vi_list = sorted(os.listdir(self.vi_folder))
        self.sar_list = sorted(os.listdir(self.sar_folder))
        print("len of vi:",len(self.vi_list))
        print("len of sar:",len(self.sar_list))

    def __getitem__(self, index):
        # gain image path
        vi_path = os.path.join(self.vi_folder, self.vi_list[index])
        sar_path = os.path.join(self.sar_folder, self.sar_list[index])

        assert os.path.basename(vi_path) == os.path.basename(sar_path), f"Mismatch sar:{os.path.basename(sar_path)} vi:{os.path.basename(vi_path)}."

        # read image as type Tensor
        vi = self.imread(path=vi_path, flags=cv2.IMREAD_GRAYSCALE)
        sar = self.imread(path=sar_path, flags=cv2.IMREAD_GRAYSCALE)

        resize = transforms.Resize([sar.shape[-2], sar.shape[-1]])
        vi = resize(vi)

        vi_sar = torch.cat([vi,sar],dim=1)
        if vi_sar.shape[-1]<=256 or vi_sar.shape[-2]<=256:
            vi_sar=TF.resize(vi_sar,256)
        vi_sar = randfilp(vi_sar)
        vi_sar = randrot(vi_sar)

        flow,disp,_ = randflow(vi_sar,10,0.1)
        vi_sar_warped = F.grid_sample(vi_sar, flow, align_corners=False, mode='bilinear')
        patch = torch.cat([vi_sar,vi_sar_warped,disp.permute(0,3,1,2)], dim=1)
        patch = self.crop(patch)

        vi, sar, vi_warped, sar_warped, disp = torch.split(patch, [3,3,3,3,2], dim=1)
        h,w = vi_sar.shape[2],vi_sar.shape[3]
        scale = (torch.FloatTensor([w,h]).unsqueeze(0).unsqueeze(0)-1)/(self.crop.size[0]*1.0-1)
        disp = disp.permute(0,2,3,1)*scale

        return sar, vi, sar_warped, vi_warped, disp

    def __len__(self):
        return len(self.vi_list)


    @staticmethod
    def imread(path, flags=cv2.IMREAD_GRAYSCALE):
        img = Image.open(path).convert('RGB')
        im_ts = TF.to_tensor(img).unsqueeze(0)
        return im_ts


class RegData_PET_MRI(torch.utils.data.Dataset):
    def __init__(self, opts, crop=lambda x: x):
        super(RegData_PET_MRI, self).__init__()
        self.PET_folder = os.path.join(opts.dataroot_PET_MRI, 'PET')
        self.MRI_folder = os.path.join(opts.dataroot_PET_MRI, "MRI")
        self.crop = torchvision.transforms.RandomCrop(256)
        
        self.PET_list = sorted(os.listdir(self.PET_folder))
        self.MRI_list = sorted(os.listdir(self.MRI_folder))
        print("len of PET:",len(self.PET_list))
        print("len of MRI:",len(self.MRI_list))

    def __getitem__(self, index):
        # gain image path
        PET_path = os.path.join(self.PET_folder, self.PET_list[index])
        MRI_path = os.path.join(self.MRI_folder, self.MRI_list[index])

        assert os.path.basename(PET_path) == os.path.basename(MRI_path), f"Mismatch MRI:{os.path.basename(MRI_path)} PET:{os.path.basename(PET_path)}."

        # read image as type Tensor
        PET = self.imread(path=PET_path, flags=cv2.IMREAD_GRAYSCALE)
        MRI = self.imread(path=MRI_path, flags=cv2.IMREAD_GRAYSCALE)

        resize = transforms.Resize([MRI.shape[-2], MRI.shape[-1]])
        PET = resize(PET)

        PET_MRI = torch.cat([PET,MRI],dim=1)
        if PET_MRI.shape[-1]<=256 or PET_MRI.shape[-2]<=256:
            PET_MRI=TF.resize(PET_MRI,256)
        PET_MRI = randfilp(PET_MRI)
        PET_MRI = randrot(PET_MRI)

        flow,disp,_ = randflow(PET_MRI,10,0.1)
        PET_MRI_warped = F.grid_sample(PET_MRI, flow, align_corners=False, mode='bilinear')
        patch = torch.cat([PET_MRI,PET_MRI_warped,disp.permute(0,3,1,2)], dim=1)
        patch = self.crop(patch)

        PET, MRI, PET_warped, MRI_warped, disp = torch.split(patch, [3,3,3,3,2], dim=1)
        h,w = PET_MRI.shape[2],PET_MRI.shape[3]
        scale = (torch.FloatTensor([w,h]).unsqueeze(0).unsqueeze(0)-1)/(self.crop.size[0]*1.0-1)
        disp = disp.permute(0,2,3,1)*scale

        return MRI, PET, MRI_warped, PET_warped, disp

    def __len__(self):
        return len(self.PET_list)


    @staticmethod
    def imread(path, flags=cv2.IMREAD_GRAYSCALE):
        img = Image.open(path).convert('RGB')
        im_ts = TF.to_tensor(img).unsqueeze(0)
        return im_ts



class RegData_SPECT_MRI(torch.utils.data.Dataset):
    def __init__(self, opts, crop=lambda x: x):
        super(RegData_SPECT_MRI, self).__init__()
        self.SPECT_folder = os.path.join(opts.dataroot_SPECT_MRI, 'SPECT')
        self.MRI_folder = os.path.join(opts.dataroot_SPECT_MRI, "MRI")
        self.crop = torchvision.transforms.RandomCrop(256)
        
        self.SPECT_list = sorted(os.listdir(self.SPECT_folder))
        self.MRI_list = sorted(os.listdir(self.MRI_folder))
        print("len of SPECT:",len(self.SPECT_list))
        print("len of MRI:",len(self.MRI_list))

    def __getitem__(self, index):
        # gain image path
        SPECT_path = os.path.join(self.SPECT_folder, self.SPECT_list[index])
        MRI_path = os.path.join(self.MRI_folder, self.MRI_list[index])

        assert os.path.basename(SPECT_path) == os.path.basename(MRI_path), f"Mismatch MRI:{os.path.basename(MRI_path)} SPECT:{os.path.basename(SPECT_path)}."

        # read image as type Tensor
        SPECT = self.imread(path=SPECT_path, flags=cv2.IMREAD_GRAYSCALE)
        MRI = self.imread(path=MRI_path, flags=cv2.IMREAD_GRAYSCALE)

        resize = transforms.Resize([MRI.shape[-2], MRI.shape[-1]])
        SPECT = resize(SPECT)

        SPECT_MRI = torch.cat([SPECT,MRI],dim=1)
        if SPECT_MRI.shape[-1]<=256 or SPECT_MRI.shape[-2]<=256:
            SPECT_MRI=TF.resize(SPECT_MRI,256)
        SPECT_MRI = randfilp(SPECT_MRI)
        SPECT_MRI = randrot(SPECT_MRI)

        flow,disp,_ = randflow(SPECT_MRI,10,0.1)
        SPECT_MRI_warped = F.grid_sample(SPECT_MRI, flow, align_corners=False, mode='bilinear')
        patch = torch.cat([SPECT_MRI,SPECT_MRI_warped,disp.permute(0,3,1,2)], dim=1)
        patch = self.crop(patch)

        SPECT, MRI, SPECT_warped, MRI_warped, disp = torch.split(patch, [3,3,3,3,2], dim=1)
        h,w = SPECT_MRI.shape[2],SPECT_MRI.shape[3]
        scale = (torch.FloatTensor([w,h]).unsqueeze(0).unsqueeze(0)-1)/(self.crop.size[0]*1.0-1)
        disp = disp.permute(0,2,3,1)*scale

        return MRI, SPECT, MRI_warped, SPECT_warped, disp

    def __len__(self):
        return len(self.SPECT_list)


    @staticmethod
    def imread(path, flags=cv2.IMREAD_GRAYSCALE):
        img = Image.open(path).convert('RGB')
        im_ts = TF.to_tensor(img).unsqueeze(0)
        return im_ts
    





class TestData(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, opts, test_path):
        super(TestData, self).__init__()
        self.target_folder = os.path.join(test_path, 'target')
        self.orign_folder = os.path.join(test_path, 'orign')
        self.crop = torchvision.transforms.RandomCrop(256)

        self.target_list = sorted(os.listdir(self.target_folder))
        self.orign_list = sorted(os.listdir(self.orign_folder))
        print("len of target:",len(self.target_list))
        print("len of orign:", len(self.orign_list))

    def __getitem__(self, index):
        # gain image path
        image_name = self.orign_list[index]
        targte_path = os.path.join(self.target_folder, image_name)
        orign_path = os.path.join(self.orign_folder, image_name)
        # read image as type Tensor
        target = self.imread(path=targte_path)
        orign = self.imread(path=orign_path)

        target_orign = torch.cat([target,orign],dim=1)
        if target_orign.shape[-1]<=256 or target_orign.shape[-2]<=256:
            target_orign=TF.resize(target_orign,256)
        target_orign = TF.resize(target_orign,(256,256))
        target, orign = torch.split(target_orign, [3,3], dim=1)

        flow,disp,_ = randflow(target_orign,10,0.1)
        target_orign_warped = F.grid_sample(target_orign, flow, align_corners=False, mode='bilinear')
        patch = torch.cat([target_orign,target_orign_warped,disp.permute(0,3,1,2)], dim=1)

        target, orign, target_warped, orign_warped, disp = torch.split(patch, [3,3,3,3,2], dim=1)

        h,w = target_orign.shape[2],target_orign.shape[3]
        scale = (torch.FloatTensor([w,h]).unsqueeze(0).unsqueeze(0)-1)/(self.crop.size[0]*1.0-1)
        disp = disp.permute(0,2,3,1)*scale


        return orign_warped, target, orign, image_name, disp

    def __len__(self):
        return len(self.target_list)

    @staticmethod
    def imread(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        im_ts = KU.image_to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.).float()
        im_ts = im_ts.unsqueeze(0)
        return im_ts