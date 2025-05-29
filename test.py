import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
import torchvision.transforms.functional as TF
import cv2
from code.dataset import TestData
from time import time
from options import MyOptions
from code.model import PGMR
import warnings
from enhance.swinir_arch import Resnet
import time

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    parser = MyOptions()
    opts = parser.parse()

    # data loader
    dataset_test = TestData(opts, test_path = opts.dataroot_test)
    p_bar_test = enumerate(dataset_test)

    # model
    model = PGMR(opts)
    model.resume(opts.resume)
    model = model.cuda()
    model.eval()

    enhancer = Resnet()
    checkpoint = torch.load(opts.resume_enhancer)
    enhancer.load_state_dict(checkpoint["params"])
    enhancer = enhancer.cuda()
    enhancer.eval()


    for idx, [orign_warped, target, orign, image_name, disp] in p_bar_test:
        img_folder = os.path.join(opts.test_dir, str(idx))
        orign_path = os.path.join(img_folder, "orign.png")
        target_path = os.path.join(img_folder, "target.png")
        orign_warped_path = os.path.join(img_folder, "orign_warped.png")
        orign_reg_path = os.path.join(img_folder, "orign_reg.png")
        bool=os.path.exists(img_folder)
        if bool:
            pass
        else:
            os.makedirs(img_folder)

        target_tensor = target.cuda().detach()
        orign_tenor = orign.cuda().detach()
        orign_warped_tensor = orign_warped.cuda().detach()

        with torch.no_grad():
            #start = time.time()
            results, disp_predict = model.registration_forward(orign_warped_tensor, target_tensor)
            results_up = enhancer(results)
            #time_used = time.time()-start

        results_np = results.cpu().numpy().squeeze(axis=0).transpose(1,2,0)*255
        results_np = cv2.cvtColor(results_np, cv2.COLOR_RGB2BGR)
        
        results_up_np = results_up.cpu().numpy().squeeze(axis=0).transpose(1,2,0)*255
        results_up_np = cv2.cvtColor(results_up_np, cv2.COLOR_RGB2BGR)

        target_np = target_tensor.cpu().numpy().squeeze(axis=0).transpose(1,2,0)*255.
        target_np = cv2.cvtColor(target_np, cv2.COLOR_RGB2BGR)

        orign_np = orign_tenor.cpu().numpy().squeeze(axis=0).transpose(1,2,0)*255.
        orign_np = cv2.cvtColor(orign_np, cv2.COLOR_RGB2BGR)

        orign_warped_np = orign_warped_tensor.cpu().numpy().squeeze(axis=0).transpose(1,2,0)*255.
        orign_warped_np = cv2.cvtColor(orign_warped_np, cv2.COLOR_RGB2BGR)

        #data save
        cv2.imwrite(target_path,target_np)
        cv2.imwrite(orign_warped_path,orign_warped_np)
        cv2.imwrite(orign_reg_path,results_up_np)
        cv2.imwrite(orign_path,orign_np)



        

        
        

        

        

        

        


        



    

    
        