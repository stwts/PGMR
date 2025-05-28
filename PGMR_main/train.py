import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from options import MyOptions
from code.dataset import RegData_MSRS, RegData_SAR, RegData_PET_MRI, RegData_SPECT_MRI
from torch.utils.data import DataLoader
from code.model import PGMR
from code.utils import Saver
from enhance.swinir_arch import Resnet
import warnings
 
warnings.filterwarnings("ignore")

def main_RF(opts):
    # data loader
    print('\n--- load dataset ---')
    dataset_MSRS = RegData_MSRS(opts)
    dataset_SAR = RegData_SAR(opts)
    dataset_PET_MRI = RegData_PET_MRI(opts)
    dataset_SPECT_MRI = RegData_SPECT_MRI(opts)

    train_loader_dataset_MSRS = torch.utils.data.DataLoader(
        dataset_MSRS, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads)
    train_loader_dataset_SAR = torch.utils.data.DataLoader(
        dataset_SAR, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads)
    train_loader_dataset_PET_MRI = torch.utils.data.DataLoader(
        dataset_PET_MRI, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads)
    train_loader_dataset_SPECT_MRI = torch.utils.data.DataLoader(
        dataset_SPECT_MRI, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads)

    # model
    print('\n--- load model ---')
    model = PGMR(opts)
    model.setgpu(opts.gpu)
    if opts.resume == None:
        ep0 = -1
        total_it = 0
    else:
        ep0, total_it = model.resume(opts.resume)
        ep0 = -1
        total_it = 0
    model.set_scheduler(opts, last_ep=ep0)
    ep0 += 1
    print('start the training at epoch %d' % (ep0))


    enhancer = Resnet()
    checkpoint = torch.load(opts.resume_enhancer)
    enhancer.load_state_dict(checkpoint["params"])
    
    enhancer = enhancer.cuda()
    enhancer.eval()

    # saver for display and output
    saver = Saver(opts)

    # train
    print('\n--- train ---')
    max_it = 500000
    for ep in range(ep0, opts.n_ep):
        #train of PET_MRI##########################################################################
        print("train of PET_MRI")
        for it, (image_ir, image_vi, image_ir_warp, image_vi_warp, deformation) in enumerate(train_loader_dataset_PET_MRI):
         # input data
            image_ir = image_ir.cuda(opts.gpu).detach()
            image_vi = image_vi.cuda(opts.gpu).detach()
            image_ir_warp = image_ir_warp.cuda(opts.gpu).detach()
            image_vi_warp = image_vi_warp.cuda(opts.gpu).detach()
            deformation = deformation.cuda(opts.gpu).detach()
            if len(image_ir.shape) > 4:
                image_ir = image_ir.squeeze(1)
                image_vi = image_vi.squeeze(1)
                image_ir_warp = image_ir_warp.squeeze(1)
                image_vi_warp = image_vi_warp.squeeze(1)
                deformation = deformation.squeeze(1)
         # update model
            model.update_RF(image_ir, image_vi, image_ir_warp,
                            image_vi_warp, deformation, orign = 'MRI', target = "PET")

        # save to display file
            if not opts.no_display_img:
                saver.write_display(total_it, model)

            if (total_it + 1) % 10 == 0:
                Reg_Img_loss = model.loss_reg_img
                Reg_Field_loss = model.loss_reg_field
                loss_smooth = model.loss_smooth
                Total_loss = model.loss_total
                print('total_it: %d (ep %d, it %d), lr %08f , Total Loss: %04f' % (
                    total_it, ep, it, model.RM_opt.param_groups[0]['lr'], Total_loss))
                print('Reg_Img_loss: {:.4}, Reg_Field_loss: {:.4}, loss_smooth: {:.4}'.format(Reg_Img_loss, Reg_Field_loss, loss_smooth))
            total_it += 1
            if total_it >= max_it:
                saver.write_img(total_it, model)
                saver.write_model(total_it, model)
                break
            
        # save result image
        saver.write_img(ep, model, "PET_MRI")
        #########################################################################################


        #train of VI-SAR##########################################################################
        print("train of VI-SAR")
        for it, (image_sar, image_vi, image_sar_warp, image_vi_warp, deformation) in enumerate(train_loader_dataset_SAR):
        # input data
            image_sar = image_sar.cuda(opts.gpu).detach()
            image_vi = image_vi.cuda(opts.gpu).detach()
            image_sar_warp = image_sar_warp.cuda(opts.gpu).detach()
            image_vi_warp = image_vi_warp.cuda(opts.gpu).detach()
            deformation = deformation.cuda(opts.gpu).detach()
            if len(image_sar.shape) > 4:
                image_sar = image_sar.squeeze(1)
                image_vi = image_vi.squeeze(1)
                image_sar_warp = image_sar_warp.squeeze(1)
                image_vi_warp = image_vi_warp.squeeze(1)
                deformation = deformation.squeeze(1)
        # update model
            model.update_RF(image_sar, image_vi, image_sar_warp,
                            image_vi_warp, deformation, orign = 'sar', target = "vi")

        # save to display file
            if not opts.no_display_img:
                saver.write_display(total_it, model)

            if (total_it + 1) % 10 == 0:
                Reg_Img_loss = model.loss_reg_img
                Reg_Field_loss = model.loss_reg_field
                loss_smooth = model.loss_smooth
                Total_loss = model.loss_total
                print('total_it: %d (ep %d, it %d), lr %08f , Total Loss: %04f' % (
                    total_it, ep, it, model.RM_opt.param_groups[0]['lr'], Total_loss))
                print('Reg_Img_loss: {:.4}, Reg_Field_loss: {:.4}, loss_smooth: {:.4}'.format(Reg_Img_loss, Reg_Field_loss, loss_smooth))
            total_it += 1
            if total_it >= max_it:
                saver.write_img(total_it, model)
                saver.write_model(total_it, model)
                break
                
        # save result image
        saver.write_img(ep, model, "VI-SAR")
        ###################################################################################


        #train of SPECT_MRI##########################################################################
        print("train of SPECT_MRI")
        for it, (image_ir, image_vi, image_ir_warp, image_vi_warp, deformation) in enumerate(train_loader_dataset_SPECT_MRI):
         # input data
            image_ir = image_ir.cuda(opts.gpu).detach()
            image_vi = image_vi.cuda(opts.gpu).detach()
            image_ir_warp = image_ir_warp.cuda(opts.gpu).detach()
            image_vi_warp = image_vi_warp.cuda(opts.gpu).detach()
            deformation = deformation.cuda(opts.gpu).detach()
            if len(image_ir.shape) > 4:
                image_ir = image_ir.squeeze(1)
                image_vi = image_vi.squeeze(1)
                image_ir_warp = image_ir_warp.squeeze(1)
                image_vi_warp = image_vi_warp.squeeze(1)
                deformation = deformation.squeeze(1)
         # update model
            model.update_RF(image_ir, image_vi, image_ir_warp,
                            image_vi_warp, deformation, orign = 'MRI', target = "SPECT")

        # save to display file
            if not opts.no_display_img:
                saver.write_display(total_it, model)

            if (total_it + 1) % 10 == 0:
                Reg_Img_loss = model.loss_reg_img
                Reg_Field_loss = model.loss_reg_field
                loss_smooth = model.loss_smooth
                Total_loss = model.loss_total
                print('total_it: %d (ep %d, it %d), lr %08f , Total Loss: %04f' % (
                    total_it, ep, it, model.RM_opt.param_groups[0]['lr'], Total_loss))
                print('Reg_Img_loss: {:.4}, Reg_Field_loss: {:.4}, loss_smooth: {:.4}'.format(Reg_Img_loss, Reg_Field_loss, loss_smooth))
            total_it += 1
            if total_it >= max_it:
                saver.write_img(total_it, model)
                saver.write_model(total_it, model)
                break
            
        # save result image
        saver.write_img(ep, model, "SPECT_MRI")
        #########################################################################################


        #train of MSRS##########################################################################
        print("train of MSRS")
        for it, (image_ir, image_vi, image_ir_warp, image_vi_warp, deformation) in enumerate(train_loader_dataset_MSRS):
        # input data
            image_ir = image_ir.cuda(opts.gpu).detach()
            image_vi = image_vi.cuda(opts.gpu).detach()
            image_ir_warp = image_ir_warp.cuda(opts.gpu).detach()
            image_vi_warp = image_vi_warp.cuda(opts.gpu).detach()
            deformation = deformation.cuda(opts.gpu).detach()
            if len(image_ir.shape) > 4:
                image_ir = image_ir.squeeze(1)
                image_vi = image_vi.squeeze(1)
                image_ir_warp = image_ir_warp.squeeze(1)
                image_vi_warp = image_vi_warp.squeeze(1)
                deformation = deformation.squeeze(1)
            # update model
            model.update_RF(image_ir, image_vi, image_ir_warp,
                                image_vi_warp, deformation, orign = 'ir', target = "vi")

            # save to display file
            if not opts.no_display_img:
                saver.write_display(total_it, model)

            if (total_it + 1) % 10 == 0:
                Reg_Img_loss = model.loss_reg_img
                Reg_Field_loss = model.loss_reg_field
                loss_smooth = model.loss_smooth
                Total_loss = model.loss_total
                print('total_it: %d (ep %d, it %d), lr %08f , Total Loss: %04f' % (
                    total_it, ep, it, model.RM_opt.param_groups[0]['lr'], Total_loss))
                print('Reg_Img_loss: {:.4}, Reg_Field_loss: {:.4}, loss_smooth: {:.4}'.format(Reg_Img_loss, Reg_Field_loss, loss_smooth))
            total_it += 1
            if total_it >= max_it:
                saver.write_img(total_it, model)
                saver.write_model(total_it, model)
                break
                
        # save result image
        saver.write_img(ep, model, "MSRS")
        ###################################################################################
        

        print(ep)
        # decay learning rate
        if opts.n_ep_decay > -1:
            model.update_lr()

        if (ep+1) % opts.model_save_freq == 0:
            saver.write_model(ep, opts.n_ep, model)

        if ep >= opts.n_ep:
            saver.write_model(ep, opts.n_ep, model)

    return

if __name__ == "__main__":
    parser = MyOptions()
    opts = parser.parse()
    main_RF(opts)