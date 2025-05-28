import argparse

class MyOptions():
  def __init__(self):
    self.parser = argparse.ArgumentParser()

    # data loader related
    self.parser.add_argument('--dataroot_MSRS', default='/train_dataset/test_MSRS', type=str)
    self.parser.add_argument('--dataroot_PET_MRI', default='/train_dataset/test_PET_MRI', type=str)
    self.parser.add_argument('--dataroot_SPECT_MRI', default='/train_dataset/test_SPECT_MRI', type=str)
    self.parser.add_argument('--dataroot_VI_SAR', default='/train_dataset/test_VI_SAR', type=str)
    self.parser.add_argument('--dataroot_test', default='/test_dataset', type=str)

    self.parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    self.parser.add_argument('--nThreads', type=int, default=1, help='# of threads for data loader')
    self.parser.add_argument('--modal_orign', type=str, default='ir')
    self.parser.add_argument('--modal_target', type=str, default='vi')

    # ouptput related
    self.parser.add_argument('--name', type=str, default='VI_test', help='folder name to save outputs')
    self.parser.add_argument('--display_dir', type=str, default='./logs', help='path for saving display results')
    self.parser.add_argument('--display_freq', type=int, default=50, help='freq (iteration) of display')
    self.parser.add_argument('--img_save_freq', type=int, default=1, help='freq (epoch) of saving images')
    self.parser.add_argument('--model_save_freq', type=int, default=2, help='freq (epoch) of saving models')
    self.parser.add_argument('--result_dir', type=str, default='/results', help='path for saving result images and models')
    self.parser.add_argument('--test_dir', type=str, default='./test_result', help='path for saving result images and models')
    self.parser.add_argument('--no_display_img', action='store_true', help='specified if no dispaly')

    # training related
    self.parser.add_argument('--resume', type=str, default="/checkpoint/regis_all.pth", help='specified the dir of saved models for resume the training')
    self.parser.add_argument('--gpu', type=int, default=0, help='gpu')
    self.parser.add_argument('--lr_policy', type=str, default='lambda', help='type of learn rate decay')
    self.parser.add_argument('--n_ep', type=int, default=100, help='number of epochs') # 400 * d_iter
    self.parser.add_argument('--n_ep_decay', type=int, default=40, help='epoch start decay learning rate, set -1 if no decay')
    self.parser.add_argument('--resume_enhancer', type=str, default="/checkpoint/enhancer.pth", help='specified the dir of saved models for resume the training')

  def parse(self):
    self.opt = self.parser.parse_args()
    args = vars(self.opt)
    print('\n--- load options ---')
    for name, value in sorted(args.items()):
      print('%s: %s' % (str(name), str(value)))
    return self.opt