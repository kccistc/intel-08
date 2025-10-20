from __future__ import print_function
import torch
from dcp_gan import DCP_GAN
from discriminator import Discriminator

import os
from math import log10
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import  torchvision.transforms  as tf
from dataset import ITSDataset
from PIL import Image
from torch.optim.lr_scheduler import MultiStepLR
import scipy.misc as misc
from torchvision import utils as uvtils
use_cuda = torch.cuda.is_available()
import cv2
from SSIM import *
from util import *

initial_rate = 0.0002
save_path = os.path.join("checkpoint", 'in_outdoor')
def weights_init(m):
    # if isinstance(m, MergeParametric):
    #     print('initializing merge layer ...')
    #     pass
    if isinstance(m, nn.Conv2d):
        print(m)
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.ConvTranspose2d):
        print(m)
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def train():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)

    train_data = ITSDataset('/home/ryu/workspace/3rd_project/DCP_GAN_dehazing/data/hazy','/home/ryu/workspace/3rd_project/DCP_GAN_dehazing/data/clear')
    test_data = ITSDataset('/home/ryu/workspace/3rd_project/DCP_GAN_dehazing/data/test/hazy','/home/ryu/workspace/3rd_project/DCP_GAN_dehazing/data/test/clear',istrain=False)
    training_data_loader = DataLoader(dataset=train_data, num_workers=4, batch_size=1,
                                      shuffle=True)
    testing_data_loader = DataLoader(dataset=test_data, num_workers=4, batch_size=1,
                                     shuffle=False)
    device = torch.device("cuda:0" if use_cuda else "cpu")

    net_G = DCP_GAN(3,3).to(device)
    net_D1 = Discriminator(in_channel=6).to(device)
    net_D2 = Discriminator(in_channel=6).to(device)
    optimizer_G = optim.Adam(net_G.parameters(),
                             lr=initial_rate,betas=(0.5, 0.99))
    optimizer_D1 = optim.Adam(net_D1.parameters(),
                             lr=initial_rate,betas=(0.5, 0.99))
    optimizer_D2 = optim.Adam(net_D2.parameters(),
                             lr=initial_rate,betas=(0.5, 0.99))
    criterionL1 = nn.L1Loss().to(device)
    criterion = SSIM().to(device)
    criterionGAN = GANLoss('vanilla').to(device)

    best_psnr = 0
    for epoch in range(10):
        # lr = adjust_learning_rate(optimizer, epoch - 1)
        for iteration, batch in enumerate(training_data_loader, 1):
            real_a, real_b,real_a_crop,real_b_crop,size_c= batch[0].to(device), batch[1].to(device),batch[2].to(device), \
                                                           batch[3].to(device),batch[4]
            fake_b = net_G(real_a)

            set_requires_grad(net_D1,True)
            set_requires_grad(net_D2, True)
            optimizer_D1.zero_grad()
            optimizer_D2.zero_grad()
            #backward_D
            # fake_b_crop = tf.ToPILImage(fake_b[0,:,:,:])

            i, j, h, w = size_c
            #Fake

            fake_AB = torch.cat((real_a, fake_b),1)
            fake_b_crop = fake_b[:,:,i:i+h,j:j+w]
            # fake_b_crop = tf.ToTensor(tf.functional.crop(fake_b_crop, i, j, h, w))

            fake_AB_crop = torch.cat((real_a_crop, fake_b_crop), 1)

            pred_fake = net_D1(fake_AB.detach())
            pred_fake_crop = net_D2(fake_AB_crop.detach())
            loss_D1_fake = criterionGAN(pred_fake, False)
            loss_D2_fake = criterionGAN(pred_fake_crop, False)
            # Real
            real_AB = torch.cat((real_a, real_b), 1)
            real_AB_crop = torch.cat((real_a_crop, real_b_crop), 1)
            pred_real = net_D1(real_AB)
            pred_real_crop = net_D2(real_AB_crop)
            loss_D1_real = criterionGAN(pred_real, True)
            loss_D2_real = criterionGAN(pred_real_crop, True)
            # combine loss and calculate gradients
            loss_D1 = (loss_D1_fake + loss_D1_real) * 0.5
            loss_D2 = (loss_D2_fake + loss_D2_real) * 0.5
            loss_D1.backward()
            loss_D2.backward()
            optimizer_D1.step()
            optimizer_D2.step()
            # backward_G
            set_requires_grad(net_D1, False)
            set_requires_grad(net_D2, False)
            optimizer_G.zero_grad()
            pred_fake = net_D1(fake_AB)
            pred_fake_crop = net_D2(fake_AB_crop)
            loss_G_GAN = criterionGAN(pred_fake, True)+criterionGAN(pred_fake_crop, True)
            loss_d = -criterion(fake_b/2.+0.5,real_b/2.+0.5)
            loss_c = criterionL1(fake_b,real_b)
            loss_G = loss_d*100+100*loss_c+loss_G_GAN
            loss_G.backward()
            optimizer_G.step()

            if iteration % 100 == 0:
                print("===> Epoch[{}]({}/{}): Loss_D1: {:.4f} Loss_D2: {:.4f} Loss_G: {:.4f} Loss_d: {:.4f} Loss_c: {:.4f}".format(
                    epoch, iteration, len(training_data_loader), loss_D1.item(), loss_D2.item(),loss_G.item(),loss_d.item(),loss_c.item()))
            if iteration%500==0:
                out_hazy = (real_a[0]).cpu().detach().permute(1, 2, 0).numpy()
                out_gt = (real_b[0]).cpu().detach().permute(1, 2, 0).numpy()
                fake_out = (fake_b[0]).cpu().detach().permute(1, 2, 0).numpy()

                out_hazy = ((out_hazy+1)/2*255).astype(np.uint8)
                out_gt = ((out_gt+1)/2 *255).astype(np.uint8)
                fake_out = ((fake_out+1)/2 * 255).astype( np.uint8)

                out_hazy = cv2.cvtColor(out_hazy,cv2.COLOR_RGB2BGR)
                out_gt = cv2.cvtColor(out_gt, cv2.COLOR_RGB2BGR)
                fake_out = cv2.cvtColor(fake_out, cv2.COLOR_RGB2BGR)

                cv2.imwrite("D:\pytorch_model\pgcgan\\samples\\{}_hazy.png".format(iteration),out_hazy)
                cv2.imwrite("D:\pytorch_model\pgcgan\\samples\\{}_gt.png".format(iteration), out_gt)
                cv2.imwrite("D:\pytorch_model\pgcgan\\samples\\{}_out.png".format(iteration), fake_out)
                # cv2.imwrite("D:\\pytorch_model\\torch_dehaze\\samples\\{}_out1.png".format(iteration), fake_out1)
                # cv2.imwrite("D:\\pytorch_model\\torch_dehaze\\samples\\{}_out2.png".format(iteration), fake_out2)

            if iteration%6500==0:
                with torch.no_grad():
                    net_G.eval()
                    avg_psnr = 0
                    for batch in testing_data_loader:
                        input, target = batch[0].to(device), batch[1].to(device)
                        prediction = net_G(input)
                        psnr = batch_PSNR(prediction,target)
                        avg_psnr += psnr
                    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
                    if avg_psnr / len(testing_data_loader)>best_psnr:
                        best_psnr = avg_psnr / len(testing_data_loader)
                        if not os.path.exists("checkpoint"):
                            os.mkdir("checkpoint")
                        if not os.path.exists(os.path.join("checkpoint", 'indoor')):
                            os.mkdir(os.path.join("checkpoint", 'indoor'))
                        model_out_path = "checkpoint/{}/netG_model_epoch_{}_{}.pth".format('indoor', epoch,iteration)
                        torch.save(net_G, model_out_path)
                        print("Checkpoint saved to {}".format("checkpoint" + 'indoor'))
        if epoch % 1 == 0:
            if not os.path.exists("checkpoint"):
                os.mkdir("checkpoint")
            if not os.path.exists(os.path.join("checkpoint", 'indoor')):
                os.mkdir(os.path.join("checkpoint", 'indoor'))
            model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format('indoor', epoch)
            torch.save(net_G, model_out_path)

            print("Checkpoint saved to {}".format("checkpoint" + 'indoor'))
if __name__ =='__main__':
    train()
