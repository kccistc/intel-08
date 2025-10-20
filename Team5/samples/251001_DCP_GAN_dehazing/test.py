from __future__ import print_function
import argparse
import os
from math import log10
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from dcp_gan import DCP_GAN

use_cuda = torch.cuda.is_available()
from glob import glob
import time
import cv2
from SSIM import *
from util import *
logdir = '/home/ryu/workspace/3rd_project/DCP_GAN_dehazing/checkpoint/indoor/'
which_model= 'netG_model_epoch_4.pth'
test_hazy_dir = '/home/ryu/workspace/3rd_project/DCP_GAN_dehazing/data/test/hazy/'
# test_hazy_dir = 'D:\Reside\RTTS\RTTS\\temp_jpg\\'
# test_hazy_dir = 'D:\SOTS\outdoor\hazy\\'
save_path = './Dehazed_result/'
# save_path = './outdoor/'
# save_path = './RTTS_500/'
def test():
    device = torch.device("cuda:0" if use_cuda else "cpu")

    net = torch.load(logdir+which_model, weights_only=False)
    if use_cuda:
        net = net.cuda()

    net.eval()


    criterion = SSIM()
    time_test = 0
    count = 0
    png_imgs = glob(os.path.join(test_hazy_dir, '*.png'))
    jpg_imgs = glob(os.path.join(test_hazy_dir, '*.jpg'))
    imgs_hazy = png_imgs + jpg_imgs
    
    for img_name in imgs_hazy:
        hazy_img = cv2.imread(img_name)
        t = hazy_img.shape
        h = t[0]
        w = t[1]
        hazy_img = cv2.resize(hazy_img,(512,512))
        img_name = os.path.basename(img_name)
        b, g, r = cv2.split(hazy_img)
        hazy_img = cv2.merge([r, g, b])
        hazy_img = (np.float32(hazy_img)/255.)*2-1
        hazy_img = np.expand_dims(hazy_img.transpose(2, 0, 1), 0)
        hazy_img = Variable(torch.Tensor(hazy_img))
        if use_cuda:
            hazy_img = hazy_img.cuda()
            # hazy_img.to(device)

        with torch.no_grad():
            if use_cuda:
                torch.cuda.synchronize()
            start_time = time.time()
            fake_b = net(hazy_img)
            # fake_b = torch.clamp(fake_b, -1., 1.)
            if use_cuda:
                torch.cuda.synchronize()
            end_time = time.time()
            dur_time = end_time - start_time
            time_test += dur_time
            if use_cuda:
                save_out = np.uint8(127.5 *( fake_b.data.cpu().numpy().squeeze()+1))
            else:
                save_out = np.uint8(127.5 * (fake_b+1).data.numpy().squeeze())
            save_out = save_out.transpose(1, 2, 0)
            b, g, r = cv2.split(save_out)
            save_out = cv2.merge([r, g, b])
            save_out = cv2.resize(save_out,(w,h))
            cv2.imwrite(save_path+img_name, save_out)
            count += 1

    if count == 0:
        print("No test images found!")
    else:
        print('Avg. time:', time_test / count)

if __name__ =='__main__':
    test()