import torch.utils.data as data
from PIL import Image
from PIL import ImageEnhance,ImageFilter
import torchvision.transforms as transforms
import numpy as np
import random
from glob2 import glob
import torchvision.transforms.functional as TF
import os
class ITSDataset(data.Dataset):
    def __init__(self, data_dir, data_dir_clean, data_trans=None, test_hazy_dir=None, test_gt_dir=None, istrain=True, flip=True):
        super(ITSDataset, self).__init__()
        # self.hazy = hazy_img_path
        # self.clear = clear_path
        # self.unhazy = unhazy_path
        self.scale_size = 286
        self.size = 256
        self.hazy_img_list = []
        self.trans_list = []
        self.clean_img_list = []
        self.isTrain = istrain
        self.Flip = flip
        png_imgs = glob(os.path.join(data_dir, '*.png'))
        jpg_imgs = glob(os.path.join(data_dir, '*.jpg'))
        imgs = png_imgs + jpg_imgs

        # imgs = imgs[:8244]

        for img_name in imgs:
            base_name = os.path.basename(img_name).split('_')[0].strip()
            if not base_name.lower().endswith('.png'):
                base_name += '.png'
            self.hazy_img_list.append(img_name)
            self.clean_img_list.append(os.path.join(data_dir_clean, base_name))


        # if self.isTrain:
        #     for img_name in imgs:
        #         name = os.path.basename(img_name).split('_')[0]
        #         self.hazy_img_list.append(img_name)
        #         # name = clean_data_dir+img.split("_")[0]+'.jpg'
        #         self.clean_img_list.append(os.path.join(data_dir_clean, name + '.png'))
        # else:
        #     imgs = glob(os.path.join(data_dir, '*.png'))
        #     for img_name in imgs:
        #         name = os.path.basename(img_name).split('_')[0]
        #         self.hazy_img_list.append(img_name)
        #         # name = clean_data_dir+img.split("_")[0]+'.jpg'
        #         self.clean_img_list.append(os.path.join(data_dir_clean, name + '.png'))

    def name(self):
        return 'ITSDataset'


    def initialize(self, opt):
        pass

    def augData(self, data, target):

        if self.isTrain:
            rand_hor = random.randint(0, 1)
            data = transforms.RandomHorizontalFlip(rand_hor)(data)
            target = transforms.RandomHorizontalFlip(rand_hor)(target)

        i, j, h, w = transforms.RandomCrop.get_params(data, output_size=(128, 128))
        data_crop = TF.crop(data, i, j, h, w)
        target_crop = TF.crop(target, i, j, h, w)

        data = transforms.ToTensor()(data)
        # data = tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(data)
        target = transforms.ToTensor()(target)
        data_crop = transforms.ToTensor()(data_crop)
        target_crop = transforms.ToTensor()(target_crop)
        data = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(data)
        target = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(target)
        data_crop = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(data_crop)
        target_crop = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(target_crop)
        return data, target,data_crop,target_crop,(i,j,h,w)

    def __getitem__(self, index):
        if self.isTrain:
            hazy_img = Image.open(self.hazy_img_list[index]).convert('RGB')
            clean_img = Image.open(self.clean_img_list[index]).convert('RGB')
            # clean_img = clean_img.resize((512, 512), Image.BICUBIC)

            i, j, h, w = transforms.RandomCrop.get_params(hazy_img, output_size=(256, 256))
            hazy_img = TF.crop(hazy_img, i, j, h, w)
            clean_img = TF.crop(clean_img, i, j, h, w)

            hazy_img, clean_img,hazy_crop,clean_crop,size_c = self.augData(hazy_img, clean_img)

            return hazy_img, clean_img,hazy_crop,clean_crop,size_c

        else:
            hazy_img = Image.open(self.hazy_img_list[index]).convert('RGB')
            clean_img = Image.open(self.clean_img_list[index]).convert('RGB')
            clean_img = clean_img.crop((10,10,630,470))
            hazy_img = hazy_img.resize((512, 512), Image.BICUBIC)
            clean_img = clean_img.resize((512, 512), Image.BICUBIC)

            transform_list = []
            transform_list += [transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5))]
            trans = transforms.Compose(transform_list)

            return trans(hazy_img), trans(clean_img)


    def __len__(self):
        return len(self.hazy_img_list)