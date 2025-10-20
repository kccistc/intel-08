import torch
import torch.nn as nn
import torch.nn.functional as F
from guided_filter_pytorch.guided_filter import GuidedFilter
import numpy as np
class DCP_GAN(nn.Module):
    def __init__(self,input_channels,output_channel,ngf=64,scale_num=3):
        super(DCP_GAN, self).__init__()
        self.in_channel = input_channels
        self.scale = scale_num
        self.out_channel = output_channel
        self.filter_num = ngf
        self.ngf = ngf
        self.dcp_g = DCPDehazeGenerator(win_size=15, r=60)
        #encoder_layers
        self.enc_conv1 = nn.Conv2d(self.in_channel,self.ngf, kernel_size=4, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(self.ngf, self.ngf*2, kernel_size=4, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(self.ngf*2, self.ngf*4, kernel_size=4, stride=2, padding=1)
        self.enc_conv4 = nn.Conv2d(self.ngf*4, self.ngf*8, kernel_size=4, stride=2, padding=1)
        self.enc_conv5 = nn.Conv2d(self.ngf*8, self.ngf*8, kernel_size=4, stride=2, padding=1)
        self.enc_conv6 = nn.Conv2d(self.ngf*8, self.ngf*8, kernel_size=4, stride=2, padding=1)
        self.enc_conv7 = nn.Conv2d(self.ngf*8, self.ngf*8, kernel_size=4, stride=2, padding=1)
        self.enc_conv8 = nn.Conv2d(self.ngf*8, self.ngf*8, kernel_size=4, stride=2, padding=1)
        #decoder_layers

        self.dec_conv8 = nn.ConvTranspose2d(self.ngf*8,self.ngf*8,kernel_size=4,stride=2,padding=1)
        self.agroup_norm8 = AGroupNorm(self.ngf*8,self.ngf*8)
        self.dec_conv7 = nn.ConvTranspose2d(self.ngf*8*2,self.ngf*8,kernel_size=4,stride=2,padding=1)
        self.agroup_norm7 = AGroupNorm(self.ngf*8,self.ngf*8)
        self.agg_7 = AggBlock(self.ngf*8*2,self.ngf*8)
        self.dec_conv6 = nn.ConvTranspose2d(self.ngf * 8 * 2, self.ngf * 8, kernel_size=4, stride=2, padding=1)
        self.agroup_norm6 = AGroupNorm(self.ngf*8,self.ngf*8)
        self.agg_6 = AggBlock(self.ngf *8 * 2, self.ngf*8)
        self.dec_conv5 = nn.ConvTranspose2d(self.ngf * 8 * 2, self.ngf * 8, kernel_size=4, stride=2, padding=1)
        self.agroup_norm5 = AGroupNorm(self.ngf*8,self.ngf*8)
        self.agg_5 = AggBlock(self.ngf * 8*2, self.ngf*8)
        self.dec_conv4 = nn.ConvTranspose2d(self.ngf * 8 * 2, self.ngf * 4, kernel_size=4, stride=2, padding=1)
        self.agroup_norm4 = AGroupNorm(self.ngf*8,self.ngf*4)
        self.agg_4 = AggBlock(self.ngf * 8*2, self.ngf*8)
        self.dec_conv3 = nn.ConvTranspose2d(self.ngf * 4 * 2, self.ngf *2, kernel_size=4, stride=2, padding=1)
        self.agroup_norm3 = AGroupNorm(self.ngf*4,self.ngf*2)
        self.agg_3 = AggBlock(self.ngf *4* 2, self.ngf*4)
        self.dec_conv2 = nn.ConvTranspose2d(self.ngf * 4, self.ngf , kernel_size=4, stride=2, padding=1)
        self.agroup_norm2 = AGroupNorm(self.ngf*2,self.ngf)
        self.agg_2 = AggBlock(self.ngf * 4, self.ngf*2)

        self.dec_conv1 =  nn.ConvTranspose2d(self.ngf * 2, 3 , kernel_size=4, stride=2, padding=1)
        # self.agroup_norm2 = AGroupNorm(self.ngf * 2, self.ngf)
        self.agg_1 = AggBlock(self.ngf * 2, self.ngf )

        self.conv_1 =   nn.Conv2d(self.ngf*8 * 2, self.ngf*8, kernel_size=1, stride=1, padding=0)


    def encoder(self, inputs):
        _encode_layers = []
        x_down1 = self.enc_conv1(inputs)
        _encode_layers.append(x_down1)

        x_down2 = F.leaky_relu(x_down1, 0.2)
        convolved = self.enc_conv2(x_down2)
        x_down2 = F.group_norm(convolved, num_groups=32)
        _encode_layers.append(x_down2)

        x_down3 = F.leaky_relu(x_down2, 0.2)
        convolved = self.enc_conv3(x_down3)
        x_down3 = F.group_norm(convolved, num_groups=32)
        _encode_layers.append(x_down3)

        x_down4 = F.leaky_relu(x_down3, 0.2)
        convolved = self.enc_conv4(x_down4)
        x_down4 = F.group_norm(convolved, num_groups=32)
        _encode_layers.append(x_down4)

        x_down5 = F.leaky_relu(x_down4, 0.2)
        convolved = self.enc_conv5(x_down5)
        x_down5 = F.group_norm(convolved, num_groups=32)
        _encode_layers.append(x_down5)

        x_down6 = F.leaky_relu(x_down5, 0.2)
        convolved = self.enc_conv6(x_down6)
        x_down6 = F.group_norm(convolved, num_groups=32)
        _encode_layers.append(x_down6)

        x_down7 = F.leaky_relu(x_down6, 0.2)
        convolved = self.enc_conv7(x_down7)
        x_down7 = F.group_norm(convolved, num_groups=32)
        _encode_layers.append(x_down7)

        x_down8 = F.leaky_relu(x_down7, 0.2)
        convolved = self.enc_conv8(x_down8)
        x_down8 = F.group_norm(convolved, num_groups=32)
        _encode_layers.append(x_down8)
        return _encode_layers
    def decoder(self,inputs_encode,inputs_encode_dcp):
        _decode_layers=[]
        input8 = torch.cat([inputs_encode[-1], inputs_encode_dcp[-1]], dim=1)
        agg8 = self.conv_1(input8)
        x_up8 = self.dec_conv8(F.relu(agg8))
        output8 = self.agroup_norm8(x_up8, agg8)
        output8 = F.dropout(output8, p=1 - 0.5)
        _decode_layers.append(output8)

        agg7 = self.agg_7(inputs_encode[-2],inputs_encode_dcp[-2])
        x_up7 = self.dec_conv7(F.relu(torch.cat([agg7,output8],dim=1)))
        output7 = self.agroup_norm7(x_up7, agg7)
        output7 = F.dropout(output7, p=1 - 0.5)
        _decode_layers.append(output7)

        agg6 = self.agg_6(inputs_encode[-3], inputs_encode_dcp[-3])
        x_up6 = self.dec_conv6(F.relu(torch.cat([agg6, output7], dim=1)))
        output6 = self.agroup_norm6(x_up6, agg6)
        output6 = F.dropout(output6, p=1 - 0.5)
        _decode_layers.append(output6)

        agg5 = self.agg_5(inputs_encode[-4], inputs_encode_dcp[-4])
        x_up5 = self.dec_conv5(F.relu(torch.cat([agg5, output6], dim=1)))
        output5 = self.agroup_norm6(x_up5, agg5)
        _decode_layers.append(output5)

        agg4 = self.agg_4(inputs_encode[-5], inputs_encode_dcp[-5])
        x_up4 = self.dec_conv4(F.relu(torch.cat([agg4, output5], dim=1)))
        output4 = self.agroup_norm4(x_up4, agg4)
        _decode_layers.append(output4)

        agg3 = self.agg_3(inputs_encode[-6], inputs_encode_dcp[-6])
        x_up3 = self.dec_conv3(F.relu(torch.cat([agg3, output4], dim=1)))
        output3 = self.agroup_norm3(x_up3, agg3)
        _decode_layers.append(output3)

        agg2 = self.agg_2(inputs_encode[-7], inputs_encode_dcp[-7])
        x_up2 = self.dec_conv2(F.relu(torch.cat([agg2, output3], dim=1)))
        output2 = self.agroup_norm2(x_up2, agg2)
        _decode_layers.append(output2)

        agg1 = self.agg_1(inputs_encode[0], inputs_encode_dcp[0])
        output1 = self.dec_conv1(F.relu(torch.cat([agg1, output2], dim=1)))
        output = F.tanh(output1)

        return output

    def forward(self,hazy_img):
        dcp = self.dcp_g(hazy_img)
        self.dcp_layers = dcp
        self.encode_layers = self.encoder(hazy_img)
        self.encode_layers_dcp = self.encoder(self.dcp_layers)
        self.decode_layers = self.decoder(self.encode_layers, self.encode_layers_dcp)
        return self.decode_layers
class DCPDehazeGenerator(nn.Module):
    """Create a DCP Dehaze generator"""
    def __init__(self, win_size=5, r=15, eps=1e-3):
        super(DCPDehazeGenerator, self).__init__()

        self.guided_filter = GuidedFilter(r=r, eps=eps)
        self.neighborhood_size = win_size
        self.omega = 0.95

    def get_dark_channel(self, img, neighborhood_size):
        shape = img.shape
        if len(shape) == 4:
            img_min,_ = torch.min(img, dim=1)

            padSize = int(np.floor(neighborhood_size/2))
            if neighborhood_size % 2 == 0:
                pads = [padSize, padSize-1 ,padSize ,padSize-1]
            else:
                pads = [padSize, padSize ,padSize ,padSize]

            img_min = F.pad(img_min, pads, mode='constant', value=1)
            dark_img = -F.max_pool2d(-img_min, kernel_size=neighborhood_size, stride=1)
        else:
            raise NotImplementedError('get_tensor_dark_channel is only for 4-d tensor [N*C*H*W]')

        dark_img = torch.unsqueeze(dark_img, dim=1)
        return dark_img

    def atmospheric_light(self, img, dark_img):
        num,chl,height,width = img.shape
        topNum = int(0.01*height*width)

        A = torch.Tensor(num,chl,1,1)
        if img.is_cuda:
            A = A.cuda()

        for num_id in range(num):
            curImg = img[num_id,...]
            curDarkImg = dark_img[num_id,0,...]

            _, indices = curDarkImg.reshape([height*width]).sort(descending=True)
            #curMask = indices < topNum

            for chl_id in range(chl):
                imgSlice = curImg[chl_id,...].reshape([height*width])
                A[num_id,chl_id,0,0] = torch.mean(imgSlice[indices[0:topNum]])

        return A


    def forward(self, x):
        if x.shape[1] > 1:
            # rgb2gray
            guidance = 0.2989 * x[:,0,:,:] + 0.5870 * x[:,1,:,:] + 0.1140 * x[:,2,:,:]
        else:
            guidance = x
        # rescale to [0,1]
        guidance = (guidance + 1)/2
        guidance = torch.unsqueeze(guidance, dim=1)
        imgPatch = (x + 1)/2

        num,chl,height,width = imgPatch.shape

        # dark_img and A with the range of [0,1]
        dark_img = self.get_dark_channel(imgPatch, self.neighborhood_size)
        A = self.atmospheric_light(imgPatch, dark_img)

        map_A = A.repeat(1,1,height,width)
        # make sure channel of trans_raw == 1
        trans_raw = 1 - self.omega*self.get_dark_channel(imgPatch/map_A, self.neighborhood_size)

        # get initial results
        T_DCP = self.guided_filter(guidance, trans_raw)
        J_DCP = (imgPatch - map_A)/T_DCP.repeat(1,3,1,1) + map_A

        return J_DCP

class AggBlock(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(AggBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel,int(in_channel/2),kernel_size=1,stride=1,padding=0)
        self.conv2 = nn.Conv2d(int(in_channel/2), out_channel, kernel_size=3, stride=1, padding=1)
    def forward(self,x,y):
        out = torch.cat([x, y], dim=1)
        out = self.conv1(out)
        out_a = self.conv2(out)
        out_a = F.sigmoid(out_a)
        out = out * out_a
        return out
class AGroupNorm(nn.Module):
    def __init__(self,in_channel,out_channel, num_groups=32, eps=1e-5):
        super(AGroupNorm, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_groups = num_groups
        self.eps = eps
        self.conv1 = nn.Conv2d(self.in_channel,self.out_channel,kernel_size=1,stride=1,padding=0)
        self.conv2 = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1, stride=1, padding=0)
    def forward(self, x,y):
        N,C,H,W = x.size()
        G = self.num_groups
        assert C % G == 0

        x = x.view(N,G,-1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N,C,H,W)
        y_pool = F.adaptive_avg_pool2d(y,1)
        weight = self.conv1(y_pool)
        bias = self.conv2(y_pool)
        return x * weight + bias