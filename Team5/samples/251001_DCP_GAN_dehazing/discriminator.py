
import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self,in_channel,ndf=64,scope='D',reuse=None):
        super(Discriminator, self).__init__()
        self.in_channel = in_channel
        self.ndf=ndf
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(self.in_channel, self.ndf, kernel_size=4, stride=2,padding=1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(self.ndf, self.ndf*2, kernel_size=4, stride=2, padding=1))
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(self.ndf*2, self.ndf*4, kernel_size=4, stride=2, padding=1))
        self.conv4 = nn.utils.spectral_norm(nn.Conv2d(self.ndf*4, self.ndf*8, kernel_size=4, stride=2, padding=1))
        self.conv5 = nn.utils.spectral_norm(nn.Conv2d(self.ndf * 8, 3, kernel_size=4, stride=1, padding=1))
    def forward(self,input):
        layers = []
        # convolved = discrim_conv(self.input, self.ndf, stride=2,use_sn=True)
        convolved = self.conv1(input)
        # convolved = conv2d(self.input,self.ndf,4,4,1,1,use_sn=True)
        rectified = F.leaky_relu(convolved, 0.2)
        layers.append(rectified)
        convolved = self.conv2(rectified)
        rectified = F.leaky_relu(convolved, 0.2)
        layers.append(rectified)
        convolved = self.conv3(rectified)
        rectified = F.leaky_relu(convolved, 0.2)
        layers.append(rectified)
        convolved = self.conv4(rectified)
        rectified = F.leaky_relu(convolved, 0.2)
        layers.append(rectified)
        convolved = self.conv5(rectified)
        output = F.sigmoid(convolved)
        layers.append(output)
        return  output