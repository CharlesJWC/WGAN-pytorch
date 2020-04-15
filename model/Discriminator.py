#-*- coding:utf-8 -*-
'''
[AI502] Deep Learning Assignment
"Wasserstein GAN" Implementation
20193640 Jungwon Choi
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#===============================================================================
''' Discriminator Network '''
class Discriminator(nn.Module):
    ''' Initialization '''
    #===========================================================================
    def __init__(self, BN=True):
        super(Discriminator, self).__init__()
        self.BN = True#BN

        # Discriminator layers
        ## layer1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128,
                            kernel_size=4, stride=2, padding=1, bias=False)
        # No batch norm
        self.lrelu1 = nn.LeakyReLU(0.2)
        ## layer2
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256,
                            kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.lrelu2 = nn.LeakyReLU(0.2)
        ## layer3
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512,
                            kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(512)
        self.lrelu3 = nn.LeakyReLU(0.2)
        ## layer4
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=1024,
                            kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(1024)
        self.lrelu4 = nn.LeakyReLU(0.2)
        ## layer5
        self.conv5 = nn.Conv2d(in_channels=1024, out_channels=1,
                            kernel_size=4, stride=1, padding=0, bias=False)
        # No batch norm
        # self.sigmoid5 = nn.Sigmoid()

        # parameters initialization
        self.params_init()

    #===========================================================================
    ''' Parameters initialization '''
    def params_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight.data, 0, 0.02)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.normal_(layer.weight.data, 1, 0.02)
                nn.init.constant_(layer.bias.data, 0)

    #===========================================================================
    ''' Forward from the generated image x '''
    def forward(self, x):
        # x : 3 x 64 x 64

        # layer 1
        out = self.conv1(x)
        out = self.lrelu1(out)
        # out : 128 x 32 x 32

        # layer 2
        out = self.conv2(out)
        if self.BN:
            out = self.bn2(out)
        out = self.lrelu2(out)
        # out : 256 x 16 x 16

        # layer 3
        out = self.conv3(out)
        if self.BN:
            out = self.bn3(out)
        out = self.lrelu3(out)
        # out : 512 x 8 x 8

        # layer 4
        out = self.conv4(out)
        if self.BN:
            out = self.bn4(out)
        out = self.lrelu4(out)
        # out : 1024 x 4 x 4

        # layer 5
        out = self.conv5(out)
        # out = self.sigmoid5(out)
        # out : 1 x 1 x 1

        return out
