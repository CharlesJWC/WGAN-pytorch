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
''' Generator Network '''
class Generator(nn.Module):
    ''' Initialization '''
    #===========================================================================
    def __init__(self, BN=True):
        super(Generator, self).__init__()
        self.BN = BN

        # Generator layers
        ## layer1
        self.deconv1 = nn.ConvTranspose2d(in_channels=100, out_channels=1024,
                            kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=1024)
        self.relu1 = nn.ReLU()
        ## layer2
        self.deconv2 = nn.ConvTranspose2d(in_channels=1024, out_channels=512,
                            kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        self.relu2 = nn.ReLU()
        ## layer3
        self.deconv3 = nn.ConvTranspose2d(in_channels=512, out_channels=256,
                            kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        ## layer4
        self.deconv4 = nn.ConvTranspose2d(in_channels=256, out_channels=128,
                            kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        ## layer5
        self.deconv5 = nn.ConvTranspose2d(in_channels=128, out_channels=3,
                            kernel_size=4, stride=2, padding=1, bias=False)
        # No batch norm
        self.tanh5 = nn.Tanh()

        # parameters initialization
        self.params_init()

    #===========================================================================
    ''' Parameters initialization '''
    def params_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.normal_(layer.weight.data, 0, 0.02)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.normal_(layer.weight.data, 1, 0.02)
                nn.init.constant_(layer.bias.data, 0)

    #===========================================================================
    ''' Forward from the noize z '''
    def forward(self, z):
        # z : 100 x 1 x 1

        # layer 1
        out = self.deconv1(z)
        if self.BN:
            out = self.bn1(out)
        out = self.relu1(out)
        # out : 1024 x 4 x 4

        # layer 2
        out = self.deconv2(out)
        if self.BN:
            out = self.bn2(out)
        out = self.relu2(out)
        # out : 512 x 8 x 8

        # layer 3
        out = self.deconv3(out)
        if self.BN:
            out = self.bn3(out)
        out = self.relu3(out)
        # out : 256 x 16 x 16

        # layer 4
        out = self.deconv4(out)
        if self.BN:
            out = self.bn4(out)
        out = self.relu4(out)
        # out : 128 x 32 x 32

        # layer 5
        out = self.deconv5(out)
        out = self.tanh5(out)
        # out : 3 x 64 x 64

        return out
