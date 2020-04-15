#-*- coding:utf-8 -*-
'''
[AI502] Deep Learning Assignment
"Wasserstein GAN" Implementation
20193640 Jungwon Choi
'''
import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import os

#===============================================================================
''' CelebA Dataloader in direct '''
class CelebA_Dataloader():
    #===========================================================================
    ''' Initialization '''
    def __init__(self, root='../dataset/CelebA', transform=None):
        self.root = root
        # if there is no user guided transform
        if transform is None:
            self.transform = transforms.Compose([
                            transforms.Resize(64),
                            transforms.CenterCrop(64),
                            transforms.ToTensor(),
                            transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
                        ])
        else:
            self.transform = transform

    def get_train_loader(self, batch_size=20, num_workers=2):
        # Load CelebA dataset
        self.dataset = dataset.ImageFolder(root=self.root,
                                           transform=self.transform)
        # Make dataLoader
        train_loader = torch.utils.data.DataLoader(self.dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers,
                                                pin_memory=True)
        return train_loader

#===============================================================================
''' Check the CelebA dataset '''
if __name__ == '__main__':
    #===========================================================================
    import torchvision.utils as utils
    #===========================================================================
    # Get images from dataloader
    celeba = CelebA_Dataloader()
    dataloader = celeba.get_train_loader(batch_size=64, num_workers=8)
    images = iter(dataloader).next()

    # Show images
    plt.figure(figsize=(8,8)); plt.axis("off"); plt.title("Train Images");
    plt.imshow(np.transpose(utils.make_grid(images[0],
                                            padding=2,
                                            normalize=True),
                                            (1,2,0)))
    plt.show()
