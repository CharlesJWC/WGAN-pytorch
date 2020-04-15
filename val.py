#-*- coding:utf-8 -*-
'''
[AI502] Deep Learning Assignment
"Wasserstein GAN" Implementation
20193640 Jungwon Choi
'''
import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import os

import torchvision.utils as utils
from visualize import FIGURE_PATH

#===============================================================================
''' Validate sequence '''
def val(generator, validate_noise, step_counter, FILE_NAME_FORMAT):
    generator.eval()
    device = next(generator.parameters()).device.index
    current_step = step_counter.current_step
    SAVE_IMG_PATH = os.path.join(FIGURE_PATH, FILE_NAME_FORMAT)

    # Check the directory of the file path
    if not os.path.exists(SAVE_IMG_PATH):
        os.makedirs(SAVE_IMG_PATH)

    IMAGE_NAME = 'fake_images_step{0}.png'.format(current_step)

    with torch.no_grad():
        #=======================================================================
        validate_noise = validate_noise.cuda(device)

        # Generate fake images from noise
        fake_images = generator(validate_noise)

        # Save the fake images
        fake_images = fake_images.detach().cpu()

        fig = plt.figure(figsize=(8,8)); plt.axis("off");
        plt.title("fake images (step: {0:d})".format(current_step));
        plt.imshow(np.transpose(utils.make_grid(fake_images,
                                                padding=2,
                                                normalize=True),
                                                (1,2,0)))
        fig.savefig(os.path.join(SAVE_IMG_PATH, IMAGE_NAME),
                bbox_inces='tight', pad_inches=0, dpi=150)
        plt.close()
        #=======================================================================
        print("[step {0:d}] fake images have been saved.".format(current_step),
                                                        end=''), print(' '*50)

    return True
