#-*- coding:utf-8 -*-
'''
[AI502] Deep Learning Assignment
"Wasserstein GAN" Implementation
20193640 Jungwon Choi
'''
import os
import sys
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
# import seaborn as sns

import PIL
from PIL import Image
from scipy.signal import savgol_filter

import torch
import torch.nn as nn
import torchvision.utils as utils
from model.Generator import Generator

FIGURE_PATH = './figures'
RESULT_PATH = './results'
CHECKPOINT_PATH ='./checkpoints/'

os.environ["CUDA_VISIBLE_DEVICES"]="3"

#===============================================================================
TEST_CHECKPOINT_LIST = [
    #---------------------------------------------------------------------------
    'DCGAN_CelebA_100_100000_64_0.000050.ckpt',
    'WGAN_CelebA_100_100000_64_0.000050.ckpt',
    'DCGAN_noBN_CelebA_100_50000_64_0.000050_edit.ckpt',
    'WGAN_noBN_CelebA_100_50000_64_0.000050_clip05.ckpt',
    #---------------------------------------------------------------------------
]

PKL_FILE_LIST = [
    'DCGAN_CelebA_100_100000_64_0.000050_results.pkl',
    'WGAN_CelebA_100_100000_64_0.000050_results.pkl',
    'DCGAN_noBN_CelebA_100_50000_64_0.000050_edit_results.pkl',
    'WGAN_noBN_CelebA_100_50000_64_0.000050_clip05_results.pkl',
    #---------------------------------------------------------------------------
    # 'WGAN_CelebA_100_100000_64_0.000050_critic10_results.pkl',
    # 'WGAN_CelebA_100_100000_64_0.000050_critic3_results.pkl',
    #---------------------------------------------------------------------------
    # 'WGAN_CelebA_100_100000_64_0.001000_results.pkl',
    # 'WGAN_CelebA_100_100000_64_0.002000_results.pkl',
    # 'WGAN_CelebA_100_100000_64_0.000100_results.pkl',
    # 'WGAN_CelebA_100_100000_64_0.000030_results.pkl',
    # 'WGAN_CelebA_100_100000_64_0.000010_results.pkl',
]

#===============================================================================
''' Generate fake images with the trained model '''
def generate_fake_images(ckpt_list):
    #===========================================================================
    for ckpt_name in ckpt_list:
        #=======================================================================
        # Parsing the hyper-parameters
        parsing_list = ckpt_name.split('.')[0].split('_')

        # Setting constants
        model_type          = parsing_list[0]

        # Step1 ================================================================
        # Make the model
        if model_type in ['WGAN', 'DCGAN']:
            generator       = Generator(BN=True)
        elif model_type in ['WGAN_noBN', 'DCGAN_noBN']:
            generator       = Generator(BN=False)
        else:
            assert False, "Please select the proper model."

        # Check DataParallel available
        if torch.cuda.device_count() > 1:
            generator = nn.DataParallel(generator)

        # Check CUDA available
        if torch.cuda.is_available():
            generator.cuda()
        print('==> Model ready.')

        # Step2 ================================================================
        # Test the model
        checkpoint = torch.load(os.path.join(CHECKPOINT_PATH, ckpt_name))
        generator.load_state_dict(checkpoint['generator_state_dict'])
        train_step = checkpoint['current_step']

        generator.eval()
        device = next(generator.parameters()).device.index

        # Set save path
        FILE_NAME_FORMAT = os.path.splitext(ckpt_name)[0]
        SAVE_IMG_PATH = os.path.join(FIGURE_PATH, FILE_NAME_FORMAT)

        # Check the directory of the file path
        if not os.path.exists(SAVE_IMG_PATH):
            os.makedirs(SAVE_IMG_PATH)

        IMAGE_NAME = 'test_fake_images_step{0}.png'.format(train_step)

        # test the model
        #-----------------------------------------------------------------------
        # Make test noise
        test_noise = torch.randn(64, 100, 1, 1)
        test_noise = test_noise.cuda(device)

        # Generate fake images from noise
        fake_images = generator(test_noise)

        # Save the fake images
        fake_images = fake_images.detach().cpu()

        fig = plt.figure(figsize=(8,8)); plt.axis("off");
        plt.title("fake images (step: {0:d})".format(train_step));
        plt.imshow(np.transpose(utils.make_grid(fake_images,
                                                padding=2,
                                                normalize=True),
                                                (1,2,0)))
        fig.savefig(os.path.join(SAVE_IMG_PATH, IMAGE_NAME),
                bbox_inces='tight', pad_inches=0, dpi=150)
        plt.close()
        #-----------------------------------------------------------------------

        # Print the result on the console
        print("model                  : {}".format(model_type))
        print('-'*50)
    print('==> Image generation done.')

#===============================================================================
def visualize_loss_graph(plk_file_list):
    for plk_file_name in plk_file_list:
        #=======================================================================
        # Load results data
        plk_file_path = os.path.join(RESULT_PATH, plk_file_name)
        with open(plk_file_path, 'rb') as pkl_file:
            result_dict = pickle.load(pkl_file)

        train_loss_G = result_dict['train_loss_G']
        train_loss_D = result_dict['train_loss_D']
        train_distance = result_dict['train_distance']

        train_loss_G_smooth = savgol_filter(train_loss_G, 101, 3)
        train_loss_D_smooth = savgol_filter(train_loss_D, 101, 3)
        train_distance_smooth = savgol_filter(train_distance, 101, 3)

        #=======================================================================
        # Save figure
        FILE_NAME_FORMAT = '_'.join(os.path.splitext(plk_file_name)[0].split('_')[:-1])
        model_type = FILE_NAME_FORMAT.split('_')[0]
        SAVE_IMG_PATH = os.path.join(FIGURE_PATH, FILE_NAME_FORMAT)

        # Check the directory of the file path
        if not os.path.exists(SAVE_IMG_PATH):
            os.makedirs(SAVE_IMG_PATH)

        # Match the data length
        MAX_LENGTH = 50000
        num_step = min(min(len(train_loss_D), len(train_loss_G)), MAX_LENGTH)
        steps = np.arange(1, num_step+1)

        train_loss_G = train_loss_G[:num_step]
        train_loss_D = train_loss_D[:num_step]
        train_distance = train_distance[:num_step]
        train_loss_G_smooth = train_loss_G_smooth[:num_step]
        train_loss_D_smooth = train_loss_D_smooth[:num_step]
        train_distance_smooth = train_distance_smooth[:num_step]

        #-----------------------------------------------------------------------
        # Generator Loss Graph
        #-----------------------------------------------------------------------
        fig = plt.figure(dpi=150)
        plt.title('Generator Loss'), plt.xlabel('Generator iteration')
        plt.ylabel('Loss')
        plt.plot(steps, train_loss_G,'b-', markersize=1, alpha=0.3)
        plt.plot(steps, train_loss_G_smooth,'b-', markersize=1, alpha=0.8,
                                        label=model_type)
        plt.xlim([0, num_step])
        plt.legend()
        file_name = "Loss_G_graph.png"
        fig.savefig(os.path.join(SAVE_IMG_PATH, file_name), format='png')
        plt.close()

        #-----------------------------------------------------------------------
        # Discriminator Loss Graph
        #-----------------------------------------------------------------------
        fig = plt.figure(dpi=150)
        plt.title('Discriminator Loss'), plt.xlabel('Discriminator iteration')
        plt.ylabel('Loss')
        plt.plot(steps, train_loss_D,'m-', markersize=1, alpha=0.3)
        plt.plot(steps, train_loss_D_smooth,'m-', markersize=1, alpha=0.8,
                                        label=model_type)
        plt.xlim([0, num_step])
        plt.legend()
        file_name = "Loss_D_graph.png"
        fig.savefig(os.path.join(SAVE_IMG_PATH, file_name), format='png')
        plt.close()

        #-----------------------------------------------------------------------
        # Distance Graph
        #-----------------------------------------------------------------------
        fig = plt.figure(dpi=150)
        plt.title('Distance Metric'), plt.xlabel('Generator iteration')
        if model_type in ['WGAN', 'WGAN_noBN']:
            plt.ylabel('Wasserstein estimate')
        else:
            plt.ylabel('JSD estimate')
        plt.plot(steps, train_distance,'g-', markersize=1, alpha=0.3)
        plt.plot(steps, train_distance_smooth,'g-', markersize=1, alpha=0.8,
                                        label=model_type)
        plt.xlim([0, num_step])
        plt.ylim(bottom=0)
        plt.legend()
        file_name = "Distance_graph.png"
        fig.savefig(os.path.join(SAVE_IMG_PATH, file_name), format='png')
        plt.close()

    print('==> Loss graph visualization done.')

#===============================================================================
if __name__ == '__main__':
    # generate_fake_images(TEST_CHECKPOINT_LIST)
    visualize_loss_graph(PKL_FILE_LIST)
    pass
