    #-*- coding:utf-8 -*-
'''
[AI502] Deep Learning Assignment
"Wasserstein GAN" Implementation
20193640 Jungwon Choi
'''

'''
# I referenced the below links,
# but I coded whole parts myself, not copy and paste.
# Reference : https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
'''

import torch
import torch.nn as nn
import torch.optim as optim

import argparse
import random
import pickle
import time
import os
import numpy as np

# Implementation files
from dataloader import CelebA_Dataloader
from model.Generator import Generator
from model.Discriminator import Discriminator
from train import train
from val import val

os.environ["CUDA_VISIBLE_DEVICES"]="1"

VERSION_CHECK_MESSAGE = 'NOW 19-11-27 07:23'

# Set the directory paths
RESULTS_PATH = './results/'
CHECKPOINT_PATH ='./checkpoints/'

#===============================================================================
# I implemented this class to count and control the iteration step.
class StepCounter():
    #===========================================================================
    ''' Initialization '''
    def __init__(self, objective_step):
        self.objective_step = objective_step
        self.current_step = 0
        self.exit_signal = False
    #===========================================================================
    ''' Count the iteration step '''
    def step(self):
        self.current_step += 1

#===============================================================================
''' Experiment1 : JSD estimate vs EM estimate '''
    # Check whether the WGAN loss is a meaningful loss metric
''' Experiment2 : DCGAN vs WGAN without batch norm '''
    # Check the WGAN loss improve the stability of the optimization process
def main(args):
    #===========================================================================
    # Set the file name format
    FILE_NAME_FORMAT = "{0}_{1}_{2:d}_{3:d}_{4:d}_{5:f}{6}".format(
                            args.model, args.dataset, args.epochs,
                            args.obj_step, args.batch_size, args.lr, args.flag)

    # Set the results file path
    RESULT_FILE_NAME = FILE_NAME_FORMAT+'_results.pkl'
    RESULT_FILE_PATH = os.path.join(RESULTS_PATH, RESULT_FILE_NAME)
    # Set the checkpoint file path
    CHECKPOINT_FILE_NAME = FILE_NAME_FORMAT+'.ckpt'
    CHECKPOINT_FILE_PATH = os.path.join(CHECKPOINT_PATH, CHECKPOINT_FILE_NAME)
    BEST_CHECKPOINT_FILE_NAME = FILE_NAME_FORMAT+'_best.ckpt'
    BEST_CHECKPOINT_FILE_PATH = os.path.join(CHECKPOINT_PATH,
                                                BEST_CHECKPOINT_FILE_NAME)

    # Set the random seed same for reproducibility
    random.seed(190811)
    torch.manual_seed(190811)
    torch.cuda.manual_seed_all(190811)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Step1 ====================================================================
    # Load dataset
    if args.dataset == 'CelebA':
        dataloader = CelebA_Dataloader()
    else:
        assert False, "Please select the proper dataset."

    train_loader = dataloader.get_train_loader(batch_size=args.batch_size,
                                                num_workers=args.num_workers)
    print('==> DataLoader ready.')

    # Step2 ====================================================================
    # Make the model
    if args.model in ['WGAN', 'DCGAN']:
        generator       = Generator(BN=True)
        discriminator   = Discriminator(BN=True)
    elif args.model in ['WGAN_noBN', 'DCGAN_noBN']:
        generator       = Generator(BN=False)
        discriminator   = Discriminator(BN=False)
    else:
        assert False, "Please select the proper model."

    # Check DataParallel available
    if torch.cuda.device_count() > 1:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

    # Check CUDA available
    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
    print('==> Model ready.')

    # Step3 ====================================================================
    # Set loss function and optimizer
    if args.model in ['DCGAN', 'DCGAN_noBN']:
        criterion = nn.BCELoss()
    else:
        criterion = None
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=args.lr)
    step_counter = StepCounter(args.obj_step)
    print('==> Criterion and optimizer ready.')

    # Step4 ====================================================================
    # Train and validate the model
    start_epoch = 0
    best_metric = float("inf")
    validate_noise = torch.randn(args.batch_size, 100, 1, 1)

    # Initialize the result lists
    train_loss_G = []
    train_loss_D = []
    train_distance = []

    if args.resume:
        assert os.path.exists(CHECKPOINT_FILE_PATH), 'No checkpoint file!'
        checkpoint = torch.load(CHECKPOINT_FILE_PATH)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        start_epoch = checkpoint['epoch']
        step_counter.current_step = checkpoint['current_step']
        train_loss_G = checkpoint['train_loss_G']
        train_loss_D = checkpoint['train_loss_D']
        train_distance = checkpoint['train_distance']
        best_metric = checkpoint['best_metric']

    # Save the training information
    result_data = {}
    result_data['model']            = args.model
    result_data['dataset']          = args.dataset
    result_data['target_epoch']     = args.epochs
    result_data['batch_size']       = args.batch_size

    # Check the directory of the file path
    if not os.path.exists(os.path.dirname(RESULT_FILE_PATH)):
        os.makedirs(os.path.dirname(RESULT_FILE_PATH))
    if not os.path.exists(os.path.dirname(CHECKPOINT_FILE_PATH)):
        os.makedirs(os.path.dirname(CHECKPOINT_FILE_PATH))

    print('==> Train ready.')

    # Validate before training (step 0)
    val(generator, validate_noise, step_counter, FILE_NAME_FORMAT)

    for epoch in range(args.epochs):
        # strat after the checkpoint epoch
        if epoch < start_epoch:
            continue
        print("\n[Epoch: {:3d}/{:3d}]".format(epoch+1, args.epochs))
        epoch_time = time.time()
        #=======================================================================
        # train the model (+ validate the model)
        tloss_G, tloss_D, tdist = train(generator, discriminator, train_loader,
                                criterion, optimizer_G, optimizer_D,
                                args.clipping, args.num_critic,
                                step_counter, validate_noise, FILE_NAME_FORMAT)
        train_loss_G.extend(tloss_G)
        train_loss_D.extend(tloss_D)
        train_distance.extend(tdist)
        #=======================================================================
        current = time.time()

        # Calculate average loss
        avg_loss_G = sum(tloss_G)/len(tloss_G)
        avg_loss_D = sum(tloss_D)/len(tloss_D)
        avg_distance = sum(tdist)/len(tdist)

        # Save the current result
        result_data['current_epoch']    = epoch
        result_data['train_loss_G']     = train_loss_G
        result_data['train_loss_D']     = train_loss_D
        result_data['train_distance']   = train_distance

        # Save result_data as pkl file
        with open(RESULT_FILE_PATH, 'wb') as pkl_file:
            pickle.dump(result_data, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

        # Save the best checkpoint
        # if avg_distance < best_metric:
        #     best_metric = avg_distance
        #     torch.save({
        #         'epoch': epoch+1,
        #         'generator_state_dict': generator.state_dict(),
        #         'discriminator_state_dict': discriminator.state_dict(),
        #         'optimizer_G_state_dict': optimizer_G.state_dict(),
        #         'optimizer_D_state_dict': optimizer_D.state_dict(),
        #         'current_step': step_counter.current_step,
        #         'best_metric': best_metric,
        #         }, BEST_CHECKPOINT_FILE_PATH)

        # Save the current checkpoint
        torch.save({
            'epoch': epoch+1,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'current_step': step_counter.current_step,
            'train_loss_G': train_loss_G,
            'train_loss_D': train_loss_D,
            'train_distance': train_distance,
            'best_metric': best_metric,
            }, CHECKPOINT_FILE_PATH)

        # Print the information on the console
        print("model                : {}".format(args.model))
        print("dataset              : {}".format(args.dataset))
        print("batch_size           : {}".format(args.batch_size))
        print("current step         : {:d}".format(step_counter.current_step))
        print("current lrate        : {:f}".format(args.lr))
        print("gen/disc loss        : {:f}/{:f}".format(avg_loss_G, avg_loss_D))
        print("distance metric      : {:f}".format(avg_distance))
        print("epoch time           : {0:.3f} sec".format(current - epoch_time))
        print("Current elapsed time : {0:.3f} sec".format(current - start))

        # If iteration step has been satisfied
        if step_counter.exit_signal:
            break

    print('==> Train done.')

    print(' '.join(['Results have been saved at', RESULT_FILE_PATH]))
    print(' '.join(['Checkpoints have been saved at', CHECKPOINT_FILE_PATH]))

#===============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WGAN Implementation')
    parser.add_argument('--model', default='WGAN', type=str,
                                help='WGAN, DCGAN, WGAN_noBN, DCGAN_noBN')
    parser.add_argument('--dataset', default='CelebA', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--obj_step', default=100000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--lr', default=0.00005, type=float)
    parser.add_argument('--clipping', default=0.01, type=float)
    parser.add_argument('--num_critic', default=5, type=int)
    parser.add_argument('--flag', default='', type=str)
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    # Code version check message
    print(VERSION_CHECK_MESSAGE)

    start = time.time()
    #===========================================================================
    main(args)
    #===========================================================================
    end = time.time()
    print("Total elapsed time: {0:.3f} sec\n".format(end - start))
    print("[Finih time]",time.strftime('%c', time.localtime(time.time())))
