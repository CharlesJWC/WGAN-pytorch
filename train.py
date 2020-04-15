#-*- coding:utf-8 -*-
'''
[AI502] Deep Learning Assignment
"Wasserstein GAN" Implementation
20193640 Jungwon Choi
'''
import torch
import torch.nn as nn
import numpy as np

from scipy.spatial.distance import jensenshannon
from val import val

#===============================================================================
''' Train sequence '''
def train(generator, discriminator, train_loader, criterion,
            optimizer_G, optimizer_D, clipping, num_critic,
            step_counter, validate_noise, FILE_NAME_FORMAT):
    generator.train()
    discriminator.train()
    device = next(generator.parameters()).device.index
    losses_G = []
    losses_D = []
    distances = []
    total_iter = len(train_loader)

    for i, (images, _) in enumerate(train_loader):
        real_images = images.cuda(device)

        #=======================================================================
        # For WGAN
        if criterion is None:
            #-------------------------------------------------------------------
            ''' Train Discriminator Network '''
            #-------------------------------------------------------------------

            # Make real & fake lables and noise for generator
            batch_size = real_images.size(0)
            real_label = torch.full((batch_size,), 1, device=device)
            fake_label = torch.full((batch_size,), 0, device=device)
            noise = torch.randn(batch_size, 100, 1, 1, device=device)

            # Empty discriminator's gradients
            discriminator.zero_grad()

            # Generate fake images from noise
            fake_images = generator(noise)

            # Discriminate the images
            output_real = discriminator(real_images)
            output_fake = discriminator(fake_images.detach())

            # Calculate loss
            loss_D =  -output_real.mean() + output_fake.mean()
            losses_D.append(loss_D.item())

            # Calculate gradients (Backpropagation)
            loss_D.backward()

            # Calculate Earth Mover(EM) distance
            distance = output_real.mean() - output_fake.mean()
            distances.append(distance)

            # Update discriminator's parameters
            optimizer_D.step()

            # Cliping the discriminator's parameters
            for parameter in discriminator.parameters():
                parameter.data.clamp_(-clipping, clipping)

            #-------------------------------------------------------------------
            ''' Train Generator Network '''
            #-------------------------------------------------------------------
            loss_G = -output_fake.mean()

            # For every num_critic iteration
            if step_counter.current_step % num_critic == 0:

                # Empty generator's gradients
                generator.zero_grad()

                # Generate fake images from noise
                fake_images = generator(noise)

                # Discriminate the images
                output_fake = discriminator(fake_images)

                # Calculate loss
                loss_G = -output_fake.mean()
                losses_G.extend([loss_G.item()]*num_critic)

                # Calculate gradients (Backpropagation)
                loss_G.backward()

                # Update generator's parameters
                optimizer_G.step()
        #=======================================================================
        # For DCGAN
        else:
            #-------------------------------------------------------------------
            ''' Train Discriminator Network '''
            # maximize log(D(x)) + log(1 - D(G(z)))
            #-------------------------------------------------------------------

            # Make real & fake lables and noise for generator
            batch_size = real_images.size(0)
            real_label = torch.full((batch_size,), 1, device=device)
            fake_label = torch.full((batch_size,), 0, device=device)
            noise = torch.randn(batch_size, 100, 1, 1, device=device)

            # <For real images> ------------------------------------------------
            # Empty discriminator's gradients
            discriminator.zero_grad()

            # Predict targets (Forward propagation)
            output = discriminator(real_images)
            pred_label = torch.sigmoid(output).view(-1)

            # Calculate loss (for real images)
            loss_D_real = criterion(pred_label, real_label)

            # Calculate gradients (Backpropagation)
            loss_D_real.backward()


            # <For fake images> ------------------------------------------------
            # Generate fake images from noise
            fake_images = generator(noise)

            # Discriminate the images
            output = discriminator(fake_images.detach())
            pred_label = torch.sigmoid(output).view(-1)

            # Calculate loss (for fake images)
            loss_D_fake = criterion(pred_label, fake_label)

            # Calculate gradients (Backpropagation)
            loss_D_fake.backward()


            # Make mixed error (Add the both of gradients) ---------------------
            loss_D = loss_D_real + loss_D_fake
            losses_D.append(loss_D.item())

            # Update discriminator's parameters
            optimizer_D.step()

            #-------------------------------------------------------------------
            ''' Train Generator Network '''
            # maximize log(D(G(z)))
            #-------------------------------------------------------------------

            # Empty generator's gradients
            generator.zero_grad()

            # Calculate loss (for fake images)
            output = discriminator(fake_images)
            pred_label = torch.sigmoid(output).view(-1)

            # Calculate loss (for fake images)
            loss_G = criterion(pred_label, real_label)
            losses_G.append(loss_G.item())

            # Calculate gradients (Backpropagation)
            loss_G.backward()

            # Update discriminator's parameters
            optimizer_G.step()

            #-------------------------------------------------------------------
            # Calculate Jenen-Shannon Divergence
            target_label = torch.full((batch_size,), 0.5)
            distance = jensenshannon(pred_label.detach().cpu(), target_label)**2
            # Ref: Jensen-shannon Distance = sqrt(Jenen-Shannon Divergence)
            distances.append(distance)

        #=======================================================================
        # Count the step
        step_counter.step()

        # Display current status
        print("[{:5d}/{:5d}]".format(i+1, total_iter), end='')
        print(" loss_G: {:f} loss_D: {:f} dist: {:f} step: {:d}   \r".format(
                loss_G, loss_D, distance, step_counter.current_step), end='')

        # validate the model
        if (step_counter.current_step % 500 == 0 or
           step_counter.current_step == step_counter.objective_step):
            val(generator, validate_noise, step_counter, FILE_NAME_FORMAT)

        # Check the current step
        if step_counter.current_step >= step_counter.objective_step:
            step_counter.exit_signal = True
            break

    #===========================================================================
    return losses_G, losses_D, distances
