# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 22:20:48 2019

@author: prnvb
"""

import numpy as np
import matplotlib.pyplot as plt

d_loss_fake419 = np.load('Results/420Eps/loss/d_loss_fake419.npy')
d_loss_real419 = np.load('Results/420Eps/loss/d_loss_real419.npy')
d_loss419 = np.load('Results/420Eps/loss/d_loss419.npy')
dc_gan_loss419 = np.load('Results/420Eps/loss/dc_gan_loss419.npy')


plt.plot(d_loss419, label='Discriminator Loss')
plt.plot(d_loss_fake419, label='Discriminator Loss - Fake')
plt.plot(d_loss_real419, label='Discriminator Loss - Real')
plt.plot(dc_gan_loss419[:,2], label='Generator Loss')
plt.legend(loc='upper right')

plt.figure()
plt.plot(d_loss_fake419, color='red', label='Discriminator Loss - Fake')
plt.plot(d_loss_real419, color='blue', label='Discriminator Loss - Real')
plt.plot(d_loss419, color='green', label='Discriminator Loss')
plt.legend(loc='upper right')

plt.figure()
plt.plot(d_loss_real419, color='magenta', label='Discriminator Loss - Real')
plt.legend(loc='upper right')

plt.figure()
plt.plot(d_loss419, color='black', label='Discriminator Loss')
plt.legend(loc='upper right')

plt.plot(dc_gan_loss419[:,0], color='blue', label='Generator Loss - L1')
plt.legend(loc='upper right')

plt.plot(dc_gan_loss419[:,1], color='orange', label='Generator Loss - Binary Cross Entropy')
plt.legend(loc='upper right')

plt.figure()
plt.plot(d_loss419, color='blue', label='Discriminator Loss')
plt.plot(dc_gan_loss419[:,2], color='red', label='Generator Loss')
plt.legend(loc='upper right')