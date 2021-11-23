# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 10:43:52 2019

@author: prnvb
"""
from typing import Tuple
import keras
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.merge import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Activation, Dropout
from keras.layers.normalization import BatchNormalization
import numpy as np
from keras.optimizers import Adam
import tensorflow as tf
from keras import backend as K
import os
from tqdm import tqdm
from keras.losses import mean_absolute_error, binary_crossentropy

import cv2 as cv
import matplotlib.pyplot as plt

def generator(in_shape: Tuple[int,int,int], out_shape: Tuple[int,int,int], filters: int):
    
    input_tensor = Input(shape = in_shape)
    noise = Input(shape = (in_shape[0], in_shape[1], 1))
    output_ch = out_shape[2]
        
    #Encoder
    x = BatchNormalization()(Conv2D(filters*1, kernel_size = (5, 5), strides = (2, 2), padding = "same")(Concatenate()([input_tensor, noise])))
    x = LeakyReLU(0.2)(x); e1 = x
    x = BatchNormalization()(Conv2D(filters*2, kernel_size = (5, 5), strides = (2, 2), padding = "same")(x))
    x = LeakyReLU(0.2)(x); e2 = x
    x = BatchNormalization()(Conv2D(filters*4, kernel_size = (5, 5), strides = (2, 2), padding = "same")(x))
    x = LeakyReLU(0.2)(x); e3 = x
    x = BatchNormalization()(Conv2D(filters*8, kernel_size = (5, 5), strides = (2, 2), padding = "same")(x))
    x = LeakyReLU(0.2)(x); e4 = x
    x = BatchNormalization()(Conv2D(filters*8, kernel_size = (4, 4), strides = (2, 2), padding = "same")(x))
    x = LeakyReLU(0.2)(x); e5 = x
    x = BatchNormalization()(Conv2D(filters*8, kernel_size = (4, 4), strides = (2, 2), padding = "same")(x))
    x = LeakyReLU(0.2)(x); e6 = x
    x = BatchNormalization()(Conv2D(filters*8, kernel_size = (4, 4), strides = (2, 2), padding = "same")(x))
    x = LeakyReLU(0.2)(x); e7 = x
    x = BatchNormalization()(Conv2D(filters*8, kernel_size = (4, 4), strides = (2, 2), padding = "same")(x))
    x = LeakyReLU(0.2)(x); 
    #e8 = x

    #Decoder
    x = BatchNormalization()(Conv2DTranspose(filters*8, kernel_size = (4, 4), strides = (2, 2), padding = "same")(x))
    x = LeakyReLU(0.2)(x); x = Concatenate()([Dropout(0.5)(x), e7])
    x = BatchNormalization()(Conv2DTranspose(filters*8, kernel_size = (4, 4), strides = (2, 2), padding = "same")(x))
    x = LeakyReLU(0.2)(x); x = Concatenate()([Dropout(0.5)(x), e6])
    x = BatchNormalization()(Conv2DTranspose(filters*8, kernel_size = (4, 4), strides = (2, 2), padding = "same")(x))
    x = LeakyReLU(0.2)(x); x = Concatenate()([Dropout(0.5)(x), e5])
    x = BatchNormalization()(Conv2DTranspose(filters*8, kernel_size = (4, 4), strides = (2, 2), padding = "same")(x))
    x = LeakyReLU(0.2)(x); x = Concatenate()([Dropout(0.5)(x), e4])
    x = BatchNormalization()(Conv2DTranspose(filters*4, kernel_size = (4, 4), strides = (2, 2), padding = "same")(x))
    x = LeakyReLU(0.2)(x); x = Concatenate()([Dropout(0.5)(x), e3])
    x = BatchNormalization()(Conv2DTranspose(filters*2, kernel_size = (4, 4), strides = (2, 2), padding = "same")(x))
    x = LeakyReLU(0.2)(x); x = Concatenate()([Dropout(0.5)(x), e2])
    x = BatchNormalization()(Conv2DTranspose(filters*1, kernel_size = (4, 4), strides = (2, 2), padding = "same")(x))
    x = LeakyReLU(0.2)(x); x = Concatenate()([Dropout(0.5)(x), e1])
    x = Conv2DTranspose(output_ch, kernel_size = (4, 4), strides = (2, 2), padding = "same")(x)
    x = Activation("tanh")(x)
    
    unet = Model(inputs=[input_tensor, noise], outputs = [x])
    return unet

def discriminator(in_shape: Tuple[int,int,int], y_shape: Tuple[int,int,int], filters: int):
    input_tensor = Input(shape=in_shape)
    input_y = Input(shape=y_shape)
    x = LeakyReLU(0.2)(Conv2D(filters*1, kernel_size=(4, 4), strides=(2, 2), padding="same")(Concatenate()([input_tensor, input_y])))
    x = LeakyReLU(0.2)(BatchNormalization()(Conv2D(filters*2, kernel_size=(4, 4), strides=(2, 2), padding="same")(x)))
    x = LeakyReLU(0.2)(BatchNormalization()(Conv2D(filters*4, kernel_size=(4, 4), strides=(2, 2), padding="same")(x)))
    x = LeakyReLU(0.2)(BatchNormalization()(Conv2D(filters*8, kernel_size=(4, 4), strides=(2, 2), padding="same")(x)))
    x = LeakyReLU(0.2)(BatchNormalization()(Conv2D(filters*8, kernel_size=(4, 4), strides=(1, 1), padding="same")(x)))
    x = LeakyReLU(0.2)(BatchNormalization()(Conv2D(filters*4, kernel_size=(4, 4), strides=(1, 1), padding="same")(x)))
    x = Activation("sigmoid")(Conv2D(1, kernel_size=(4, 4), strides=(1, 1), padding="same")(x))

    disc = Model(inputs=[input_tensor, input_y], outputs=[x])
    return disc

def cGAN_Model(gen, disc):
    input_gen = Input((512, 512, 7))
    noise = Input((512, 512, 1))
    output_gen = gen([input_gen, noise])
    #input_disc = Concatenate()([input_gen, output_gen])
    disc.trainable = False
    output_disc = disc([input_gen, output_gen])
    cGAN = Model(inputs = [input_gen, noise], outputs = [output_gen, output_disc])
    return cGAN

def generate_noise(in_shape: Tuple[int,int,int]):
    return np.random.normal(0, 1, size=in_shape)


gen_model = generator((512, 512, 7), (512, 512, 3), 64)
disc_model = discriminator((512, 512, 7), (512, 512, 3), 64)

gen_optimizer = keras.optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999) 
disc_optimizer = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

gen_model.compile(loss = 'mae', optimizer = gen_optimizer)
disc_model.compile(loss = 'binary_crossentropy', optimizer = disc_optimizer)

gen_model.load_weights('D:/dev/GeoGAN/Results/420Eps/Weights/420_epochs_gen.h5')
disc_model.load_weights('D:/dev/GeoGAN/Results/420Eps/Weights/420_epochs_disc.h5')


gen_model.load_weights('D:/dev/GeoGAN/Trained-Weights/720_epochs_gen.h5')
disc_model.load_weights('D:/dev/GeoGAN/Trained-Weights/720_epochs_disc.h5')


X_test = np.load('DataSet/Data/Resized/npy/OR/16/X16_11Norm_Test.npy').astype(np.float32)
Y_test = np.load('DataSet/Data/Resized/npy/OR/16/Y16_11Norm_Test.npy').astype(np.float32)

X_test = np.load('D:/dev/GeoGAN/DataSet/Data/Resized/npy/OR/16/NoFlip/X16_11Norm_Train.npy').astype(np.float32)
Y_test = np.load('D:/dev/GeoGAN/DataSet/Data/Resized/npy/OR/16/NoFlip/Y16_11Norm_Train.npy').astype(np.float32)

X_test = np.load('D:/dev/GeoGAN/DataSet/Data/Resized/npy/OR/16/NoFlip/X16_11Norm_Train_Full.npy').astype(np.float32)
Y_test = np.load('D:/dev/GeoGAN/DataSet/Data/Resized/npy/OR/16/NoFlip/Y16_11Norm_Train_Full.npy').astype(np.float32)


plt.ioff()
plt.ion()
my_dpi=100


infra_id = k
road = k
trees = k
water = k

k = 119

masks = X_test[k, :, :, 3:]
img = X_test[k, :, :, :3]

fig = plt.figure(frameon=False)
fig.set_size_inches(9.6, 9.03)
ax = plt.Axes(fig, [0.,0.,1.,1.])
ax.set_axis_off()
fig.add_axes(ax)

ax.imshow((img+1)/2)
fig.savefig('D:\\dev\\GeoGAN\\Seminar Report\\Images\\GGAN\\Results\\'+str(k)+'.jpg', transparent=True)

ax.imshow( (Y_test[k, :, :, :]+1)/2)
fig.savefig('D:\\dev\\GeoGAN\\Seminar Report\\Images\\GGAN\\Results\\'+str(k)+'-expected.jpg', transparent=True)

ax.imshow(masks[:, :, 0], cmap='gray')
fig.savefig('D:\\dev\\GeoGAN\\Seminar Report\\Images\\GGAN\\Results\\'+str(k)+'-Infra.jpg', transparent=True)

ax.imshow(masks[:, :, 1], cmap='gray')
fig.savefig('D:\\dev\\GeoGAN\\Seminar Report\\Images\\GGAN\\Results\\'+str(k)+'-Roads.jpg', transparent=True)

ax.imshow(masks[:, :, 2], cmap='gray')
fig.savefig('D:\\dev\\GeoGAN\\Seminar Report\\Images\\GGAN\\Results\\'+str(k)+'-Trees.jpg', transparent=True)

ax.imshow(masks[:, :, 3], cmap='gray')
fig.savefig('D:\\dev\\GeoGAN\\Seminar Report\\Images\\GGAN\\Results\\'+str(k)+'-Water.jpg', transparent=True)


noise = generate_noise((1, 512, 512, 1))
ip = X_test[k:k+1,:,:,:]
ip[:, :, :, 3:] = 0
g = gen_model.predict( [ip, noise] ).reshape((512, 512, 3))
ax.imshow((g+1)/2)
fig.savefig('D:\\dev\\GeoGAN\\Seminar Report\\Images\\GGAN\\Results\\'+str(k)+'-gen-output-allzero-420.jpg', transparent=True)




for k in range(0, 48):
    noise = generate_noise((1, 512, 512, 1))
    
    #ip = np.concatenate((X_test[k:k+1, :, :, 0:3], X_test[infra_id:infra_id+1, :, :, 3:4]), axis=-1)
    #ip = np.concatenate((ip, X_test[road:road+1, :, :, 4:5]), axis=-1)
    #ip = np.concatenate((ip, X_test[trees:trees+1, :, :, 5:6]), axis=-1)
    #ip = np.concatenate((ip, X_test[water:water+1, :, :, 6:7]), axis=-1)
    
    ip = X_test[k:k+1,:,:,:]
    
    #ip[:, :, :, -1] = 0
    
    g = gen_model.predict( [ip, noise] ).reshape((512, 512, 3))
    
    plt.figure()
    
    plt.subplots_adjust(left = 0, right=1, wspace = 0.1, hspace = 0, bottom=0, top=1)
    
    plt.subplot(251).set_title('Input Image')
    plt.subplot(251).axis('off')
    plt.imshow(((ip[0, :, :, 0:3])+1)/2)
    
    plt.subplot(252).set_title('Infra Mask')
    plt.subplot(252).axis('off')
    plt.imshow(ip[0, :, :, 3], cmap='gray')
    
    plt.subplot(253).set_title('Road Mask')
    plt.subplot(253).axis('off')
    plt.imshow(ip[0, :, :, 4], cmap='gray')
    
    plt.subplot(254).set_title('Trees Mask')
    plt.subplot(254).axis('off')
    plt.imshow(ip[0, :, :, 5], cmap='gray')
    
    plt.subplot(255).set_title('Water Mask')
    plt.subplot(255).axis('off')
    plt.imshow(ip[0, :, :, 6], cmap='gray')
    
    plt.subplot(257).set_title('Expected')
    plt.subplot(257).axis('off')
    plt.imshow(((Y_test[k, :, :, :])+1)/2)
    
    plt.subplot(259).set_title('Generated')
    plt.subplot(259).axis('off')
    plt.imshow((g+1)/2)

    plt.savefig('D:/dev/GeoGAN/Results/420/Train/'+str(k), dpi=my_dpi)
    print(k)