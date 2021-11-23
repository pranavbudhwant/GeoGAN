# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 15:49:14 2019

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

def l1_loss(y_true, y_pred): #MAE
    return  tf.reduce_mean(tf.abs(y_true - y_pred))

def discriminator_on_generator_loss(y_true, y_pred): #Binary Crossentropy
    #BATCH_SIZE=10 
    return K.mean(K.binary_crossentropy(K.flatten(y_pred), K.flatten(y_true)), axis = -1) 
    #return K.mean(K.binary_crossentropy(K.flatten(y_pred), K.ones_like(K.flatten(y_pred))), axis=-1)

def train(epochs, batch_size):
    Y_fake = np.zeros((batch_size,32,32,1))
    Y_real = np.random.uniform(0.7, 1, (batch_size,32,32,1))
        
    for num_epochs in range(epochs):
        for num_batch in tqdm(range(int(X_train_B.shape[0]/batch_size))):
            
            X_before = X_train_B[num_batch*batch_size : (num_batch + 1 )*batch_size , : , : , :]            
            X_after = X_train_A[num_batch*batch_size : (num_batch + 1 )*batch_size , : , : , : ]
            
            noise = generate_noise((batch_size, 512, 512, 1))
            gen_images = gen_model.predict([X_before, noise])
        
            disc_model.trainable = True
            
            #d_loss_real = disc_model.train_on_batch(np.concatenate([X_before, X_after], axis = -1), Y_real)            
            #d_loss_fake = disc_model.train_on_batch(np.concatenate([X_before, gen_images], axis = -1), Y_fake)
            dloss_real = disc_model.train_on_batch([X_before, X_after], Y_real)
            dloss_fake = disc_model.train_on_batch([X_before, gen_images], Y_fake)
            
            dloss = (0.5 * np.add(dloss_real, dloss_fake))
            
            print('\n\nD_Loss_real: ', dloss_real)
            print('D_Loss_fake: ', dloss_fake)
            print('D_Loss: ', dloss)
            d_loss_real.append(dloss_real)
            d_loss_fake.append(dloss_fake)
            d_loss.append(dloss)
            
            disc_model.trainable = False
            
            for _ in range(2):
                gan_loss = (cgan_model.train_on_batch([X_before, noise], [X_after, Y_real]))
                print('GAN_Loss: ', gan_loss)
                dc_gan_loss.append(gan_loss)
    
        #Save weights of generator and discriminator 
        #path_gen = os.path.join('D:/dev/GeoGAN/Weights', str(num_epochs)+'_epochs_gen.h5')
        #path_disc = os.path.join('D:/dev/GeoGAN/Weights', str(num_epochs)+'_epochs_disc.h5')
        #gen_model.save_weights(path_gen)
        #disc_model.save_weights(path_disc) 


if __name__ == '__main__':
    
    cgan_optimizer = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    gen_optimizer = keras.optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999) 
    loss_weights = [10, 1]
    disc_optimizer = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    gen_model = generator((512, 512, 7), (512, 512, 3), 64)
    disc_model = discriminator((512, 512, 7), (512, 512, 3), 64)
    cgan_model = cGAN_Model(gen_model, disc_model)
    
    print("Generator Summary")
    gen_model.summary()
    
    print("Discriminator Summary")
    disc_model.summary()
    
    print("Complete Model Summary")
    cgan_model.summary()
    
    gen_model.compile(loss = 'mae', optimizer = gen_optimizer)
    disc_model.compile(loss = 'binary_crossentropy', optimizer = disc_optimizer)
    cgan_model.compile(loss = [mean_absolute_error, binary_crossentropy], loss_weights = loss_weights, optimizer = cgan_optimizer)
    
    print(cgan_model.metrics)
    
    #Data Preprocessing:
    X_train_B = np.load('D:/dev/GeoGAN/DataSet/Data/Resized/npy/OR/Shuffled/X_train_01Norm.npy')
    X_train_A = np.load('D:/dev/GeoGAN/DataSet/Data/Resized/npy/OR/Shuffled/Y_train_01Norm.npy')
    
    #Train function : 
    # 1. Generate images 
    # 2. Train Discriminator as per Conditional GANs 
    # 3. discriminator.trainable = False
    # 4. Train DCGAN model 
    
    d_loss = []
    d_loss_real = []
    d_loss_fake = []
    dc_gan_loss = []
    dc = 0 
    d_l = 0
    
    train(2, 1)
    
    import matplotlib.pyplot as plt
    plt.plot(d_loss_real, label='d_loss_real')
    plt.plot(d_loss_fake, label='d_loss_fake')
    plt.plot(d_loss, label='d_loss')
    plt.plot(dc_gan_loss)
    plt.show()