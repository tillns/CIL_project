"""DCGAN Model

This file contains two functions which create the generator model and the discriminator model. The code is based on the 
DCGAN tutorial on the TensorFlow website (see https://www.tensorflow.org/alpha/tutorials/generative/dcgan). 

The values for most hyperparameters are from the original DCGAN paper and code: 
https://arxiv.org/abs/1511.06434
https://github.com/Newmu/dcgan_code/blob/master/imagenet/load_pretrained.py (original code)
https://github.com/carpedm20/DCGAN-tensorflow/blob/master/ops.py (referenced public TensorFlow implementation)

"""


from __future__ import absolute_import, division, print_function, unicode_literals

# !pip install -q tensorflow-gpu==2.0.0-alpha0
import tensorflow as tf

from tensorflow.keras import layers


def make_generator_model():
    
    """Creates a convolutional neural network which serves as generator model
    
    
    Returns
    -------
    model : tf.keras.Sequential
        The generator model
        
    """
    
    model = tf.keras.Sequential()
    
    
    model.add(layers.Dense(64 * 64 * 42, use_bias=False, 
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.0002, seed=None),
        input_shape=(34,))) # consider changing latent dimension from 100 to e.g. 20
    model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5))
    model.add(layers.ReLU())
    model.add(layers.Reshape((64, 64, 42)))
    
    assert model.output_shape == (None, 64, 64, 42) # NOTE: None is the batch size
    
    
    model.add(layers.UpSampling2D(size=(2, 2), interpolation='nearest'))

    model.add(layers.Conv2D(
        32, (5, 5), strides=(1, 1), padding="same", use_bias=False,
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.0002, seed=None)))
    model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5))
    model.add(layers.ReLU())
    
    assert model.output_shape == (None, 128, 128, 32)
    
    
    model.add(layers.UpSampling2D(size=(2, 2), interpolation='nearest'))
    
    model.add(layers.Conv2D(
        24, (5, 5), strides=(1, 1), padding="same", use_bias=False, 
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.0002, seed=None)))
    model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5))
    model.add(layers.ReLU())
    
    assert model.output_shape == (None, 256, 256, 24)
    
    
    model.add(layers.UpSampling2D(size=(2, 2), interpolation='nearest'))
    
    model.add(layers.Conv2D(
        16, (5, 5), strides=(1, 1), padding="same", use_bias=False, 
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.0002, seed=None)))
    model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5))
    model.add(layers.ReLU())
    
    assert model.output_shape == (None, 512, 512, 16)
    
    
    model.add(layers.UpSampling2D(size=(2, 2), interpolation='nearest'))
    
    model.add(layers.Conv2D(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
    
    assert model.output_shape == (None, 1024, 1024, 1)
    

    return model


def make_discriminator_model():
    
    """Creates a convolutional neural network which serves as discriminator model
    
    
    Returns
    -------
    model : tf.keras.Sequential
        The discriminator model
        
    """
    
    model = tf.keras.Sequential()
    
    
    model.add(layers.Conv2D(
        16, (5, 5), strides=(2, 2), padding='same', input_shape=[1024, 1024, 1], use_bias=False, 
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.0002, seed=None)))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    assert model.output_shape == (None, 512, 512, 16)
    

    model.add(layers.Conv2D(
        24, (5, 5), strides=(2, 2), padding='same', use_bias=False, 
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.0002, seed=None)))
    model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    assert model.output_shape == (None, 256, 256, 24)
    
    
    model.add(layers.Conv2D(
        32, (5, 5), strides=(2, 2), padding='same', use_bias=False, 
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.0002, seed=None)))
    model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    assert model.output_shape == (None, 128, 128, 32)
    
    
    model.add(layers.Conv2D(
        42, (5, 5), strides=(2, 2), padding='same', use_bias=False, 
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.0002, seed=None)))
    model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    assert model.output_shape == (None, 64, 64, 42)
    
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model
