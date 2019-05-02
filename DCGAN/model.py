
import tensorflow as tf
from tensorflow.keras import layers


def make_generator_model():
    
    model = tf.keras.Sequential()
    
    model.add(layers.Dense(63 * 63 * 64, use_bias=True, 
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.0002, seed=None),
        bias_initializer=tf.keras.initializers.Constant(0.0),
        input_shape=(100,)))
    model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5))
    model.add(layers.ReLU())
    model.add(layers.Reshape((63, 63, 64)))
    
    assert model.output_shape == (None, 63, 63, 64) # note: None is the batch size

    model.add(layers.Conv2DTranspose(
        32, (5, 5), strides=(2, 2), padding="same", use_bias=False, output_padding=0,
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.0002, seed=None)))
    model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5))
    model.add(layers.ReLU())
    
    assert model.output_shape == (None, 125, 125, 32)
    
    model.add(layers.Conv2DTranspose(
        16, (5, 5), strides=(2, 2), padding="same", use_bias=False, 
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.0002, seed=None)))
    model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5))
    model.add(layers.ReLU())
    
    assert model.output_shape == (None, 250, 250, 16)
    
    model.add(layers.Conv2DTranspose(
        8, (5, 5), strides=(2, 2), padding="same", use_bias=False, 
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.0002, seed=None)))
    model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5))
    model.add(layers.ReLU())
    
    assert model.output_shape == (None, 500, 500, 8)
    
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    
    assert model.output_shape == (None, 1000, 1000, 1)

    return model


def make_discriminator_model():
    
    model = tf.keras.Sequential()
    
    model.add(layers.Conv2D(
        32, (5, 5), strides=(2, 2), padding='same', input_shape=[1000, 1000, 1], use_bias=False, 
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.0002, seed=None)))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    assert model.output_shape == (None, 500, 500, 32)

    model.add(layers.Conv2D(
        64, (5, 5), strides=(2, 2), padding='same', use_bias=False, 
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.0002, seed=None)))
    model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    assert model.output_shape == (None, 250, 250, 64)
    
    model.add(layers.Conv2D(
        128, (5, 5), strides=(2, 2), padding='same', use_bias=False, 
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.0002, seed=None)))
    model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    assert model.output_shape == (None, 125, 125, 128)
    
    model.add(layers.Conv2D(
        256, (5, 5), strides=(2, 2), padding='same', use_bias=False, 
        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.0002, seed=None)))
    model.add(layers.BatchNormalization(momentum=0.9, epsilon=1e-5))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    assert model.output_shape == (None, 63, 63, 256)

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


def discriminator_loss(real_output, fake_output):
    
    # This method returns a helper function to compute cross entropy loss
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    
    # This method returns a helper function to compute cross entropy loss
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    return cross_entropy(tf.ones_like(fake_output), fake_output)

