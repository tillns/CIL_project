"""DCGAN Train

This file contains functions to train the DCGAN. The code is based on the DCGAN tutorial on the TensorFlow website (see
https://www.tensorflow.org/alpha/tutorials/generative/dcgan).
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# !pip install -q tensorflow-gpu==2.0.0-alpha0
import tensorflow as tf

import os

from dcgan_utils import load_train_test_dataset

from dcgan_model import make_generator_model
from dcgan_model import make_discriminator_model

from argparse import ArgumentParser


# File paths

parser_dcgan = ArgumentParser()

parser_dcgan.add_argument("--data_directory", type=str, required=True, 
                          help="Required. The directory where the data set is stored.")

parser_dcgan.add_argument("--path_ckpt", type=str, required=False, default="ckpt/checkpoint",
                          help="Optinal. The path to the checkpoints which are stored during training.")
parser_dcgan.add_argument("--path_ckpt_stable", type=str, required=False, default="ckpt_stable", 
                          help="Optional. The path to the checkpoint directory for image generation.")

parser_dcgan.add_argument("--output_dir_generated_images", type=str, required=False, default="generated", 
                          help="Optional. The directory where generated images are written to.")

parser_dcgan.add_argument("--frac_train", type=float, required=False, default=0.9, 
                          help="Optional. The validation split for training.")


# Hyperparameters
_generator_optimizer = tf.keras.optimizers.Adam(lr=2e-4, beta_1=0.5)
_discriminator_optimizer = tf.keras.optimizers.Adam(lr=2e-4, beta_1=0.5)

_epochs = 120
_batch_size = 24 # NOTE: avoid OOM

_latent_dim = 34 # NOTE: avoid OOM


# Metrics
_gen_loss_mean = tf.keras.metrics.Mean('gen_loss_mean', dtype=tf.float32)
_disc_loss_mean = tf.keras.metrics.Mean('disc_loss_mean', dtype=tf.float32)


def _compute_generator_loss(fake_output):
    """Computes the loss of the generator model

    Parameters
    ----------
    fake_output : tf.Tensor
        The output of the discriminator for a batch of generated images


    Returns
    -------
    gen_loss : tf.Tensor
        The computed loss
    """

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

    return gen_loss


def _compute_discriminator_loss(real_output, fake_output):
    """Computes the loss of the discriminator model

    Parameters
    ----------
    real_output : tf.Tensor
        The output of the discriminator for a batch of real images
    fake_output : tf.Tensor
        The output of the discriminator for a batch of generated images


    Returns
    -------
    disc_loss : tf.Tensor
        The computed loss
    """

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    disc_loss = real_loss + fake_loss

    return disc_loss


@tf.function
def train_step(batch_images, generator, discriminator):
    """Trains the generator model and the discriminator model of the DCGAN on a batch of images


    Parameters
    ----------
    batch_images : tf.Tensor
        The batch of images
    generator : tf.keras.Sequential
        The generator model
    discriminator : tf.keras.Sequential
        The discriminator model
    """

    noise = tf.random.normal([_batch_size, _latent_dim]) # dim(z) = _latent_dim


    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        generated_images = generator(noise, training=True)

        real_output = discriminator(batch_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = _compute_generator_loss(fake_output)
        disc_loss = _compute_discriminator_loss(real_output, fake_output)


    gradients_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    _generator_optimizer.apply_gradients(zip(gradients_generator, generator.trainable_variables))
    _discriminator_optimizer.apply_gradients(zip(gradients_discriminator, discriminator.trainable_variables))


def train_dcgan(arguments, generator, discriminator):
    """Trains the DCGAN on a all labeled images and on all scored images with score >= 2.61


    Parameters
    ----------
    arguments : argparse-arguments
        The given command-line arguments
    generator : tf.keras.Sequential
        The generator model
    discriminator : tf.keras.Sequential
        The discriminator model
    """

    train_dataset, test_dataset = load_train_test_dataset(arguments, _batch_size)

    _gen_loss_mean.reset_states()
    _disc_loss_mean.reset_states()

    gen_loss_test = []
    disc_loss_test = []

    checkpoint = tf.train.Checkpoint(generator=generator, discriminator=discriminator)


    for epoch in range(1, _epochs + 1):

        for batch_images in train_dataset:

            train_step(batch_images, generator, discriminator)


        for batch_images in test_dataset:

            noise = tf.random.normal([_batch_size, _latent_dim]) # dim(z) = _latent_dim

            generated_images = generator(noise)

            real_output = discriminator(batch_images)
            fake_output = discriminator(generated_images)

            gen_loss = _compute_generator_loss(fake_output)
            disc_loss = _compute_discriminator_loss(real_output, fake_output)

            _gen_loss_mean.update_state(gen_loss)
            _disc_loss_mean.update_state(disc_loss)

        gen_loss_test.append(_gen_loss_mean.result())
        disc_loss_test.append(_disc_loss_mean.result())

        _gen_loss_mean.reset_states()
        _disc_loss_mean.reset_states()


        if epoch >= 80 and epoch % 5 == 0:

            checkpoint.save(file_prefix=os.path.join(os.path.dirname(__file__), arguments.path_ckpt))


    return gen_loss_test, disc_loss_test


if __name__ == "__main__":

    arguments = parser_dcgan.parse_args()


    generator = make_generator_model()
    discriminator = make_discriminator_model()


    gen_loss_test, disc_loss_test = train_dcgan(arguments, generator, discriminator)

