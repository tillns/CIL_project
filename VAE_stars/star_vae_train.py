"""Variational Autoencoder Train

This file contains functions to train the variational autoencoder model.
"""


from __future__ import absolute_import, division, print_function, unicode_literals

from star_vae import StarVAE
from star_vae_utils import load_train_test_dataset

import tensorflow as tf

import os

from argparse import ArgumentParser


# File paths

parser_vae = ArgumentParser()

parser_vae.add_argument("--data_directory", type=str, required=True, 
                        help="Required. The directory where the data set is stored.")
parser_vae.add_argument("--path_ckpt_generative", type=str, required=False, default="ckpt_generative/checkpoint", 
                        help="Optinal. The path to the checkpoint which is stored after training.")
parser_vae.add_argument("--path_ckpt_generative_stable", type=str, required=False, default="ckpt_generative_stable/checkpoint", 
                        help="Optional. The path to the checkpoint which is loaded for image generation.")
parser_vae.add_argument("--output_dir_generated_images", type=str, required=False, default="generated", 
                        help="Optional. The directory where generated images are written to.")
parser_vae.add_argument("--frac_train", type=float, required=False, default=0.9, 
                        help="Optional. The validation split for training.")


# Hyperparameters

_optimizer = tf.keras.optimizers.Adadelta(lr=1)

_epochs = 60
_batch_size = 100

_latent_dim = 16 # NOTE: dimension of latent space (bad quality if too small)

_c1 = 20.0
_c2 = 0.5


# Metrics

_KL_div_mean = tf.keras.metrics.Mean('KL_div_mean', dtype=tf.float32)
_reconstr_error_mean = tf.keras.metrics.Mean('reconstr_error_mean', dtype=tf.float32)


def _compute_annealing_factor(curr_epoch, num_epochs):

    """Computes an annealing factor for the current epoch.

    The annealing factor is used as a variable weight for the KL divergence term of the loss
    function. See https://arxiv.org/pdf/1511.06349.pdf for reference.

    Parameters
    ----------
    curr_epoch : int
        The current epoch
    num_epochs : int
        The total number of epochs


    Returns
    -------
    annealing_factor : float
        The annealing factor

    """

    frac = float(curr_epoch) / float(num_epochs)

    annealing_factor = 1.0 / (1.0 + tf.exp(-_c1 * (frac - _c2)))

    return annealing_factor


def _compute_KL_divergence(mean, logvar, raxis=1):

    """Computes the KL divergence of a Gaussian distribution and the standard normal distribution


    Parameters
    ----------
    mean : tf.Tensor
        A batch of mean values (vectors)
    logvar : tf.Tensor
        A batch of logarithms of variances (vectors)


    Returns
    -------
    KL_div : tf.Tensor
        The computed KL divergences

    """

    KL_div = tf.reduce_sum(-0.5 * (1 + logvar - mean ** 2 - tf.exp(logvar)), axis=raxis)

    return KL_div


def _compute_loss(model, x, annealing_factor):

    """Computes the loss of the star variational autoencoder model for a batch of star images


    Parameters
    ----------
    model : tf.keras.Model
        The model
    x : tf.Tensor
        The batch of star images
    annealing_factor : float
        The annealing factor which is a variable weight for the KL divergence term of the loss


    Returns
    -------
    loss : tf.Tensor
        The computed loss

    """

    mean, logvar = model.encode(x)

    # reconstruction error
    z = model.reparameterize(mean, logvar)

    x_logit = model.decode(z)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])

    # KL divergence
    KL_div = _compute_KL_divergence(mean, logvar)


    # NOTE: update metrics
    _reconstr_error_mean.update_state(logpx_z)
    _KL_div_mean.update_state(KL_div)


    # NOTE: negation causes ascent -> descent
    # NOTE: reduce mean is expectation over batch of data
    loss = -tf.reduce_mean(logpx_z - annealing_factor * KL_div)

    return loss


def _compute_gradients(model, x, annealing_factor):

    with tf.GradientTape() as tape:

        loss = _compute_loss(model, x, annealing_factor)

    return tape.gradient(loss, model.trainable_variables)


def _apply_gradients(optimizer, gradients, variables):

    optimizer.apply_gradients(zip(gradients, variables))


def train_star_vae(path_csv, path_labeled_images, frac_train, model):

    """Trains the star variational autoencoder model on the dataset of labeled cosmology images


    Parameters
    ----------
    path_csv : str
        The path to the CSV file containing the image labels
    path_labeled_images : str
        The path to the directory containing the labeled images
    frac_train : int
        The train / test split for the star images
    model : tf.keras.Model
        The model


    Returns
    -------
    KL_div_train : list
        The mean KL divergence loss on the train dataset during each epoch
    reconstr_error_train : list
        The mean reconstruction error loss on the train dataset during each epoch
    KL_div_test : list
        The mean KL divergence loss on the test dataset during each epoch
    reconstr_error_test : list
        The mean reconstruction error loss on the test dataset during each epoch

    """

    KL_div_train = []; reconstr_error_train = []

    KL_div_test = []; reconstr_error_test = []


    train_dataset, test_dataset = load_train_test_dataset(
        path_csv, path_labeled_images, frac_train, _batch_size)


    for epoch in range(1, _epochs + 1):

        annealing_factor = _compute_annealing_factor(epoch, _epochs)


        for x in train_dataset:

            gradients = _compute_gradients(model, x, annealing_factor)
            _apply_gradients(_optimizer, gradients, model.trainable_variables)

        KL_div_train.append(_KL_div_mean.result())
        print("KL div train: {}".format(_KL_div_mean.result()))
        reconstr_error_train.append(_reconstr_error_mean.result())
        print("reconstr error train: {}".format(_reconstr_error_mean.result()))

        # NOTE: reset metrics
        _KL_div_mean.reset_states()
        _reconstr_error_mean.reset_states()


        for x in test_dataset:

            _compute_loss(model, x, annealing_factor)

        KL_div_test.append(_KL_div_mean.result())
        print("KL div test: {}".format(_KL_div_mean.result()))
        reconstr_error_test.append(_reconstr_error_mean.result())
        print("reconstr error test: {}".format(_reconstr_error_mean.result()))

        # NOTE: reset metrics
        _KL_div_mean.reset_states()
        _reconstr_error_mean.reset_states()


    return KL_div_train, reconstr_error_train, KL_div_test, reconstr_error_test


def save_ckpt_generative(model, path_ckpt):

    model.save_weights(path_ckpt)


if __name__ == "__main__":

    arguments = parser_vae.parse_args()

    data_directory = arguments.data_directory
    path_ckpt_generative = arguments.path_ckpt_generative
    frac_train = arguments.frac_train


    model = StarVAE(_latent_dim)
    
    
    path_csv = os.path.join(data_directory, "labeled.csv")
    
    path_labeled_images = os.path.join(data_directory, "labeled")

    
    train_star_vae(path_csv, path_labeled_images, frac_train, model)


    save_ckpt_generative(model.generative_net, os.path.join(os.path.dirname(__file__), path_ckpt_generative))
    
