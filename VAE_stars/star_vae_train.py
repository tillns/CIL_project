"""Variational Autoencoder Train

This file contains functions to train the variational autoencoder model.

"""


from __future__ import absolute_import, division, print_function, unicode_literals

from star_vae import StarVAE
from star_vae_utils import load_train_test_dataset

# !pip install -q tensorflow-gpu==2.0.0-alpha0
import tensorflow as tf
import numpy as np

from argparse import ArgumentParser


# File paths

parser_vae = ArgumentParser()

parser_vae.add_argument("--path_csv", default="/cluster/home/hannepfa/cosmology_aux_data/labeled.csv", type=str)
parser_vae.add_argument("--dir_labeled_images", default="/cluster/home/hannepfa/cosmology_aux_data/labeled/", type=str)
parser_vae.add_argument("--path_json_generative", 
                        default="/cluster/home/hannepfa/CIL_project/VAE_stars/json_generative/model_config.json", type=str)
parser_vae.add_argument("--path_ckpt_generative", 
                        default="/cluster/home/hannepfa/CIL_project/VAE_stars/ckpt_generative/checkpoint", type=str)
parser_vae.add_argument("--frac_train", default=0.9, type=float)

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
  
    frac = float(curr_epoch) / float(num_epochs)
  
    return 1.0 / (1.0 + tf.exp(_c1 * (frac - _c2)))
        
        
def _compute_KL_divergence(mean, logvar, raxis=1):
    
    return tf.reduce_sum(-0.5 * (1 + logvar - mean ** 2 - tf.exp(logvar)), axis=raxis)
    
    
def _compute_loss(model, x, annealing_factor):
    
    mean, logvar = model.encode(x)
    
    # KL divergence
    KL_div = _compute_KL_divergence(mean, logvar)

    # reconstruction error
    z = model.reparameterize(mean, logvar)
  
    x_logit = model.decode(z)
    
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    
    
    # NOTE: update metrics
    _KL_div_mean.update_state(KL_div)
    _reconstr_error_mean.update_state(logpx_z)
  

    # NOTE: negation causes ascent -> descent
    # NOTE: reduce mean is expectation over batch of data
    return -tf.reduce_mean(logpx_z - annealing_factor * KL_div)


def _compute_gradients(model, x, annealing):
  
    with tf.GradientTape() as tape:
        loss = _compute_loss(model, x, annealing)
  
    return tape.gradient(loss, model.trainable_variables)


def _apply_gradients(optimizer, gradients, variables):
    
    optimizer.apply_gradients(zip(gradients, variables))

    
def train_star_vae(path_csv, path_labeled_images, frac_train, model):
        
    KL_div_train = [] 
    reconstr_error_train = []
    
    KL_div_test = [] 
    reconstr_error_test = []    
        
        
    train_dataset, test_dataset = load_train_test_dataset(
        path_csv, path_labeled_images, frac_train, _batch_size)
            
            
    for epoch in range(1, _epochs + 1):
            
        annealing_factor = _compute_annealing_factor(epoch, _epochs)
            
        
        for x in train_dataset: # train dataset
    
            gradients = _compute_gradients(model, x, annealing_factor)
            _apply_gradients(_optimizer, gradients, model.trainable_variables)
            
        KL_div_train.append(_KL_div_mean.result())
        reconstr_error_train.append(_reconstr_error_mean.result())
            
        # NOTE: reset metrics
        _KL_div_mean.reset_states()
        _reconstr_error_mean.reset_states()
        
        
        for x in test_dataset: # test dataset
                
            _compute_loss(model, x, annealing_factor)
        
        KL_div_test.append(_KL_div_mean.result())
        reconstr_error_test.append(_reconstr_error_mean.result())
            
        # NOTE: reset metrics
        _KL_div_mean.reset_states()
        _reconstr_error_mean.reset_states()
            
            
    return KL_div_train, reconstr_error_train, KL_div_test, reconstr_error_test
            
            
def save_ckpt_generative(model, path_ckpt):
    
    model.save_weights(path_ckpt)

    
def save_json_generative(model, path_json):
    
    json_config = model.to_json()
    
    with open(path_json, 'w') as json_file:
        
        json_file.write(json_config)
    
        
if __name__ == "__main__":
    
    arguments = parser_vae.parse_args()
    
    path_csv = arguments.path_csv
    dir_labeled_images = arguments.dir_labeled_images
    path_json_generative = arguments.path_json_generative
    path_ckpt_generative = arguments.path_ckpt_generative
    frac_train = arguments.frac_train
    
    
    model = StarVAE(_latent_dim)
        
    train_star_vae(path_csv, dir_labeled_images, frac_train, model)
    
    
    save_json_generative(model.generative_net, path_json_generative)
    
    save_ckpt_generative(model.generative_net, path_ckpt_generative)
    
    
    
    
    
    
    