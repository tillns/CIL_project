"""Variational Autoencoder Train

This file contains a class to train the variational autoencoder model.

"""


from __future__ import absolute_import, division, print_function, unicode_literals

from star_vae import StarVAE
from star_vae_utils import load_train_test_dataset

# !pip install -q tensorflow-gpu==2.0.0-alpha0
import tensorflow as tf

import numpy as np


class StarVAETrainer():
    
    def __init__(self, latent_dim):
        super(VAETrainer, self).__init__()
                
        # Hyperparameters
        self.optimizer = tf.keras.optimizers.Adadelta(lr=1)
        
        self.epochs = 60
        self.batch_size = 100
        
        self.latent_dim = 16 # NOTE: dimension of latent space (bad quality if too small)
        
        self.c1 = 20.0
        self.c2 = 0.5
        
        # Model
        self.model = StarVAE(self.latent_dim)
        
        # TensorBoard summaries
        self.KL_div_mean = tf.keras.metrics.Mean('KL_div_mean', dtype=tf.float32)
        self.reconstr_error_mean = tf.keras.metrics.Mean('reconstr_error_mean', dtype=tf.float32)
        
        
    def _compute_gradients(model, x, annealing):
  
        with tf.GradientTape() as tape:
            loss = compute_loss(model, x, annealing)
  
        return tape.gradient(loss, model.trainable_variables)


    def _apply_gradients(optimizer, gradients, variables):
  
        optimizer.apply_gradients(zip(gradients, variables))
    
    
    def _compute_annealing_factor(curr_epoch, num_epochs):
  
        frac = float(curr_epoch) / float(num_epochs)
  
        return 1.0 / (1.0 + tf.exp(-self.c1 * (frac - self.c2)))
        
        
    def _compute_KL_divergence(mean, logvar, raxis=1):
    
        return tf.reduce_sum(-0.5 * (1 + logvar - mean ** 2 - tf.exp(logvar)), axis=raxis)
    
    
    def _compute_loss(self, model, x, annealing_factor):
    
        mean, logvar = model.encode(x)
    
        # KL divergence
        KL_div = _compute_KL_divergence(mean, logvar)

        # reconstruction error
        z = model.reparameterize(mean, logvar)
  
        x_logit = model.decode(z)
    
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    
    
        # NOTE: TensorBoard summaries
        self.KL_div_mean(KL_div)
        self.reconstr_error_mean(logpx_z)
  

        # NOTE: negation causes ascent -> descent
        # NOTE: reduce mean is expectation over batch of data
        return -tf.reduce_mean(logpx_z - annealing_factor * KL_div)
    
    
    def train(self, path_csv, path_labeled_images, frac_train):
        
        KL_div_train = []; reconstr_error_train = []
        KL_div_test = []; reconstr_error_test = []
        
        
        train_dataset, test_dataset = 
            _load_train_test_dataset(path_csv, path_labeled_images, frac_train, self.batch_size)
            
            
        for epoch in range(1, self.epochs + 1):
            
            annealing_factor = _compute_annealing_factor(epoch, self.epochs)
            
            for x in train_dataset:
    
                gradients = _compute_gradients(self.model, x, annealing_factor)
                _apply_gradients(self.optimizer, gradients, self.model.trainable_variables)
            
            # store losses
            KL_div_train.append(self.KL_div_mean.result())
            reconstr_error_train.append(self.reconstr_error_mean.result())
            
            # reset metrics
            self.KL_div_mean.reset_states()
            self.reconstr_error_mean.reset_states()
            
            for x in test_dataset:
                
                compute_loss(self.model, x, annealing_factor)
            
            # store losses
            KL_div_test.append(self.KL_div_mean.result())
            reconstr_error_test.append(self.reconstr_error_mean.result())
            
            # reset metrics
            self.KL_div_mean.reset_states()
            self.reconstr_error_mean.reset_states()
            
            
    return KL_div_train, reconstr_error_train, KL_div_test, reconstr_error_test
            
            
    def save_trained_model(self, path_ckpt):
        self.model.save_weights(path_ckpt)
        
        
    def load_pretrained_model(self, path_ckpt):
        self.model.load_weights(path_ckpt)
        
              
    def load_stable_pretrained_model(self, path_ckpt_stable):   
        self.model.load_weights(path_ckpt_stable)

