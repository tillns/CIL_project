"""Variational Autoencoder Model

This file contains a variational autoencoder model which is used to generate 28x28 images of stars.

"""


from __future__ import absolute_import, division, print_function, unicode_literals

# !pip install -q tensorflow-gpu==2.0.0-alpha0
import tensorflow as tf


class StarVAE(tf.keras.Model):
  
    def __init__(self, latent_dim):
        super(StarVAE, self).__init__()
    
        self.latent_dim = latent_dim
    
        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(500, activation=None),
                tf.keras.layers.BatchNormalization(momentum=0.99),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(latent_dim + latent_dim) 
            ]
        )
        
        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(500, activation=None),
                tf.keras.layers.BatchNormalization(momentum=0.99),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(28 * 28 * 1, activation=None),
                tf.keras.layers.Reshape(target_shape=(28, 28, 1))
            ]
        )

        
        def encode(self, x):
            
            mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
            
            return mean, logvar

  
        def decode(self, z, apply_sigmoid=False):

            logits = self.generative_net(z)
    
            if apply_sigmoid:
                probs = tf.sigmoid(logits)
                return probs
            
            return logits
  
  
        def reparameterize(self, mean, logvar):
    
            eps = tf.random.normal(shape=mean.shape)
        
            return eps * tf.exp(logvar * .5) + mean
  
  
        def sample(self, eps=None):
            
            if eps is None:
                eps = tf.random.normal(shape=(100, self.latent_dim))
            
            return self.decode(eps)
  
  
        def save_ckpt_generative(self, path):
            
            self.generative_net.save_weights(path)
      
    
        def load_ckpt_generative(self, path):
            
            self.generative_net.load_weights(path)
            
            