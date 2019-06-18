"""Variational Autoencoder Model

This file contains a variational autoencoder model which is used to generate 28x28 images of stars.

"""


from __future__ import absolute_import, division, print_function, unicode_literals

# !pip install -q tensorflow-gpu==2.0.0-alpha0
import tensorflow as tf


class StarVAE(tf.keras.Model):
    
    """Star VAE
    
    This class represents a variational autoencoder model for 28x28 images of stars. This class is based on
    the CVAE tutorial on the TensorFlow website (see https://www.tensorflow.org/beta/tutorials/generative/cvae).

    This class contains the following public functions:
        #encode
        #decode
        #reparameterize
        #sample
        #save_ckpt
        #load_ckpt
    """
  
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
        """Encodes a batch of images into multivariate normal distributions in latent space.
                
        Parameters
        ----------
        x : tf.Tensor
            The batch of images
                
                
        Returns
        -------
        mean : tf.Tensor
            The mean vectors of the distributions in latent space
        logvar: tf.Tensor
            The logarithm of the variance vectors of the distributions in latent space
                
        """
            
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
            
        return mean, logvar
        
        
    def reparameterize(self, mean, logvar):
            
        """Samples latent vectors from a batch of multivariate normal distributions. This is the 
        implementation of the reparameterization trick by Kingma and Welling (https://arxiv.org/abs/1312.6114).
                
        Parameters
        ----------
        mean : tf.Tensor
            A batch of mean values (vectors)
        logvar : tf.Tensor
            A batch of logarithms of variances (vectors)
            
                
        Returns
        -------
        eps : tf.Tensor
            A batch of vectors in latent space, sampled from the batch of multivariate normal distributions
                
        """
    
        eps = tf.random.normal(shape=mean.shape)
        
        return eps * tf.exp(logvar * .5) + mean
        
  
    def decode(self, z, apply_sigmoid=False):
        
        """Decodes a batch of vectors in latent space to images. The pixels of the images are 
        assumed to be Bernoulli distributed random variables.
                
        Parameters
        ----------
        z : tf.Tensor
            The batch of vectors in latent space
        apply_sigmoid : bool, optimal
            If set to True, the function returns probabilities between 0 and 1 for each pixel
            
                
        Returns
        -------
        logits : tf.Tensor
            Logarithmic probabilities for each pixel. If apply_sigmoid is set to True, 
            actual probabilites between 0 and 1 are returned.
                
        """

        logits = self.generative_net(z)
    
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
            
        return logits
  
  
    def sample(self):
            
        """Samples 100 images by decoding 100 random normally distributed latent vectors.
                
        Returns
        -------
        samples : tf.Tensor
            100 images sampled from the model
                
        """
            
        eps = None
            
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
            
        return self.decode(eps)
        
        
    def save_ckpt(self, path_ckpt):
            
        """Saves the model weights to the specified checkpoint file.
                
        Parameters
        ----------
        path_ckpt : str
            The path to the checkpoint including filename without file extension.
                
        """
            
        self.save_weights(path_ckpt)
            
        
    def load_ckpt(self, path_ckpt):
            
        """Loads the model weights from the specified checkpoint file.
                
        Parameters
        ----------
        path_ckpt : str
            The path to the checkpoint including filename without file extension.
                
        """
            
        self.load_weights(path_ckpt)
            
            