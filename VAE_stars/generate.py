"""Star Image Generation

This script loads the pretrained variational autoencoder model and generates 100 star images

"""


from __future__ import absolute_import, division, print_function, unicode_literals

from star_vae import StarVAE
from star_vae_train import parser_vae

import tensorflow as tf
import numpy as np

import cv2


if __name__ == "__main__":
    
    arguments = parser_vae.parse_args()
    
    
    path_ckpt_generative = arguments.path_ckpt_generative
 
    output_dir_generated_images = arguments.output_dir_generated_images

    
    model = StarVAE(16) # NOTE: latent dimension is explicitly set 
    generative_net = model.generative_net

    generative_net.load_weights(path_ckpt_generative)
    
  
    num_samples = 100    
    
<<<<<<< HEAD
    z = tf.random.normal(shape=(num_samples, 16)) # NOTE: latent dimension is explicitly set 
=======
    num_samples = 100
            
    # NOTE: latent dimension is explicitly set
    z = tf.random.normal(shape=(num_samples, 16))
>>>>>>> 27d254335dcf658fd5b534021562c4e256c0d810
    
    samples_logits = generative_net(z)
    
    for j in range(100):
        
        sample = (np.array(tf.sigmoid(samples_logits[j, :, :, :])).reshape(28, 28) * 255).astype(np.uint8)
        
        cv2.imwrite(output_dir_generated_images + "image" + str(j) + ".png", sample)
        
        