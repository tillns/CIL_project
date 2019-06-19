"""Star Image Generation

This script loads the pretrained variational autoencoder model and generates 100 star images

"""


from __future__ import absolute_import, division, print_function, unicode_literals

from star_vae import StarVAE
from star_vae_train import parser_vae
from star_vae_train import train_star_vae

# !pip install -q tensorflow-gpu==2.0.0-alpha0
import tensorflow as tf

import numpy as np
import cv2
    
    
if __name__ == "__main__":
    
    arguments = parser_vae.parse_args()
    
    path_ckpt_generative_stable = arguments.path_ckpt_generative_stable
    output_dir_generated_images = arguments.output_dir_generated_images

    
    model = StarVAE(16) # NOTE: latent dimension is explicitly set 

    generative_net = model.generative_net
    
    generative_net.load_weights(path_ckpt_generative_stable)
    

    num_samples = 100    
    
    z = tf.random.normal(shape=(num_samples, 16)) # NOTE: latent dimension is explicitly set 
                
    samples_logits = generative_net(z)
    
    for j in range(100):
        
        sample = (np.array(tf.sigmoid(samples_logits[j, :, :, :])).reshape(28, 28) * 255).astype(np.uint8)
        
        cv2.imwrite(output_dir_generated_images + "image" + str(j) + ".png", sample)
        
        