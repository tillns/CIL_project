"""Complete Image Creation

This script loads the pretrained variational autoencoder model and creates 100 complete images

"""


from __future__ import absolute_import, division, print_function, unicode_literals

from star_vae import StarVAE
from star_vae_train import parser_vae

# !pip install -q tensorflow-gpu==2.0.0-alpha0
import tensorflow as tf
import numpy as np

from random import gauss, randint

import cv2


# Parameters for number of stars per image

_mean = 6.27

_std_dev = 2.64


def _nonnegative_int(decimal):
    
    if decimal <= 0:
        return 0
    
    if decimal - int(decimal) < 0.5:
        return int(decimal)
    
    return int(decimal) + 1


def create_complete_images(output_dir, generative_net, background=0, num_images=100):
    
    
    for i in range(num_images):
        
        image_padded = np.full((1028, 1028), background).astype(np.uint8)
                
        
        num_stars = _nonnegative_int(gauss(_mean, _std_dev))
                
        for j in range(num_stars):
            
            x = randint(0, 1000 - 1)
            y = randint(0, 1000 - 1)
            
            
            z = tf.random.normal(shape=(1, 16)) # NOTE: latent dimension is explicitly set 
            
            star_image = (np.array(tf.sigmoid(generative_net(z))).reshape(28, 28) * 255).astype(np.uint8)
            
            
            image_padded[x : x + 28, y : y + 28] = star_image
            
            
        image = image_padded[14 : 14 + 1000, 14 : 14 + 1000]
            
        cv2.imwrite(output_dir + "image" + str(i) + ".png", image)


if __name__ == "__main__":
    
    arguments = parser_vae.parse_args()
    
    path_ckpt_generative_stable = arguments.path_ckpt_generative_stable
    output_dir_generated_images = arguments.output_dir_generated_images
    
    
    model = StarVAE(16) # NOTE: latent dimension is explicitly set 

    generative_net = model.generative_net
    
    generative_net.load_weights(path_ckpt_generative_stable)
    
    
    create_complete_images(output_dir_generated_images, generative_net)

