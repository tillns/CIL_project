"""Star Image Generation

This script loads the pretrained variational autoencoder model and generates 100 star images

"""


from __future__ import absolute_import, division, print_function, unicode_literals

from star_vae_train import parser_vae

import tensorflow as tf
import numpy as np

import cv2


if __name__ == "__main__":
    
    arguments = parser_vae.parse_args()
    
    
    path_json_generative = arguments.path_json_generative
    path_ckpt_generative = arguments.path_ckpt_generative
    
    output_dir_generated_images = arguments.output_dir_generated_images
    
    
    with open(path_json_generative) as json_file:
        
        json_config = json_file.read()

    
    generative_net = tf.keras.models.model_from_json(json_config)
    
    generative_net.load_weights(path_ckpt_generative)
    
    
    num_samples = 100
            
    # NOTE: latent dimension is explicitly set
    z = tf.random.normal(shape=(num_samples, 16))
    
    samples_logits = generative_net(z)
    
    for j in range(100):
        
        sample = (np.array(tf.sigmoid(samples_logits[j, :, :, :])).reshape(28, 28) * 255).astype(np.uint8)
        
        cv2.imwrite(output_dir_generated_images + "image" + str(j) + ".png", sample)
        
        
        
        
        
        
    
    
    
    