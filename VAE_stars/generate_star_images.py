"""Star Image Generation

This script loads the pretrained variational autoencoder model and generates 100 star images
"""

from __future__ import absolute_import, division, print_function, unicode_literals

from star_vae import StarVAE
from star_vae_train import parser_vae
from star_vae_train import train_star_vae

import tensorflow as tf

import numpy as np
import cv2

import os


if __name__ == "__main__":

    arguments = parser_vae.parse_args()

    path_ckpt_generative_stable = os.path.join(os.path.dirname(__file__), arguments.path_ckpt_generative_stable)
    output_dir_generated_images = os.path.join(os.path.dirname(__file__), arguments.output_dir_generated_images)


    model = StarVAE(16) # NOTE: latent dimension is explicitly set

    generative_net = model.generative_net

    generative_net.load_weights(path_ckpt_generative_stable)
    
    
    if not os.path.exists(output_dir_generated_images):
        os.makedirs(output_dir_generated_images)

    num_samples = 100

    z = tf.random.normal(shape=(num_samples, 16)) # NOTE: latent dimension is explicitly set

    samples_logits = generative_net(z)

    for j in range(100):

        sample = (np.array(tf.sigmoid(samples_logits[j, :, :, :])).reshape(28, 28) * 255).astype(np.uint8)

        cv2.imwrite(os.path.join(output_dir_generated_images, "image" + str(j) + ".png"), sample)
        
