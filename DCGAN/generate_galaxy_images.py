"""Galaxy Image Generation

This script loads the pretrained generator model of the DCGAN and generates 100 galaxy images

"""


from __future__ import absolute_import, division, print_function, unicode_literals

# !pip install -q tensorflow-gpu==2.0.0-alpha0
import tensorflow as tf
import numpy as np

import os

import cv2

from dcgan_train import parser_dcgan

from dcgan_model import make_generator_model
from dcgan_model import make_discriminator_model


def create_galaxy_images(output_dir, generator, num_images=100):
    
    
    for i in range(num_images):
        
        noise = tf.random.normal([1, 34]) # NOTE: dim(z) = 34
        
        generated_image = generator(noise)
        
        
        np_image = (np.array(generated_image[0, 12 : 12 + 1000, 12 : 12 + 1000, :], 
                             dtype=np.float32).reshape(1000, 1000) * 127.5 + 127.5).astype(np.uint8)
        
        cv2.imwrite(os.path.join(output_dir_generated_images, "image" + str(i) + ".png"), np_image)
        

if __name__ == "__main__":

    
    arguments = parser_dcgan.parse_args()
    
    path_ckpt_stable = arguments.path_ckpt_stable
    output_dir_generated_images = arguments.output_dir_generated_images
    
    
    generator = make_generator_model()
    discriminator = make_discriminator_model()
    
    checkpoint = tf.train.Checkpoint(generator=generator, discriminator=discriminator)

    checkpoint.restore(tf.train.latest_checkpoint(path_ckpt_stable))

    
    create_galaxy_images(output_dir_generated_images, generator)
    
    