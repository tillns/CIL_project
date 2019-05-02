
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import os

import time

from util import load_labeled_images
from model import make_generator_model
from model import make_discriminator_model
from model import generator_loss
from model import discriminator_loss


path_csv = "/cluster/home/hannepfa/cosmology_aux_data/labeled.csv"
path_labeled_images = "/cluster/home/hannepfa/cosmology_aux_data/labeled"

path_checkpoint_dir = "/cluster/home/hannepfa/checkpoints"


imgs = load_labeled_images(path_csv, path_labeled_images, augment=False)


generator = make_generator_model()
discriminator = make_discriminator_model()


generator_optimizer = tf.keras.optimizers.Adam(lr=2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(lr=2e-4, beta_1=0.5)


checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, 
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images, batch_size):
    noise = tf.random.normal([batch_size, 100]) # dim(z) = 100

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    
def train(epochs, batch_size):
  
    num_batches = len(imgs) // batch_size
  
    for epoch in range(epochs):
    
        for iteration in range(num_batches):    
    
            batch = imgs[iteration * batch_size : (iteration + 1) * batch_size]
                      
            train_step(batch, batch_size)
            
        
        if (epoch + 1) % 20 == 0:
            
            checkpoint.save(file_prefix=os.path.join(path_checkpoint_dir, "ckpt"))


epochs = 100
batch_size = 24

train(epochs, batch_size)
        

  