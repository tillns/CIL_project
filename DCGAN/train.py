
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import model
import util

imgs = load_labeled_images()

generator = make_generator_model()
discriminator = make_discriminator_model()


# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(lr=2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(lr=2e-4, beta_1=0.5)


# TODO: save checkpoint data
checkpoint_dir = "/cluster/home/hannepfa/checkpoints"

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
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
  
    num_batches = imgs.shape[0] // batch_size
  
    for epoch in range(epochs):
    
        for iteration in range(num_batches):    
    
            images = imgs[iteration * batch_size : (iteration + 1) * batch_size]
                      
            train_step(images, batch_size)
        
        # TODO: save checkpoint data
        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        
epochs = 65
batch_size = 32

train(epochs, batch_size)      

  