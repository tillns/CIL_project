"""
This is a simple deep convolutional autoencoder that serves the purpose to find a meaningful representation of the star
patches. This representation is then used in the kmeans.py file to cluster the images. For that, the saved encoder model
can be found in the latest folder in checkpoints. The autoencoder probably overfits a lot without augmentation, but this
is not bad since only a distinguishable representation of the given data is required without any need for generality.

Input is --image_dir the directory containing the 28x28 images without any labels.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

# Helper libraries
import numpy as np
import os
from PIL import Image
from datetime import datetime
import yaml
import argparse


def load_dataset(image_directory, image_size, image_channels):
    images = []
    for img_name in sorted(os.listdir(image_directory)):
        img = Image.open(os.path.join(image_directory, img_name)).resize((image_size, image_size))
        img_np = np.array(img, dtype=np.float32).reshape((image_size, image_size, image_channels))/255
        images.append(img_np)
    return np.stack(images)


if __name__ == '__main__':
    home_dir = os.path.expanduser("~")
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', '--image_dir', type=str, default=os.path.join(home_dir,
                        "CIL_project/extracted_stars/labeled1_and_scoredover3"))
    args = parser.parse_args()
    with open("config.yaml", 'r') as stream:
        conf = yaml.full_load(stream)

    image_size = conf['image_size']
    image_channels = 1
    ae_dir = os.path.join(home_dir, "CIL_project/AE_plus_KMeans")
    image_directory = args.image_dir
    conf['image_dir'] = image_directory

    period_to_save_cp = conf['period_to_save_cp']
    epochs = conf['epochs']
    learning_rate = conf['lr']
    batch_size = conf['batch_size']

    # checkpoint callback
    checkpoint_dir = os.path.join(ae_dir, "checkpoints")
    cp_dir_time = os.path.join(checkpoint_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(cp_dir_time):
        os.makedirs(cp_dir_time)
    cp_path = os.path.join(cp_dir_time, "cp-{epoch:04d}.ckpt")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(cp_path, save_weights_only=True,
                                                     verbose=1, period=period_to_save_cp)

    # tensorboard callback
    tb_path = os.path.join(cp_dir_time, "summary")
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_path)

    # Model architecture taken from https://www.kaggle.com/vikramtiwari/autoencoders-using-tf-keras-mnist
    input_img = tf.keras.layers.Input(shape=(image_size, image_size, image_channels))
    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    decoded = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = tf.keras.models.Model(input_img, decoded)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    autoencoder.compile(
        loss='binary_crossentropy',
        optimizer=optimizer
    )
    # only necessary when neither batch norm nor dropout is used.
    # See https://stackoverflow.com/questions/52107555/different-loss-function-for-validation-set-in-keras
    autoencoder.outputs[0]._uses_learning_phase = True
    autoencoder.summary()

    json_config = autoencoder.to_json()
    with open(os.path.join(cp_dir_time, 'ae_config.json'), 'w') as json_file:
        json_file.write(json_config)

    with open(os.path.join(cp_dir_time, 'model_summary.txt'), 'w') as file:
        autoencoder.summary(print_fn=lambda mylambda: file.write(mylambda + '\n'))

    cp_command = 'cp {} {}'.format(os.path.join(ae_dir, "{}"), cp_dir_time)
    os.system(cp_command.format("ae.py"))
    os.system(cp_command.format("config.yaml"))

    train_images = load_dataset(image_directory, image_size, image_channels)
    autoencoder.fit(train_images, train_images, batch_size=batch_size, epochs=epochs,
                    callbacks=[tensorboard_callback, cp_callback], shuffle=True)
    encoder = tf.keras.models.Model(input_img, encoded)
    json_config = encoder.to_json()
    with open(os.path.join(cp_dir_time, 'encoder_config.json'), 'w') as json_file:
        json_file.write(json_config)
    


