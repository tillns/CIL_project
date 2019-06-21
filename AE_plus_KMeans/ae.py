from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and tf.keras
import tensorflow as tf


# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os
from random import randint
import PIL
from PIL import Image
import sys
import math
from datetime import datetime
from random import shuffle
from sklearn.linear_model import LinearRegression
import csv
import yaml
import gc



def get_random_indices(begin=0, end=9600, num_indices=25):
    list1 = list(range(begin, end))
    shuffle(list1)
    return list1[:num_indices]


def isInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False



with open("config.yaml", 'r') as stream:
    conf = yaml.full_load(stream)


image_size = conf['image_size']
image_channels = 1
home_dir = os.path.expanduser("~")
ae_dir = os.path.join(home_dir, "CIL_project/AE_plus_KMeans")

image_directory = os.path.join(home_dir, "CIL_project/extracted_stars_Hannes")
period_to_save_cp = conf['period_to_save_cp']
tot_num_epochs = conf['tot_num_epochs']



def load_dataset():
    images = []
    for img_name in sorted(os.listdir(image_directory)):
        img = Image.open(os.path.join(image_directory, img_name)).resize((image_size, image_size))
        img_np = np.array(img, dtype=np.float32).reshape((image_size, image_size, image_channels))/255

        images.append(img_np)
    return np.stack(images)




class FactorLayer(tf.keras.layers.Layer):
    def __init__(self, factor):
        super(FactorLayer, self).__init__()
        self.factor = factor

    def call(self, x):
        return self.factor * x

    def get_config(self):
        return {'factor': self.factor}


def getNormLayer(norm_type='batch', momentum=0.9, epsilon=1e-5):
    if norm_type == 'batch':
        return tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)
    return FactorLayer(1)


class SigmoidLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(SigmoidLayer, self).__init__()

    def call(self, x):
        return tf.keras.activations.sigmoid(x)

    def get_config(self):
        return {}


class CrossEntropy(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        from tensorflow.python.ops import math_ops
        from tensorflow.python.framework import ops
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        return - math_ops.log(1 - math_ops.abs(y_true - y_pred) / label_range)


class CustomLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(CustomLoss, self).__init__()
        train_loss_type = conf['train_loss_type']
        if train_loss_type == "CE":
            self.train_loss = CrossEntropy()  # from_logits=True ?
        elif train_loss_type == "MSE":
            self.train_loss = tf.keras.losses.MeanSquaredError()
        else:
            self.train_loss = tf.keras.losses.MeanAbsoluteError()

        val_loss_type = conf['val_loss_type']
        if val_loss_type == "CE":
            self.val_loss = CrossEntropy()  # from_logits=True ?
        elif val_loss_type == "MSE":
            self.val_loss = tf.keras.losses.MeanSquaredError()
        else:
            self.val_loss = tf.keras.losses.MeanAbsoluteError()

    def call(self, y_true, y_pred):
        return tf.keras.backend.in_train_phase(self.train_loss(y_true, y_pred), self.val_loss(y_true, y_pred))


# checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
checkpoint_dir = os.path.join(ae_dir, "checkpoints")


cp_dir_time = os.path.join(checkpoint_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
if not os.path.exists(cp_dir_time):
    os.makedirs(cp_dir_time)
cp_path = os.path.join(cp_dir_time, "cp-{epoch:04d}.ckpt")
cp_callback = tf.keras.callbacks.ModelCheckpoint(cp_path, save_weights_only=True,
                                                 verbose=1, period=period_to_save_cp)

tb_path = os.path.join(cp_dir_time, "summary")
if not os.path.exists(tb_path):
    os.makedirs(tb_path)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_path)

"""## Train the model

As the model trains, the loss and accuracy metrics are displayed. This model reaches an accuracy of about 0.88 (or 88%) on the training data.
"""
learning_rate = conf['lr']
optimizer = tf.keras.optimizers.Adam(learning_rate)

"""As I see it, there is a bug in the keras library that forbids the labels y from training and test set to have different unique values (which is here the case because y is a continuous label)"""

batch_size = conf['batch_size']

input_img = tf.keras.layers.Input(shape=(28, 28, 1)) # adapt this if using `channels_first` image data format

x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(x)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
decoded = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = tf.keras.models.Model(input_img, decoded)
autoencoder.compile(
    #loss=CustomLoss(),  # keras.losses.mean_squared_error
    loss='binary_crossentropy',
    optimizer=optimizer
)
# only necessary when neither batch norm nor dropout is used.
# See https://stackoverflow.com/questions/52107555/different-loss-function-for-validation-set-in-keras
autoencoder.outputs[0]._uses_learning_phase = True
autoencoder.summary()

# model.save(os.path.join(cp_dir_time, 'model.h5'))
json_config = autoencoder.to_json()
with open(os.path.join(cp_dir_time, 'ae_config.json'), 'w') as json_file:
    json_file.write(json_config)

with open(os.path.join(cp_dir_time, 'model_summary.txt'), 'w') as file:
    autoencoder.summary(print_fn=lambda mylambda: file.write(mylambda + '\n'))

cp_command = 'cp {} {}'.format(os.path.join(ae_dir, "{}"), cp_dir_time)
os.system(cp_command.format("ae.py"))
os.system(cp_command.format("config.yaml"))

train_images = load_dataset()
# train the network

autoencoder.fit(train_images, train_images, batch_size=batch_size, epochs=tot_num_epochs,
                callbacks=[tensorboard_callback, cp_callback], shuffle=True)
encoder = tf.keras.models.Model(input_img, encoded)
json_config = encoder.to_json()
with open(os.path.join(cp_dir_time, 'encoder_config.json'), 'w') as json_file:
    json_file.write(json_config)
    


