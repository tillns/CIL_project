"""Classifier

This file trains a CNN classifier and evaluates it against a test set. It also predicts
the scores for a query data set and saves them to a file.

"""

from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and tf.keras
import tensorflow as tf


# Helper libraries
import numpy as np
import os
import PIL
from PIL import Image
import sys
import math
from datetime import datetime
from sklearn.linear_model import LinearRegression
import csv
import yaml
import gc
import argparse
import cv2
from albumentations import (
    Compose, HorizontalFlip, ShiftScaleRotate, VerticalFlip
)
from CustomLayers import ResBlock, Padder, get_custom_objects, getNormLayer
from utils import get_epoch_and_path, get_specific_cp, load_dataset, FFT_augm, Augm_Sequence, \
    get_max_val_fft


class CrossEntropy(tf.keras.losses.Loss):
    # Cross Entropy loss
    def call(self, y_true, y_pred):
        from tensorflow.python.ops import math_ops
        from tensorflow.python.framework import ops
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        return - math_ops.log(1 - math_ops.abs(y_true - y_pred) / label_range)


class CustomLoss(tf.keras.losses.Loss):
    # Defines the loss based on the config file
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


def get_model():
    """Defines and returns the CNN model based on the config file.
    
    Returns
    -------
    model : sequential tf model
        CNN model
    """
    features = conf['features']
    last_features = image_channels
    res = image_size
    resexp = [pow(2, x) for x in range(10)]

    model = tf.keras.Sequential(name='till_model')
    while res > conf['min_res']:
        if res == 250:
            closestexpres = 256
            model.add(Padder(padding=closestexpres - res,
                             input_shape=(res, res, last_features)))  # 250 -> 256
            res = closestexpres
        for i in range(conf['num_blocks_per_res']):
            downsample = (i == 0 and conf['downsample_with_first_conv']) or \
                         (i == conf['num_blocks_per_res'] - 1 and not conf['downsample_with_first_conv'])
            model.add(ResBlock(conf, downsample, features, last_features, input_shape=(res, res, last_features)))
        last_features = features
        if features < conf['max_features']:
            features *= 2
        res = res // conf['downsample_factor']

    model.add(tf.keras.layers.Flatten())
    for i in range(1, conf['num_dense_layers']):
        model.add(tf.keras.layers.Dense(conf['latent_features'], use_bias=conf['use_bias']))
        model.add(getNormLayer(conf['norm_type']))
        model.add(tf.keras.layers.LeakyReLU(alpha=conf['lrelu_alpha']))
        if conf['dropout'] > 0:
            model.add(tf.keras.layers.Dropout(conf['dropout']))
    if conf['num_dense_layers'] == 1:
        model.add(tf.keras.layers.Dropout(conf['dropout']))
    model.add(tf.keras.layers.Dense(1, use_bias=conf['use_bias']))
    # model.add(SigmoidLayer())
    # model.add(FactorLayer(8))  # to make output range from 0 to 8
    return model


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--is_cluster', type=bool, default=False, help='Set to true if code runs on cluster.')
    parser.add_argument('-T', '--test_on_query', type=bool, default=False)
    parser.add_argument('-R', '--restore_ckpt', type=bool, default=False, help="Can't be false if test_on_query is True")
    parser.add_argument('-P', '--ckpt_path', type=str, default=None, help=
                        "Complete path to ckpt file ending with .data_00001...")

    args = parser.parse_args()

    # load in the configurations from the config file
    with open("config.yaml", 'r') as stream:
        conf = yaml.full_load(stream)

    # get the settings from both the command line arguments and the config file
    test_on_query = args.test_on_query
    restore_checkpoint = True if test_on_query else args.restore_ckpt
    image_size = conf['image_size']
    image_channels = 1
    home_dir = os.path.expanduser("~")
    classifier_dir = os.path.join(home_dir, "CIL_project/Classifier")
    scored_directory = os.path.join(home_dir, "dataset/cil-cosmology-2018/cosmology_aux_data_170429/scored")
    label_path = os.path.join(home_dir, "dataset/cil-cosmology-2018/cosmology_aux_data_170429/scored.csv")
    query_directory = os.path.join(home_dir, "dataset/cil-cosmology-2018/cosmology_aux_data_170429/query")
    image_directory = query_directory if test_on_query else scored_directory
    percentage_train = conf['percentage_train'] if not test_on_query else 1
    print("Searching for images in {}".format(image_directory))
    use_dummy_dataset = False
    period_to_save_cp = conf['period_to_save_cp'] if percentage_train == 1 else 1
    tot_num_epochs = conf['tot_num_epochs']
    label_range = 8  # labels go from 0 to 8
    save_np_to_mem = image_size > 250 and not args.is_cluster

    # checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_dir = os.path.join(classifier_dir, "checkpoints/res{}".format(image_size))

    """
    Restoring the checkpoint
    """
    if restore_checkpoint:
        if args.ckpt_path is not None:
            epoch_start, specific_path = get_epoch_and_path(args.ckpt_path)
            if os.path.exists(specific_path):
                specific_path = specific_path[:-len(".data-00000-of-00001")]
            global cp_dir_time
            cp_dir_time = os.path.dirname(specific_path)
        else:
            epoch_start, specific_path = get_specific_cp()
        if specific_path is not None:
            # model_path = os.path.join("/".join(specific_path.split("/")[:-1]), "model.h5")
            model_path = os.path.join("/".join(specific_path.split("/")[:-1]), "model_config.json")
            if not os.path.exists(model_path):
                sys.exit("Couldn't locate model")
            # model = tf.keras.models.load_model(os.path.join(model_path), custom_objects={'Padder': Padder, 'FactorLayer': FactorLayer, 'SigmoidLayer':SigmoidLayer})
            with open(model_path) as json_file:
                json_config = json_file.read()
            custom_objects = get_custom_objects()

            model = tf.keras.models.model_from_json(json_config, custom_objects=custom_objects)
            model.load_weights(specific_path)
            model.summary()
        else:
            print("Couldn't find a checkpoint. Starting from scratch.")

    else:
        model = get_model()

    """
    Testing model on the query data set.
    """
    if test_on_query:
        with open(os.path.join(cp_dir_time, "config.yaml"), 'r') as stream:
            conf = yaml.full_load(stream)
        image_size = conf['image_size']
        batch_size = conf['batch_size']
        train_images, _, _, _, img_list = load_dataset(conf, save_np_to_mem, classifier_dir, test_on_query,
                                                       label_path, image_directory)
        AUGMENTATIONS_TEST = Compose([FFT_augm(conf['use_fft'])])
        test_data = Augm_Sequence(train_images, None, batch_size, AUGMENTATIONS_TEST, shuffle=False)
        with open(os.path.join(cp_dir_time, 'query_compl{}.csv'.format(epoch_start)), 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='"', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(['Id', 'Predicted'])
            print("Beginning query")
            num_iterations = test_data.__len__()
            for i in range(num_iterations):
                current, _ = test_data.__getitem__(i)
                # current = tf.expand_dims(current, 0)
                score = np.clip(model(current, training=False).numpy()[:, 0], 0, label_range)
                for j in range(score.shape[0]):
                    filewriter.writerow([img_list[i * batch_size + j].split(".")[0], score[j]])
                print("\rScored image {}/{}".format(min((i + 1) * batch_size, train_images.shape[0]),
                                                    train_images.shape[0]), end="")

    else:
        """
        Train model on the scored data set.
        """
        cp_dir_time = os.path.join(checkpoint_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
        if not os.path.exists(cp_dir_time):
            os.makedirs(cp_dir_time)
        cp_path = os.path.join(cp_dir_time, "cp-{epoch:04d}.ckpt")
        cp_callback = tf.keras.callbacks.ModelCheckpoint(cp_path,
                                                         save_weights_only=True, save_best_only=percentage_train<1,
                                                         verbose=1, period=period_to_save_cp)

        tb_path = os.path.join(cp_dir_time, "summary")
        if not os.path.exists(tb_path):
            os.makedirs(tb_path)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_path)

        linear_model = LinearRegression()

        """## Train the model
        
        As the model trains, the loss and accuracy metrics are displayed. This model reaches an accuracy of about 0.88 (or 88%) on the training data.
        """
        learning_rate = conf['lr']
        optimizer = tf.keras.optimizers.Adam(learning_rate, decay=1/tot_num_epochs)

        """As I see it, there is a bug in the keras library that forbids the labels y from training and test set to have different unique values (which is here the case because y is a continuous label)"""

        epochs = conf['num_epochs_for_lin_regression']
        batch_size = conf['batch_size']
        model.compile(
            loss=CustomLoss(),  # keras.losses.mean_squared_error
            optimizer=optimizer,
        )
        model.summary()
        for layer in model.layers:
            if "res_block" in layer.name:
                print(" {} summary:".format(layer.name))
                layer.model.summary()
                if conf['residual']:
                    layer.projection_model.summary()
        # only necessary when neither batch norm nor dropout is used.
        # See https://stackoverflow.com/questions/52107555/different-loss-function-for-validation-set-in-keras
        model.outputs[0]._uses_learning_phase = True

        # save the configs to the checkpoints folder
        if not restore_checkpoint:
            # model.save(os.path.join(cp_dir_time, 'model.h5'))
            json_config = model.to_json()
            with open(os.path.join(cp_dir_time, 'model_config.json'), 'w') as json_file:
                json_file.write(json_config)

            with open(os.path.join(cp_dir_time, 'model_summary.txt'), 'w') as file:
                model.summary(print_fn=lambda mylambda: file.write(mylambda + '\n'))
                for layer in model.layers:
                    if "res_block" in layer.name:
                        file.write(" {} summary:".format(layer.name))
                        layer.model.summary(print_fn=lambda mylambda: file.write(mylambda + '\n'))
                        if conf['residual']:
                            layer.projection_model.summary(print_fn=lambda mylambda: file.write(mylambda + '\n'))

            cp_command = 'cp {} {}'.format(os.path.join(classifier_dir, "{}"), cp_dir_time)
            os.system(cp_command.format("classifier.py"))
            os.system(cp_command.format("config.yaml"))

        compose_list = [HorizontalFlip(p=0.5), VerticalFlip(p=0.5), ShiftScaleRotate(shift_limit=0.2, scale_limit=0,
                        rotate_limit=0, border_mode=cv2.BORDER_REFLECT_101, p=0.8)]
        if not conf['transform_before']:
            compose_list.append(FFT_augm(conf['use_fft']))
        AUGMENTATIONS_TRAIN = Compose(compose_list)

        AUGMENTATIONS_VAL = Compose([]) if conf['transform_before'] else Compose([FFT_augm(conf['use_fft'])])

        counter = 0
        if restore_checkpoint:
            counter = epoch_start
        no_val_impr = False
        val_data = np.array([]).reshape((-1, 1))
        train_images, train_labels, test_images, test_labels, _ = load_dataset(conf, save_np_to_mem, classifier_dir,
                                                                               test_on_query, label_path,
                                                                               image_directory)
        while True:
            # train the network
            if percentage_train < 1:
                validation_data = Augm_Sequence(test_images, test_labels, batch_size, AUGMENTATIONS_VAL, shuffle=False)
            else:
                validation_data = None
            H = model.fit_generator(Augm_Sequence(train_images, train_labels, batch_size, AUGMENTATIONS_TRAIN, shuffle=True),
                                    initial_epoch=counter, epochs=counter + epochs, shuffle=False,  # check shuffle=False
                                    callbacks=[tensorboard_callback, cp_callback], validation_data=validation_data)
            #test_model()
            # save whole model after first run through
            counter += epochs
            if validation_data is not None:
                val_data = np.append(val_data, np.array(H.history['val_loss']).reshape((-1, 1)), 0)
                if counter % conf['num_epochs_for_lin_regression'] == 0:
                    val_indices = np.array(list(range(val_data.shape[0]))).reshape((-1, 1))
                    linear_model.fit(val_indices, val_data)
                    if linear_model.coef_ >= 0:
                        if no_val_impr:
                            print("No improvement on evaluation data on more than one round in a row.")
                            if counter >= tot_num_epochs:
                                print("Stopping training.")
                                break
                        else:
                            print("There was NO improvement on evaluation data on this set...")
                            no_val_impr = True
                    else:
                        print("There was AN improvement on the evaluation data on this set.")
                        no_val_impr = False
                    val_data = np.array([val_data[-1, 0]]).reshape((-1, 1))

            elif counter >= tot_num_epochs:
                break

        print("All done")
