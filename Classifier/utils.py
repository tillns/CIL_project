"""Utils for the CNN Classifier

This module contains the following public functions:
    #get_pad
    #getNormLayer
    #get_random_indices
    #isInt
    #get_numpy
    #get_max_val_fft
    #transform
    #load_dataset
    #get_latest_cp
    #get_epoch_and_path
    #get_specific_cp
    
And the following classes:
    #FFT_augm
    #Augm_Sequence
"""

from random import randint
import tensorflow as tf
from CustomLayers import Pixel_norm, FactorLayer
from random import shuffle
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence
from albumentations import DualTransform
import sys


def get_pad(x, total_padding=0):
    if total_padding == 0:
        return x
    elif total_padding > 0:
        rand1 = randint(0, total_padding)
        rand2 = randint(0, total_padding)
        return tf.pad(x, tf.constant([[0, 0], [rand1, total_padding - rand1], [rand2, total_padding - rand2], [0, 0]]))
    else:
        total_padding = abs(total_padding)
        rand1 = randint(0, total_padding)
        rand2 = randint(0, total_padding)
        s = x.shape
        return x[:, rand1:s[1] - total_padding + rand1, rand2:s[2] - total_padding + rand2]


def getNormLayer(norm_type='batch', momentum=0.9, epsilon=1e-5):
    if norm_type == 'pixel':
        return Pixel_norm(epsilon)
    if norm_type == 'batch':
        return tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)
    return FactorLayer(1)

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


def get_numpy(mode, num_images, path, save_np_to_mem, image_size):
    if save_np_to_mem:
        return np.memmap(path, dtype=np.float32, mode=mode,
                         shape=(num_images, image_size, image_size, 1))
    return np.zeros((num_images, image_size, image_size, 1), dtype=np.float32)


def get_max_val_fft():
    return 497.75647


def transform(img_np, use_fft):
    if use_fft:
        img_np = 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(img_np))) ** 2 + np.power(1.0, -20)) / get_max_val_fft()
    else:
        img_np = img_np / 255
    return img_np

class FFT_augm(DualTransform):

    def __init__(self, use_fft=True):
        super(FFT_augm, self).__init__(True, 1)
        self.use_fft = use_fft

    def apply(self, img, **params):
        return transform(img, self.use_fft)


def load_dataset(conf, save_np_to_mem, classifier_dir, test_on_query, label_path, image_directory):
    train_images, train_labels, test_images, test_labels = None, None, None, None
    percentage_train = conf['percentage_train']
    image_size = conf['image_size']
    try:
        f = open(label_path, 'r')
        print("Found Labels")
        label_list = []
        for line in f:
            if not "Id,Actual" in line:
                split_line = line.split(",")
                split_line[-1] = float(split_line[-1])
                label_list.append(split_line)
        label_list = sorted(label_list)

        img_list = []
        for filename in os.listdir(image_directory):
            if filename.endswith(".png") and not filename.startswith("._"):
                img_list.append(filename)

        img_list = sorted(img_list)
        dataset_len = len(img_list)
        if not test_on_query:
            assert len(label_list) == dataset_len

    except FileNotFoundError:
        sys.exit("Dataset not found.")
    np_data_path = os.path.join(classifier_dir, "numpy_data/{}_p{:.1f}_s{}.dat".format('{}', percentage_train,
                                                                                       image_size))
    trainset_path = np_data_path.format("queryset") if test_on_query else np_data_path.format("trainset")
    testset_path = np_data_path.format('testset')
    both_paths_exist = save_np_to_mem and os.path.exists(trainset_path) and (percentage_train == 1 or os.path.exists(testset_path))
    mode = 'r+' if both_paths_exist else 'w+'
    train_images = get_numpy(mode, int(dataset_len*percentage_train), trainset_path)
    train_labels = np.zeros(shape=(train_images.shape[0],), dtype=np.float32)
    if percentage_train < 1:
        test_images = get_numpy(mode, dataset_len - int(dataset_len * percentage_train), testset_path)
        test_labels = np.zeros(shape=(test_images.shape[0],), dtype=np.float32)
    for num in range(dataset_len):
        if not both_paths_exist:
            img = Image.open(os.path.join(image_directory, img_list[num])).resize((image_size, image_size))
            img_np = np.array(img, dtype=np.float32).reshape((image_size, image_size, 1))
            if conf['transform_before']:
                transform(img_np)
        if num < int(dataset_len * percentage_train):
            train_labels[num] = label_list[num][1]
            if not both_paths_exist:
                train_images[num] = img_np

        else:
            test_labels[num - int(dataset_len * percentage_train)] = label_list[num][1]
            if not both_paths_exist:
                test_images[num - int(dataset_len * percentage_train)] = img_np
        print("\rLoaded image {}/{}".format(num + 1, dataset_len), end="")
    print("")
    return train_images, train_labels, test_images, test_labels, img_list
    # following 4 lines gave min_val = 0 and max_val = 497.75647 as used in get_max_val_fft
    # max_test = -1000 if percentage_train == 1 else np.max(test_images)
    # min_test = 1000 if percentage_train == 1 else np.min(test_images)
    # max_val = max(np.max(train_images), max_test)
    # min_val = min(np.min(train_images), min_test)


class Augm_Sequence(Sequence):
    # set y_set to None for test set
    def __init__(self, x_set, y_set, batch_size, augmentations, shuffle=False):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.augment = augmentations
        self.shuffle = shuffle
        self.indices = list(range(self.x.shape[0]))

    def __len__(self):
        return int(np.ceil(self.x.shape[0] / self.batch_size))

    def __getitem__(self, idx):
        max_idx = min(self.x.shape[0], (idx + 1)*self.batch_size)
        batch_x = self.x[self.indices[idx * self.batch_size:max_idx]]
        x_stacked = np.float32(np.stack([self.augment(image=x)["image"] for x in batch_x], axis=0))
        if self.y is not None:
            batch_y = self.y[self.indices[idx * self.batch_size:max_idx]]
            return x_stacked, np.array(batch_y)
        return x_stacked, None

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def get_latest_cp(checkpoint_dir, epoch=None):
    if not os.path.exists(checkpoint_dir):
        return None
    list_dir = os.listdir(checkpoint_dir)
    if len(list_dir) == 0:
        return None
    list_dir = sorted(list_dir)
    if epoch is not None:
        specific_cp_path = os.path.join(checkpoint_dir, os.path.join(list_dir[-1], "cp-{0:04d}.ckpt".format(epoch)))
        return specific_cp_path
    return tf.train.latest_checkpoint(os.path.join(checkpoint_dir, list_dir[-1]))


def get_epoch_and_path(path):
    if path is None:
        return 0, None
    filename = path.split("/")[-1]
    epoch = int(filename.split("-")[1].split(".")[0])
    return epoch, path


def get_specific_cp(checkpoint_dir):
    while True:
        user_input = input("Enter the path of checkpoint file or leave empty to use latest checkpoint.")
        if len(user_input) == 0:
            print("Using latest checkpoint from latest directory.")
            specific_path = get_latest_cp(checkpoint_dir)
            break
        elif isInt(user_input):
            specific_path = get_latest_cp(checkpoint_dir, int(user_input))
            break
        if os.path.exists(user_input):
            specific_path = user_input[:-20]
            break
        if os.path.exists(user_input + ".data-00000-of-00001"):
            specific_path = user_input
            break
        else:
            print("Please provide a valid path")

    global cp_dir_time
    cp_dir_time = os.path.dirname(specific_path)
    return get_epoch_and_path(specific_path)