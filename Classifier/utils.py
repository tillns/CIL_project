"""Utils for the CNN Classifier

This module contains the following public functions:
    #isInt
    #get_numpy
    #transform
    #load_dataset
    #get_latest_cp
    #get_epoch_and_path
    #get_specific_cp
"""

import tensorflow as tf
import os
from PIL import Image
import numpy as np
import sys


def isInt(s):
    """
    :param s: obj
    :return: True if s is (castable to) an integer; False otherwise
    """
    try:
        int(s)
        return True
    except ValueError:
        return False


def get_numpy(mode, num_images, path, save_np_to_mem, image_size):
    """
    Due to local RAM restrictions, the numpy data had to be saved to disk. On the other hand, on the cluster the data
    could not be saved to disk due to disk restrictions, but there was enough RAM available. This method returns a numpy
    memmap, which is a mapping to the saved file, or a normal numpy array if the conditions allow for it.
    :param mode: for memmap, 'r+' for reading files (when they've already been created) and 'w+' for creating new ones
    :param num_images: number of images (e.g. 9600 for scored data)
    :param path: path to numpy memmap
    :param save_np_to_mem: Whether to use memmap or not
    :param image_size: size of the images (e.g. 1000 for original data)
    :return: either numpy memmap or array
    """
    if save_np_to_mem:
        return np.memmap(path, dtype=np.float32, mode=mode,
                         shape=(num_images, image_size, image_size, 1))
    return np.zeros((num_images, image_size, image_size, 1), dtype=np.float32)


def transform(img_np, use_fft):
    """
    :param img_np: input numpy array ranging from 0 to 255
    :param use_fft: whether to apply fast Fourier transform on input
    :return: numpy image array transformed to range from 0 to 1
    """
    if use_fft:
        # + 1, so it's never negative
        # 497.75647 is the absolute maximum on scored and query set and very specific to formula
        img_np = np.float32(20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(img_np))) ** 2 + 1) / 497.75647)
    else:
        img_np = img_np / 255
    return img_np


def load_dataset(conf, save_np_to_mem, classifier_dir, test_on_query, label_path, image_directory, percentage_train):
    """
    Load images inside a directory and split them into training and validation set including labels for both
    :param conf: classifier configuration
    :param save_np_to_mem: whether to use memmap (see get_numpy() for an explanation)
    :param classifier_dir: directory of the classifier project
    :param test_on_query: Whether to test classifier on query set
    :param label_path: path to labels (ending with .csv)
    :param image_directory: directory of the images to load
    :param percentage_train: split between training length and whole dataset length
    :return: training images and labels, validation images and labels, and list of image names
    """
    train_images, train_labels, test_images, test_labels = None, None, None, None
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
    fft_string = "_fft" if conf['use_fft'] else ""
    np_data_path = os.path.join(classifier_dir, "numpy_data/{}_p{:.1f}_s{}{}.dat".format('{}', percentage_train,
                                                                                         image_size, fft_string))
    trainset_path = np_data_path.format("queryset") if test_on_query else np_data_path.format("trainset")
    testset_path = np_data_path.format('testset')
    both_paths_exist = save_np_to_mem and os.path.exists(trainset_path) and (percentage_train == 1 or os.path.exists(testset_path))
    mode = 'r+' if both_paths_exist else 'w+'
    train_images = get_numpy(mode, int(dataset_len*percentage_train), trainset_path, save_np_to_mem, image_size)
    train_labels = np.zeros(shape=(train_images.shape[0],), dtype=np.float32)
    if percentage_train < 1:
        test_images = get_numpy(mode, dataset_len - int(dataset_len * percentage_train), testset_path, save_np_to_mem, image_size)
        test_labels = np.zeros(shape=(test_images.shape[0],), dtype=np.float32)
    for num in range(dataset_len):
        if not both_paths_exist:
            img = Image.open(os.path.join(image_directory, img_list[num])).resize((image_size, image_size))
            img_np = np.array(img, dtype=np.float32).reshape((image_size, image_size, 1))
            img_np = transform(img_np, conf['use_fft'])
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
    # following 4 lines gave min_val = 0 (unsurprisingly) and max_val = 497.75647 as used in transform
    # max_test = -1000 if percentage_train == 1 else np.max(test_images)
    # min_test = 1000 if percentage_train == 1 else np.min(test_images)
    # max_val = max(np.max(train_images), max_test)
    # min_val = min(np.min(train_images), min_test)


def get_latest_cp(checkpoint_dir, epoch=None):
    """
    Loads specific or last checkpoint of (alphabetically) last checkpoint sub directory
    :param checkpoint_dir: directory containing all runs
    :param epoch: (optional) checkpoint epoch number
    :return: path to checkpoint
    """
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
    """
    :param path: path to checkpoint
    :return: epoch to corresponding path, and path
    """
    if path is None:
        return 0, None
    filename = path.split("/")[-1]
    epoch = int(filename.split("-")[1].split(".")[0])
    return epoch, path


def get_specific_cp(checkpoint_dir):
    """
    If no checkpoint path was provided in the arguments, this method is called to retrieve the path.
    :param checkpoint_dir: Path containing all runs
    :return: epoch and path to checkpoint
    """
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