"""
This module provides useful methods for the cDCGAN. It implements the following methods:
    #transform_norm
    #detransform_norm
    #load_dataset
    #one_hot
    #generate_and_save_images
"""
import cv2
import os
import numpy as np
from PIL import Image


def transform_norm(numpy_image_array, conf):
    """
    :param numpy_image_array: numpy image array ranging from 0 to 255
    :param conf: cDCGAN configuration
    :return: copy of the numpy image array, linearly transformed to range from conf['vmin'] to conf['vmax']
    """
    return numpy_image_array / 255.0 * (conf['vmax']-conf['vmin']) + conf['vmin']


def detransform_norm(numpy_image_array, conf):
    """
    :param numpy_image_array: numpy image array ranging from conf['vmin'] to conf['vmax']
    :param conf: cDCGAN configuration
    :return: copy of the numpy image array, linearly transformed to range from 0 to 255
    """
    return (numpy_image_array - conf['vmin']) / (conf['vmax']-conf['vmin']) * 255.0


def load_dataset(conf, image_directory, image_size):
    """
    :param conf: cDCGAN configuration
    :param image_directory: directory which directly contains the images for the unconditional setting, and the folders
                            for each class, each containing its corresponding images for the conditional setting.
    :param image_size: Size of the images to load
    :return: training images, validation images, training labels, and validation labels
    """
    images = []
    if not conf['conditional']:
        for filename in os.listdir(image_directory):
            if filename.endswith(".png") and not filename.startswith("._"):
                img = Image.open(os.path.join(image_directory, filename)).resize((image_size, image_size))
                img_np = transform_norm(np.array(img, dtype=np.float32).reshape((image_size, image_size, 1)), conf)
                images.append(img_np)
        dataset = np.stack(images)
    else:
        label_list = []
        num_classes = len(os.listdir(image_directory))
        conf['num_classes'] = num_classes
        for label, folder_name in enumerate(sorted(os.listdir(image_directory))):
            folder_path = os.path.join(image_directory, folder_name)
            label_vec = np.zeros(num_classes)
            label_vec[label] += 1
            for img_name in sorted(os.listdir(folder_path)):
                img = Image.open(os.path.join(folder_path, img_name)).resize((image_size, image_size))
                img_np = transform_norm(np.array(img, dtype=np.float32).reshape((image_size, image_size, 1)), conf)
                images.append(img_np)
                label_list.append(label_vec)
        dataset = np.stack(images)
        labels = np.stack(label_list)
        # dataset and labels are shuffled together, s.t. training and test set have same distribution
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
        dataset = dataset[indices]
        labels = labels[indices]

    train_len = round(conf['percentage_train'] * len(dataset))
    print("Loaded all {} images".format(len(dataset)))
    train_images = dataset[:train_len]
    train_labels = labels[:train_len] if conf['conditional'] else None
    if conf['percentage_train']:
        test_images = dataset[train_len:]
        test_labels = labels[train_len:] if conf['conditional'] else None
    else:
        test_images = None
        test_labels = None

    return train_images, test_images, train_labels, test_labels


#  taken from PixelCNN code
def one_hot(batch_y, num_classes):
    """
    :param batch_y: batch-sized vector containing class label for each image
    :param num_classes: total number of classes
    :return: batch_sized matrix, for each image a vector with value 1 at the class label entry and 0 otherwise
    """
    y_ = np.zeros((batch_y.shape[0], num_classes))
    y_[np.arange(batch_y.shape[0]), batch_y] = 1
    return y_


def generate_and_save_images(model, epoch, test_input, conf, save_image_path):
    """
    Generates a bunch of images from a given latent input with the generative model.
    :param model: keras model
    :param epoch: number of current epoch (used for saving)
    :param test_input: latent input (should preferably be kept constant throughout training for comparison purposes)
    :param conf: model configuration
    :param save_image_path: path to folder in which to save generated images.
    """
    tot = test_input.shape[0]
    per_row = int(np.sqrt(tot)) if not conf['conditional'] else conf['num_classes']
    if conf['conditional']:
        labels = one_hot(np.array(list(range(conf['num_classes']))*(tot//conf['num_classes'])), conf['num_classes'])
        predictions = model([test_input, labels], training=False)
    else:
        predictions = model(test_input, training=False)
    predictions = detransform_norm(predictions, conf)
    res = predictions.shape[1]
    depth = predictions.shape[3]

    dist = int(0.25 * res)
    outres = per_row * res + dist * (per_row + 1)
    output_image = np.zeros((outres, outres, depth)) + 127.5
    for k in range(tot):
        i = k // per_row
        j = k % per_row
        output_image[dist * (i + 1) + i * res:dist * (i + 1) + res * (i + 1),
                     dist * (j + 1) + j * res:dist * (j + 1) + res * (j + 1)] = predictions[k]

    cv2.imwrite(os.path.join(save_image_path, 'image_at_epoch_{:05d}.png'.format(epoch)), output_image)