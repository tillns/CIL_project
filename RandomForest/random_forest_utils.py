"""Random Forest Utils

These methods are for retrieving, storing and loading the feature and label vectors of the scored
images of the dataset.

This module contains the following public functions:
    #get_train_and_test_data
    #get_query_data
"""

import os
import csv
import random
import numpy as np
import numpy.ma as ma
import PIL.Image
import sys
import cv2
import math
import yaml
import pywt
home_dir = os.path.expanduser("~")
sys.path.insert(0, os.path.join(home_dir, "CIL_project/utils"))
from roi_utils import roi_histograms

def get_train_and_test_data(data_directory, num_features, split_ratio, num_imgs_to_load):
    """Gets the preprocessed train and test data

    Parameters
    ----------
    data_directory : str
        The directory where the dataset is stored
    num_features : int
        The number of features per image
    split_ratio : float
        The train-test split ratio of the data to be stored
    num_imgs_to_load : int
        The maximal number of images which should be loaded

    Returns
    -------
    train_features : numpy array
        The feature matrix of the train set
    train_labels : numpy array
        The labels of the train set
    test_features : numpy array
        The feature matrix of the test set
    test_labels : numpy array
        The labels of the test set
    """

    print("\tcreate scored image features...")
    train_features, train_labels, test_features, test_labels = \
        _preprocess_scored_data(data_directory, num_features, split_ratio, num_imgs_to_load)

    return train_features, train_labels, test_features, test_labels

def get_query_data(data_directory, num_features, split_ratio):
    """Gets the preprocessed query data.

    Parameters
    ----------
    data_directory : str
        The directory where the dataset is stored
    num_features : int
        The number of features per image

    Returns
    -------
    query_features : numpy array
        The feature matrix of the query set
    query_ids : numpy array
        The ids of the query set
    """

    print("\tcreate query image features...")
    query_features, query_ids = _load_query_data(data_directory, num_features)

    return query_features, query_ids

def _preprocess_scored_data(data_directory, num_features, split_ratio, num_imgs_to_load):
    """Preprocesses the scored image data

    The data is read from the data_directory, preprocessed and put into feature matrices and label
    vectors.

    Parameters
    ----------
    data_directory : str
        The directory where the dataset is stored
    num_features : int
        The number of features per image
    split_ratio : float
        The train-test split ratio of the data to be stored
    num_imgs_to_load : int
        The maximal number of images which should be loaded

    Returns
    -------
    train_features : numpy array
        The feature matrix of the train set
    train_labels : numpy array
        The labels of the train set
    test_features : numpy array
        The feature matrix of the test set
    test_labels : numpy array
        The labels of the test set
    """

    csv_path = os.path.join(data_directory, "scored.csv")
    ids_with_scores = []

    count = 0
    with open(csv_path, 'r') as csv_data:
        csv_reader = csv.reader(csv_data)
        next(csv_reader)    # skip first row (labels)
        for row in csv_reader:
            if count < num_imgs_to_load:
                ids_with_scores.append(row)
                count += 1

    # seed random to always get the same train-test split
    random.seed(1234)
    random.shuffle(ids_with_scores)

    num_images = len(ids_with_scores)
    num_train = int(num_images * split_ratio)
    num_test = num_images - num_train

    train_labels = np.array([score for (id, score) in ids_with_scores[:num_train]], dtype=np.float64)
    print("\t\tloading train images...")
    train_features = _load_and_preprocess_images(os.path.join(data_directory, "scored"), [id for (id, score) in ids_with_scores[:num_train]], num_features)

    test_labels = np.array([score for (id, score) in ids_with_scores[num_train:]], dtype=np.float64)
    print("\t\tloading test images...")
    test_features = _load_and_preprocess_images(os.path.join(data_directory, "scored"), [id for (id, score) in ids_with_scores[num_train:]], num_features)

    return train_features, train_labels, test_features, test_labels

def _load_query_data(data_directory, num_features):
    """Loads the query image data

    The data is read from the data_directory, preprocessed and put into a feature matrix and a
    vector containing the ids

    Parameters
    ----------
    data_directory : str
        The directory where the dataset is stored
    num_features : int
        The number of features per image

    Returns
    -------
    query_features : numpy array
        The feature matrix of the query set
    query_ids : numpy array
        The ids of the query set
    """

    image_ids = []
    for filename in os.listdir(os.path.join(data_directory, "query")):
        if filename.endswith(".png") and not filename.startswith("._"):
            image_ids.append(int(filename.split(".")[0]))

    query_ids = np.array(image_ids, dtype=np.uint32)
    print("\t\tloading query images...")
    query_features = _load_and_preprocess_images(os.path.join(data_directory, "query"), image_ids, num_features)

    return query_features, query_ids

def _load_and_preprocess_images(image_directory, image_ids, num_features):
    """Loads and preprocesses the image data

    Preprocessing is done by computing a histogram of the pixel values of the image where
    number of bins = num_features

    Parameters
    ----------
    image_directory : str
        The directory where the images are stored
    image_ids : list
        A list containing the ids of the images to be loaded and preprocessed
    num_features : int
        The number of features per image

    Returns
    -------
    image_feature_matrix : numpy array
        Matrix containing the preprocessed features of all images with id in image_ids
    """

    with open("config.yaml", 'r') as stream:
        conf = yaml.full_load(stream)

    image_feature_matrix = np.zeros((len(image_ids), num_features), dtype=np.uint32)

    print("\t\t\tloading {} images...".format(len(image_ids)))
    for i, id in enumerate(image_ids):
        if (i+1) % 400 == 0:
            print("\t\t\t\t{} images loaded...".format(i+1))

        image = PIL.Image.open(os.path.join(image_directory, "{}.png".format(id)))
        histograms = roi_histograms(image, conf)
        image_feature_matrix[i] = np.concatenate(histograms)

    print("\t\t\tfinished loading images!")
    return image_feature_matrix
