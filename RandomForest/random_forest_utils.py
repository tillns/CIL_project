"""Random Forest Utils

These methods are for retrieving, storing and loading the feature and label vectors of the scored
images of the dataset.

This module contains the following public functions:
    #get_train_data
    #get_test_data
    #get_train_and_test_data
    #get_query_data
"""

import os
import csv
import random
import numpy as np
import PIL.Image

def get_train_data(numpy_data_directory, data_directory, num_features, split_ratio):
    """Gets the preprocessed train data

    Parameters
    ----------
    numpy_data_directory : str
        The directory where the numpy data is stored or should be stored
    data_directory : str
        The directory where the dataset is stored
    num_features : int
        The number of features per image
    split_ratio : float
        The train-test split ratio of the data to be stored

    Returns
    -------
    train_features : numpy-array
        The feature matrix of the train set
    train_labels : numpy-array
        The labels of the train set
    """

    print("load train data...")
    train_features, train_labels, _, _ = get_train_and_test_data(numpy_data_directory, data_directory, num_features, split_ratio)

    return train_features, train_labels

def get_test_data(numpy_data_directory, data_directory, num_features, split_ratio):
    """Gets the preprocessed test data

    Parameters
    ----------
    numpy_data_directory : str
        The directory where the numpy data is stored or should be stored
    data_directory : str
        The directory where the dataset is stored
    num_features : int
        The number of features per image
    split_ratio : float
        The train-test split ratio of the data to be stored

    Returns
    -------
    test_features : numpy-array
        The feature matrix of the test set
    test_labels : numpy-array
        The labels of the test set
    """

    print("load test data...")
    _, _, test_features, test_labels = get_train_and_test_data(numpy_data_directory, data_directory, num_features, split_ratio)

    return test_features, test_labels

def get_train_and_test_data(numpy_data_directory, data_directory, num_features, split_ratio):
    """Gets the preprocessed train and test data

    If the data is not already preprocessed (i.e. stored in the numpy_data_directory),
    the data is newly calculated and saved to the numpy_data_directory.

    Parameters
    ----------
    numpy_data_directory : str
        The directory where the numpy data is stored or should be stored
    data_directory : str
        The directory where the dataset is stored
    num_features : int
        The number of features per image
    split_ratio : float
        The train-test split ratio of the data to be stored

    Returns
    -------
    train_features : numpy-array
        The feature matrix of the train set
    train_labels : numpy-array
        The labels of the train set
    test_features : numpy-array
        The feature matrix of the test set
    test_labels : numpy-array
        The labels of the test set
    """

    try:
        # try loading th data from disk
        train_features = np.load(os.path.join(numpy_data_directory, _np_train_file_name(num_features, split_ratio) + ".npy"))
        train_labels = np.load(os.path.join(numpy_data_directory, _np_train_file_name(num_features, split_ratio, True) + ".npy"))
        test_features = np.load(os.path.join(numpy_data_directory, _np_test_file_name(num_features, split_ratio) + ".npy"))
        test_labels = np.load(os.path.join(numpy_data_directory, _np_test_file_name(num_features, split_ratio, True) + ".npy"))
    except:
        # if it doesn't work, recalculate the data and store it to disk
        print("\tcreate scored image features...")
        train_features, train_labels, test_features, test_labels = _preprocess_scored_data(data_directory, num_features, split_ratio)
        _save_np_data(numpy_data_directory, train_features, _np_train_file_name(num_features, split_ratio))
        _save_np_data(numpy_data_directory, train_labels, _np_train_file_name(num_features, split_ratio, True))
        _save_np_data(numpy_data_directory, test_features, _np_test_file_name(num_features, split_ratio))
        _save_np_data(numpy_data_directory, test_labels, _np_test_file_name(num_features, split_ratio, True))

    return train_features, train_labels, test_features, test_labels

def get_query_data(numpy_data_directory, data_directory, num_features):
    """Gets the preprocessed query data

    If the data is not already preprocessed (i.e. stored in the numpy_data_directory),
    the data is newly calculated and saved to the numpy_data_directory.

    Parameters
    ----------
    numpy_data_directory : str
        The directory where the numpy data is stored or should be stored
    data_directory : str
        The directory where the dataset is stored
    num_features : int
        The number of features per image

    Returns
    -------
    query_features : numpy-array
        The feature matrix of the query set
    query_ids : numpy-array
        The ids of the query set
    """

    print("load query data...")
    try:
        # try loading the data from disk
        query_features = np.load(os.path.join(numpy_data_directory, _np_query_file_name(num_features, split_ratio) + ".npy"))
        query_ids = np.load(os.path.join(numpy_data_directory, _np_query_file_name(num_features, True) + ".npy"))
    except:
        # if it doesn't work, recalculate the data and store it to disk
        print("\tcreate query image features...")
        query_features, query_ids = _load_query_data(data_directory, num_features)
        _save_np_data(numpy_data_directory, query_features, _np_query_file_name(num_features))
        _save_np_data(numpy_data_directory, query_ids, _np_query_file_name(num_features, True))

    return query_features, query_ids

def _preprocess_scored_data(data_directory, num_features, split_ratio):
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

    Returns
    -------
    train_features : numpy-array
        The feature matrix of the train set
    train_labels : numpy-array
        The labels of the train set
    test_features : numpy-array
        The feature matrix of the test set
    test_labels : numpy-array
        The labels of the test set
    """

    csv_path = os.path.join(data_directory, "scored.csv")
    ids_with_scores = []

    with open(csv_path, 'r') as csv_data:
        csv_reader = csv.reader(csv_data)
        next(csv_reader)    # skip first row (labels)
        for row in csv_reader:
            ids_with_scores.append(row)

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
    query_features : numpy-array
        The feature matrix of the query set
    query_ids : numpy-array
        The ids of the query set
    """

    image_ids = [int(file_name.split(".")[0]) for file_name in os.listdir(os.path.join(data_directory, "query"))]

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
    image_feature_matrix : numpy-array
        Matrix containing the preprocessed features of all images with id in image_ids
    """

    image_feature_matrix = np.zeros((len(image_ids), num_features), dtype=np.uint32)

    print("\t\t\tloading {} images...".format(len(image_ids)))
    for i, id in enumerate(image_ids):
        if (i+1) % 400 == 0:
            print("\t\t\t\t{} images loaded...".format(i+1))

        image = PIL.Image.open(os.path.join(image_directory, "{}.png".format(id)))
        np_image = np.array(image.getdata(), dtype=np.uint8)
        image_features, _ = np.histogram(np_image, bins=num_features, range=(0, 255))
        image_feature_matrix[i] = image_features

    print("\t\t\tfinished loading images!")
    return image_feature_matrix

def _save_np_data(numpy_data_directory, data, file_name):
    """Saves a numpy-array to disk

    Parameters
    ----------
    numpy_data_directory : str
        The directory in which the data should be stored
    data : numpy-array
        The data to be stored
    file_name : str
        The name of the file in which the data should be stored
    """
    if not os.path.exists(numpy_data_directory):
        os.makedirs(numpy_data_directory)
    np.save(os.path.join(numpy_data_directory, file_name), data)

def _np_train_file_name(num_features, split_ratio, labels=False):
    """Returns the name of a train data file

    Parameters
    ----------
    num_features : int
        The number of features of the data to be stored
    split_ratio : float
        The train-test split ratio of the data to be stored
    labels : bool, optional
        Wheter the data stored are labels or not (default is False)

    Returns
    -------
    file_name : str
        Name of the train file for the given parameters
    """

    return "train_{}_{}_{}".format("labels" if labels else "features", num_features, split_ratio)

def _np_test_file_name(num_features, split_ratio, labels=False):
    """Returns the name of a test data file

    Parameters
    ----------
    num_features : int
        The number of features of the data to be stored
    split_ratio : float
        The train-test split ratio of the data to be stored
    labels : bool, optional
        Wheter the data stored are labels or not (default is False)

    Returns
    -------
    file_name : str
        Name of the test file for the given parameters
    """

    return "test_{}_{}_{}".format("labels" if labels else "features", num_features, split_ratio)

def _np_query_file_name(num_features, ids=False):
    """Returns the name of a query data file

    Parameters
    ----------
    num_features : int
        The number of features of the data to be stored
    ids : bool, optional
        Wheter the data stored are ids or not (default is False)

    Returns
    -------
    file_name : str
        Name of the query file for the given parameters
    """

    return "query_{}_{}".format("ids" if ids else "features", num_features)
