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
import numpy.ma as ma
import PIL.Image

import sys

import cv2
import math

import yaml

import pywt
# run this in this directory:
#
# pip install Cython
# python setup.py build_ext -i
try:
    import compute_hist
    use_grad = True
except ModuleNotFoundError:
    use_grad = False

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
    train_features : numpy array
        The feature matrix of the train set
    train_labels : numpy array
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
    test_features : numpy array
        The feature matrix of the test set
    test_labels : numpy array
        The labels of the test set
    """

    print("load test data...")
    _, _, test_features, test_labels = get_train_and_test_data(numpy_data_directory, data_directory, num_features, split_ratio)

    return test_features, test_labels

def get_train_and_test_data(numpy_data_directory, data_directory, num_features, split_ratio, num_imgs_to_load=1500):
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
    train_features : numpy array
        The feature matrix of the train set
    train_labels : numpy array
        The labels of the train set
    test_features : numpy array
        The feature matrix of the test set
    test_labels : numpy array
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
        train_features, train_labels, test_features, test_labels = \
            _preprocess_scored_data(data_directory, num_features, split_ratio, num_imgs_to_load)
        _save_np_data(numpy_data_directory, train_features, _np_train_file_name(num_features, split_ratio))
        _save_np_data(numpy_data_directory, train_labels, _np_train_file_name(num_features, split_ratio, True))
        _save_np_data(numpy_data_directory, test_features, _np_test_file_name(num_features, split_ratio))
        _save_np_data(numpy_data_directory, test_labels, _np_test_file_name(num_features, split_ratio, True))

    return train_features, train_labels, test_features, test_labels

def get_query_data(numpy_data_directory, data_directory, num_features, split_ratio, filetype='png'):
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
    query_features : numpy array
        The feature matrix of the query set
    query_ids : numpy array
        The ids of the query set
    """

    print("load query data...")
    try:
        # try loading the data from disk
        query_features = np.load(os.path.join(numpy_data_directory, _np_query_file_name(num_features) + ".npy"))
        query_ids = np.load(os.path.join(numpy_data_directory, _np_query_file_name(num_features, True) + ".npy"))
    except:
        # if it doesn't work, recalculate the data and store it to disk
        print("\tcreate query image features...")
        query_features, query_ids = _load_query_data(data_directory, num_features, filetype)
        _save_np_data(numpy_data_directory, query_features, _np_query_file_name(num_features))
        _save_np_data(numpy_data_directory, query_ids, _np_query_file_name(num_features, True))

    return query_features, query_ids

def _preprocess_scored_data(data_directory, num_features, split_ratio, num_imgs_to_load=1500):
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

    cnt = 0

    # IMPORTANT: ADJUST 1500 TO e.g. 15000 TO LOAD ALL IMAGES
    # TODO: remove this before handing it in

    with open(csv_path, 'r') as csv_data:
        csv_reader = csv.reader(csv_data)
        next(csv_reader)    # skip first row (labels)
        for row in csv_reader:
            if cnt < num_imgs_to_load:
                ids_with_scores.append(row)
                cnt = cnt + 1

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

def _load_query_data(data_directory, num_features, filetype='png'):
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
    query_features = _load_and_preprocess_images(os.path.join(data_directory, "query"), image_ids, num_features, filetype)

    return query_features, query_ids

def _load_and_preprocess_images(image_directory, image_ids, num_features, filetype='png'):
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
    filetype : str
        type of the images to be preprocessed

    Returns
    -------
    image_feature_matrix : numpy array
        Matrix containing the preprocessed features of all images with id in image_ids
    """

    with open("config.yaml", 'r') as stream:
        conf = yaml.full_load(stream)
    roi_conf = conf['ROI_options']

    image_feature_matrix = np.zeros((len(image_ids), num_features), dtype=np.uint32)

    print("\t\t\tloading {} images...".format(len(image_ids)))
    for i, id in enumerate(image_ids):
        if (i+1) % 400 == 0:
            print("\t\t\t\t{} images loaded...".format(i+1))

        image = PIL.Image.open(os.path.join(image_directory, "{}.{}".format(id, filetype)))

        if conf['histogram_type'] == 'ROI':
            histograms = roi_histograms(image, conf)
        else:
            raise ValueError("config.yaml: histogram_type has no valid value")

        image_feature_matrix[i] = np.concatenate(histograms)

    print("\t\t\tfinished loading images!")
    return image_feature_matrix

def _save_np_data(numpy_data_directory, data, file_name):
    """Saves a numpy array to disk

    Parameters
    ----------
    numpy_data_directory : str
        The directory in which the data should be stored
    data : numpy array
        The data to be stored
    file_name : str
        The name of the file in which the data should be stored
    """
    if not os.path.exists(numpy_data_directory):
        os.makedirs(numpy_data_directory)
    #np.save(os.path.join(numpy_data_directory, file_name), data)

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

def _compute_histogram_from_mask(mask, image, num_bins, range):
    """Computes a histogram for the image region defined by the mask for each channel

    Parameters
    ----------
    mask : numpy array
        boolean mask. Shape (H, W).
    image : numpy array
        original image. Shape (H, W, C).
    num_bins : int
        the bins argument for the histogram
    range : tuple
        the range argument for the histogram


    Returns
    -------
    hist : list
        list of length bins, containing the histogram of the masked image values
    """

    # Apply binary mask to your array, you will get array with shape (N, C)
    region = image[mask]

    hist, _ = np.histogram(region, bins=num_bins, range=range)
    return hist


def _create_circular_mask(h, w, center=None, radius=None, invert_mask=False):
    """ Computes a circular mask

    Get a circular mask on an image given its height, width and optionally center and radius of the
    mask.

    Paramters
    ---------
    h : int
        height of the image
    w : int
        width of the image
    center : tuple
        tuple of ints, representing the center of the generated mask
    radius : int or tuple
        either the radius of a circular mask or the min- and max-radius of the mask
    invert_mask : bool
        wheter the generated mask should be inverted before returning it

    Returns
    -------
    mask : numpy array
        a numpy array of boolean values
    """

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    if isinstance(radius, list) and len(radius) == 2:
        if radius[0] < 0:
            radius[0] = np.abs(radius[0])
            radius[1] = np.abs(radius[1])
            invert_mask = True
        mask = (max(radius) <= dist_from_center) | (dist_from_center <= min(radius)) if invert_mask else \
            (max(radius) >= dist_from_center) & (dist_from_center >= min(radius))
    else:
        if radius < 0:
            radius = np.abs(radius)
            invert_mask = True
        mask = dist_from_center >= radius if invert_mask else dist_from_center <= radius
    return mask


def _create_angle_mask(h, w, angle_l, angle_h):
    
    """ Computes a mask for an interval of angles

    Returns an image mask masking all points which are not at an angle inside [angle_l, angle_h] or
    [angle_l - 180.0, angle_h - 180.0] to the center of the image. The angles are measured counter-clockwise 
    from the horizontal.
    

    Paramters
    ---------
    h : int
        The image height
    w : int
        The image width
    angle_l : float
        The first angle of the interval. Has to be inside [0.0, angle_h].
    angle_h : float
        The second angle of the interval. Has to be inside [angle_l, 180.0].


    Returns
    -------
    mask : array-like
        The image mask
    
    """

    center = [w // 2, h // 2]
                
    Y, X = np.ogrid[:h, :w]
    
    
    angle = np.degrees(np.arctan2((-(Y - center[1])), (X-center[0])))
    
    
    mask_angle = ((angle_l <= angle) * (angle <=  angle_h)) + \
                    ((angle_l - 180.0 <= angle) * (angle <= angle_h - 180.0))
    
    mask_angle[w // 2 , h // 2] = True # center
    
    
    mask_angle = np.invert(mask_angle)
   
    return mask_angle


def roi_histograms(image, conf):
    roi_conf = conf['ROI_options']
    np_image = np.array(image, dtype=np.uint8)
    range_normal = (0, 255)
    eps = np.power(10.0, -15.0)
    psd_fft = np.fft.fft2(np.asarray(image, dtype=np.float32))
    psd_fft_shift = np.abs(np.fft.fftshift(psd_fft))**2
    psd_log = 10 * np.log10(psd_fft_shift + eps)
    psd_noshifted = np.abs(psd_fft)**2
    psd_noshifted_log = 10 * np.log10(psd_noshifted + eps)
    ranges_fft = [(range_fft[0], range_fft[1]) for range_fft in conf['ranges']]

    hists = []

    if roi_conf['whole_img']['include']:
        if not roi_conf['whole_img']['prepr_fft']:
            hists.append(np.histogram(np_image, bins=roi_conf['whole_img']['num_bins'], range=range_normal)[0])
        else:
            for num_range, range_fft in enumerate(ranges_fft):
                hists.append(np.histogram(psd_log, bins=roi_conf['whole_img']['num_bins'][num_range], range=range_fft)[0])


    # example for subset of frequencies (the frequency spectrum is very symmetric)
    # adding these additional histograms gave an improvement of about 5 %
    if roi_conf['quarter_img']['include']:
        num_rows, num_cols = psd_fft_shift.shape

        if roi_conf['quarter_img']['prepr_fft']:
            quarter_psd = psd_fft_shift[num_rows // 2 : num_rows, num_cols // 2 : num_cols]
            quarter = 10 * np.log(quarter_psd + np.power(1.0, -120))
            quarter_range = range_fft

        else:
            quarter = np_image[num_rows // 2 : num_rows, num_cols // 2 : num_cols]
            quarter_range = range_normal

        image_features_4, _ = np.histogram(quarter[0:250, 250:500][0:125, 0:125],
                                               bins=roi_conf['quarter_img']['num_bins'], range=quarter_range)
        image_features_5, _ = np.histogram(quarter[0:250, 250:500][0:125, 125:250],
                                               bins=roi_conf['quarter_img']['num_bins'], range=quarter_range)
        image_features_6, _ = np.histogram(quarter[0:250, 250:500][125:250, 0:125],
                                               bins=roi_conf['quarter_img']['num_bins'], range=quarter_range)
        image_features_7, _ = np.histogram(quarter[0:250, 250:500][125:250, 125:250],
                                               bins=roi_conf['quarter_img']['num_bins'], range=quarter_range)
        hists.extend([image_features_4, image_features_5, image_features_6, image_features_7])

    if roi_conf['radial']['include']:
        for num_rad, radius in enumerate(roi_conf['radial']['radii']):
            mask = _create_circular_mask(1000, 1000, [500, 500], radius)
            if roi_conf['radial']['prepr_fft']:
                if roi_conf['radial']['shift_fft']:
                    for num_range, range_fft in enumerate(ranges_fft):
                        hists.append(_compute_histogram_from_mask(mask, psd_log, roi_conf['radial']['num_bins'][num_rad][num_range],
                                                                  range_fft))
                else:
                    hists.append(_compute_histogram_from_mask(mask, psd_noshifted_log,
                                                              roi_conf['radial']['num_bins'][num_rad], range_fft))
            else:
                hists.append(_compute_histogram_from_mask(mask, np_image, roi_conf['radial']['num_bins'][num_rad],
                                                          range_normal))

    if roi_conf['grad']['include'] and use_grad:
        # Todo: I think this only uses data without fft, add fft

        grad_img = ((psd_log + 150.0) / 2.0).astype(np.uint8)
        gx = cv2.Sobel(grad_img, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(grad_img, cv2.CV_32F, 0, 1, ksize=1)
        
        msk = _create_circular_mask(1000, 1000, [500, 500], radius=120)

        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        
        mag_mskd = ma.masked_array(mag, mask=msk)
        angle_mskd = ma.masked_array(angle, mask=msk)

        grad_hist = np.zeros(9).astype(np.float32)
        
        compute_hist.compute_hist_func(mag_mskd[0:500,500:1000].compressed().flatten(), 
                                       angle_mskd[0:500,500:1000].compressed().flatten(), 
                                       grad_hist, angle_mskd[0:500,500:1000].compressed().size, roi_conf['grad']['num_bins'])

        hists.append(grad_hist)

    if roi_conf['angle']['include']:
        for num_it, num_bins in enumerate(roi_conf['angle']['num_bins']):
            mask = _create_angle_mask(1000, 1000, roi_conf['angle']['angle_l'][num_it],
                                      roi_conf['angle']['angle_h'][num_it])
            hist = _compute_histogram_from_mask(mask, psd_log, num_bins, range_fft) \
                if roi_conf['angle']['prepr_fft'] else \
                _compute_histogram_from_mask(mask, np_image, num_bins, range_normal)
            hists.append(hist)
    return hists
