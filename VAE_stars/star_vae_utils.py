"""Variational Autoencoder Utilities

This file contains utility functions to create the dataset which is used to train the variational autoencoder model.

"""


from __future__ import absolute_import, division, print_function, unicode_literals

# !pip install -q tensorflow-gpu==2.0.0-alpha0
import tensorflow as tf

import os
import numpy as np

import cv2


def _extract_stars_28x28(image):
    
    """ Extracts stars from an image
    
    Detects all stars within the given image, extracts them and centers them within patches of size 28x28 with a black background.
    
    
    Parameters
    ----------
    image : np.ndarray
        The image from which the stars are extracted. The dimensions of the image are assumed to be 1000x1000. The 
        image is assumed to be grayscale.
        
        
    Returns
    -------
    patches : list
        A list containing the resulting 28x28 patches.
        
    """
  
    _, image_binary = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
  
    contours, _ = cv2.findContours(image_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    
    
    # tuples (x, y, w, h)
    bounding_rects = [cv2.boundingRect(c) for c in contours] 
  
    bounding_rects_filtered = [r for r in bounding_rects 
                               if not(r[0] <= 0 or r[0] + r[2] >= 1000 or r[1] <= 0 or r[1] + r[3] >= 1000)]
  
    patches = []
  

    for r in bounding_rects_filtered:
    
        x = r[0]; y = r[1]; w = r[2]; h = r[3]
    
        if(w > 28 or h > 28):
            continue
    
        star = image[y : y + h, x : x + w]
    
        if np.amax(star) < 30:
            continue
    
        patch = np.zeros((28, 28), dtype=np.float32)
    
        padding_y = (28 - h) // 2
        padding_x = (28 - w) // 2
    
        patch[padding_y : padding_y + h, padding_x : padding_x + w] = star
    
        patches.append(patch)
  
    
    return patches


def _load_stars_28x28(path_csv, path_labeled_images):
    
    """ Creates a list of star images
    
    Loads the labeled images and extracts a list of approximately 6000 star images with dimensions 28x28.
    
    Parameters
    ----------
    path_csv : str
        The path to the CSV file containing the image labels
    path_labeled_images : str
        The path to the directory containing the labeled images
        
        
    Returns
    -------
    images : list
        The list of star images
        
    """ 
    
    try:
        csv_file = open(path_csv, "r")

        list_id_label = [] # contains string tuples (id, label)

        for line in csv_file:
            if not "Id,Actual" in line:
                line = line.rstrip() # remove trailing newline character            
                list_id_label.append(line.split(","))

        list_id_label = sorted(list_id_label)

        labels = np.zeros(len(list_id_label)) # contains labels

        for idx, elem in enumerate(list_id_label):
            labels[idx] = float(elem[1])

        list_filenames = [] # contains filenames of images

        for filename in os.listdir(path_labeled_images):
            if filename.endswith(".png") and not filename.startswith("."):
                list_filenames.append(filename)

        list_filenames = sorted(list_filenames)

        assert len(labels) == len(list_filenames)

        # i suppose an image needs approximately
        # sizeof(np.float32) * 1 * 1000 * 1000 B of RAM

        # a list works well in terms of performance
        images = [] # images
        
                
        for idx, filename in enumerate(list_filenames):
            
            if labels[idx] == 1.0: # include only images with label == 1.0
      
                img = cv2.imread(path_labeled_images + filename, cv2.IMREAD_GRAYSCALE)
    
    
                patches = _extract_stars_28x28(img)
      
                for p in patches:
          
                    p = p.reshape((28, 28, 1))
                    p = np.divide(p, 255.0)
                  
                    images.append(p)
                  
        
        # conversion allows for batches to be extracted
        return np.stack(images)

    except:
        print("ERROR: failed to load labeled images.")
        
        
def load_train_test_dataset(path_csv, path_labeled_images, frac_train, batch_size):
    
    """ Creates a train and a test dataset of star images
    
    Loads the labeled images, extracts a list of approximately 6000 star images with dimensions 28x28 and
    creates a test and a train dataset based on these.
    
    
    Parameters
    ----------
    path_csv : str
        The path to the CSV file containing the image labels
    path_labeled_images : str
        The path to the directory containing the labeled images
    frac_train : int
        The train / test split for the star images
    batch_size : int
        The desired batch size for tf.data.Dataset
        
        
    Returns
    -------
    train_dataset : tf.data.Dataset
        The dataset for training
    test_dataset : tf.data.Dataset
        The dataset for testing
        
    """ 
    
    star_images = _load_stars_28x28(path_csv, path_labeled_images)
    num_star_images = len(star_images)
        
    num_images_train = int(num_star_images * frac_train)
    num_images_test = num_star_images - num_images_train
        
    train_dataset = tf.data.Dataset.from_tensor_slices(
        star_images[:num_images_train, :, :, :]).shuffle(num_images_train).batch(batch_size)
    
    test_dataset = tf.data.Dataset.from_tensor_slices(
        star_images[num_images_train:, :, :, :]).shuffle(num_images_test).batch(batch_size)
    
    return train_dataset, test_dataset

