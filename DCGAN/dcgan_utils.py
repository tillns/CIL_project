"""DCGAN Utilities

This file contains utility functions to create the dataset which is used to train the DCGAN.

"""


from __future__ import absolute_import, division, print_function, unicode_literals

# !pip install -q tensorflow-gpu==2.0.0-alpha0
import tensorflow as tf

import os
import numpy as np

import cv2


def _load_labeled_images(arguments):
    
    """ Loads all labeled images and returns them inside a list
    
    
    Parameters
    ----------
    arguments : argparse-arguments
        The command-line arguments
        
        
    Returns
    -------
    images : list
        The list of images
        
    """ 
    
    try:
        csv_file = open(arguments.path_csv_labeled, "r")

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

        for filename in os.listdir(arguments.dir_labeled_images):
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
      
                img = cv2.imread(os.path.join(arguments.dir_labeled_images, filename), cv2.IMREAD_GRAYSCALE)
        
        
                arr = np.array(img, dtype=np.float32).reshape((1000, 1000, 1))

                # map the image data from [0, 255] to [-1.0, 1.0]
                arr = np.subtract(arr, 127.5)
                arr = np.divide(arr, 127.5)

                
                images.append(arr)

                
        return images

    except:
        print("ERROR: failed to load labeled images.")


def _load_scored_images(arguments):
    
    """ Loads all scored images with score >= 2.61 and returns them inside a list
    
    
    Parameters
    ----------
    arguments : argparse-arguments
        The command-line arguments
        
        
    Returns
    -------
    images : list
        The list of images
        
    """ 
    
    try:
        csv_file = open(arguments.path_csv_scored, "r")

        list_id_score = [] # contains string tuples (id, score)

        for line in csv_file:
            if not "Id,Actual" in line:
                line = line.rstrip() # remove trailing newline character            
                list_id_score.append(line.split(","))

        list_id_score = sorted(list_id_score)

        scores = np.zeros(len(list_id_score)) # contains scores

        for idx, elem in enumerate(list_id_score):
            scores[idx] = float(elem[1])

        list_filenames = [] # contains filenames of images

        for filename in os.listdir(arguments.dir_scored_images):
            if filename.endswith(".png") and not filename.startswith("."):
                list_filenames.append(filename)

        list_filenames = sorted(list_filenames)

        assert len(scores) == len(list_filenames)

        # i suppose an image needs approximately
        # sizeof(np.float32) * 1 * 1000 * 1000 B of RAM

        # a list works well in terms of performance
        images = [] # images
        
                
        for idx, filename in enumerate(list_filenames):
            
            if scores[idx] >= 2.61: # include only images with score >= 2.61
      
                img = cv2.imread(os.path.join(arguments.dir_scored_images, filename), cv2.IMREAD_GRAYSCALE)
        
        
                arr = np.array(img, dtype=np.float32).reshape((1000, 1000, 1))

                # map the image data from [0, 255] to [-1.0, 1.0]
                arr = np.subtract(arr, 127.5)
                arr = np.divide(arr, 127.5)

                
                images.append(arr)

                
        return images

    except:
        print("ERROR: failed to load scored images.")        
        
        
def _load_images(arguments):
    
    """ Loads all labeled images and all scored images with score >= 2.61 and returns them as array
    
    
    Parameters
    ----------
    arguments : argparse-arguments
        The command-line arguments
        
        
    Returns
    -------
    images : np.ndarray
        The array of images
        
    """ 
    
    labeled_images = _load_labeled_images(arguments)
    scored_images = _load_scored_images(arguments)
    
    scored_images.extend(labeled_images)
    
    return np.stack(scored_images)  # conversion allows for batches to be extracted


def load_train_test_dataset(arguments, batch_size):
    
    """ Creates a train and a test dataset of galaxy images
    
    Loads all labeled images and all scored images with score >= 2.61 and creates a test and a train 
    dataset based on these.
    
    
    Parameters
    ----------
    arguments : argparse-arguments
        The command-line arguments
    batch_size : int
        The desired batch size for tf.data.Dataset
        
        
    Returns
    -------
    train_dataset : tf.data.Dataset
        The dataset for training
    test_dataset : tf.data.Dataset
        The dataset for testing
        
    """ 
    
    images = _load_images(arguments)
    num_images = len(images)
        
    num_images_train = int(num_images * arguments.frac_train)
    num_images_test = num_images - num_images_train
    
    print("num images train: {}".format(num_images_train))
    print("num images test: {}".format(num_images_test))
        
    train_dataset = tf.data.Dataset.from_tensor_slices(
        images[:num_images_train, :, :, :]).shuffle(num_images_train).batch(batch_size)
    
    test_dataset = tf.data.Dataset.from_tensor_slices(
        images[num_images_train:, :, :, :]).shuffle(num_images_test).batch(batch_size)
    
    return train_dataset, test_dataset

        