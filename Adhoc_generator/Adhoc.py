"""Adhoc Generator

This file generates cosmology images based on the labelled set given in the kaggle competition.
The adhoc method here randomly places stars that it has detected from the labelled images 
and places them randomly onto a black image.
"""

import numpy as np
import csv
import sys
from PIL import Image
import random
import cv2
import matplotlib.pyplot as plt
import os
import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="Complete path to cosmology_aux_data_170429 dir")
parser.add_argument('-T', '--threshold', type=int, default=4)

# fix the seed for the random number generator
random.seed(9999)

def choose_img(data):
    """Chooses a random image to use as source for a star

    Parameters
    ----------
    data : list
        a list containing the id and label of each image

    Returns
    -------
    src : numpy.ndarray
        A randomly chosen image
    """
    
    num_images = len(data)
    random_img_idx = random.randint(0, num_images-1)
    src = cv2.imread(data_path + '/' + data[random_img_idx] + '.png')
    return src

def detect_stars(src):
    """Detects how many stars are in an image

    Parameters
    ----------
    src : numpy.ndarray
        Image to check the number of stars

    Returns
    -------
    filtered_contours : list of arrays
        List of contours of the stars in the image
    """
    img = np.copy(src) 
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold and detect contours
    thresh = cv2.threshold(imgray, THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh,
                                    cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]

    # filter contours by area
    min_area = 2
    filtered_contours = [c for c in contours
                             if cv2.contourArea(c) >= min_area]
    
    return filtered_contours

def add_random_star(dst, src):
    """Adds a random star from src to dst

    Parameters
    ----------
    src : numpy.ndarray
        Image to take a sample star from
    dst : numpy.ndarray
        Image to add a star to

    Returns
    -------
    dst : numpy.ndarray
        Image with the added star
    """
    
    # filter contours by area
    filtered_contours = detect_stars(src)
    area_contours = [cv2.contourArea(c) for c in filtered_contours]
    # get bounding rectangles
    boundings = [cv2.boundingRect(c) for c in filtered_contours] # (x,y,w,h)

    # add a star from the src image
    random_star = random.randint(0,len(boundings)-1)
    rx = random.randint(0,src.shape[0]-1)
    ry = random.randint(0,src.shape[1]-1)
    xlength = min([rx+boundings[random_star][2], dst.shape[0]])
    ylength = min([ry+boundings[random_star][3], dst.shape[1]])
    dst[ry:ylength, rx:xlength] += \
        src[boundings[random_star][1]:boundings[random_star][1]+ylength-ry, 
            boundings[random_star][0]:boundings[random_star][0]+xlength-rx]
    
    return dst

def find_num_stars(data):
    """Finds the least and highest amount of stars in a cosmology image from the dataset

    Parameters
    ----------
    data : list
        a list containing the id and label of each image

    Returns
    -------
    min_stars : int
        The least amount of stars found in an image in the dataset
    max_stars : int
        The highest amount of stars found in an image in the dataset
    """
    
    min_stars = sys.maxsize
    max_stars = 0
    
    for d in data:
        src = cv2.imread(data_path + '/'  + d + '.png')
        num_stars = len(detect_stars(src))
        if (num_stars > max_stars):
            max_stars = num_stars
        if (num_stars < min_stars):
            min_stars = num_stars
    
    return min_stars, max_stars

def generate_image(data, min_stars, max_stars):
    """Generates a cosmology image using the adhoc method

    Parameters
    ----------
    data : list
        a list containing the id and label of each image
    min_stars : int
        The least amount of stars found in an image in the dataset
    max_stars : int
        The highest amount of stars found in an image in the dataset

    Returns
    -------
    dst : numpy.ndarray
        Adhoc generated cosmology image
    """
    
    # initialize src and dst image
    src = choose_img(data)
    dst = np.zeros((src.shape[0], src.shape[1], src.shape[2]), np.uint8)
    
    # choose number of stars to add
    num_stars = random.randint(min_stars, max_stars)
    
    # add stars
    for i in range(num_stars):
        dst = add_random_star(dst, src)
        # get next random source image to take stars from
        src = choose_img(data) 
    
    # make sure the highest value doesn't go over 254
    dst = np.clip(dst, 0, 254)
    
    return dst


if __name__ == '__main__':
    args = parser.parse_args()
    data_path = args.data_path
    THRESHOLD = args.threshold
    
    # load in data
    with open(data_path + '.csv') as csv_file:
        # 1201 lines, 1 header, 1000 cosmology images, 200 others
        csv_reader = csv.reader(csv_file, delimiter=',')
        header = next(csv_reader)
        data = list(csv_reader)
        # filter the data, such that we only have cosmology images
        data = [x[0] for x in data if x[1] != '0.0']

    # initizalize number of stars
    print('Finding lower and upper bound for number of stars...')
    min_stars, max_stars = find_num_stars(data)

    if not os.path.exists("./Images/"):
        os.mkdir("./Images/")

    # generate images
    print('Generating images...')
    for i in range(100):
        new_img = generate_image(data, min_stars, max_stars)
        cv2.imwrite( "./Images/" + repr(i) + ".jpg", new_img );
        if i == 99:
            print('\r\tGenerated images ' + repr(i+1) + '/100')
        else:
            print('\r\tGenerated images ' + repr(i+1) + '/100', end="\r")
