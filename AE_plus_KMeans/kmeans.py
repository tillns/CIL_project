from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and tf.keras
import tensorflow as tf


# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os
from random import randint
import PIL
from PIL import Image
import sys
import math
from datetime import datetime
from random import shuffle
from sklearn.linear_model import LinearRegression
import csv
import yaml
import gc
import argparse
from sklearn.cluster import KMeans

image_size = 28
image_channels = 1
home_dir = os.path.expanduser("~")
ae_dir = os.path.join(home_dir, "CIL_project/AE_plus_KMeans")

image_directory = os.path.join(home_dir, "CIL_project/extracted_stars_Hannes")

def load_dataset():
    images = []
    for img_name in sorted(os.listdir(image_directory)):
        img = Image.open(os.path.join(image_directory, img_name)).resize((image_size, image_size))
        img_np = np.array(img, dtype=np.float32).reshape((image_size, image_size, image_channels))/255

        images.append(img_np)
    return np.stack(images)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--encoder_path', type=str, default=None, help='Path to encoder json')
    args = parser.parse_args()

    if args.encoder_path is None:
        args.encoder_path = input("Please provide the full path to the encoder json")
    experiment_dir = os.path.dirname(args.encoder_path)
    with open(os.path.join(experiment_dir, "config.yaml"), 'r') as stream:
        conf = yaml.full_load(stream)

    with open(args.encoder_path) as json_file:
        json_config = json_file.read()
    encoder = tf.keras.models.model_from_json(json_config)
    x = load_dataset()
    latent = encoder(x).numpy()
    latent = np.reshape(latent, (latent.shape[0], -1))
    kmeans = KMeans(n_clusters=10, n_init=20, n_jobs=4)
    # Train K-Means.
    y_pred_kmeans = kmeans.fit_predict(latent)
    print(y_pred_kmeans)