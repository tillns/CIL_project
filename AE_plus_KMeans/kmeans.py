"""KMEANS

Uses the encoder from ae.py to encode the dataset, which the autoencoder was trained on, into a compact latent space.
The combined latent space for all images is then clustered into a set number of clusters using kmeans.

This module takes the following arguments:
required:
--encoder_path the full path to the json file of the trained encoder (stored in a folder in the checkpoints directory)
optional
--num_clusters the wished number of discrete clusters.
--target_dir Directory in which to save clustered stars.

Following public methods are implemented:
    #copy_clustered_images

"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

# Helper libraries
import numpy as np
import os
from datetime import datetime
import yaml
import argparse
from sklearn.cluster import KMeans
from ae import load_dataset


def copy_clustered_images(y_pred_kmeans, image_directory, clustered_img_dir_base):
    """
    Copies the images from their original directory to their corresponding cluster target directory
    :param y_pred_kmeans: tensor with cluster prediction for each image
    :param image_directory: original image directory
    :param clustered_img_dir_base: base directory of the target clusters (will contain folder for each cluster)
    """

    for num_img, img_name in enumerate(sorted(os.listdir(image_directory))):
        img_path = os.path.join(image_directory, img_name)
        new_img_path = os.path.join(clustered_img_dir_base, "{}/".format(y_pred_kmeans[num_img]))
        os.system("cp {} {}".format(img_path, new_img_path))


if __name__ == '__main__':
    cil_dir = os.path.dirname(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--encoder_path', type=str, required=True,
                        help='Full path to json file of the trained encoder (stored in the checkpoints directory).')
    parser.add_argument('--target_dir', type=str, default=os.path.join(cil_dir,
                        "images/clustered_stars/{}".format(datetime.now().strftime("%Y%m%d-%H%M%S"))),
                        help='Directory in which to save clustered stars.')
    parser.add_argument('-N', '--num_clusters', type=int, default=5,
                        help='Integer specifying number of clusters. Default is 5.')
    args = parser.parse_args()
    with open(os.path.join(os.path.dirname(args.encoder_path), "config.yaml"), 'r') as stream:
        conf = yaml.full_load(stream)

    image_size = conf['image_size']
    image_channels = 1

    num_clusters = args.num_clusters
    image_directory = os.path.join(cil_dir, conf['image_dir'])
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)
    for i in range(num_clusters):
        os.makedirs(os.path.join(args.target_dir, "{}".format(i)))

    with open(args.encoder_path) as json_file:
        json_config = json_file.read()
    encoder = tf.keras.models.model_from_json(json_config)
    x = load_dataset(image_directory, image_size, image_channels)
    latent = encoder(x, training=False).numpy()
    latent = np.reshape(latent, (latent.shape[0], -1))
    kmeans = KMeans(n_clusters=num_clusters, n_init=20, n_jobs=4)
    # Train K-Means.
    y_pred_kmeans = kmeans.fit_predict(latent)
    copy_clustered_images(y_pred_kmeans, image_directory, args.target_dir)
