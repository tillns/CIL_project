"""KMEANS

Uses the encoder from ae.py to encode the dataset, which the autoencoder was trained on, into a compact latent space.
The combined latent space for all images is then clustered into a set number of clusters using kmeans.

Input --encoder_path the full path to the json file of the trained encoder (stored in a folder in the checkpoints
directory); --num_clusters the wished number of discrete clusters.

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

parser = argparse.ArgumentParser()
parser.add_argument('-C', '--encoder_path', type=str, default=None, required=True, help='Full path to json file of the trained encoder (stored in the checkpoints directory).')
parser.add_argument('-N', '--num_clusters', type=int, default=5, help='(Optional) Integer specifying number of clusters. Default is 5.')

def copy_clustered_images(y_pred_kmeans, image_directory, clustered_img_dir_base):
    """Dummy Title

    TODO: This is a description of this method/module. Description is optional.

    Parameters
    ----------
    foo : int
    	foo does whatever

    Returns
    -------
    bar : str
    	bar is also whatever
    """

    for num_img, img_name in enumerate(sorted(os.listdir(image_directory))):
        img_path = os.path.join(image_directory, img_name)
        new_img_path = os.path.join(clustered_img_dir_base, "{}/".format(y_pred_kmeans[num_img]))
        os.system("cp {} {}".format(img_path, new_img_path))


if __name__ == '__main__':
    args = parser.parse_args()
    with open(os.path.join(os.path.dirname(args.encoder_path), "config.yaml"), 'r') as stream:
        conf = yaml.full_load(stream)

    image_size = conf['image_size']
    image_channels = 1
    home_dir = os.path.expanduser("~")
    ae_dir = os.path.join(home_dir, "CIL_project/AE_plus_KMeans")
    num_clusters = args.num_clusters

    image_directory = conf['image_dir']
    clustered_img_dir_base = os.path.join(ae_dir,
                                          "clustered_images/{}".format(datetime.now().strftime("%Y%m%d-%H%M%S")))
    os.makedirs(clustered_img_dir_base)
    for i in range(num_clusters):
        os.makedirs(os.path.join(clustered_img_dir_base, "{}".format(i)))

    # TODO why is the config loaded? It is never used, correct?
    experiment_dir = os.path.dirname(args.encoder_path)
    with open(os.path.join(experiment_dir, "config.yaml"), 'r') as stream:
        conf = yaml.full_load(stream)

    with open(args.encoder_path) as json_file:
        json_config = json_file.read()
    encoder = tf.keras.models.model_from_json(json_config)
    x = load_dataset(image_directory, image_size, image_channels)
    latent = encoder(x, training=False).numpy()
    latent = np.reshape(latent, (latent.shape[0], -1))
    kmeans = KMeans(n_clusters=num_clusters, n_init=20, n_jobs=4)
    # Train K-Means.
    y_pred_kmeans = kmeans.fit_predict(latent)
    copy_clustered_images(y_pred_kmeans, image_directory, clustered_img_dir_base)
