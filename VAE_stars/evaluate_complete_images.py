"""Complete Image Evaluation

This script uses the trained random forest model to evaluate a set of 100 complete images

"""

from __future__ import absolute_import, division, print_function, unicode_literals

from star_vae import StarVAE
from star_vae_train import parser_vae

import joblib

import numpy as np
import cv2


if __name__ == "__main__":

    arguments = parser_vae.parse_args()

    output_dir_generated_images = arguments.output_dir_generated_images
    path_pretrained_random_forest = arguments.path_pretrained_random_forest
    path_scorefile = arguments.path_scorefile


    random_forest_model = joblib.load(path_pretrained_random_forest)

    hist_list = []


    num_images = 100

    for i in range(num_images):

        complete_image = cv2.imread(output_dir_generated_images + "image" + str(i) + ".png", cv2.IMREAD_GRAYSCALE).astype(np.uint8)

        hist_list.append(np.histogram(complete_image, bins=10, range=(0, 255))[0])


    hist_tensor = np.stack(hist_list)

    scores = random_forest_model.predict(hist_tensor)


    np.savetxt(path_scorefile, scores, delimiter=",", fmt='%f')
