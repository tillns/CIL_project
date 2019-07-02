""" Stars Extractor

Takes as input --img_dir a directory containing the 1000x1000 star images. You may use
create_dir_for_labeled_star_images.py to create a directory with only fitting images (without the low-score ones).
The second input --target_dir specifies where to save the 28x28 extracted star patches. It also prints some basic
numbers concerning distribution of the stars.
"""

import os
import sys
import numpy as np
from PIL import Image
import cv2
import argparse
from stars_clustered_distribution import get_mean, get_std

home_dir = os.path.expanduser("~")
parser = argparse.ArgumentParser()
# TODO remove hard coded paths and add help and required flags
parser.add_argument('--img_dir', type=str, default=os.path.join(home_dir,
                    "dataset/cil-cosmology-2018/cosmology_aux_data_170429/labeled1_and_scoredover3"))
parser.add_argument('--target_dir', type=str, default=os.path.join(home_dir,
                    "CIL_project/extracted_stars/labeled1_and_scoredover3"))


def _extract_stars_28x28(image):
    """ Extracts stars from an image

    Detects all stars within the given image, extracts them and centers them within patches of size 28x28 with a black
    background.


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
                               if not (r[0] <= 0 or r[0] + r[2] >= 1000 or r[1] <= 0 or r[1] + r[3] >= 1000)]
    patches = []

    for r in bounding_rects_filtered:

        x = r[0];
        y = r[1];
        w = r[2];
        h = r[3]

        if (w > 28 or h > 28):  # first filter
            continue

        star = image[y: y + h, x: x + w]

        if np.amax(star) < 30:  # second filter
            continue

        patch = np.zeros((28, 28), dtype=np.float32)

        padding_y = (28 - h) // 2
        padding_x = (28 - w) // 2

        patch[padding_y: padding_y + h, padding_x: padding_x + w] = star

        patches.append(patch)

    return patches


if __name__ == '__main__':
    args = parser.parse_args()

    img_dir = args.img_dir
    stars_dir = args.target_dir
    if not os.path.exists(stars_dir):
        os.makedirs(stars_dir)
    img_size = 1000
    num_stars_per_img = []
    max_brightnesses = []
    img_list = os.listdir(img_dir)
    for num, img_name in enumerate(sorted(img_list)):
        img_pil = Image.open(os.path.join(img_dir, img_name)).resize((img_size, img_size))
        img_np = np.array(img_pil, dtype=np.uint8)
        patches = _extract_stars_28x28(img_np)
        num_stars_per_img.append(len(patches))
        for num_patch, patch in enumerate(patches):
            patch = np.uint8(patch)
            max_brightnesses.append(np.max(patch))
            patch_pil = Image.fromarray(patch, 'L')
            patch_pil.save(os.path.join(stars_dir, "{}_star{}.png".format(img_name.split(".png")[0], num_patch)))

    print("Num stars per image: {} +- {}".format(get_mean(num_stars_per_img), get_std(num_stars_per_img)))
    print("Max brightness: {} +- {}".format(get_mean(max_brightnesses), get_std(max_brightnesses)))
