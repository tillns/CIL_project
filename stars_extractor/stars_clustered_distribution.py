"""
This file serves to measure the distributions of the different kinds of stars in an image. It assumes a Gaussian
distribution and measures mean and std of every kind of star. For image generation, a distribution containing only
integeres >= 0 is required. Hence, this file approximates a fitting distribution iteratively. The resulting numbers
may be used for the complete image creator in the GAN project.

Takes as arguments:
required:
--unclustered_stars_dir the directory containing directly the star patch images
--clustered_stars_dir the directory containing the cluster folders each containing their star patch images (Note that
these images MUST come from the unclustered directory. The clustering may be achieved with the AE_plus_Kmeans project.)
optional:
--precision relative precision for distribution approximation (the lower, the longer it will take).

Implements public functions:
    #find_arg_of_dir
    #find_good_distr_approx_iteratively
"""

import os
import sys
import glob
import numpy as np
import argparse
cil_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(cil_dir, "cDCGAN"))
from create_complete_images import round_pos_int, get_classes_dict


def find_arg_of_dir(directory):
    """
    :param directory: Directory that belongs to one of the categories
    :return: argument (number) of the directory
    """
    for num_cat, cat in enumerate(categories):
        cat = os.path.join(clustered_stars_dir, cat)
        if cat == directory:
            return num_cat


def find_good_distr_approx_iteratively(mean, std, precision=0.05):
    """ Iteratively approximates a good gaussian distribution that only contains integers >= 0 to the measured mean and
    standard deviation up to a set precision.

    :param mean: float, measured mean of number of stars of specific kind in an image
    :param std: float, measured standard deviation of number of stars of specific kind in an image
    :param precision: float, relative precision for distribution approximation (the lower, the longer it will take)
    :return: mean, float, approximated mean of number of stars of specific kind in an image;
             std, float, approximated standard deviation of number of stars of specific kind in an image
             (These numbers may equal the input if they already approximate the distribution well enough within the set
             precision).
    """

    current_mean = mean
    current_std = std
    current_add = 0.1
    std_mode = True
    while True:
        list_rand = []
        for i in range(100000):
            list_rand.append(round_pos_int(np.random.normal(current_mean, current_std)))
        result_mean = np.mean(list_rand)
        result_std = np.std(list_rand)
        if np.abs(1-result_mean/mean) < precision and np.abs(1-result_std/std) < precision:
            return current_mean, current_std
        if std_mode and result_std > std:
            current_add /= 10
            std_mode = not std_mode
        elif not std_mode and result_mean < mean:
            current_add /= 10
            std_mode = not std_mode
        else:
            if std_mode:
                current_std += current_add
            else:
                current_mean -= current_add


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--unclustered_stars_dir', type=str, required=True, help="Directory that directly contains the "
                        "unclustered extracted star patches")
    parser.add_argument('--clustered_stars_dir', type=str, required=True, help="Directory that contains the cluster "
                        "folders which contain the corresponding extracted star patches, which have to match with the "
                        "unclustered ones")
    parser.add_argument('--precision', type=float, default=0.05, help="Precision for approximation.")
    args = parser.parse_args()
    unclustered_stars_dir = args.unclustered_stars_dir
    clustered_stars_dir = args.clustered_stars_dir

    categories = sorted(os.listdir(clustered_stars_dir))
    num_classes = len(categories)
    num_stars_per_img_per_cat = {}
    for file_name in sorted(os.listdir(unclustered_stars_dir)):
        full_img_name = file_name.split("_")[0]
        if full_img_name not in num_stars_per_img_per_cat:
            num_stars_per_img_per_cat[full_img_name] = get_classes_dict()
        file_list = glob.glob(os.path.join(clustered_stars_dir, "*/{}".format(file_name)))
        label = find_arg_of_dir(os.path.dirname(file_list[0]))
        num_stars_per_img_per_cat[full_img_name][label] += 1

    list_per_cat = get_classes_dict(kind='list')
    list_per_star = []
    for name_key in num_stars_per_img_per_cat:
        num_stars = 0
        for cat_key in num_stars_per_img_per_cat[name_key]:
            list_per_cat[cat_key].append(num_stars_per_img_per_cat[name_key][cat_key])
            num_stars += num_stars_per_img_per_cat[name_key][cat_key]
        list_per_star.append(num_stars)

    for cat_key in list_per_cat:
        print("Distribution of stars for cat {}: {} +- {}". format(cat_key, np.mean(list_per_cat[cat_key]),
                                                                   np.std(list_per_cat[cat_key])))
    print("Num stars per image: {} +- {}".format(np.mean(list_per_star), np.std(list_per_star)))

    distr_per_cat = get_classes_dict()
    new_distr_per_cat = get_classes_dict()
    for cat_key in list_per_cat:
        distr_per_cat[cat_key] = (np.mean(list_per_cat[cat_key]), np.std(list_per_cat[cat_key]))
        new_distr_per_cat[cat_key] = find_good_distr_approx_iteratively(distr_per_cat[cat_key][0],
                                                                        distr_per_cat[cat_key][1], args.precision)
        print("Adjusted distribution for cat {}: ({}, {})".format(cat_key, new_distr_per_cat[cat_key][0],
                                                                  new_distr_per_cat[cat_key][1]))

    # the following only checks if approximated distribution resembles measured one
    print("Sanity check:")
    list_rand = get_classes_dict('list')
    num_els = 100000
    list_measured_num_stars = [0] * num_els
    for i in range(num_els):
        for cat_key in new_distr_per_cat:
            list_rand[cat_key].append(round_pos_int(np.random.normal(new_distr_per_cat[cat_key][0],
                                                                     new_distr_per_cat[cat_key][1])))
            list_measured_num_stars[i] += list_rand[cat_key][i]

    for cat_key in list_rand:
        print("New Measured distr for cat {}: {} +- {}".format(cat_key, np.mean(list_rand[cat_key]),
                                                               np.std(list_rand[cat_key])))

    print("New measured num stars per image: {} +- {}".format(np.mean(list_measured_num_stars),
                                                              np.std(list_measured_num_stars)))
