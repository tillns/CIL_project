import os
import sys
import numpy as np
from PIL import Image
import cv2

home_dir = os.path.expanduser("~")
img_dir = os.path.join(home_dir, "dataset/cil-cosmology-2018/cosmology_aux_data_170429/labeled1_and_scoredover3")
stars_dir = os.path.join(home_dir, "CIL_project/extracted_stars_Hannes")
if not os.path.exists(stars_dir):
    os.makedirs(stars_dir)
patch_size = 28  # assumed to be even
dist_stars = 28  # stars may be closer than patch_size
background_threshold = 254
img_size = 1000
num_stars_per_img = []
num_patches_per_img = []
max_brightnesses = []
img_list = os.listdir(img_dir)
use_Hannes_approach = True


def overlap(x, y, stars_pos_list):
    for (xi, yi) in stars_pos_list:
        if xi <= x < xi+dist_stars and yi <= y < yi+dist_stars:
            return True
    return False


def get_mean(some_list):
    return sum(some_list)/len(some_list)



def get_std(some_list):
    sq_sum = 0
    mean = get_mean(some_list)
    for el in some_list:
        sq_sum += pow(el-mean, 2)
    return np.sqrt(sq_sum/(len(some_list)-1))


def get_sourrounding_pixels(x, y):
    pixel_list = []
    adds = list(range(-dist_stars, dist_stars))
    for addx in adds:
        for addy in adds:
            if addx == addy == 0:
                continue
            xi = x + addx
            yi = y + addy
            if 0 <= xi < img_size and 0 <= yi < img_size:
                pixel_list.append((xi, yi))
    return pixel_list


def local_max(x, y, img_np):
    value = img_np[x, y]
    surrounding_pixels = get_sourrounding_pixels(x, y)
    for xi, yi in surrounding_pixels:
        if value < img_np[xi, yi]:
            return False
    return True


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

for num, img_name in enumerate(sorted(img_list)):
    img_pil = Image.open(os.path.join(img_dir, img_name)).resize((img_size, img_size))
    img_np = np.array(img_pil, dtype=np.uint8)
    if use_Hannes_approach:
        patches = _extract_stars_28x28(img_np)
        for num_patch, patch in enumerate(patches):
            patch_pil = Image.fromarray(np.uint8(patch), 'L')
            # patch_pil.show("{} star number {}".format(img_name, counter))
            patch_pil.save(os.path.join(stars_dir, "{}_star{}.png".format(img_name.split(".png")[0], num_patch)))

    else:
        counter = 0
        patch_counter = 0
        stars_pos_list = []
        for x in range(img_size):
            for y in range(img_size):
                value = img_np[x, y]
                if value > background_threshold and local_max(x, y, img_np) and not overlap(x, y, stars_pos_list):
                        stars_pos_list.append((x, y))
                        max_brightnesses.append(value)
                        counter += 1

                        if x - patch_size//2 >= 0 and x + patch_size//2 <= img_size and y - patch_size//2 >= 0 and y + patch_size//2 <= img_size:
                            patch_pil = Image.fromarray(img_np[x-patch_size//2:x+patch_size//2, y-patch_size//2:y+patch_size//2])
                            #patch_pil.show("{} star number {}".format(img_name, counter))
                            patch_pil.save(os.path.join(stars_dir, "{}_star{}.png".format(img_name.split(".png")[0], counter)))
                            patch_counter += 1

        num_stars_per_img.append(counter)
        num_patches_per_img.append(patch_counter)
        print("\rFinished image {}/{} with {}/{} stars saved".format(num+1, len(img_list), patch_counter, counter), end="")


print("")
print("Num stars per image: {} +- {}".format(get_mean(num_stars_per_img), get_std(num_stars_per_img)))
print("Max brightness: {} +- {}".format(get_mean(max_brightnesses), get_std(max_brightnesses)))
print("Num patchable stars per image: {} +- {}".format(get_mean(num_patches_per_img), get_std(num_patches_per_img)))
