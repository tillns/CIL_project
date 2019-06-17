import os
import sys
import numpy as np
from PIL import Image

home_dir = os.path.expanduser("~")
img_dir = os.path.join(home_dir, "dataset/cil-cosmology-2018/cosmology_aux_data_170429/only_labeled1")
stars_dir = os.path.join(home_dir, "CIL_project/extracted_stars")
if not os.path.exists(stars_dir):
    os.makedirs(stars_dir)
patch_size = 28  # assumed to be even
dist_stars = 10  # stars may be closer than patch_size
background_threshold = 10
img_size = 1000
num_stars_per_img = []
num_patches_per_img = []
max_brightnesses = []
img_list = os.listdir(img_dir)


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
    adds = [-3, -2, -1, 0, 1, 2, 3]
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


for num, img_name in enumerate(sorted(img_list)):
    img_pil = Image.open(os.path.join(img_dir, img_name)).resize((img_size, img_size))
    img_np = np.array(img_pil, dtype=np.uint8)
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
