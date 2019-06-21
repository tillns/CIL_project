import os
import glob
import numpy as np
import tensorflow as tf

def get_mean(some_list):
    return sum(some_list)/len(some_list)


def get_std(some_list):
    sq_sum = 0
    mean = get_mean(some_list)
    for el in some_list:
        sq_sum += pow(el-mean, 2)
    return np.sqrt(sq_sum/(len(some_list)-1))


def find_arg_of_dir(dir):
    for num_cat, cat in enumerate(categories):
        cat = os.path.join(clustered_stars_dir, cat)
        if cat == dir:
            return num_cat

def get_classes_dict(kind='int'):
    dict_to_return = {}
    for i in range(num_classes):
        dict_to_return[i] = 0 if kind == 'int' else []
    return dict_to_return

def round_pos_int(some_decimal):
    if some_decimal < 0:
        return 0
    if some_decimal - int(some_decimal) < 0.5:
        return int(some_decimal)
    return int(some_decimal) + 1

def round_int(some_decimal):
    if some_decimal < 0:
        if some_decimal - int(some_decimal) > -0.5:
            return int(some_decimal)
        return int(some_decimal) - 1
    if some_decimal - int(some_decimal) < 0.5:
        return int(some_decimal)
    return int(some_decimal) + 1

def find_good_distr_approx_iteratively(mean, std):
    current_mean = mean
    current_std = std
    precision = 0.05
    current_add = 0.1
    std_mode = True
    while True:
        list_rand = []
        for i in range(100000):
            list_rand.append(round_pos_int(np.random.normal(current_mean, current_std)))
        result_mean = get_mean(list_rand)
        result_std = get_std(list_rand)
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

    home_dir = os.path.expanduser("~")
    unclustered_stars_dir = os.path.join(home_dir, "CIL_project/extracted_stars_Hannes")
    clustered_stars_dir = os.path.join(home_dir,
                                       "CIL_project/AE_plus_KMeans/clustered_images/labeled1_and_scoredover3_5cats")

    categories = sorted(os.listdir(clustered_stars_dir))
    num_classes = len(categories)
    num_stars_per_img_per_cat = {}
    for file_name in sorted(os.listdir(unclustered_stars_dir)):
        full_img_name = file_name.split("_")[0]
        if not full_img_name in num_stars_per_img_per_cat:
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

    distr_per_cat = get_classes_dict()
    new_distr_per_cat = get_classes_dict()
    for cat_key in list_per_cat:
        distr_per_cat[cat_key] = (get_mean(list_per_cat[cat_key]), get_std(list_per_cat[cat_key]))
        new_distr_per_cat[cat_key] = find_good_distr_approx_iteratively(distr_per_cat[cat_key][0],
                                                                        distr_per_cat[cat_key][1])
        print("Adjusted distribution for cat {}: ({}, {})".format(cat_key, new_distr_per_cat[cat_key][0],
                                                                         new_distr_per_cat[cat_key][1]))


    list_rand = get_classes_dict('list')
    num_els = 100000
    list_measured_num_stars = [0] * num_els
    for i in range(num_els):
        for cat_key in new_distr_per_cat:
            list_rand[cat_key].append(round_pos_int(np.random.normal(new_distr_per_cat[cat_key][0],
                                                                     new_distr_per_cat[cat_key][1])))
            list_measured_num_stars[i] += list_rand[cat_key][i]

    for cat_key in list_rand:
        print("New Measured distr for cat {}: {} +- {}".format(cat_key, get_mean(list_rand[cat_key]),
                                                               get_std(list_rand[cat_key])))


    print("Num stars per image: {} +- {}".format(get_mean(list_per_star), get_std(list_per_star)))
    print("New measured num stars per image: {} +- {}".format(get_mean(list_measured_num_stars),
                                                              get_std(list_measured_num_stars)))