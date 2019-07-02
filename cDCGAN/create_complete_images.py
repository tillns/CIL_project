import os
import pickle
import numpy as np
import tensorflow as tf
import argparse
from random import gauss, randint
import matplotlib.pyplot as plt
from img_scorer import score_tensor_with_rf, score_tensor_with_keras_model, load_rf_with_conf, load_km_with_conf, km_transform
from gan_utils import one_hot, detransform_norm


image_channels = 1
image_size = 1000
home_dir = os.path.expanduser("~")

num_stars_per_cat_per_num_cats = {1: {0: (6.390691114245416, 2.527025670888084)},
                                  5: {0: (0.6685472496473907, 0.593681773720953),
                                      1: (3.097790314997649, 1.8328121002868505),
                                      2: (0.6472778561354019, 1.0348597763036949),
                                      3: (1.2030277385989656, 1.3678544263436558),
                                      4: (0.3940479548660083, 0.9661977591756937)}}


def overlap(x, y, stars_pos_list, dist_stars):
    """
    Checks if a new star overlaps with one of the existing stars in the image
    :param x: new star's x-pos
    :param y: new star's y-pos
    :param stars_pos_list: list of existing stars' pos (each pos a tuple with x and y)
    :param dist_stars: the minimum distance allowed between two stars
    :return: True if new star is closer than dist_stars to at least one of the existing stars in stars_pos_list;
             False otherwise
    """
    for (xi, yi) in stars_pos_list:
        if xi <= x < xi+dist_stars and yi <= y < yi+dist_stars:
            return True
    return False


def round_pos_int(some_decimal):
    """
    :param some_decimal: input decimal
    :return: closest unsigned integer, e.g. -0.7 -> 0, 0.2 -> 0, 0.7 -> 1
    """
    if some_decimal < 0:
        return 0
    if some_decimal - int(some_decimal) < 0.5:
        return int(some_decimal)
    return int(some_decimal) + 1


def get_classes_dict(num_classes, kind='int'):
    """
    :param num_classes: Number of star classes
    :param kind: 'int' or 'list'
    :return: a dictionary with a key for each class containing either an integer or a list as value
    """
    dict_to_return = {}
    for i in range(num_classes):
        dict_to_return[i] = 0 if kind == 'int' else []
    return dict_to_return


def save_obj(obj, name):
    """
    Save an obj in the current directory as name.pkl
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """
    :return: loaded obj in the current directory called name.pkl
    """
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def create_complete_images(gen_model, vmin=0, num_images_to_create=100, num_classes=1, load_dict=True):
    """
    Given a generative model (keras kind), this method creates an image tensor containing num_images_to_create images,
    each composed of a black background (vmin) and certain number of stars following the approximated cosmology data
    distribution. If num_classes is 1, an unconditional model is assumed (without a label as input); if num_classes
    is bigger than 1, a conditional model is assumed (with label as input).
    :param gen_model: (keras kind) generative model
    :param vmin: background value
    :param num_images_to_create: number of images to return
    :param num_classes: number of star classes (1 for unconditional), only for 1 and 5, the distributions have been
                        approximated
    :param load_dict: True: saved data for the number of stars, their latent code and position is loaded (if available)
                      False: New latents, number of stars and positions are drawn from the given distributions
    :return: image_tensor of size [num_images_to_create, image_size, image_size, 1],
             and also the drawn star latents and their positions
    """
    if load_dict:
        latents_per_pic = load_obj("latents_per_pic")
        stars_pos = load_obj("stars_pos")
    else:
        num_stars_per_pic = get_classes_dict(num_classes, 'list')
        latents_per_pic = get_classes_dict(num_classes, 'list')
        stars_pos = [[] for i in range(num_images_to_create)]
        for cat_key in num_stars_per_pic:
            for i in range(num_images_to_create):
                latents_per_pic[cat_key].append(0)
                num_stars_per_pic[cat_key].append(round_pos_int(np.random.normal(num_stars_per_cat_per_num_cats
                                                                                 [num_classes][cat_key][0],
                                                                                 num_stars_per_cat_per_num_cats
                                                                                 [num_classes][cat_key][1])))

    star_patch_size = gen_model.output.shape[1]
    image_tensor = np.zeros((num_images_to_create, image_size+star_patch_size, image_size+star_patch_size,
                             image_channels))
    image_tensor = image_tensor + vmin
    for num_img in range(num_images_to_create):
        num_latents = gen_model.input.shape[1] if not isinstance(gen_model.input, list) else gen_model.input[0].shape[1]
        star_img_list = []
        for cat_key in latents_per_pic:

            if load_dict and latents_per_pic[cat_key][num_img] != 0:
                random_latent = latents_per_pic[cat_key][num_img]
            elif not load_dict and num_stars_per_pic[cat_key][num_img] > 0:
                random_latent = tf.random.normal([num_stars_per_pic[cat_key][num_img], num_latents])
                latents_per_pic[cat_key][num_img] = random_latent
            else:
                continue
            if num_classes == 1:
                star_img_list.append(gen_model(random_latent, training=False).numpy())
            else:
                labels = one_hot(np.array([cat_key] * random_latent.shape[0]), num_classes)
                star_img_list.append(gen_model([random_latent, labels], training=False).numpy())

        star_imgs = np.concatenate(star_img_list, axis=0) if len(star_img_list) > 0 else None

        if star_imgs is not None:
            star_pos_list = []
            num_stars = star_imgs.shape[0]
            for i in range(num_stars):
                if not load_dict:
                    while True:
                        x = randint(0, image_size-1)
                        y = randint(0, image_size-1)
                        if not overlap(x, y, star_pos_list, dist_stars=star_patch_size):
                            stars_pos[num_img].append((x, y))
                            break
                    star_pos_list.append((x, y))
                else:
                    x, y = stars_pos[num_img][i]
                image_tensor[num_img, x:x+star_patch_size, y:y+star_patch_size] = star_imgs[i]

    return image_tensor[:, star_patch_size // 2:star_patch_size // 2 + image_size,
                        star_patch_size // 2:star_patch_size // 2 + image_size], latents_per_pic, stars_pos


def find_and_save_best_setting(gen_model, conf, num_classes, rf_model, rf_conf, km_model, km_conf, best_score):
    """
    Loops infinitely to find best latent and pos setting for given model.
    :param gen_model: generative model that produces star patches (conditional or unconditional)
    :param conf: configuraiton of gen_model
    :param num_classes: number of classes for stars (1 means unconditional gen_model)
    :param rf_model: Random Forest classifier model
    :param rf_conf: configuration of rf_model
    :param km_model: keras classifier model
    :param km_conf: configuration of km_model
    :param best_score: Score threshold. Only scores above are counted as new best settings.
    """
    counter = 0
    while True:
        rf_score, nn_score, image_tensor, latents_per_pic, stars_pos \
            = create_and_score_images(gen_model, conf, num_classes, rf_model, rf_conf, km_model, km_conf,
                                      load_dict=False)
        current_score = (np.mean(rf_score) + np.mean(nn_score)) / 2
        if current_score > best_score:
            best_score = current_score
            save_obj(latents_per_pic, "latents_per_pic")
            save_obj(stars_pos, "stars_pos")
            for num_img in range(image_tensor.shape[0]):
                img_np2 = image_tensor[num_img, :, :, 0]
                plt.imsave(os.path.join(os.path.join(home_dir, "CIL_project/cDCGAN/compl_images"),
                                        "{}.png".format(num_img)), img_np2, cmap='gray', vmin=0, vmax=255)
        counter += 1
        print("\rFinished round {} Current best score: {}".format(counter, best_score), end="")


def create_and_score_images(gen_model, conf, num_classes, rf_model, rf_conf, km_model, km_conf, load_dict=True):
    """
    Creates an image tensor with 100 images using the gen_model and scores it using rf_model and km_model
    :param gen_model: generative model that produces star patches (conditional or unconditional)
    :param conf: configuraiton of gen_model
    :param num_classes: number of classes for stars (1 means unconditional gen_model)
    :param rf_model: Random Forest classifier model
    :param rf_conf: configuration of rf_model
    :param km_model: keras classifier model
    :param km_conf: configuration of km_model
    :param load_dict: True: load latents and pos; False: draw new ones
    :return: score tensor containing the score for each image from rf and km, the image tensor, the latents, and pos
    """
    image_tensor, latents, pos = create_complete_images(gen_model, vmin=conf['vmin'], num_images_to_create=100,
                                                        num_classes=num_classes, load_dict=load_dict)

    image_tensor = detransform_norm(image_tensor, conf)
    rf_score = score_tensor_with_rf(image_tensor.copy(), rf_model, rf_conf)
    km_score = score_tensor_with_keras_model(km_transform(image_tensor.copy(), km_conf['use_fft']), km_model,
                                             km_conf['batch_size'])
    return rf_score, km_score, image_tensor, latents, pos


def main(args):
    """
    Either loops infinitely to find best random latents and position for stars or creates a single set of images and
    scores it.
    :param args: parsed arguments
    """
    if args.checkpoint_path is None:
        args.checkpoint_path = input("Please provide the full path to the checkpoint file ending "
                                     "with data-00000-of-00001")

    gen_model, conf = load_km_with_conf(args.checkpoint_path, model_name="gen_config.json")
    num_classes = 1 if not conf['conditional'] else gen_model.input[1].shape[1]
    km_model, km_conf = load_km_with_conf(args.nn_path)
    rf_model, rf_conf = load_rf_with_conf(args.rf_path)
    if args.find_best_num_stars_per_image:
        find_and_save_best_setting(gen_model, conf, num_classes, rf_model, rf_conf, km_model, km_conf, args.min_score)
    else:
        rf_score, km_score, _, _, _ = create_and_score_images(gen_model, conf, num_classes, rf_model, rf_conf, km_model,
                                                              km_conf, load_dict=True)
        score = np.concatenate((np.expand_dims(rf_score, axis=1), km_score))
        print("Score RF: {} +- {}. Score NN; {} +- {}".format(np.mean(rf_score), np.std(rf_score), np.mean(km_score),
                                                              np.std(km_score)))
        print("Mean score {} +- {}".format(np.mean(score), np.std(score)))


# todo: make savable latents and pos dependent on num_classes
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--checkpoint_path', type=str, default=os.path.join(home_dir,
                        "CIL_project/cDCGAN/checkpoints/advanced_upsampling2d/cp_gen_epoch185.data-00000-of-00001"),
                        help='Whole path to checkpoint file ending with data-00000-of-00001')
    parser.add_argument('--nn_path', type=str, default=os.path.join(home_dir, "CIL_project/Classifier/reference_run/"
                        "fft_4convs_8features_MAE/cp-0140.ckpt"),
                        help='Whole path to classfier nn checkpoint file ending with .data-00001....')
    parser.add_argument('--rf_path', type=str, default=os.path.join(home_dir, "CIL_project/RandomForest/final_model/"
                        "random_forest_96_1.sav"), help='Whole path rf model (.sav)')
    parser.add_argument('-f', '--find_best_num_stars_per_image', type=bool, default=True)
    parser.add_argument('--min_score', type=float, default=2.948633233877742,
                        help="Don't save latents and images for scores below min_score.")
    args = parser.parse_args()

    main(args)

