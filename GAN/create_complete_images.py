import os
import pickle
import numpy as np
import tensorflow as tf
import argparse
import yaml
from Models import get_custom_objects
from random import gauss, randint
import matplotlib.pyplot as plt
from img_scorer import score_tensor_with_rf, score_tensor_with_keras_model, load_rf_with_conf, load_km_with_conf, km_transform


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
    for (xi, yi) in stars_pos_list:
        if xi <= x < xi+dist_stars and yi <= y < yi+dist_stars:
            return True
    return False


def round_pos_int(some_decimal):
    if some_decimal < 0:
        return 0
    if some_decimal - int(some_decimal) < 0.5:
        return int(some_decimal)
    return int(some_decimal) + 1


def get_classes_dict(num_classes, kind='int'):
    dict_to_return = {}
    for i in range(num_classes):
        dict_to_return[i] = 0 if kind == 'int' else []
    return dict_to_return


def one_hot(batch_y, num_classes):
    y_ = np.zeros((batch_y.shape[0], num_classes))
    y_[np.arange(batch_y.shape[0]), batch_y] = 1
    return y_


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def create_complete_images(gen_model, vmin=0, num_images_to_create=100, num_classes=1, load_dict=True):

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
                num_stars_per_pic[cat_key].append(round_pos_int(np.random.normal(num_stars_per_cat_per_num_cats[num_classes][cat_key][0],
                                                                                 num_stars_per_cat_per_num_cats[num_classes][cat_key][1])))

    star_patch_size = gen_model.output.shape[1]
    image_tensor = np.zeros((num_images_to_create, image_size+star_patch_size, image_size+star_patch_size,
                             image_channels))
    image_tensor = image_tensor + vmin
    for num_img in range(num_images_to_create):
        num_latents = gen_model.input.shape[1] if not isinstance(gen_model.input, list) else gen_model.input[0].shape[1]
        #if num_classes == 1:

            #random_latent = tf.random.normal([num_stars_per_pic[1][num_img], num_latents])
            #star_imgs = gen_model(random_latent, training=False).numpy() if num_stars_per_pic[1][num_img] > 0 else None
        #else:
        star_img_list = []
        for cat_key in latents_per_pic:

            if load_dict and latents_per_pic[cat_key][num_img] != 0:
                random_latent = latents_per_pic[cat_key][num_img]
            elif not load_dict and num_stars_per_pic[cat_key][num_img] > 0:
                random_latent = tf.random.normal([num_stars_per_pic[cat_key][num_img], num_latents])
                latents_per_pic[cat_key][num_img] = random_latent
            else:
                continue
            labels = one_hot(np.array([cat_key]*random_latent.shape[0]), num_classes)
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

    if load_dict:
        return image_tensor[:, star_patch_size//2:star_patch_size//2+image_size,
                            star_patch_size//2:star_patch_size//2+image_size]
    else:
        return image_tensor[:, star_patch_size // 2:star_patch_size // 2 + image_size,
               star_patch_size // 2:star_patch_size // 2 + image_size], latents_per_pic, stars_pos


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--checkpoint_path', type=str, default=os.path.join(home_dir,
                        "CIL_project/GAN/checkpoints/20190620-211644/cp_gen_epoch400.data-00000-of-00001"),
                        help='Whole path to checkpoint file ending with data-00000-of-00001')
    parser.add_argument('--nn_path', type=str, default=os.path.join(home_dir, "CIL_project/Classifier/reference_run/"
                        "fft_4convs_8features_MAE/cp-0140.ckpt"),
                        help='Whole path to classfier nn checkpoint file ending with .data-00001....')
    parser.add_argument('--rf_path', type=str, default=os.path.join(home_dir, "CIL_project/RandomForest/final_model/"
                        "random_forest_96_1.sav"), help='Whole path rf model (.sav)')
    parser.add_argument('-f', '--find_best_num_stars_per_image', type=bool, default=False)
    args = parser.parse_args()

    if args.checkpoint_path is None:
        args.checkpoint_path = input("Please provide the full path to the checkpoint file ending "
                                     "with data-00000-of-00001")
    experiment_dir = os.path.dirname(args.checkpoint_path)
    with open(os.path.join(experiment_dir, "config.yaml"), 'r') as stream:
        conf = yaml.full_load(stream)

    gen_model_path = os.path.join(experiment_dir, "gen_config.json")
    with open(gen_model_path) as json_file:
        json_config = json_file.read()
    custom_objects = get_custom_objects()
    gen_model = tf.keras.models.model_from_json(json_config, custom_objects=custom_objects)
    gen_model.load_weights(args.checkpoint_path[:-len('.data-00000-of-00001')])
    num_classes = 1 if not conf['conditional'] else gen_model.input[1].shape[1]
    best_score = 0
    num_rounds = 2000
    km_model, km_conf = load_km_with_conf(args.nn_path)
    rf_model, rf_conf = load_rf_with_conf(args.rf_path)
    if args.find_best_num_stars_per_image:
        for i in range(num_rounds):
            image_tensor, latents_per_pic, stars_pos = create_complete_images(gen_model, vmin=conf['vmin'],
                                                                  num_images_to_create=100,
                                                                  num_classes=num_classes, load_dict=False)
            # todo: when GAN class has been rewritten, use detransform method
            image_tensor = (image_tensor - conf['vmin']) / (conf['vmax'] - conf['vmin'])*255
            nn_score = np.mean(score_tensor_with_keras_model(km_transform(image_tensor.copy(),
                                                                          km_conf['use_fft']), km_model,
                                                             km_conf['batch_size']))
            rf_score = np.mean(score_tensor_with_rf(image_tensor[:, :, :, 0], rf_model, rf_conf))

            current_score = (rf_score+nn_score)/2
            if current_score > best_score:
                best_image_tensor = image_tensor
                best_latents_per_pic = latents_per_pic
                best_stars_pos = stars_pos
                best_score = current_score
            print("\rFinished round {}/{} Current best score: {}".format(i+1, num_rounds, best_score), end="")
        image_tensor = best_image_tensor

        save_obj(best_latents_per_pic, "latents_per_pic")
        save_obj(best_stars_pos, "stars_pos")
        for num_img in range(image_tensor.shape[0]):
            img_np2 = image_tensor[num_img, :, :, 0]
            plt.imsave(os.path.join(os.path.join(home_dir, "CIL_project/GAN/compl_images"),
                                    "{}.png".format(num_img)), img_np2, cmap='gray', vmin=0, vmax=255)
    else:
        image_tensor = create_complete_images(gen_model, vmin=conf['vmin'], num_images_to_create=100,
                                              num_classes=num_classes, load_dict=True)
        image_tensor = (image_tensor - conf['vmin']) / (conf['vmax'] - conf['vmin']) * 255
        rf_score = score_tensor_with_rf(image_tensor[:, :, :, 0], rf_model, rf_conf)
        nn_score = score_tensor_with_keras_model(km_transform(image_tensor.copy(), km_conf['use_fft']), km_model,
                                                 km_conf['batch_size'])
        score = np.concatenate((np.expand_dims(rf_score, axis=1), nn_score))
        print("Score RF: {} +- {}".format(np.mean(score), np.std(score)))
