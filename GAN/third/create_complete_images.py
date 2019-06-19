import os
import sys
from PIL import Image
import numpy as np
import tensorflow as tf
import argparse
import json
import yaml
from Models import TanhLayer, SigmoidLayer
from random import gauss, randint
import matplotlib.pyplot as plt
import joblib

image_channels = 1
image_size = 1000
home_dir = os.path.expanduser("~")

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

def score_tensor(image_tensor):
    hist_list = []
    for i in range(image_tensor.shape[0]):
        hist_list.append(np.histogram(image_tensor[i], bins=10)[0])
    hist_tensor = np.stack(hist_list)
    val_model = joblib.load(os.path.join(os.path.join(home_dir, "CIL_project/RandomForest"), "random_forest_10_0.9.sav"))
    score = val_model.predict(hist_tensor)
    label_eight = np.ones((image_tensor.shape[0], 1)) * 8
    gen_val_loss = tf.reduce_mean(tf.square(score - label_eight))
    print("MSE loss of generated images: {}".format(gen_val_loss))


def create_complete_images(gen_model, vmin=0, num_images_to_create=100):
    num_stars_per_pic = []
    for i in range(num_images_to_create):
        num_stars_per_pic.append(round_pos_int(gauss(12.3, 4)))

    star_patch_size = gen_model.output.shape[1]
    image_tensor = np.zeros((num_images_to_create, image_size+star_patch_size, image_size+star_patch_size,
                             image_channels))
    image_tensor = image_tensor + vmin
    for num_img, num_stars in enumerate(num_stars_per_pic):
        random_latent = tf.random.normal([num_stars, gen_model.input.shape[1]])
        star_imgs = gen_model(random_latent).numpy()
        star_pos_list = []
        for i in range(num_stars):
            while True:
                x = randint(0, image_size-1)
                y = randint(0, image_size-1)
                if not overlap(x, y, star_pos_list, dist_stars=star_patch_size):
                    break
            star_pos_list.append((x, y))
            image_tensor[num_img, x:x+star_patch_size, y:y+star_patch_size] = star_imgs[i]

    return image_tensor[:, star_patch_size//2:star_patch_size//2+image_size,
                        star_patch_size//2:star_patch_size//2+image_size]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--checkpoint_path', type=str, default=None, help='Whole path to checkpoint file ending with data-00000-of-00001')
    args = parser.parse_args()

    if args.checkpoint_path is None:
        args.checkpoint_path = input("Please provide the full path to the checkpoint file ending with data-00000-of-00001")
    experiment_dir = os.path.dirname(args.checkpoint_path)
    with open(os.path.join(experiment_dir, "config.yaml"), 'r') as stream:
        conf = yaml.full_load(stream)

    gen_model_path = os.path.join(experiment_dir, "gen_config.json")
    with open(gen_model_path) as json_file:
        json_config = json_file.read()
    custom_objects = {'TanhLayer': TanhLayer} if conf['vmin'] == -1 else {'Sigmoidayer': SigmoidLayer}
    gen_model = tf.keras.models.model_from_json(json_config, custom_objects=custom_objects)
    gen_model.load_weights(args.checkpoint_path[:-len('.data-00000-of-00001')])
    image_tensor = create_complete_images(gen_model, vmin=conf['vmin'], num_images_to_create=100)

    if True:
        for num_img in range(image_tensor.shape[0]):
            img_np2 = image_tensor[num_img, :, :, 0]
            #plt.imshow(img_np2, cmap='gray', vmin=conf['vmin'], vmax=conf['vmax'])
            plt.imsave(os.path.join(os.path.join(home_dir, "CIL_project/GAN/third/compl_images"),
                                    "{}.png".format(num_img)), img_np2, cmap='gray', vmin=conf['vmin'], vmax=conf['vmax'])
            #plt.show()
    score_tensor(image_tensor)