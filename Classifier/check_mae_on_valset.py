"""
A small model that only serves for scoring the validation set with a given tf keras model and calculate mean and std MAE
using the labels.
It takes the following arguments:
required:
--dataset_dir Complete path to cosmology_aux_data_170429 directory
optional:
--cp_path Whole path to nn checkpoint file ending with .data-00000-of-00001
"""

import os
from utils import load_dataset
from CustomLayers import get_custom_objects
import argparse
import yaml
import numpy as np
import tensorflow as tf


if __name__ == '__main__':
    cil_dir = os.path.dirname(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--cp_path', type=str, default=os.path.join(cil_dir, "Classifier/reference_run/cp-0140.ckpt"),
                        help='Whole path to nn checkpoint file ending with .data-00000-of-00001')
    parser.add_argument("--dataset_dir", type=str, required=True, help="Complete path to cosmology_aux_data_170429 dir")
    args = parser.parse_args()
    scored_directory = os.path.join(args.dataset_dir, "scored")
    label_path = os.path.join(args.dataset_dir, "scored.csv")

    if args.cp_path.endswith('.data-00000-of-00001'):
        ckpt_path = args.cp_path[:-len('.data-00000-of-00001')]

    model_path = os.path.join(os.path.dirname(args.cp_path), "model_config.json")
    with open(model_path) as json_file:
        json_config = json_file.read()
    model = tf.keras.models.model_from_json(json_config, custom_objects=get_custom_objects())
    model.load_weights(args.cp_path)
    model.summary()
    with open(os.path.join(os.path.dirname(args.cp_path), "config.yaml"), 'r') as stream:
        conf = yaml.full_load(stream)

    train_images, train_labels, test_images, test_labels, img_list = \
        load_dataset(conf, True, os.path.join(cil_dir, "Classifier"), False, label_path, scored_directory,
                     conf['percentage_train'])

    predictions_list = []
    for i in range(int(np.ceil(test_images.shape[0]/conf['batch_size']))):
        current_prediction = model(test_images[i*conf['batch_size']:min((i+1)*conf['batch_size'],
                                                                        test_images.shape[0])], training=False).numpy()
        predictions_list.append(np.clip(current_prediction, 0, 8))

    predictions_np = np.concatenate(predictions_list)[:, 0]
    diff = np.abs(predictions_np - test_labels)
    print("MAE on test images: {} +- {}".format(np.mean(diff), np.std(diff)))