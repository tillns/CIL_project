import os
import sys
from PIL import Image
import numpy as np
import tensorflow as tf
import argparse
import json
import yaml
from Models import TanhLayer, SigmoidLayer


def create_complete_images(gen_model):
    return True


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
    create_complete_images(gen_model)