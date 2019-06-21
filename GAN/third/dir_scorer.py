import os
from create_complete_images import score_tensor_with_rf, score_tensor_with_keras_model
from Models import TanhLayer, SigmoidLayer, Padder, FactorLayer
import yaml
import tensorflow as tf
import argparse
from PIL import Image
import numpy as np



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir_path', type=str, default=None, help='Whole path to dir with images')
    args = parser.parse_args()

    if args.dir_path is None:
        args.dir_path = input("Please provide the full path to the dir with images")
    home_dir = os.path.expanduser("~")
    ckpt_path = os.path.join(home_dir, "CIL_project/Classifier/checkpoints/3convs/cp-0137.ckpt")
    model_path = os.path.join("/".join(ckpt_path.split("/")[:-1]), "model_config.json")
    with open(os.path.join("/".join(ckpt_path.split("/")[:-1]), "config.yaml"), 'r') as stream:
        conf = yaml.full_load(stream)
    with open(model_path) as json_file:
        json_config = json_file.read()
    custom_objects = {'Padder': Padder, 'FactorLayer': FactorLayer, 'SigmoidLayer': SigmoidLayer}
    model = tf.keras.models.model_from_json(json_config, custom_objects=custom_objects)
    model.load_weights(ckpt_path)
    res_for_rf = 1000
    res_for_keras_model = conf['image_size']

    images_rf = []
    images_kmodel = []
    for filename in os.listdir(args.dir_path):
        if filename.endswith(".png") and not filename.startswith("._"):
            img = Image.open(os.path.join(args.dir_path, filename), 'L').resize((1000, 1000))
            img_rf = img.resize((res_for_rf, res_for_rf))
            img_np = np.array(img_rf, dtype=np.float32).reshape((res_for_rf, res_for_rf, 1))/255
            images_rf.append(img_np)
            img_km = img.resize((res_for_keras_model, res_for_keras_model))
            img_np = np.array(img_km, dtype=np.float32).reshape((res_for_keras_model, res_for_keras_model, 1)) / 255
            images_kmodel.append(img_np)

    tensor_rf = np.stack(images_rf)
    tensor_km = np.stack(images_kmodel)
    score_rf = score_tensor_with_rf(tensor_rf)
    score_km = score_tensor_with_keras_model(tensor_km, model, conf['batch_size'])
    print("MSE. Random Forest: {}; Classifier NN: {}".format(score_rf, score_km))

