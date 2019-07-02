import os
import yaml
import tensorflow as tf
import argparse
from PIL import Image
import numpy as np
import joblib
import math
import sys
home_dir = os.path.expanduser("~")
sys.path.insert(0, os.path.join(home_dir, "CIL_project/Classifier"))
sys.path.insert(1, os.path.join(home_dir, "CIL_project/RandomForest"))
from CustomLayers import get_custom_objects
from utils import transform as km_transform
from random_forest_utils import roi_histograms


def load_rf_with_conf(rf_path):
    rf_model = joblib.load(rf_path)
    with open(os.path.join(os.path.dirname(rf_path), "config.yaml"), 'r') as stream:
        conf = yaml.full_load(stream)

    return rf_model, conf


def load_km_with_conf(ckpt_path):
    if ckpt_path.endswith('.data-00000-of-00001'):
        ckpt_path = ckpt_path[:-len('.data-00000-of-00001')]
    model_path = os.path.join("/".join(ckpt_path.split("/")[:-1]), "model_config.json")
    # todo: remove
    sys.path.insert(2, os.path.join(os.path.dirname(ckpt_path), "code"))

    with open(os.path.join("/".join(ckpt_path.split("/")[:-1]), "config.yaml"), 'r') as stream:
        conf = yaml.full_load(stream)
    with open(model_path) as json_file:
        json_config = json_file.read()
    custom_objects = get_custom_objects()
    model = tf.keras.models.model_from_json(json_config, custom_objects=custom_objects)
    model.load_weights(ckpt_path)

    return model, conf


def score_tensor_with_rf(image_tensor, rf_model, conf):

    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor[:, :, :, 0]
    hist_list = []
    for i in range(image_tensor.shape[0]):
        hist_list.append(np.concatenate(roi_histograms(image_tensor[i], conf)))
    hist_tensor = np.stack(hist_list)
    score_tensor = rf_model.predict(hist_tensor)
    return score_tensor


def score_tensor_with_keras_model(image_tensor, model, batch_size):
    score_list = []
    for it in range(math.ceil(image_tensor.shape[0]/batch_size)):
        x_ = image_tensor[it*batch_size:min((it+1)*batch_size, image_tensor.shape[0])]
        score_list.append(model(x_, training=False).numpy())
    score_tensor = np.concatenate(score_list)
    return score_tensor


def get_rf_and_km_img(img_path, res_for_rf, res_for_km, conf):
    img = Image.open(img_path).resize((1000, 1000)).convert('L')
    img_rf = img.resize((res_for_rf, res_for_rf))
    img_np_rf = np.array(img_rf, dtype=np.float32).reshape((res_for_rf, res_for_rf))
    img_km = img.resize((res_for_km, res_for_km))
    use_fft = ('use_fft' in conf and conf['use_fft'])
    img_np_km = km_transform(np.array(img_km, dtype=np.float32).reshape((res_for_km, res_for_km, 1)), use_fft=use_fft)
    return img_np_rf, img_np_km


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None,
                        help='Whole path to dir with images or path to individual image.')
    parser.add_argument('--nn_path', type=str, default=os.path.join(home_dir, "CIL_project/Classifier/reference_run/"
                        "fft_4convs_8features_MAE/cp-0140.ckpt.data-00000-of-00001"),
                        help='Whole path to nn checkpoint file ending with .data-00001....')
    parser.add_argument('--rf_path', type=str, default=os.path.join(home_dir, "/home/tillns/CIL_project/RandomForest/"
                        "final_model/random_forest_96_1.sav"), help='Whole path rf model (.sav)')
    args = parser.parse_args()

    if args.path is None:
        args.path = input("Please provide the full path to the dir with images or to an individual image.")
    ckpt_path = args.nn_path
    km_model, km_conf = load_km_with_conf(ckpt_path)
    rf_model, rf_conf = load_rf_with_conf(args.rf_path)
    res_for_rf = 1000
    res_for_keras_model = km_conf['image_size']

    images_rf = []
    images_kmodel = []
    if os.path.isfile(args.path):
        img_rf, img_km = get_rf_and_km_img(args.path, res_for_rf, res_for_keras_model, km_conf)
        images_rf.append(img_rf)
        images_kmodel.append(img_km)
    else:
        for filename in os.listdir(args.path):
            if (filename.endswith(".png") or filename.endswith(".jpg")) and not filename.startswith("._"):
                img_path = os.path.join(args.path, filename)
                img_rf, img_km = get_rf_and_km_img(img_path, res_for_rf, res_for_keras_model, km_conf)
                images_rf.append(img_rf)
                images_kmodel.append(img_km)

    tensor_rf = np.stack(images_rf)
    tensor_km = np.stack(images_kmodel)
    score_rf = score_tensor_with_rf(tensor_rf, rf_model, rf_conf)
    score_km = score_tensor_with_keras_model(tensor_km, km_model, km_conf['batch_size'])
    print("Mean Score. Random Forest: {} +- {}; Classifier NN: {} +- {}".format(np.mean(score_rf), np.std(score_rf),
                                                                                np.mean(score_km), np.std(score_km)))

