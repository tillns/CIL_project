import os
from Models import TanhLayer, SigmoidLayer, Padder, FactorLayer, ResBlock
import yaml
import tensorflow as tf
import argparse
from PIL import Image
import numpy as np
import joblib
import math

def preprocess(img_np, use_fft=True, kind='km'):
    max_val_fft = 497.75647
    if use_fft:
        img_np = 20 * np.log(np.abs(np.fft.fftshift(np.fft.fft2(img_np))) ** 2 + np.power(1.0, -20))
        if kind == 'km':
            img_np = img_np / max_val_fft
    else:
        if kind == 'km':
            img_np = img_np / 255
    return np.float32(img_np)

def score_tensor_with_rf(image_tensor, val_model=None):
    hist_list = []
    for i in range(image_tensor.shape[0]):
        hist_list.append(np.histogram(image_tensor[i], bins=10)[0])
    hist_tensor = np.stack(hist_list)
    if val_model is None:
        val_model = joblib.load(os.path.join(os.path.join(home_dir, "CIL_project/RandomForest"),
                                             "random_forest_10_0.9.sav"))
    score = val_model.predict(hist_tensor)
    label_eight = np.ones((image_tensor.shape[0], 1)) * 8
    score = tf.reduce_mean(tf.abs(score - label_eight))
    #print("MSE loss of generated images: {}".format(gen_val_loss))
    return score

def score_tensor_with_keras_model(image_tensor, model, batch_size):
    score_list = []
    for it in range(math.ceil(image_tensor.shape[0]/batch_size)):
        x_ = image_tensor[it*batch_size:min((it+1)*batch_size, image_tensor.shape[0])]
        score_list.append(model(x_, training=False).numpy())
    score_tensor = np.concatenate(score_list)
    label_eight = np.ones((image_tensor.shape[0], 1)) * 8
    score = np.mean(np.abs(score_tensor - label_eight))
    return score

def get_rf_and_km_img(img_path, res_for_rf, res_for_km, conf):
    img = Image.open(img_path).resize((1000, 1000)).convert('L')
    img_rf = img.resize((res_for_rf, res_for_rf))
    img_np_rf = preprocess(np.array(img_rf, dtype=np.float32).reshape((res_for_rf, res_for_rf, 1)), kind='rf')
    img_km = img.resize((res_for_km, res_for_km))
    use_fft = ('use_fft' in conf and conf['use_fft'])
    img_np_km = preprocess(np.array(img_km, dtype=np.float32).reshape((res_for_km, res_for_km, 1)), use_fft=use_fft)
    return img_np_rf, img_np_km


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--path', type=str, default=None,
                        help='Whole path to dir with images or path to individual image.')
    args = parser.parse_args()

    if args.path is None:
        args.path = input("Please provide the full path to the dir with images or to an individual image.")
    home_dir = os.path.expanduser("~")
    ckpt_path = os.path.join(home_dir, "CIL_project/Classifier/checkpoints/"
                                       "res1000/fft_3convs_2dense/cp-0182.ckpt")
    model_path = os.path.join("/".join(ckpt_path.split("/")[:-1]), "model_config.json")
    with open(os.path.join("/".join(ckpt_path.split("/")[:-1]), "config.yaml"), 'r') as stream:
        conf = yaml.full_load(stream)
    with open(model_path) as json_file:
        json_config = json_file.read()
    custom_objects = {'Padder': Padder, 'FactorLayer': FactorLayer, 'SigmoidLayer': SigmoidLayer,
                      'TanhLayer': TanhLayer, 'ResBlock': ResBlock}
    model = tf.keras.models.model_from_json(json_config, custom_objects=custom_objects)
    model.load_weights(ckpt_path)
    res_for_rf = 1000
    res_for_keras_model = conf['image_size']

    images_rf = []
    images_kmodel = []
    if os.path.isfile(args.path):
        img_rf, img_km = get_rf_and_km_img(args.path, res_for_rf, res_for_keras_model, conf)
        images_rf.append(img_rf)
        images_kmodel.append(img_km)
    else:
        for filename in os.listdir(args.path):
            if (filename.endswith(".png") or filename.endswith(".jpg")) and not filename.startswith("._"):
                img_path = os.path.join(args.path, filename)
                img_rf, img_km = get_rf_and_km_img(img_path, res_for_rf, res_for_keras_model, conf)
                images_rf.append(img_rf)
                images_kmodel.append(img_km)

    tensor_rf = np.stack(images_rf)
    tensor_km = np.stack(images_kmodel)
    score_rf = score_tensor_with_rf(tensor_rf)
    score_km = score_tensor_with_keras_model(tensor_km, model, conf['batch_size'])
    print("MAE. Random Forest: {}; Classifier NN: {}".format(score_rf, score_km))

