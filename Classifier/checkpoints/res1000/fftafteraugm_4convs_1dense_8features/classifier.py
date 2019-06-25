from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and tf.keras
import tensorflow as tf


# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os
from random import randint
import PIL
from PIL import Image
import sys
import math
from datetime import datetime
from random import shuffle
from sklearn.linear_model import LinearRegression
import csv
import yaml
import gc
import argparse
from tensorflow.python.keras.utils.data_utils import Sequence
import cv2
from albumentations import (
    Compose, HorizontalFlip, ShiftScaleRotate, VerticalFlip, DualTransform
)


print(tf.__version__)

"""## Import dataset"""


def get_shuffled_train_plus_val_list():
    list1 = list(range(round(dataset_len * percentage_train)))
    list2 = list(range(round(dataset_len * percentage_train), dataset_len))
    if not test_on_query:
        shuffle(list1)
        shuffle(list2)
    return list1[:train_len] + list2[:test_len]


def get_random_indices(begin=0, end=9600, num_indices=25):
    list1 = list(range(begin, end))
    shuffle(list1)
    return list1[:num_indices]


def isInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def test_model():
    indices = get_random_indices()
    plt.figure(figsize=(10, 10))
    for i in range(25):
        ind = indices[i]
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        label = label_list[ind][1]
        img = Image.open(os.path.join(scored_directory, label_list[ind][0] + ".png")).resize((image_size, image_size))
        img_np = np.array(img, dtype=np.float32).reshape((1, image_size, image_size, image_channels)) / 255
        score = model(img_np, training=False)
        plt.imshow(img_np[0, :, :, 0], cmap='gray', vmin=0, vmax=1)
        plt.xlabel("{}\no: {}\ny: {}".format(label_list[ind][0], score.numpy()[0, 0], label))
    plt.show()

parser = argparse.ArgumentParser()
if __name__ == '__main__':

    parser.add_argument('-C', '--is_cluster', type=bool, default=False, help='Set to true if code runs on cluster.')
    parser.add_argument('-T', '--test_on_query', type=bool, default=False)
    parser.add_argument('-R', '--restore_ckpt', type=bool, default=False, help="Can't be false if test_on_query is True")
    parser.add_argument('-P', '--ckpt_path', type=str, default=None, help="Complete path to ckpt file ending with .data_00001...")

args = parser.parse_args()

with open("config.yaml", 'r') as stream:
    conf = yaml.full_load(stream)

test_on_query = args.test_on_query
restore_checkpoint = True if test_on_query else args.restore_ckpt
image_size = conf['image_size']
image_channels = 1
home_dir = os.path.expanduser("~")
classifier_dir = os.path.join(home_dir, "CIL_project/Classifier")
scored_directory = os.path.join(home_dir, "dataset/cil-cosmology-2018/cosmology_aux_data_170429/scored")
label_path = os.path.join(home_dir, "dataset/cil-cosmology-2018/cosmology_aux_data_170429/scored.csv")
query_directory = os.path.join(home_dir, "dataset/cil-cosmology-2018/cosmology_aux_data_170429/query")
# query_directory = os.path.join(home_dir, "CIL_project/GAN/third/compl_images")
image_directory = query_directory if test_on_query else scored_directory
percentage_train = conf['percentage_train'] if not test_on_query else 1
print("Searching for images in {}".format(image_directory))
use_dummy_dataset = False
period_to_save_cp = conf['period_to_save_cp'] if percentage_train == 1 else 1
tot_num_epochs = conf['tot_num_epochs']
label_range = 8  # labels go from 0 to 8
save_np_to_mem = image_size > 250 and not args.is_cluster
max_val_fft = 497.75647

try:
    f = open(label_path, 'r')
    print("Found Labels")
    label_list = []
    for line in f:
        if not "Id,Actual" in line:
            split_line = line.split(",")
            split_line[-1] = float(split_line[-1])
            label_list.append(split_line)
    label_list = sorted(label_list)

    imgs = []
    img_list = []
    for filename in os.listdir(image_directory):
        if filename.endswith(".png") and not filename.startswith("._"):
            img_list.append(filename)

    img_list = sorted(img_list)
    dataset_len = len(img_list)
    train_len = round(percentage_train * dataset_len)
    test_len = dataset_len - train_len
    if not test_on_query:
        assert len(label_list) == dataset_len

except FileNotFoundError:
    print("Dataset not found. Using dummy dataset")
    use_dummy_dataset = True


def get_numpy(mode, num_images, path):
    if save_np_to_mem:
        return np.memmap(path, dtype=np.float32, mode=mode,
                         shape=(num_images, image_size, image_size, 1))
    return np.zeros((num_images, image_size, image_size, 1), dtype=np.float32)


class FFT_augm(DualTransform):

    def __init__(self, max_val_fft, use_fft=True):
        super(FFT_augm, self).__init__(True, 1)
        self.max_val_fft = max_val_fft
        self.use_fft = use_fft

    def apply(self, img, **params):
        if self.use_fft:
            img_np = 20*np.log(np.abs(np.fft.fftshift(np.fft.fft2(img)))**2+np.power(1.0, -20)) / self.max_val_fft
        else:
            img_np = img / 255
        return img_np

def load_dataset():
    global train_images, train_labels, test_images, test_labels
    if not use_dummy_dataset:
        if not "use_fft" in conf:
            conf['use_fft'] = False
        np_data_path = os.path.join(classifier_dir,"numpy_data/{}_p{:.1f}_s{}.dat".format('{}', percentage_train,
                                                                                            image_size))
        trainset_path = np_data_path.format("queryset") if test_on_query else np_data_path.format("trainset")
        testset_path = np_data_path.format('testset')
        both_paths_exist = save_np_to_mem and os.path.exists(trainset_path) and (percentage_train == 1 or os.path.exists(testset_path))
        mode = 'r+' if both_paths_exist else 'w+'
        train_images = get_numpy(mode, int(dataset_len*percentage_train), trainset_path)
        train_labels = np.zeros(shape=(train_images.shape[0],), dtype=np.float32)
        if percentage_train < 1:
            test_images = get_numpy(mode, dataset_len - int(dataset_len * percentage_train), testset_path)
            test_labels = np.zeros(shape=(test_images.shape[0],), dtype=np.float32)
        for num in range(dataset_len):
            if not both_paths_exist:
                img = Image.open(os.path.join(image_directory, img_list[num])).resize((image_size, image_size))
                img_np = np.array(img, dtype=np.float32).reshape((image_size, image_size, image_channels))
                #if conf['use_fft']:
                    #img_np = 20*np.log(np.abs(np.fft.fftshift(np.fft.fft2(img_np)))**2+np.power(1.0, -20)) / max_val_fft
                #else:
                    #img_np = img_np / 255
            if num < int(dataset_len * percentage_train):
                train_labels[num] = label_list[num][1]
                if not both_paths_exist:
                    train_images[num] = img_np

            else:
                test_labels[num - int(dataset_len * percentage_train)] = label_list[num][1]
                if not both_paths_exist:
                    test_images[num - int(dataset_len * percentage_train)] = img_np
            print("\rLoaded image {}/{}".format(num + 1, dataset_len), end="")
        print("")
        # following 4 lines gave min_val = 0 and max_val = 497.75647 as used above
        # max_test = -1000 if percentage_train == 1 else np.max(test_images)
        # min_test = 1000 if percentage_train == 1 else np.min(test_images)
        # max_val = max(np.max(train_images), max_test)
        # min_val = min(np.min(train_images), min_test)



    else:
        train_labels = np.ones((train_len, 1))
        test_labels = np.ones((test_len, 1))
        train_images = np.random.normal(0, 1, (train_len, image_size, image_size, image_channels))
        test_images = np.random.normal(0, 1, (test_len, image_size, image_size, image_channels))


"""Loading the dataset returns four NumPy arrays:

* The `train_images` and `train_labels` arrays are the *training set*—the data the model uses to learn.
* The model is tested against the *test set*, the `test_images`, and `test_labels` arrays.

The images are 1000x1000 NumPy arrays, with pixel values ranging between 0 and 255. The *labels* are floats ranging from 0 to 8, with 8 denoting a star image completely representing the distribution. We will consider scaling the labels by 1/8 s.t. the reange can be easily represented by a classifier ending with a sigmoid activation.

## Build the model

Building the neural network requires configuring the layers of the model, then compiling the model. Two types of models will be evaluated and compared. The first one downsamples further (from Till) than the second one (from Hannes).
"""


def get_pad(x, total_padding=0):
    if total_padding == 0:
        return x
    elif total_padding > 0:
        rand1 = randint(0, total_padding)
        rand2 = randint(0, total_padding)
        return tf.pad(x, tf.constant([[0, 0], [rand1, total_padding - rand1], [rand2, total_padding - rand2], [0, 0]]))
    else:
        total_padding = abs(total_padding)
        rand1 = randint(0, total_padding)
        rand2 = randint(0, total_padding)
        s = x.shape
        return x[:, rand1:s[1] - total_padding + rand1, rand2:s[2] - total_padding + rand2]


class Padder(tf.keras.layers.Layer):
    def __init__(self, padding=6, **kwargs):
        super(Padder, self).__init__(**kwargs)
        self.padding = padding

    def call(self, x):
        return get_pad(x, self.padding)

    def get_config(self):
        #return {'padding': self.padding}
        config = {'padding': self.padding}
        base_config = super(Padder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        output_shape[1] += self.padding
        output_shape[2] += self.padding
        return output_shape


class Pixel_norm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-8):
        super(Pixel_norm, self).__init__()
        self.epsilon = epsilon

    def call(self, x):
        # print("input shape in pixel norm layer: {}".format(x.shape))
        return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + self.epsilon)

    def get_config(self):
        return {'epsilon': self.epsilon}


class FactorLayer(tf.keras.layers.Layer):
    def __init__(self, factor):
        super(FactorLayer, self).__init__()
        self.factor = factor

    def call(self, x):
        return self.factor * x

    def get_config(self):
        return {'factor': self.factor}


def getNormLayer(norm_type='batch', momentum=0.9, epsilon=1e-5):
    if norm_type == 'pixel':
        return Pixel_norm(epsilon)
    if norm_type == 'batch':
        return tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)
    return FactorLayer(1)


class SigmoidLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(SigmoidLayer, self).__init__()

    def call(self, x):
        return tf.keras.activations.sigmoid(x)

    def get_config(self):
        return {}


class CrossEntropy(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        from tensorflow.python.ops import math_ops
        from tensorflow.python.framework import ops
        y_pred = ops.convert_to_tensor(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        return - math_ops.log(1 - math_ops.abs(y_true - y_pred) / label_range)


class CustomLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(CustomLoss, self).__init__()
        train_loss_type = conf['train_loss_type']
        if train_loss_type == "CE":
            self.train_loss = CrossEntropy()  # from_logits=True ?
        elif train_loss_type == "MSE":
            self.train_loss = tf.keras.losses.MeanSquaredError()
        else:
            self.train_loss = tf.keras.losses.MeanAbsoluteError()

        val_loss_type = conf['val_loss_type']
        if val_loss_type == "CE":
            self.val_loss = CrossEntropy()  # from_logits=True ?
        elif val_loss_type == "MSE":
            self.val_loss = tf.keras.losses.MeanSquaredError()
        else:
            self.val_loss = tf.keras.losses.MeanAbsoluteError()

    def call(self, y_true, y_pred):
        return tf.keras.backend.in_train_phase(self.train_loss(y_true, y_pred), self.val_loss(y_true, y_pred))


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, conf, downsample, features, last_features, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.conf = conf
        self.model = tf.keras.Sequential()
        self.downsample = downsample
        self.features = features
        self.last_features = last_features
        self.projection = features != last_features or downsample
        downsample_stride = 1 if conf['use_max_pool'] else conf['downsample_factor']
        for j in range(conf['num_convs_per_block']):
            downsample_this_conv = downsample and ((j == conf['num_convs_per_block'] - 1 and
                                                    not conf['downsample_with_first_conv']) or
                                                   (j == 0 and conf['downsample_with_first_conv']))
            strides = downsample_stride if downsample_this_conv else 1
            self.model.add(tf.keras.layers.Conv2D(features, (conf['kernel'], conf['kernel']),
                                                  padding='same', strides=strides, use_bias=conf['use_bias']))
            self.model.add(getNormLayer(conf['norm_type']))
            self.model.add(tf.keras.layers.LeakyReLU(alpha=conf['lrelu_alpha']))
            if downsample_this_conv and conf['use_max_pool']:
                self.model.add(tf.keras.layers.MaxPool2D(pool_size=conf['downsample_factor']))
        if conf['residual']:
            self.projection_model = tf.keras.Sequential()
            if self.projection:
                if self.downsample and not self.conf['use_max_pool']:
                    self.projection_model.add(tf.keras.layers.Conv2D(self.features, (self.conf['downsample_factor'],
                                                                          self.conf['downsample_factor']),
                                                          padding='same', strides=self.conf['downsample_factor'],
                                                          use_bias=self.conf['use_bias']))
                elif conf['use_max_pool']:
                    self.projection_model.add(tf.keras.layers.Conv2D(self.features, (1, 1), padding='same', strides=1,
                                                          use_bias=self.conf['use_bias']))
                    if self.downsample:
                        self.projection_model.add(tf.keras.layers.MaxPool2D(pool_size=self.conf['downsample_factor']))

    def call(self, x, training=False):
        input_tensor = x
        x = self.model(x, training=training)
        if self.conf['residual']:
            x += self.projection_model(input_tensor, training=training)
        return x

    def get_config(self):
        config = {'conf' : self.conf, 'downsample': self.downsample,
                  'features' : self.features, 'last_features' : self.last_features}
        base_config = super(ResBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Augm_Sequence(Sequence):
    def __init__(self, x_set, y_set, batch_size, augmentations, shuffle=False):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.augment = augmentations
        self.shuffle = shuffle
        self.indices = list(range(self.x.shape[0]))

    def __len__(self):
        return int(np.ceil(self.x.shape[0] / batch_size))

    def __getitem__(self, idx):
        max_idx = min(self.x.shape[0], (idx + 1)*self.batch_size)
        batch_x = self.x[self.indices[idx * self.batch_size:max_idx]]
        batch_y = self.y[self.indices[idx * self.batch_size:max_idx]]

        return np.stack([
            self.augment(image=x)["image"] for x in batch_x
        ], axis=0), np.array(batch_y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)



# params for model 1
def get_model():
    features = conf['features']
    last_features = image_channels
    res = image_size
    resexp = [pow(2, x) for x in range(10)]

    model = tf.keras.Sequential(name='till_model')
    while res > conf['min_res']:
        if res % conf['downsample_factor'] != 0:
            closestexpres = min(resexp, key=lambda x: abs(x - res))
            model.add(Padder(padding=closestexpres - res,
                             input_shape=(res, res, last_features)))  # 125 -> 128 or maybe 25 -> 32...
            res = closestexpres
        for i in range(conf['num_blocks_per_res']):
            downsample = (i == 0 and conf['downsample_with_first_conv']) or \
                         (i == conf['num_blocks_per_res'] - 1 and not conf['downsample_with_first_conv'])
            model.add(ResBlock(conf, downsample, features, last_features, input_shape=(res, res, last_features)))
        last_features = features
        if features < conf['max_features']:
            features *= 2
        res = res // conf['downsample_factor']

    model.add(tf.keras.layers.Flatten())
    for i in range(1, conf['num_dense_layers']):
        model.add(tf.keras.layers.Dense(conf['latent_features'], use_bias=conf['use_bias']))
        model.add(getNormLayer(conf['norm_type']))
        model.add(tf.keras.layers.LeakyReLU(alpha=conf['lrelu_alpha']))
        if conf['dropout'] > 0:
            model.add(tf.keras.layers.Dropout(conf['dropout']))
    if conf['num_dense_layers'] == 1:
        model.add(tf.keras.layers.Dropout(conf['dropout']))
    model.add(tf.keras.layers.Dense(1, use_bias=conf['use_bias']))
    # model.add(SigmoidLayer())
    # model.add(FactorLayer(8))  # to make output range from 0 to 8
    return model


def get_latest_cp(epoch=None):
    if not os.path.exists(checkpoint_dir):
        return None
    list_dir = os.listdir(checkpoint_dir)
    if len(list_dir) == 0:
        return None
    list_dir = sorted(list_dir)
    if epoch is not None:
        specific_cp_path = os.path.join(checkpoint_dir, os.path.join(list_dir[-1], "cp-{0:04d}.ckpt".format(epoch)))
        return specific_cp_path
    return tf.train.latest_checkpoint(os.path.join(checkpoint_dir, list_dir[-1]))


def get_epoch_and_path(path):
    if path is None:
        return 0, None
    filename = path.split("/")[-1]
    epoch = int(filename.split("-")[1].split(".")[0])
    return epoch, path


def get_specific_cp():
    while True:
        user_input = input("Enter the path of checkpoint file or leave empty to use latest checkpoint.")
        if len(user_input) == 0:
            print("Using latest checkpoint from latest directory.")
            specific_path = get_latest_cp()
            break
        elif isInt(user_input):
            specific_path = get_latest_cp(int(user_input))
            break
        if os.path.exists(user_input):
            specific_path = user_input[:-20]
            break
        if os.path.exists(user_input + ".data-00000-of-00001"):
            specific_path = user_input
            break
        else:
            print("Please provide a valid path")

    global cp_dir_time
    cp_dir_time = os.path.dirname(specific_path)
    return get_epoch_and_path(specific_path)


"""Choose which model to use"""

# checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
checkpoint_dir = os.path.join(classifier_dir, "checkpoints/res{}".format(image_size))

if restore_checkpoint:
    if args.ckpt_path is not None:
        epoch_start, specific_path = get_epoch_and_path(args.ckpt_path)
        if os.path.exists(specific_path):
            specific_path = specific_path[:-len(".data-00000-of-00001")]
        global cp_dir_time
        cp_dir_time = os.path.dirname(specific_path)
    else:
        epoch_start, specific_path = get_specific_cp()
    if specific_path is not None:
        # model_path = os.path.join("/".join(specific_path.split("/")[:-1]), "model.h5")
        model_path = os.path.join("/".join(specific_path.split("/")[:-1]), "model_config.json")
        if not os.path.exists(model_path):
            sys.exit("Couldn't locate model")
        # model = tf.keras.models.load_model(os.path.join(model_path), custom_objects={'Padder': Padder, 'FactorLayer': FactorLayer, 'SigmoidLayer':SigmoidLayer})
        with open(model_path) as json_file:
            json_config = json_file.read()
        custom_objects = {'Padder': Padder, 'FactorLayer': FactorLayer, 'SigmoidLayer': SigmoidLayer, 'ResBlock': ResBlock}
        if conf['norm_type'] == 'pixel':
            custom_objects['Pixel_norm'] = Pixel_norm
        model = tf.keras.models.model_from_json(json_config, custom_objects=custom_objects)
        model.load_weights(specific_path)
        model.summary()
    else:
        print("Couldn't find a checkpoint. Starting from scratch.")

else:
    model = get_model()

if test_on_query:
    with open(os.path.join(cp_dir_time, "config.yaml"), 'r') as stream:
        conf = yaml.full_load(stream)
    image_size = conf['image_size']
    batch_size = conf['batch_size']
    if not args.is_cluster:
        test_model()
    load_dataset()
    with open(os.path.join(cp_dir_time, 'query_compl{}.csv'.format(epoch_start)), 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['Id', 'Predicted'])
        print("Beginning query")
        num_iterations = math.ceil(train_images.shape[0] / batch_size)
        for i in range(num_iterations):
            current = train_images[i * batch_size:min((i + 1) * batch_size, train_images.shape[0])]
            # current = tf.expand_dims(current, 0)
            score = np.clip(model(current, training=False).numpy()[:, 0], 0, label_range)
            for j in range(score.shape[0]):
                filewriter.writerow([img_list[i * batch_size + j].split(".")[0], score[j]])
            print("\rScored image {}/{}".format(min((i + 1) * batch_size, train_images.shape[0]), dataset_len), end="")


else:
    cp_dir_time = os.path.join(checkpoint_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(cp_dir_time):
        os.makedirs(cp_dir_time)
    cp_path = os.path.join(cp_dir_time, "cp-{epoch:04d}.ckpt")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(cp_path,
                                                     save_weights_only=True, save_best_only=percentage_train<1,
                                                     verbose=1, period=period_to_save_cp)

    tb_path = os.path.join(cp_dir_time, "summary")
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_path)

    linear_model = LinearRegression()

    """## Train the model
    
    As the model trains, the loss and accuracy metrics are displayed. This model reaches an accuracy of about 0.88 (or 88%) on the training data.
    """
    learning_rate = conf['lr']
    optimizer = tf.keras.optimizers.Adam(learning_rate, decay=1/tot_num_epochs)

    """As I see it, there is a bug in the keras library that forbids the labels y from training and test set to have different unique values (which is here the case because y is a continuous label)"""

    epochs = conf['num_epochs_for_lin_regression']
    batch_size = conf['batch_size']
    model.compile(
        loss=CustomLoss(),  # keras.losses.mean_squared_error
        optimizer=optimizer,
    )
    model.summary()
    for layer in model.layers:
        if "res_block" in layer.name:
            print(" {} summary:".format(layer.name))
            layer.model.summary()
            if conf['residual']:
                layer.projection_model.summary()
    # only necessary when neither batch norm nor dropout is used.
    # See https://stackoverflow.com/questions/52107555/different-loss-function-for-validation-set-in-keras
    model.outputs[0]._uses_learning_phase = True

    # save the configs to the checkpoints folder
    if not restore_checkpoint:
        # model.save(os.path.join(cp_dir_time, 'model.h5'))
        json_config = model.to_json()
        with open(os.path.join(cp_dir_time, 'model_config.json'), 'w') as json_file:
            json_file.write(json_config)

        with open(os.path.join(cp_dir_time, 'model_summary.txt'), 'w') as file:
            model.summary(print_fn=lambda mylambda: file.write(mylambda + '\n'))
            for layer in model.layers:
                if "res_block" in layer.name:
                    file.write(" {} summary:".format(layer.name))
                    layer.model.summary(print_fn=lambda mylambda: file.write(mylambda + '\n'))
                    if conf['residual']:
                        layer.projection_model.summary(print_fn=lambda mylambda: file.write(mylambda + '\n'))

        cp_command = 'cp {} {}'.format(os.path.join(classifier_dir, "{}"), cp_dir_time)
        os.system(cp_command.format("classifier.py"))
        os.system(cp_command.format("config.yaml"))

    AUGMENTATIONS_TRAIN = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(
            shift_limit=0.2, scale_limit=0,
            rotate_limit=0, border_mode=cv2.BORDER_REFLECT_101, p=0.8),
        FFT_augm(max_val_fft, conf['use_fft'])
    ])

    AUGMENTATIONS_TEST = Compose([FFT_augm(max_val_fft, conf['use_fft'])])

    counter = 0
    if restore_checkpoint:
        counter = epoch_start
    no_val_impr = False
    val_data = np.array([]).reshape((-1, 1))
    load_dataset()
    while True:
        # train the network
        if percentage_train < 1:
            validation_data = Augm_Sequence(test_images, test_labels, batch_size, AUGMENTATIONS_TEST, shuffle=False)
        else:
            validation_data = None
        H = model.fit_generator(Augm_Sequence(train_images, train_labels, batch_size, AUGMENTATIONS_TRAIN, shuffle=True),
                                initial_epoch=counter, epochs=counter + epochs, shuffle=False,  # check shuffle=False
                                callbacks=[tensorboard_callback, cp_callback], validation_data=validation_data)
        #test_model()
        # save whole model after first run through
        counter += epochs
        if validation_data is not None:
            val_data = np.append(val_data, np.array(H.history['val_loss']).reshape((-1, 1)), 0)
            if counter % conf['num_epochs_for_lin_regression'] == 0:
                val_indices = np.array(list(range(val_data.shape[0]))).reshape((-1, 1))
                linear_model.fit(val_indices, val_data)
                if linear_model.coef_ >= 0:
                    if no_val_impr:
                        print("No improvement on evaluation data on more than one round in a row.")
                        if counter >= tot_num_epochs:
                            print("Stopping training.")
                            break
                    else:
                        print("There was NO improvement on evaluation data on this set...")
                        no_val_impr = True
                else:
                    print("There was AN improvement on the evaluation data on this set.")
                    no_val_impr = False
                val_data = np.array([val_data[-1, 0]]).reshape((-1, 1))

        elif counter >= tot_num_epochs:
            break

    print("All done")
