from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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

print(tf.__version__)

"""## Import dataset"""

def get_shuffled_train_plus_val_list():
  list1 = list(range(round(dataset_len*percentage_train)))
  list2 = list(range(round(dataset_len*percentage_train), dataset_len))
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
  plt.figure(figsize=(10,10))
  for i in range(25):
    ind = indices[i]
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    label = label_list[ind][1]
    img = Image.open(os.path.join(scored_directory, label_list[ind][0]+".png"))
    img_np = np.array(img, dtype=np.float32).reshape((1, image_size, image_size, image_channels)) / 255
    score = model(img_np)
    plt.imshow(img_np[0,:,:,0], cmap='gray', vmin=0, vmax=1)
    plt.xlabel("{}\no: {}\ny: {}".format(label_list[ind][0], score.numpy()[0, 0], label))
  plt.show()


with open("config.yaml", 'r') as stream:
  conf = yaml.load(stream)

test_on_query = False
restore_checkpoint = test_on_query
image_size = conf['image_size']
image_channels = 1
home_dir = os.path.expanduser("~")
classifier_dir = os.path.join(home_dir, "CIL_project/Classifier")
scored_directory = os.path.join(home_dir, "dataset/cil-cosmology-2018/cosmology_aux_data_170429/scored")
label_path = os.path.join(home_dir, "dataset/cil-cosmology-2018/cosmology_aux_data_170429/scored.csv")
query_directory = os.path.join(home_dir, "dataset/cil-cosmology-2018/cosmology_aux_data_170429/query")
image_directory = query_directory if test_on_query else scored_directory
percentage_train = conf['percentage_train'] if not test_on_query else 1
print("Searching for images in {}".format(image_directory))
max_elements = conf['max_elements']
whole_len = max_elements
break_point = round(percentage_train*whole_len)
use_dummy_dataset = False
use_bias = conf['use_bias']
period_to_save_cp = conf['period_to_save_cp']
tot_num_epochs = conf['tot_num_epochs']

try:
  f=open(label_path,'r')
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
  whole_len = min(dataset_len, max_elements) if not test_on_query else dataset_len
  train_len = round(percentage_train*whole_len)
  test_len = whole_len-train_len
  if not test_on_query:
    assert len(label_list) == dataset_len
  
except FileNotFoundError:
  print("Dataset not found. Using dummy dataset")
  use_dummy_dataset = True


def load_dataset():
  global train_images, train_labels, test_images, test_labels
  if not use_dummy_dataset:
    trainset_path = 'trainset{0:.1f}.dat'.format(percentage_train)
    testset_path = 'testset{0:.1f}.dat'.format(1 - percentage_train)
    both_paths_exist = os.path.exists(trainset_path) and (percentage_train == 1 or os.path.exists(testset_path))
    mode = 'r+' if both_paths_exist else 'w+'
    train_images = np.memmap(trainset_path, dtype=np.float32, mode=mode,
                             shape=(int(dataset_len * percentage_train), image_size, image_size, 1))
    train_labels = np.zeros(shape=(train_images.shape[0],), dtype=np.float32)
    if percentage_train < 1:
      test_images = np.memmap(testset_path, dtype=np.float32, mode=mode,
                              shape=(dataset_len - int(dataset_len * percentage_train), image_size, image_size, 1))
      test_labels = np.zeros(shape=(test_images.shape[0],), dtype=np.float32)
    indices = get_shuffled_train_plus_val_list()
    for num, ind in enumerate(indices):
      if not both_paths_exist:
        img = Image.open(os.path.join(image_directory, img_list[ind]))
        img_np = np.array(img, dtype=np.float32).reshape((image_size, image_size, image_channels))/255
      if num < int(dataset_len*percentage_train):
        train_labels[num] = label_list[ind][1]
        if not both_paths_exist:
          train_images[num] = img_np

      else:
        test_labels[num-int(dataset_len*percentage_train)] = label_list[ind][1]
        if not both_paths_exist:
          test_images[num-int(dataset_len*percentage_train)] = img_np
      print("\rLoaded image {}/{}".format(num+1, whole_len), end="")
    print("")

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
    return tf.pad(x, tf.constant([[0, 0], [rand1, total_padding-rand1], [rand2, total_padding-rand2], [0, 0]])) 
  else:
    total_padding = abs(total_padding)
    rand1 = randint(0, total_padding)
    rand2 = randint(0, total_padding)
    s = x.shape
    return x[:, rand1:s[1]-total_padding+rand1, rand2:s[2]-total_padding+rand2]

class Padder(tf.keras.layers.Layer):
  def __init__(self, padding=6):
    super(Padder, self).__init__()
    self.padding = padding
  
  def call(self, x):
    return get_pad(x, self.padding)

  def get_config(self):
    return {'padding': self.padding}


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
    return self.factor*x

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


# parames for model 1
def get_model():
    cconf = conf['till']
    kernel = (cconf['kernel'], cconf['kernel'])
    features = cconf['features']
    max_features = cconf['max_features']
    min_res = cconf['min_res']
    norm_type = cconf['norm_type']

    res = image_size
    res2 = [pow(2, x) for x in range(10)]
    model = keras.Sequential(name='till_model')
    while res > min_res:
      if res%2 != 0:
        closest2res = min(res2, key=lambda x:abs(x-res))
        model.add(Padder(padding=closest2res-res))  # 125 -> 128 or maybe 25 -> 32...
        res = closest2res
      model.add(tf.keras.layers.Conv2D(features, kernel, padding='same', strides=2, use_bias=use_bias,
                                       input_shape=(res, res, image_channels)))
      model.add(getNormLayer(norm_type))
      model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
      if features < max_features:
        features *= 2
      res = res//2

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))
    model.add(SigmoidLayer())
    model.add(FactorLayer(8))  # to make output range from 0 to 8
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
    if os.path.exists(user_input) or os.path.exists(user_input + ".data-00000-of-00001"):
      specific_path = user_input
      break
    else:
      print("Please provide a valid path")

  global cp_dir_time
  cp_dir_time = os.path.dirname(specific_path)
  return get_epoch_and_path(specific_path)


"""Choose which model to use"""

#checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
checkpoint_dir = os.path.join(classifier_dir, "checkpoints")

if restore_checkpoint:
  epoch_start, specific_path = get_specific_cp()
  if specific_path is not None:
    #model_path = os.path.join("/".join(specific_path.split("/")[:-1]), "model.h5")
    model_path = os.path.join("/".join(specific_path.split("/")[:-1]), "model_config.json")
    if not os.path.exists(model_path):
      sys.exit("Couldn't locate model")
    #model = tf.keras.models.load_model(os.path.join(model_path), custom_objects={'Padder': Padder, 'FactorLayer': FactorLayer, 'SigmoidLayer':SigmoidLayer})
    with open(model_path) as json_file:
      json_config = json_file.read()
    model = keras.models.model_from_json(json_config, custom_objects={'Padder': Padder, 'FactorLayer': FactorLayer, 'SigmoidLayer':SigmoidLayer})
    model.load_weights(specific_path)
    print(model.summary())
    test_model()
  else:
    print("Couldn't find a checkpoint. Starting from scratch.")

else:
    model = get_model()


if test_on_query:
  load_dataset()
  with open(os.path.join(cp_dir_time, 'query.csv'), 'w') as csvfile:
      filewriter = csv.writer(csvfile, delimiter=',',
                              quotechar='"', quoting=csv.QUOTE_MINIMAL)
      filewriter.writerow(['Id', 'Predicted'])
      print("Beginning query")
      for i in range(train_images.shape[0]):
          current = train_images[i]
          current = tf.expand_dims(current, 0)
          score = model(current)
          filewriter.writerow([img_list[i].split(".")[0], score.numpy()[0, 0]])
          print("\rScored image {}/{}".format(i+1, dataset_len), end="")


else:
  cp_dir_time = os.path.join(checkpoint_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
  if not os.path.exists(cp_dir_time):
    os.makedirs(cp_dir_time)
  cp_path = os.path.join(cp_dir_time, "cp-{epoch:04d}.ckpt")
  cp_callback = tf.keras.callbacks.ModelCheckpoint(cp_path,
                                                   save_weights_only=True,
                                                   verbose=1, period=period_to_save_cp)

  tb_path = os.path.join(cp_dir_time, "summary")
  if not os.path.exists(tb_path):
    os.makedirs(tb_path)
  tensorboard_callback = keras.callbacks.TensorBoard(log_dir=tb_path)

  linear_model = LinearRegression()

  """## Train the model
  
  As the model trains, the loss and accuracy metrics are displayed. This model reaches an accuracy of about 0.88 (or 88%) on the training data.
  """

  loss_type = conf['loss_type']
  learning_rate = conf['lr']
  if loss_type == "BCE":
    loss_func = tf.keras.losses.BinaryCrossentropy()  # from_logits=True ?
  elif loss_type == "MSE":
    loss_func = tf.keras.losses.MeanSquaredError()

  optimizer = tf.keras.optimizers.Adam(learning_rate)

  """As I see it, there is a bug in the keras library that forbids the labels y from training and test set to have different unique values (which is here the case because y is a continuous label)"""

  epochs = conf['num_epochs_for_lin_regression']
  batch_size = conf['batch_size']
  model.compile(
      loss=loss_func, # keras.losses.mean_squared_error
      optimizer=optimizer,
  )

  # save the configs to the checkpoints folder
  if not restore_checkpoint:
    # model.save(os.path.join(cp_dir_time, 'model.h5'))
    json_config = model.to_json()
    with open(os.path.join(cp_dir_time,'model_config.json'), 'w') as json_file:
      json_file.write(json_config)

    with open(os.path.join(cp_dir_time, 'model_summary.txt'), 'w') as file:
      model.summary(print_fn=lambda mylambda: file.write(mylambda + '\n'))

    cp_command = 'cp {} {}'.format(os.path.join(classifier_dir, "{}"), cp_dir_time)
    os.system(cp_command.format("classifier.py"))
    os.system(cp_command.format("config.yaml"))


  aug = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=360,
                                                        zoom_range=0.15,
      width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
      horizontal_flip=True, vertical_flip=True, fill_mode="nearest",
                           validation_split=0)
  val = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=0,
                                                        zoom_range=0,
      width_shift_range=0, height_shift_range=0, shear_range=0,
      horizontal_flip=False, vertical_flip=False, fill_mode="nearest",
                           validation_split=0)


  counter = 0
  if restore_checkpoint:
    counter = epoch_start
  no_val_impr = False
  val_data = np.array([]).reshape((-1, 1))
  load_dataset()
  while True:
    # train the network
    if percentage_train < 1:
      validation_data = val.flow(test_images, test_labels, batch_size=batch_size, shuffle=False)
    else:
      validation_data = None
    H = model.fit_generator(aug.flow(train_images, train_labels, batch_size=batch_size),
                            steps_per_epoch=whole_len // batch_size, initial_epoch=counter,
                            epochs=counter+epochs, callbacks=[tensorboard_callback, cp_callback], shuffle=True,
                            validation_data=validation_data)
    # save whole model after first run through
    counter += epochs
    if validation_data is not None:
      val_data = np.append(val_data, np.array(H.history['val_loss']).reshape((-1, 1)), 0)
      if counter % conf['num_epochs_for_lin_regression'] == 0:
        val_indices = np.array(list(range(val_data.shape[0]))).reshape((-1, 1))
        linear_model.fit(val_indices, val_data)
        if linear_model.coef_ >= 0:
          if no_val_impr:
            print("No improvement on evaluation data on two sets in a row. Stopping training.")
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