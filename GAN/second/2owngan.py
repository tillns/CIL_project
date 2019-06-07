from __future__ import absolute_import, division, print_function, unicode_literals
#!pip3 install tensorflow-gpu==2.0.0-alpha0
import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import os, itertools, pickle
from PIL import Image
import math
from IPython import display
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from random import randint
import sys
from collections import OrderedDict

#tf.enable_eager_execution()

"""### Set initial parameters"""

resolutions = [4, 8, 16, 32, 64, 125, 250, 500, 1000]
original_image_size = resolutions[-1]
image_channels = 1
home_dir = os.path.expanduser("~")
image_directory = os.path.join(home_dir, "dataset/cil-cosmology-2018/cosmology_aux_data_170429/labeled")
label_path = os.path.join(home_dir, "dataset/cil-cosmology-2018/cosmology_aux_data_170429/labeled.csv")
print(label_path)

normal_conv_kernel_dim = 3
normal_kernel = (normal_conv_kernel_dim, normal_conv_kernel_dim)
downsample_conv_kernel_dim = 4
lrelu_factor = 0.2
batch_size = 16
lr = 0.001
num_epochs = 100
GAN_loss_type = 'WGAN_GP'  # WGAN_GP, LSGAN or else
gen_normalization='pixel'
# defined in load_dataset
dataset_length = 0
num_iterations = 0
downsample_size = 0

max_features = 64
min_features = 2
start_res = resolutions[0]
end_res = original_image_size  
num_res_change = 8
num_const_feature_layers = 4
mbstd_group_size = 4  # don't really know what that is  
# different layers, filled in build functions
to_rgb = []
gen_convs = []
gen_dense = []
from_rgb = []
dis_convs = []
dis_dense = []
dis_pool = tf.keras.layers.AveragePooling2D()
combined_res_change_and_conv = False  # can't be true yet
cross_fade_alpha = 1
# all numbers that 1000 can be evenly downsampled to
even_downsample_sizes = [1, 2, 4, 5, 8, 10, 20, 25, 40, 50, 100, 125, 200, 250, 500, 1000]
gen_losses = []
dis_losses = []
use_gen_bias = True
gen_output_activation = None  # should be none 
do_validation = True
percentage_train = 0.9
models = ["dis", "gen"]

def print_np(img):
  _, h, w, _ = img.shape
  for i in range(h):
    for j in range(w):
      print("{}, ".format(img[0, i, j, 0]), end="")
    print("")
  

def load_dataset(image_size=64, patch_divide_factor=4, print_images=False):
  factor_to_downsample = max(1024//image_size//patch_divide_factor, 1)
  global downsample_size
  downsample_size = original_image_size//factor_to_downsample  
  # only allow even downsampling to avoid strange interpolation
  downsample_size = min(even_downsample_sizes, key=lambda x:abs(x-downsample_size))
  #downsample_size = image_size  # comment out
  print("Downsampling input images to: {}x{}".format(downsample_size, downsample_size))
  try:
    f=open(label_path,'r')
    label_list = []
    for line in f:
      if not "Id,Actual" in line:
        label_list.append(line.split(","))
    label_list = sorted(label_list)
    labels = np.zeros(len(label_list))
    for ind, label in enumerate(label_list):
      labels[ind] = label[1]

    imgs = np.empty((0, downsample_size, downsample_size, image_channels)) 
    img_list = []
    for filename in os.listdir(image_directory):
      if filename.endswith(".png") and not filename.startswith("._"):
         img_list.append(filename)


    img_list = sorted(img_list)
    img_array = []
    for ind, filename in enumerate(img_list):
      if labels[ind] == 1:
        img = Image.open(os.path.join(image_directory, filename)).resize((downsample_size, downsample_size))
        img_np = np.array(img).astype(np.float32).reshape((downsample_size, downsample_size, image_channels))
        img_np = (img_np-127.5)/127.5  # images range from -1 to 1
        if print_images:
          img_np3 = img_np.reshape((downsample_size, downsample_size))
          plt.imshow(img_np3 * 127.5 + 127.5, cmap='gray')
          plt.show()
          print(filename)
          #print_np(img_np)
          #print("max: {}, min: {}".format(np.amax(img_np), np.amin(img_np)))
        img_array.append(img_np)
        #imgs = np.append(imgs, img_np, axis=0)
    
    if do_validation:
      break_point = round(percentage_train*len(img_array))
      imgs = np.stack(img_array[:break_point])
      test_imgs = np.stack(img_array[break_point:])
    else:
      imgs = np.stack(img_array)
      test_imgs = None

  except FileNotFoundError:
    print("Dataset not found. Using dummy dataset")
    labels = np.ones(100)
    #imgs = np.random.normal(0, 1, (1000, downsample_size, downsample_size, image_channels))
    imgs = np.zeros((100, downsample_size, downsample_size, image_channels))
    test_imgs = np.zeros((min(20, batch_size), downsample_size, downsample_size, image_channels))
    
    
  global num_iterations
  global num_test_iterations
  global train_len
  global test_len
  train_len = int(imgs.shape[0])
  test_len = int(test_imgs.shape[0]) if do_validation else 0
  num_iterations = math.ceil(train_len / batch_size)
  num_test_iterations = math.ceil(test_len/batch_size)
  print("Training it per epoch: {}. Test it: {}".format(num_iterations, num_test_iterations))

  if False:
    plt.figure(figsize=(10,10))
    for i in range(25):
      plt.subplot(5,5,i+1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(imgs[i,:,:,0], cmap='gray', vmin=-1, vmax=1)
      plt.xlabel(1)
    plt.show()
  return imgs, test_imgs

"""### Load dataset in appropriate size"""

#load_dataset(4)

"""## Create the models

Both the generator and discriminator are defined using the [Keras Sequential API](https://www.tensorflow.org/guide/keras#sequential_model).

###First fill the layer arrays
"""

def build_dis_layers(strided_conv=False):
  features = min_features

  # downsampling convs (reduce resolution, increase features)
  for num_downsample in range(num_res_change):
    # this layer has to be cross faded for res changes
    from_rgb.append(tf.keras.layers.Conv2D(features, kernel_size=(1, 1), padding='same', name="dis_from_rgb{}".format(num_downsample)))

    if end_res//pow(2, num_downsample) == 125:
      # padding: 125x125 -> 131x131, then valid padding conv 4x4: 131x131 -> 128x128. CHECK
      dis_convs.append(tf.keras.layers.Conv2D(features, kernel_size=(4, 4), padding='valid', name="dis_special4conv{}".format(2*num_downsample)))
    else:
      dis_convs.append(tf.keras.layers.Conv2D(features, normal_kernel, padding='same', name="dis_conv{}".format(2*num_downsample)))
    if features<max_features:
      features=features*2
    if strided_conv:
      # add strided conv
      pass
    else:
      dis_convs.append(tf.keras.layers.Conv2D(features, normal_kernel, padding='same', name="dis_conv{}".format(2*num_downsample+1)))

  from_rgb.append(tf.keras.layers.Conv2D(features, kernel_size=(1, 1), padding='same', name="dis_from_rgb{}".format(num_res_change)))
  dis_convs.append(tf.keras.layers.Conv2D(features, normal_kernel, strides=(1, 1), padding='same', name="dis_conv{}".format(2*num_res_change)))
  dis_dense.append(tf.keras.layers.Dense(features))
  dis_dense.append(tf.keras.layers.Dense(1))
  
def build_gen_layers(transp=False):
  features = max_features
  # first special block
  gen_dense.append(tf.keras.layers.Dense(features*start_res*start_res, use_bias=use_gen_bias))
  gen_convs.append(tf.keras.layers.Conv2D(features, normal_kernel, padding='same', name="gen_conv{}".format(0), use_bias=use_gen_bias))
  to_rgb.append(tf.keras.layers.Conv2D(image_channels, kernel_size=(1, 1), padding='same', name="gen_to_rgb{}".format(0), use_bias=use_gen_bias, activation=gen_output_activation))

  # rest normal blocks
  for num_upsample in range(num_res_change):
    if features>min_features: 
      features = features//2

    if transp:
      # Put transposed conv here. Should be more efficient but could also produce checkerboards, we'll see
      pass
    else:
      gen_convs.append(tf.keras.layers.Conv2D(features, normal_kernel, padding='same', name="gen_conv{}".format(2*num_upsample+1), use_bias=use_gen_bias))
    if resolutions[num_upsample+1]==125:
      # this should decrease the spatial resolution from 128 to 125 to produce 1000x1000 images in the end
      gen_convs.append(tf.keras.layers.Conv2D(features, kernel_size=(4, 4), padding='valid', name="gen_special4conv{}".format(2*num_upsample+2), use_bias=use_gen_bias))
    else:
      gen_convs.append(tf.keras.layers.Conv2D(features, normal_kernel, padding='same', name="gen_conv{}".format(2*num_upsample+2), use_bias=use_gen_bias))
    # every block is followed by a to_rgb conv that are faded into one another
    to_rgb.append(tf.keras.layers.Conv2D(image_channels, kernel_size=(1, 1), padding='same', name="gen_to_rgb{}".format(num_upsample+1), use_bias=use_gen_bias, activation=gen_output_activation))

trou = tf.zeros([32, 128, 128, 16])
#trou = get_pad(trou, -3)
print(trou.shape)

"""###Define some custom layers"""

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
  
def get_cut(x, cut_size):
  spatial_res = x.shape[1]
  assert spatial_res == x.shape[2]
  return get_pad(x, -(spatial_res-cut_size))

# in the progression paper, they apply the activation (lrelu) BEFORE the pixel normalization
def get_activated_norm(normalization='pixel', epsilon=1e-8, lrelu_factor=0.2):
  if normalization == 'pixel':
    return tf.keras.layers.LeakyReLU(lrelu_factor), Pixel_norm(epsilon)
  elif normalization == 'batch':
    return tf.keras.layers.BatchNormalization(), tf.keras.layers.LeakyReLU(lrelu_factor)
  else:    
    return tf.keras.layers.LeakyReLU(lrelu_factor), None

def add_activated_norm(model, normalization='pixel', epsilon=1e-8, lrelu_factor=0.2):
  add1, add2 = get_activated_norm(normalization, epsilon, lrelu_factor)
  model.add(add1)
  if add2 is not None: 
    model.add(add2)
    
class Pixel_norm(tf.keras.layers.Layer):
  def __init__(self, epsilon=1e-8):
    super(Pixel_norm, self).__init__()
    self.epsilon = epsilon
    
  def call(self, x):
    # print("input shape in pixel norm layer: {}".format(x.shape))
    return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=3, keepdims=True) + self.epsilon) 


# nearest neighbor upscaling. copied from progressive GAN paper; adjusted bc dimensions were ordered differently.      
class Upscale2D(tf.keras.layers.Layer):
  def __init__(self, factor=2):
    super(Upscale2D, self).__init__()
    self.factor = factor
  
  def call(self, x):
    assert isinstance(self.factor, int) and self.factor >= 1
    if self.factor == 1: 
      return x
    s = x.shape
    x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
    x = tf.tile(x, [1, 1, self.factor, 1, self.factor, 1])
    x = tf.reshape(x, [-1, s[1] * self.factor, s[2] * self.factor, s[3]])
    return x
  
class Minibatch_stddev(tf.keras.layers.Layer):
  def __init__(self, group_size=2):
    super(Minibatch_stddev, self).__init__()
    self.group_size = min(group_size, batch_size)
  
  def call(self, x):
    # print("Got to minibatch stddev")
    s = x.shape                                             # [NCHW]  Input shape.
    y = tf.reshape(x, (self.group_size, -1, s[1], s[2], s[3]))   # [GMCHW] Split minibatch into M groups of size G.
    y = tf.cast(y, tf.float32)                              # [GMCHW] Cast to FP32.
    y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMCHW] Subtract mean over group.
    y = tf.reduce_mean(tf.square(y), axis=0)                # [MCHW]  Calc variance over group.
    y = tf.sqrt(y + 1e-8)                                   # [MCHW]  Calc stddev over group.
    y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)      # [M111]  Take average over fmaps and pixels.
    y = tf.cast(y, x.dtype)                                 # [M111]  Cast back to original data type.
    y = tf.tile(y, [self.group_size, s[1], s[2], 1])             # [N1HW]  Replicate over group and pixels.
    return tf.concat([x, y], axis=3)                        # [NCHW]  Append as new fmap.
  
# This is kind of a spetial implementation as the padding is TOTAL, so both sides combined
class Padder(tf.keras.layers.Layer):
  def __init__(self, padding=6):
    super(Padder, self).__init__()
    self.padding = padding
  
  def call(self, x):
    #if self.padding != 0:
      #print("Padding {} with total {}".format(x.shape, self.padding))
    return get_pad(x, self.padding)

# cross_fade_alpha needs to be a global variable, s.t. it may change with each call
class GenFadeBlock(tf.keras.Model):
  def __init__(self, num_upsample):
    super(GenFadeBlock, self).__init__(name='')

    actnorm1, actnorm2 = get_activated_norm(normalization=gen_normalization)
    self.myseq = tf.keras.Sequential([
        #Upscale2D(),
        gen_convs[2*num_upsample+1],
        actnorm1, actnorm2,
        gen_convs[2*num_upsample+2],
        actnorm1, actnorm2,
        to_rgb[num_upsample+1]  #Todo: check
    ])
    
    if resolutions[num_upsample+1] == 125:  # maybe need some padding in next gen_fade as well?
      gen_padder = Padder(-3)
    else:
      gen_padder = Padder(0)
    self.cross_fade_seq = tf.keras.Sequential([
        gen_padder,
        to_rgb[num_upsample]  #Todo: check
    ])

  def call(self, input_tensor, training=False):
    x = self.myseq(input_tensor)
    x = cross_fade_alpha*x + (1-cross_fade_alpha)*self.cross_fade_seq(input_tensor)
    return x
  def get_config(self):
    return_string = []
    return_string.append("Gen fade block normal sequence config:")
    for layer in self.myseq.layers:
      return_string.append(layer.get_config())
      
    return_string.append("Fade layer:")
    for layer in self.cross_fade_seq.layers:
      return_string.append(layer.get_config())
    return_string.append("Gen fade block end")
    return return_string
  
class DisFadeBlock(tf.keras.Model):
  def __init__(self, num_downsample_blocks, cross_fade=False, pad=False):
    super(DisFadeBlock, self).__init__(name='')
    self.cross_fade = cross_fade
    actnorm1, _ = get_activated_norm(normalization='None')
    if pad:
      disPadLayer = Padder(padding=6)
    else:
      disPadLayer = Padder(padding=0)
    self.myseq = tf.keras.Sequential([
        from_rgb[-1-num_downsample_blocks],
        actnorm1, 
        disPadLayer,
        dis_convs[2*num_res_change-2*num_downsample_blocks],
        actnorm1,
        dis_convs[2*num_res_change-2*num_downsample_blocks+1],
        actnorm1,
        tf.keras.layers.AveragePooling2D()
    ])
    print("Dis fade myseq: {}".format(self.myseq))
    if self.cross_fade:
      if pad:
        cross_padder = Padder(padding=3)
      else:
        cross_padder = Padder(padding=0)
      self.cross_fade_seq = tf.keras.Sequential([
          cross_padder,
          tf.keras.layers.AveragePooling2D(),
          from_rgb[-num_downsample_blocks],
          actnorm1
      ])

  def call(self, input_tensor, training=False):
    x = self.myseq(input_tensor)
    if self.cross_fade:
      x = cross_fade_alpha*x + (1-cross_fade_alpha)*self.cross_fade_seq(input_tensor)
    return x
  def get_config(self):
    return_string = []
    return_string.append("Dis fade block normal sequence config:")
    for layer in self.myseq.layers:
      return_string.append(layer.get_config())
    if self.cross_fade:
      return_string.append("Fade sequence:")
      for layer in self.cross_fade_seq.layers:
        return_string.append(layer.get_config())
    return_string.append("Dis fade block end")
    return return_string

"""### The Discriminator"""

def make_discriminator_model(strided_conv=False, num_downsample_blocks=0, cross_fade=False):
  model = tf.keras.Sequential(name='dis')
  # reshape layer only serves for specifying input shape
  model.add(tf.keras.layers.Reshape((resolutions[num_downsample_blocks], resolutions[num_downsample_blocks], image_channels), 
                                    input_shape=(resolutions[num_downsample_blocks], resolutions[num_downsample_blocks], image_channels)))
  if num_downsample_blocks > 0:
    model.add(DisFadeBlock(num_downsample_blocks, cross_fade, pad=(resolutions[num_downsample_blocks]==125)))
  else:
    model.add(from_rgb[-1-num_downsample_blocks])
    add_activated_norm(model, normalization='None')
  for num_downsample in range(1, num_downsample_blocks):  # 0 case handled by DisFadeBlock
    if resolutions[num_downsample_blocks-num_downsample] == 125: 
      print("Adding Dis padder") 
      model.add(Padder())
    model.add(dis_convs[2*num_res_change-2*num_downsample_blocks+2*num_downsample])
    add_activated_norm(model, normalization='None')
    model.add(dis_convs[2*num_res_change-2*num_downsample_blocks+2*num_downsample+1])
    add_activated_norm(model, normalization='None')
    model.add(tf.keras.layers.AveragePooling2D())

  if mbstd_group_size > 1:
    model.add(Minibatch_stddev(mbstd_group_size))
  model.add(dis_convs[-1])
  model.add(tf.keras.layers.Reshape((max_features*start_res*start_res,)))
  model.add(dis_dense[0])
  add_activated_norm(model, normalization='None')
  model.add(dis_dense[1])

  return model

"""### The Generator"""

def make_generator_model(transp=False, num_upsample_blocks=0, cross_fade=False):  
  model = tf.keras.Sequential(name='gen')
  model.add(tf.keras.layers.Reshape((max_features,), input_shape=(1, 1, max_features)))
  model.add(gen_dense[0])
  model.add(tf.keras.layers.Reshape((start_res, start_res, max_features)))
  add_activated_norm(model, normalization=gen_normalization)
  model.add(gen_convs[0])
  add_activated_norm(model, normalization=gen_normalization)
  for num_upsample in range(num_upsample_blocks):
    model.add(Upscale2D())
    if num_upsample == num_upsample_blocks-1 and cross_fade:
      model.add(GenFadeBlock(num_upsample))
    else:
      model.add(gen_convs[2*num_upsample+1])
      add_activated_norm(model, normalization=gen_normalization)
      model.add(gen_convs[2*num_upsample+2])
      add_activated_norm(model, normalization=gen_normalization)
  if not cross_fade:
    model.add(to_rgb[num_upsample_blocks])
    
  return model

"""## Define the loss and optimizers

Define loss functions and optimizers for both models.
"""

def lerp(a, b, t):
  with tf.name_scope('Lerp'):
    return a + (b - a) * t
      
def exp2(x):
  with tf.name_scope('Exp2'):
    return tf.exp(x * np.float32(np.log(2.0)))
  
def is_tf_expression(x):
  return isinstance(x, tf.Tensor) or isinstance(x, tf.Variable) or isinstance(x, tf.Operation)
      
def apply_loss_scaling(value, use_loss_scaling=False):
  assert is_tf_expression(value)
  if not use_loss_scaling:
      return value
  return value * exp2(value)

# Undo the effect of dynamic loss scaling for the given expression.
def undo_loss_scaling(value):
  assert is_tf_expression(value, use_loss_scaling=False)
  if not use_loss_scaling:
      return value
  return value * exp2(-value)
  
      
def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

# wgangp losses. I crossed out the label penalty terms and adjusted the other stuff
# to tf 2.0 and left out autosummary (for tensorboard) stuff. 
# I think the label penalty was introduced in
# https://arxiv.org/pdf/1610.09585.pdf the ac-gan paper. For multi-class generation,
# this term is useful, so it's probably not necessary in this work
def G_wgangp(labels, fake_scores_out,
    cond_weight = 1.0): # Weight of the conditioning term.

    return -cond_weight*tf.reduce_mean(fake_scores_out)

def D_wgangp(minibatch_size, reals, fake_images_out, real_scores_out, fake_scores_out, discriminator,
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0,      # Target value for gradient magnitudes.
    cond_weight     = 1.0):     # Weight of the conditioning terms.

    loss = tf.reduce_mean(fake_scores_out) - tf.reduce_mean(real_scores_out)  # this is the normal WGAN loss

    with tf.name_scope('GradientPenalty'):
      alpha = tf.random.uniform(shape=[minibatch_size], minval=0., maxval=1.)
      differences = fake_images_out - reals
      interpolates = reals + (differences.mul(alpha, axis=0))
      with tf.GradientTape() as t:
        t.watch(interpolates)
        pred = discriminator(interpolates)
      grad = t.gradient(pred, interpolates)
      norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
      gradient_penalty = tf.reduce_mean((norm - 1.)**2)
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))  # this is the gradient penalty term

    with tf.name_scope('EpsilonPenalty'):
      epsilon_penalty = tf.reduce_mean(tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon  # this is the small additional loss mentioned to avoid drifts from zero

    return loss

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

"""### Discriminator loss

This method quantifies how well the discriminator is able to distinguish real images from fakes. It compares the discriminator's predictions on real images to an array of 1s, and the discriminator's predictions on fake (generated) images to an array of 0s. The default loss uses the loss from https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/generative/dcgan.ipynb#scrollTo=90BIcCKcDMxz --- likewise of gen loss. I'm not completely sure what kind of GAN objective that is, tho.
"""

def discriminator_loss(D_real_logits, D_fake_logits, x, generated_image, discriminator):
  label_zero = tf.zeros([x.shape[0], 1])
  label_one = tf.ones([x.shape[0], 1])
  if GAN_loss_type == 'LSGAN':
     return tf.reduce_mean(tf.square(D_fake_logits - label_zero)) + tf.reduce_mean(tf.square(D_real_logits - label_one))
  elif GAN_loss_type == 'WGAN_GP':
    return D_wgangp(batch_size, x, generated_image, D_real_logits, D_fake_logits, discriminator)
  else:
    real_loss = cross_entropy(label_one, D_real_logits)
    fake_loss = cross_entropy(label_zero, D_fake_logits)
    total_loss = real_loss + fake_loss
    return total_loss

"""### Generator loss
The generator's loss quantifies how well it was able to trick the discriminator. Intuitively, if the generator is performing well, the discriminator will classify the fake images as real (or 1). Here, we will compare the discriminators decisions on the generated images to an array of 1s.
"""

def generator_loss(D_fake_logits):
  label_one = tf.ones([D_fake_logits.shape[0], 1])
  if GAN_loss_type == 'LSGAN':
    return tf.reduce_mean(tf.square(D_fake_logits - label_one))
  elif GAN_loss_type == 'WGAN_GP':
    return G_wgangp(label_one, D_fake_logits)
  else:
    return cross_entropy(label_one, D_fake_logits)
  
def alternative_gen_loss(generated_image, real_image):
  return tf.reduce_mean(tf.abs(generated_image - real_image))

"""The discriminator and the generator optimizers are different since we will train two networks separately."""

#  check that other settings are standard
generator_optimizer = tf.keras.optimizers.Adam(lr)
discriminator_optimizer = tf.keras.optimizers.Adam(lr)

"""Tensorboard initialization"""

from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras import callbacks as cbks
class CallbackList(object):
  def __init__(self, callbacks):
    assert len(callbacks) == len(models)
    self.callbacks = []
    for model_kind in models:
      for callback in callbacks:
        if model_kind in callback.log_dir:
          self.callbacks.append(callback)

    
  def set_model(self, model_list):
    assert len(model_list) == len(self.callbacks)
    assert len(models) == len(model_list)
    for model_kind in models:
      for i, model in enumerate(model_list):
        if model_kind == model.name:
          self.callbacks[i].set_model(model)

      
      
  def set_params(self, batch_size, num_epochs, num_iterations, train_len, verbose, do_validation):
    gen_metrics = ['gen_loss']
    dis_metrics=['dis_loss']
    if do_validation:
      dis_metrics.append('dis_val_loss')
    self.params = {'dis': {
        'batch_size': batch_size,
        'epochs': num_epochs,
        'steps': num_iterations,
        'samples': train_len,
        'verbose': verbose,
        'do_validation': do_validation,
         'metrics': dis_metrics,  # not sure here 
      }, 'gen': {
        'batch_size': batch_size,
        'epochs': num_epochs,
        'steps': num_iterations,
        'samples': train_len,
        'verbose': verbose,
        'do_validation': False,
         'metrics': gen_metrics,  # not sure here 
      }, 'comb': {
        'batch_size': batch_size,
        'epochs': num_epochs,
        'steps': num_iterations,
        'samples': train_len,
        'verbose': verbose,
        'do_validation': do_validation,
         'metrics': gen_metrics + dis_metrics,  # not sure here 
      }} 
    for ind, callback in enumerate(self.callbacks):
        callback.set_params(self.params[models[ind]])

      
  def get_params(self, type='comb'):
    return self.params[type]
      
  def _call_begin_hook(self, mode):
    for callback in self.callbacks:
      if mode == 'train': 
        callback.on_train_begin()
      else:
        if "dis" in callback.log_dir:
          callback.on_test_begin()
      
  def _call_end_hook(self, mode):
    for callback in self.callbacks:
      if mode == 'train': 
        callback.on_train_end()
      else:
        if "dis" in callback.log_dir:
          callback.on_test_end()
    
  def stop_training(self, bool_var):
    for callback in self.callbacks:
      callback.model.stop_training = bool_var
      
  def duplicate_logs_for_models(self, logs):
    if models[0] in logs:
      return logs
    duplicated_logs = {}
    for model_kind in models:
      duplicated_logs[model_kind] = logs
    return duplicated_logs
    
  def on_epoch_begin(self, epoch, logs=None):
    self.duplicate_logs_for_models(logs)
    for i, callback in enumerate(self.callbacks):
      callback.on_epoch_begin(epoch, logs[models[i]])
      
  def on_epoch_end(self, epoch, logs=None):
    self.duplicate_logs_for_models(logs)
    #print("Logs on epoch end: {}".format(logs))
    for i, callback in enumerate(self.callbacks):
      #print("Logs at {}: {}".format(models[i], logs[models[i]]))
      callback.on_epoch_end(epoch, logs[models[i]])
      
  def _call_batch_hook(self, state_str, beginorend, iteration, logs=None, modelkind=None):
    self.duplicate_logs_for_models(logs)
    for i, callback in enumerate(self.callbacks):
      if modelkind is None or modelkind in callback.log_dir:
        if state_str == 'train':
          if beginorend == 'begin':
            callback.on_train_batch_begin(iteration, logs[models[i]])
          else:
            callback._enable_trace()  # I have absolutely no idea why, but it works here
            callback.on_batch_end(iteration, logs[models[i]])
        else:
          if "dis" in callback.log_dir:  # only discriminator is tested
            if beginorend == 'begin':
              callback.on_test_batch_begin(iteration, logs[models[i]])
            else:
              callback.on_test_batch_end(iteration, logs[models[i]])
              
  def print_models(self, modelkind=None):
    for callback in self.callbacks:
      if modelkind is None or modelkind in callback.log_dir:
        print("log dir: {}".format(callback.log_dir))
        print_model(callback.model)

from datetime import datetime
checkpoint_dir = os.path.join(home_dir, "GAN/Checkpoints")
tb_path = os.path.join(home_dir, "GAN/summary/")
if not os.path.exists(tb_path):
  os.makedirs(tb_path)
tb_path = os.path.join(tb_path, datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_path_gen = os.path.join(tb_path, "gen")
tb_path_dis = os.path.join(tb_path, "dis")
tb_callback_gen = tf.keras.callbacks.TensorBoard(log_dir=tb_path_gen)
tb_callback_dis = tf.keras.callbacks.TensorBoard(log_dir=tb_path_dis)
callbacks = CallbackList([tb_callback_dis, tb_callback_gen])

"""## Define the training loop"""

num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, 1, 1, max_features])

"""The training loop begins with generator receiving a random seed as input. That seed is used to produce an image. The discriminator is then used to classify real images (drawn from the training set) and fakes images (produced by the generator). The loss is calculated for each of these models, and the gradients are used to update the generator and discriminator."""

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
#@tf.function
ratio_gen_dis=1
def train_step(images, generator, discriminator, iteration, step, progbar, res_change=None):
  #for gen_step in range(ratio_gen_dis):
  current_batch_size = images.shape[0]
  noise = tf.random.normal([current_batch_size, 1, 1, max_features])
  batch_logs = callbacks.duplicate_logs_for_models({'batch': step, 'size': current_batch_size})
  #print("Batch logs: {}".format(batch_logs))
  callbacks._call_batch_hook("train", 'begin', step, logs=batch_logs)
  progbar.on_batch_begin(iteration, batch_logs['dis'])  # as of now, the progress bar will only show the dis state

  with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
    generated_images = generator(noise, training=True)
    fake_output = discriminator(generated_images, training=True)
    gen_loss = generator_loss(fake_output)
    batch_logs['gen']['gen_loss'] = gen_loss
    #gen_loss = alternative_gen_loss(generated_images, images)

    #if gen_step == ratio_gen_dis-1:
    real_output = discriminator(images, training=True)
    dis_loss = discriminator_loss(real_output, fake_output, images, generated_images, discriminator)
    batch_logs['dis']['dis_loss'] = dis_loss
    

  #if gen_step == ratio_gen_dis-1:
  gradients_of_discriminator = disc_tape.gradient(dis_loss, discriminator.trainable_variables)    
  discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
  
  gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
  generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
  callbacks._call_batch_hook('train', 'end', iteration, batch_logs)
  progbar.on_batch_end(iteration, {**batch_logs['gen'], **batch_logs['dis']})
  return gen_loss, dis_loss

dict1 = {'batch': 0, 'size': 24, 'gen_loss': 0.3}
dict2 = {'batch': 0, 'size': 24, 'dis_loss': 0.4}
dict3 = {**dict1, **dict2}
print(dict3)

def plot_losses():
  x_axis = np.arange(len(gen_losses))
  plt.plot(x_axis, gen_losses, label="Gen loss")
  plt.plot(x_axis, dis_losses, label="Dis loss")
  plt.legend()
  plt.show
def get_mean(some_list):
  return sum(some_list)/len(some_list)

def print_model(model):
  print("\n")
  for layer in model.layers:
    layerconfig = layer.get_config();
    if isinstance(layerconfig, list):
      for indconf in layerconfig:
        print(indconf)
    else:
      print(layerconfig)
  print("\n")

def train(imgs, test_imgs, num_res_change_layers=0, cross_fade=False, device='/device:GPU:0', image_size=64):
  global cross_fade_alpha
  cross_fade_alpha = 1
  step = 2*num_res_change_layers*num_epochs*num_iterations
  if cross_fade:
    step = step - num_epochs*num_iterations
  discriminator = make_discriminator_model(strided_conv=combined_res_change_and_conv, num_downsample_blocks=num_res_change_layers, cross_fade=cross_fade)
  generator = make_generator_model(transp=combined_res_change_and_conv, num_upsample_blocks=num_res_change_layers, cross_fade=cross_fade)
  callbacks.set_model([discriminator, generator])
  callbacks.set_params(batch_size, step//num_iterations+num_epochs, num_iterations, train_len, 1, do_validation)
  checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
  progbar = training_utils.get_progbar(generator, 'steps')
  progbar.params = callbacks.get_params()
  progbar.params['verbose'] = 1
  
  print_model(discriminator)
  print_model(generator)
    
  #callbacks.set_params(batch_size, num_epochs, num_iterations, train_len, 1, do_validation, None,)  # not sure bout metrics=None
  callbacks.stop_training(False)
  callbacks._call_begin_hook('train')
  progbar.on_train_begin()
  #callbacks.print_models()
  #print("Graph: {}".format(tf.keras.backend.get_session().graph))
    
  for epoch in range(num_epochs):
    tot_num_epoch = step//num_iterations
    gen_losses = 0
    dis_losses = 0
    generator.reset_metrics()
    discriminator.reset_metrics()
    epoch_logs = callbacks.duplicate_logs_for_models({})
    callbacks.on_epoch_begin(tot_num_epoch, epoch_logs)
    progbar.on_epoch_begin(tot_num_epoch, epoch_logs)
    start = time.time()
    cross_fade_alpha = 1
    np.random.shuffle(imgs)
    for iteration in range(num_iterations):
      x_ = imgs[iteration*batch_size:min((iteration+1)*batch_size, train_len)]
      x_ = x_.astype(np.float32)
      # draw a random patch from the input
      x_ = get_cut(x_, image_size)
      for i in range(2):
        if randint(0, 1):
          x_ = np.flip(x_, axis=i+1)  # randomly flip x- and y-axis
          
      if False:
        img_np3 = x_[0, :, :, 0]
        plt.imshow(img_np3 * 127.5 + 127.5, cmap='gray')
        plt.show()
      # consider adding random rotation as well
      
      #print("x_ shape: {}".format(x_.shape))
      if cross_fade:
        cross_fade_alpha = (epoch*num_iterations+iteration+1)/(num_epochs*num_iterations) 
      if epoch == 0 and iteration == 0 and cross_fade:
        res_change = image_size
      else:
        res_change = None
      gen_loss, dis_loss = train_step(x_, generator, discriminator, iteration, step, progbar, res_change)
      gen_losses += gen_loss
      dis_losses += dis_loss
      step = step + 1

    epoch_logs = {'gen': {'gen_loss': gen_losses/num_iterations}, 'dis': {'dis_loss': dis_losses/num_iterations}}
    # only test dis on one randomly drawn batch of test data per epoch
    if do_validation:
      callbacks._call_begin_hook('test')
      batch_logs = {'dis': {'batch': 0, 'size': batch_size}}
      callbacks._call_batch_hook('test', 'begin', 0, batch_logs)
      index = randint(0, test_len-batch_size)
      x_ = test_imgs[index:index+batch_size]
      x_ = x_.astype(np.float32)
      # maybe adjust something here, s.t. test images don't contain something random
      x_ = get_cut(x_, image_size)
      output = discriminator(x_, training=False)
      if False:
        plt.figure(figsize=(10, 10))
        for i in range(batch_size):
          plt.subplot(4, 4, i + 1)
          plt.xticks([])
          plt.yticks([])
          plt.grid(False)
          plt.imshow(x_[i, :, :, 0], cmap='gray', vmin=-1, vmax=1)
          plt.xlabel(output.numpy()[i, 0])
        plt.show()
      loss = tf.reduce_mean(tf.square(tf.ones([x_.shape[0], 1])-output))
      batch_logs['dis']['loss'] = dis_loss
      callbacks._call_batch_hook('test', 'end', 0, batch_logs)
      #callbacks.write_log("dis", "test", "loss", loss, step)
      callbacks._call_end_hook('test')
      epoch_logs['dis']['dis_val_loss'] = loss
    
    # Save the model every few epochs
    if (epoch + 1) % 20 == 0:
        
      try:
        if cross_fade:
          fade = "fade"
        else:
          fade = "nofade"
        checkpoint_path = os.path.join(checkpoint_dir, "size{}_{}_epoch{}".format(image_size, fade, epoch+1))
        checkpoint.save(checkpoint_path)
        print("Checkpoint saved as: {}".format(checkpoint_path))
      except Error:
        print("Something went wrong with saving the checkpoint:\n" + Error)

    #print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    
    
    callbacks.on_epoch_end(tot_num_epoch, epoch_logs)
    progbar.on_epoch_end(tot_num_epoch, {**epoch_logs['gen'], **epoch_logs['dis']})

    # Generate after every epoch
    generate_and_save_images(generator, step//num_iterations, seed)
  callbacks._call_end_hook('train')

"""**Generate and save images**"""

import cv2
save_image_path = os.path.join(home_dir, "GAN/outputs")
save_image_path = os.path.join(save_image_path, datetime.now().strftime("%Y%m%d-%H%M%S"))
if not os.path.exists(save_image_path):
  os.makedirs(save_image_path)
def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)
  predictions = predictions*127.5+127.5
  tot = predictions.shape[0]
  res = predictions.shape[1]
  depth = predictions.shape[3]
  per_row = int(math.sqrt(tot))
  dist = int(0.25*res)
  outres = per_row*res+dist*(per_row+1)
  output_image = np.zeros((outres, outres, depth))+127.5
  for k in range(tot):
    i = k // per_row
    j = k % per_row
    output_image[dist*(j+1)+j*res:dist*(j+1)+res*(j+1), dist*(i+1)+i*res:dist*(i+1)+res*(i+1)] = predictions[k]
    
  font = cv2.FONT_HERSHEY_SIMPLEX
  bottomLeft = (dist//2, dist//2)
  fontScale=1
  fontColor=(255,255,255)
  lineType=2
  #cv2.putText(output_image, "{:05d}".format(epoch), bottomLeft, font, fontScale, fontColor, lineType)
  cv2.imwrite(os.path.join(save_image_path,'image_at_epoch_{:05d}.png'.format(epoch)), output_image)
  
  
  
  '''
  size_figure_grid = 4
  fig = plt.figure(figsize=(size_figure_grid, size_figure_grid))

  for i in range(tot):
      plt.subplot(per_row, per_row, i+1)
      plt.imshow(np.clip(predictions[i, :, :, 0]*127.5+127.5, 0, 255), cmap='gray', vmin=0, vmax=255)
      plt.xticks([])
      plt.yticks([])
      if i == per_row*(per_row-1):
        plt.xlabel("Images at epoch {:05d}".format(epoch))

  print("Saving {}".format(epoch))
  plt.savefig(os.path.join(save_image_path,'image_at_epoch_{:05d}.png'.format(epoch)))
  plt.close()
  #plt.show()
  '''

"""Use `imageio` to create an animated gif using the images saved during training."""

def make_gif():
  anim_file = os.path.join(save_image_path, 'dcgan.gif')

  '''
  with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob(os.path.join(save_image_path, 'image*.png'))
    filenames = sorted(filenames)
    last = -1
    for i,filename in enumerate(filenames):
      frame = 2*(i**0.5)
      if round(frame) > round(last):
        last = frame
      else:
        continue
      image = imageio.imread(filename)
      writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)
    '''
  epoch_imgs = []
  filenames = glob.glob(os.path.join(save_image_path, 'image*.png'))
  filenames = sorted(filenames)
  image = imageio.imread(filenames[-1])
  biggest_res = image.shape
  font = cv2.FONT_HERSHEY_SIMPLEX
  bottomLeft = (biggest_res[0]//4, biggest_res[1]//4)
  fontScale=1
  fontColor=(255,255,255)
  lineType=2
  for i, filename in enumerate(filenames):
    image = imageio.imread(filename)
    image = cv2.resize(image, (biggest_res[0], biggest_res[1]))
    cv2.putText(image, "{:05d}".format(i), bottomLeft, font, fontScale, fontColor, lineType)
    epoch_imgs.append(image)
    
  imageio.mimsave(anim_file, epoch_imgs, 'GIF', duration=0.2)

  import IPython
  if IPython.version_info > (6,2,0,''):
    print("Python version high enough, but gif doesn't seem to show...")
    display.Image(filename=anim_file)
  else:
    print("Python version to old to display file directly. Trying html...")
  IPython.display.HTML('<img src="{}">'.format(anim_file))

"""## Train the model
Call the `train()` method defined above to train the generator and discriminator simultaneously. Note, training GANs can be tricky. It's important that the generator and discriminator do not overpower each other (e.g., that they train at a similar rate).

At the beginning of the training, the generated images look like random noise. As training progresses, the generated digits will look increasingly real. After about 50 epochs, they resemble MNIST digits. This may take about one minute / epoch with the default settings on Colab.
"""

build_dis_layers(combined_res_change_and_conv)
build_gen_layers(combined_res_change_and_conv)
current_res = start_res
for counter in range(num_res_change):
  print("Current Resolution: {}x{}".format(current_res, current_res))
  imgs, test_imgs = load_dataset(current_res)
  print("Dataset shape: {}".format(imgs.shape))
  if counter == 0:
    train(imgs, test_imgs, counter, False, image_size=current_res)
  else:
    train(imgs, test_imgs, counter, True, image_size=current_res)
    train(imgs, test_imgs, counter, False, image_size=current_res)

  current_res = 2*current_res
  if current_res == 128:
    current_res = 125  # little res change s.t. final image size is 1000, not 1024
  make_gif()


"""Restore the latest checkpoint."""

#checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

"""If you're working in Colab you can download the animation with the code below:"""