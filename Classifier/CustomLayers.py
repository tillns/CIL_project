"""Custom Layers for the CNN Classifier

These classes define custom layers for the CNN Classifier.

This module contains the following public functions:
    #get_pad
    #getNormLayer
    #get_custom_objects

and the following classes:
    #ResBlock
    #Padder
    #Pixel_norm
    #FactorLayer
"""

import tensorflow as tf
from random import randint


def get_pad(x, total_padding=0, training=True):
    """
    :param x: input tensor
    :param total_padding: integer that may also be negative, in which case the input is sliced in its spatial dimensions
    :param training: if True, x is randomly padded; if False, x is padded by total_padding//2 on one side and by the
                     rest on the other side.
    :return: padded tensor
    """

    if total_padding == 0:
        return x
    elif total_padding > 0:
        rand1 = randint(0, total_padding)
        rand2 = randint(0, total_padding)
        return tf.cond(training, lambda: tf.pad(x, tf.constant([[0, 0], [rand1, total_padding - rand1],
                                                                [rand2, total_padding - rand2], [0, 0]])),
                       lambda : tf.pad(x, tf.constant([[0, 0], [total_padding//2, total_padding - total_padding//2],
                                                        [total_padding//2, total_padding - total_padding//2], [0, 0]])))
    else:
        total_padding = abs(total_padding)
        rand1 = randint(0, total_padding)
        rand2 = randint(0, total_padding)
        s = x.shape
        return tf.cond(training, lambda: x[:, rand1:s[1] - total_padding + rand1, rand2:s[2] - total_padding + rand2],
                       lambda: x[:, total_padding//2:s[1] - total_padding + total_padding//2,
                                 total_padding//2:s[2] - total_padding + total_padding//2])


def getNormLayer(norm_type='batch', momentum=0.9, epsilon=1e-5):
    """
    :param norm_type: 'batch' for BatchNorm, 'pixel' for PixelNorm and anything else for no norm
    :param momentum: momentum for BatchNorm
    :param epsilon: epsilon for Batch- and PixelNorm
    :return: the specified norm layer
    """

    if norm_type == 'pixel':
        return Pixel_norm(epsilon)
    if norm_type == 'batch':
        return tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)
    return FactorLayer(1)


class ResBlock(tf.keras.layers.Layer):
    """
    Block of combined convolutions including normalizations and non-linearities. It allows for a residual connection
    between the input and the output.
    """

    def __init__(self, conf, downsample, features, last_features, **kwargs):
        """
        :param conf: configuration of classifier model
        :param downsample: bool whether to downsample input or not
        :param features: number of feature maps used in convolutions
        :param last_features: input's number of features (when using a residual connection and the features differ, the
                              input has to be projected using a small convolution in order for the features to match)
        :param kwargs: should contain the input dimensions
        """

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

    @staticmethod
    def get_name():
        return "ResBlock"


class Padder(tf.keras.layers.Layer):
    """
    Layer that pads its input with a total of padding in both height and width. E.g. an input of size bx125x125xc will
    be padded to bx128x128xc when padding is set to 3.
    """

    def __init__(self, padding=6, **kwargs):
        super(Padder, self).__init__(**kwargs)
        self.padding = padding

    def call(self, x, training=False):
        return get_pad(x, self.padding, training=training)

    def get_config(self):
        config = {'padding': self.padding}
        base_config = super(Padder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        output_shape[1] += self.padding
        output_shape[2] += self.padding
        return output_shape

    @staticmethod
    def get_name():
        return "Padder"


class Pixel_norm(tf.keras.layers.Layer):
    """
    A different normalization layer than BatchNorm. Adopted from the project "Progressive Growing of GANs for Improved
    Quality, Stability, and Variation" (https://github.com/tkarras/progressive_growing_of_gans).
    """

    def __init__(self, epsilon=1e-8):
        super(Pixel_norm, self).__init__()
        self.epsilon = epsilon

    def call(self, x):
        return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + self.epsilon)

    def get_config(self):
        return {'epsilon': self.epsilon}

    @staticmethod
    def get_name():
        return "Pixel_norm"


class FactorLayer(tf.keras.layers.Layer):
    """
    A layer that multiplies its input with factor.
    """

    def __init__(self, factor):
        super(FactorLayer, self).__init__()
        self.factor = factor

    def call(self, x):
        return self.factor * x

    def get_config(self):
        return {'factor': self.factor}

    @staticmethod
    def get_name():
        return "FactorLayer"


def get_custom_objects():
    """
    This is necessary for loading a keras model that contains custom layers. Use like so:
    with open(model_path) as json_file:
        json_config = json_file.read()
    model = tf.keras.models.model_from_json(json_config, custom_objects=get_custom_objects())
    :return: list of all custom layers defined in this module
    """
    
    layers = [ResBlock, Padder, Pixel_norm, FactorLayer]
    return_dict = {}
    for layer in layers:
        return_dict[layer.get_name()] = layer
    return return_dict
