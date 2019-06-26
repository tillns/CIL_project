import tensorflow as tf
from utils import getNormLayer, get_pad


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

    @staticmethod
    def get_name():
        return "ResBlock"


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

    @staticmethod
    def get_name():
        return "Padder"


class Pixel_norm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-8):
        super(Pixel_norm, self).__init__()
        self.epsilon = epsilon

    def call(self, x):
        # print("input shape in pixel norm layer: {}".format(x.shape))
        return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + self.epsilon)

    def get_config(self):
        return {'epsilon': self.epsilon}

    @staticmethod
    def get_name():
        return "Pixel_norm"


class FactorLayer(tf.keras.layers.Layer):
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
    layers = [ResBlock, Padder, Pixel_norm, FactorLayer]
    return_dict = {}
    for layer in layers:
        return_dict[layer.get_name()] = layer
    return return_dict
