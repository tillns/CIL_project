import tensorflow as tf
from random import randint

"""###Define some custom layers"""


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

class TanhLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(TanhLayer, self).__init__()

    def call(self, x):
        return tf.keras.activations.tanh(x)

    def get_config(self):
        return {}


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

    def get_config(self):
        config = {'factor': self.factor}
        base_config = super(Upscale2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# This is kind of a spetial implementation as the padding is TOTAL, so both sides combined
class Padder(tf.keras.layers.Layer):
    def __init__(self, padding=6, **kwargs):
        super(Padder, self).__init__(**kwargs)
        self.padding = padding

    def call(self, x):
        return get_pad(x, self.padding)

    def get_config(self):
        config = {'padding': self.padding}
        base_config = super(Padder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        output_shape[1] += self.padding
        output_shape[2] += self.padding
        return output_shape


class Models():
    def __init__(self, conf):
        self.conf = conf
        self.model_kind = 3 if conf['image_size'] == 28 else conf['model_kind']
        self.dis_features=[]


    def get_discriminator_model(self):
        conf = self.conf
        dconf = conf['dis']
        features = conf['features']
        res = conf['image_size']
        model = tf.keras.Sequential(name='dis')
        if self.model_kind == 1:
            resexp = [pow(2, x) for x in range(10)]
            if conf['weight_reg_factor'] == 0:
                kernel_regularizer = None
            elif conf['weight_reg_kind'] == 'l2':
                kernel_regularizer = tf.keras.regularizers.l2(conf['weight_reg_factor'])
            else:
                kernel_regularizer = tf.keras.regularizers.l1(conf['weight_reg_factor'])

            while res > conf['min_res']:
                self.dis_features.append(features)
                if res % 2 != 0:
                    closestexpres = min(resexp, key=lambda x: abs(x - res))
                    model.add(Padder(padding=closestexpres - res,
                                     input_shape=(res, res, conf['image_channels'])))  # 125 -> 128
                    res = closestexpres
                for i in range(conf['num_convs_per_res']):
                    strides = 2 if i == 0 and dconf['strided_conv'] else 1
                    model.add(tf.keras.layers.Conv2D(features, (conf['kernel'], conf['kernel']),
                                                     kernel_regularizer=kernel_regularizer, padding='same', strides=strides,
                                                     use_bias=dconf['use_bias'], input_shape=(res, res, conf['image_channels'])))
                    # depth wrong for following convs, but doesn't seem to matter, so I'll let it be
                    model.add(getNormLayer(conf['norm_type']))
                    model.add(tf.keras.layers.LeakyReLU(alpha=conf['lrelu_alpha']))
                    if i == 0 and not dconf['strided_conv']:
                        model.add(tf.keras.layers.MaxPool2D())
                if features < conf['max_features']:
                    features *= 2
                res = res // 2

            model.add(tf.keras.layers.Flatten())
            for i in range(1, dconf['num_dense_layers']):
                model.add(tf.keras.layers.Dense(features, use_bias=dconf['use_bias'], kernel_regularizer=kernel_regularizer))
                model.add(getNormLayer(conf['norm_type']))
                model.add(tf.keras.layers.LeakyReLU(alpha=conf['lrelu_alpha']))
                if dconf['dropout'] > 0:
                    model.add(tf.keras.layers.Dropout(dconf['dropout']))
            model.add(tf.keras.layers.Dense(1, use_bias=dconf['use_bias'], kernel_regularizer=kernel_regularizer))
            model.add(SigmoidLayer())
        elif self.model_kind == 2:
            while res > conf['min_res']:
                self.dis_features.append(features)
                if res == 125:
                    model.add(Padder(padding=3, input_shape=[res, res, 1]))
                    res = 128
                model.add(tf.keras.layers.Conv2D(features, (5, 5), strides=2, padding='same'))
                model.add(tf.keras.layers.LeakyReLU(alpha=conf['lrelu_alpha']))
                model.add(tf.keras.layers.Dropout(dconf['dropout']))
                if features < conf['max_features']:
                    features *= 2
                res = res // 2

            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(1))
            #model.add(SigmoidLayer())

        elif self.model_kind == 3:
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                             input_shape=[28, 28, 1]))
            model.add(tf.keras.layers.LeakyReLU())
            model.add(tf.keras.layers.Dropout(0.3))

            model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
            model.add(tf.keras.layers.LeakyReLU())
            model.add(tf.keras.layers.Dropout(0.3))

            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(1))
        return model

    def get_generator_model(self):
        conf = self.conf
        gconf = conf['gen']
        counter = 1
        res = conf['min_res']
        model = tf.keras.Sequential(name='gen')
        if self.model_kind == 1:
            model.add(tf.keras.layers.Dense(self.dis_features[-1]*res*res, use_bias=gconf['use_bias'],
                                            input_shape=(gconf['input_neurons'],)))
            model.add(getNormLayer(conf['norm_type']))
            model.add(tf.keras.layers.LeakyReLU(alpha=conf['lrelu_alpha']))
            model.add(tf.keras.layers.Reshape((res, res, self.dis_features[-1])))
            while res < conf['image_size']:
                features = self.dis_features[-counter]
                res *= 2
                counter += 1
                for i in range(conf['num_convs_per_res']):
                    if i == conf['num_convs_per_res'] - 1 and res >= conf['image_size']:
                        features = conf['image_channels']
                    if i == conf['num_convs_per_res']-1 and not gconf['transp_conv']:
                        model.add(Upscale2D(factor=2))
                    if i == conf['num_convs_per_res']-1 and gconf['transp_conv']:
                        model.add(tf.keras.layers.Conv2DTranspose(features, (conf['kernel'], conf['kernel']),
                                                         padding='same', strides=2, use_bias=gconf['use_bias']))
                    else:
                        model.add(tf.keras.layers.Conv2D(features, (conf['kernel'], conf['kernel']),
                                                         padding='same', strides=1, use_bias=gconf['use_bias']))

                    if features != conf['image_channels']:
                        model.add(getNormLayer(conf['norm_type']))
                        model.add(tf.keras.layers.LeakyReLU(alpha=conf['lrelu_alpha']))
                    else:
                        if conf['vmin'] == 0 and conf['vmax'] == 1:
                            model.add(SigmoidLayer())
                        elif conf['vmin'] == -1 and conf['vmax'] == 1:
                            model.add(TanhLayer())

                if res == 128:
                    model.add(Padder(padding=-3))  # 128 -> 125
                    res = 125

        elif self.model_kind == 2:
            model = tf.keras.Sequential(name='gen')
            res = conf['min_res']
            model.add(tf.keras.layers.Dense(res*res*self.dis_features[-1], use_bias=False, input_shape=(gconf['input_neurons'],)))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.LeakyReLU(alpha=conf['lrelu_alpha']))

            model.add(tf.keras.layers.Reshape((res, res, self.dis_features[-1])))

            counter = 1
            while res < conf['image_size']:
                res *= 2
                features = self.dis_features[-counter] if res < conf['image_size'] else conf['image_channels']
                counter += 1
                model.add(tf.keras.layers.Conv2DTranspose(features, (5, 5), strides=2, padding='same', use_bias=False))
                if features != conf['image_channels']:
                    model.add(tf.keras.layers.BatchNormalization())
                    model.add(tf.keras.layers.LeakyReLU(alpha=conf['lrelu_alpha']))
                else:
                    if conf['vmin'] == 0 and conf['vmax'] == 1:
                        model.add(SigmoidLayer())
                    elif conf['vmin'] == -1 and conf['vmax'] == 1:
                        model.add(TanhLayer())
                    model.add(Padder(padding=-3))

        elif self.model_kind == 3:
            model.add(tf.keras.layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.LeakyReLU())

            model.add(tf.keras.layers.Reshape((7, 7, 256)))
            assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

            model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
            assert model.output_shape == (None, 7, 7, 128)
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.LeakyReLU())

            model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
            assert model.output_shape == (None, 14, 14, 64)
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.LeakyReLU())

            model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same',
                                                      use_bias=False, activation='tanh'))
            assert model.output_shape == (None, 28, 28, 1)

        return model