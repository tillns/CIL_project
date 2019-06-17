from __future__ import absolute_import, division, print_function, unicode_literals
# !pip3 install tensorflow-gpu==2.0.0-alpha0
import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time
import os, itertools, pickle
from PIL import Image
import math
from IPython import display
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from random import randint
import sys
from collections import OrderedDict
import yaml
from random import shuffle
from datetime import datetime
from tensorflow.python.keras.engine import training_utils
import cv2
from Models import Models, FactorLayer, Padder, SigmoidLayer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # disables some annoying tensorflow warnings
"""### Set initial parameters"""

with open("config.yaml", 'r') as stream:
    conf = yaml.full_load(stream)
dconf = conf['dis']
gconf = conf['gen']
resolutions = [4, 8, 16, 32, 64, 125, 250, 500, 1000]
image_size = conf['image_size']
min_res = conf['min_res']
image_channels = 1
home_dir = os.path.expanduser("~")
gan_dir = os.path.join(home_dir, "CIL_project/GAN/third")
classifier_cp_dir = os.path.join(home_dir, "CIL_project/Classifier/checkpoints")
labeled_directory = os.path.join(home_dir, "dataset/cil-cosmology-2018/cosmology_aux_data_170429/labeled")
label_path = os.path.join(home_dir, "dataset/cil-cosmology-2018/cosmology_aux_data_170429/labeled.csv")
star_directory = os.path.join(home_dir, "CIL_project/extracted_stars")
if image_size == 28:
    image_directory = star_directory
else:
    image_directory = labeled_directory

max_features = conf['max_features']
# all numbers that 1000 can be evenly downsampled to
even_downsample_sizes = [1, 2, 4, 5, 8, 10, 20, 25, 40, 50, 100, 125, 200, 250, 500, 1000]
dis_features = []
gen_output_activation = None  # should be none
percentage_train = conf['percentage_train']
do_validation = percentage_train < 1
models = ["dis", "gen"]
gan_loss = conf['gan_loss']
batch_size = conf['batch_size']
num_epochs = conf['num_epochs']
ratio_gen_dis = conf['ratio_gen_dis']
vmin = conf['vmin']
vmax = conf['vmax']


def transform(numpy_image_array):
    return numpy_image_array / 255.0 * (vmax-vmin) + vmin


def detransform(numpy_image_array):
    return (numpy_image_array - vmin) / (vmax-vmin) * 255.0


try:
    img_list = []
    for filename in os.listdir(image_directory):
        if filename.endswith(".png") and not filename.startswith("._"):
            img_list.append(filename)

    img_list = sorted(img_list)
except FileNotFoundError:
    sys.exit("Can't find dataset")


def load_dataset():
    global train_images, test_images, dataset_len, train_len, test_len, num_train_it, num_test_it
    images = []
    for num in range(len(img_list)):
        img = Image.open(os.path.join(image_directory, img_list[num])).resize((image_size, image_size))
        img_np = transform(np.array(img, dtype=np.float32).reshape((image_size, image_size, image_channels)))
        images.append(img_np)
    dataset_len = len(images)
    train_len = round(percentage_train * dataset_len)
    test_len = dataset_len - train_len
    num_train_it = math.ceil(train_len/batch_size)
    num_test_it = math.ceil(test_len/batch_size)

    print("Loaded all {} images".format(dataset_len))
    train_images = np.stack(images[:train_len])
    if do_validation:
        test_images = np.stack(images[train_len:])
        assert test_images.shape[0] == test_len
    else:
        test_images = None


# wgangp losses. I crossed out the label penalty terms and adjusted the other stuff
# to tf 2.0 and left out autosummary (for tensorboard) stuff. 
# I think the label penalty was introduced in
# https://arxiv.org/pdf/1610.09585.pdf the ac-gan paper. For multi-class generation,
# this term is useful, so it's probably not necessary in this work
def G_wgangp(labels, fake_scores_out,
             cond_weight=1.0):  # Weight of the conditioning term.

    return -cond_weight * tf.reduce_mean(fake_scores_out)


def D_wgangp(reals, fake_images_out, real_scores_out, fake_scores_out, discriminator,
             wgan_lambda=10.0,  # Weight for the gradient penalty term.
             wgan_epsilon=0.001,  # Weight for the epsilon term, \epsilon_{drift}.
             wgan_target=1.0,  # Target value for gradient magnitudes.
             cond_weight=1.0):  # Weight of the conditioning terms.

    loss = tf.reduce_mean(fake_scores_out) - tf.reduce_mean(real_scores_out)  # this is the normal WGAN loss

    with tf.name_scope('GradientPenalty'):
        alpha = tf.random.uniform(shape=[reals.shape[0]], minval=0., maxval=1.)
        differences = fake_images_out - reals
        interpolates = reals + (differences.mul(alpha, axis=0))
        with tf.GradientTape() as t:
            t.watch(interpolates)
            pred = discriminator(interpolates)
        grad = t.gradient(pred, interpolates)
        norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
        gradient_penalty = tf.reduce_mean((norm - 1.) ** 2)
    loss += gradient_penalty * (wgan_lambda / (wgan_target ** 2))  # this is the gradient penalty term

    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = tf.reduce_mean(tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon  # this is the small additional loss mentioned to avoid drifts from zero

    return loss


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(D_real_logits, D_fake_logits, x, generated_image, discriminator):
    label_zero = tf.zeros([x.shape[0], 1])
    label_one = tf.ones([x.shape[0], 1])

    if gan_loss == 'lsgan':
        return tf.reduce_mean(tf.square(D_fake_logits - label_zero)) + tf.reduce_mean(
            tf.square(D_real_logits - label_one))
    elif gan_loss == 'wgan_gp':
        return D_wgangp(x, generated_image, D_real_logits, D_fake_logits, discriminator)
    else:
        real_loss = cross_entropy(label_one, D_real_logits)
        fake_loss = cross_entropy(label_zero, D_fake_logits)
        return (real_loss + fake_loss)/2


def generator_loss(D_fake_logits):
    label_one = tf.ones([D_fake_logits.shape[0], 1])
    if gan_loss == 'lsgan':
        return tf.reduce_mean(tf.square(D_fake_logits - label_one))
    elif gan_loss == 'wgan_gp':
        return G_wgangp(label_one, D_fake_logits)
    else:
        return cross_entropy(label_one, D_fake_logits)


def alternative_gen_loss(generated_image, real_image):
    return tf.reduce_mean(tf.abs(generated_image - real_image))


generator_optimizer = tf.keras.optimizers.Adam(conf['lr'])
discriminator_optimizer = tf.keras.optimizers.Adam(conf['lr'])


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

    def set_params(self, verbose):
        gen_metrics = ['gen_loss']
        dis_metrics = ['dis_loss']
        if do_validation:
            dis_metrics.append('dis_val_loss')
            dis_metrics.append('gen_val_loss')
        self.params = {'dis': {
            'batch_size': batch_size,
            'epochs': num_epochs,
            'steps': num_train_it,
            'samples': train_len,
            'verbose': verbose,
            'do_validation': do_validation,
            'metrics': dis_metrics,  # not sure here
        }, 'gen': {
            'batch_size': batch_size,
            'epochs': num_epochs,
            'steps': num_train_it,
            'samples': train_len,
            'verbose': verbose,
            'do_validation': do_validation,
            'metrics': gen_metrics,  # not sure here
        }, 'comb': {
            'batch_size': batch_size,
            'epochs': num_epochs,
            'steps': num_train_it,
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
        # print("Logs on epoch end: {}".format(logs))
        for i, callback in enumerate(self.callbacks):
            # print("Logs at {}: {}".format(models[i], logs[models[i]]))
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


def get_mean(some_list):
    return sum(some_list) / len(some_list)


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    predictions = detransform(predictions)
    tot = predictions.shape[0]
    res = predictions.shape[1]
    depth = predictions.shape[3]
    per_row = int(math.sqrt(tot))
    dist = int(0.25 * res)
    outres = per_row * res + dist * (per_row + 1)
    output_image = np.zeros((outres, outres, depth)) + 127.5
    for k in range(tot):
        i = k // per_row
        j = k % per_row
        output_image[dist * (j + 1) + j * res:dist * (j + 1) + res * (j + 1),
        dist * (i + 1) + i * res:dist * (i + 1) + res * (i + 1)] = predictions[k]

    cv2.imwrite(os.path.join(save_image_path, 'image_at_epoch_{:05d}.png'.format(epoch)), output_image)


"""Use `imageio` to create an animated gif using the images saved during training."""


def make_gif():
    anim_file = os.path.join(save_image_path, 'dcgan.gif')
    epoch_imgs = []
    filenames = glob.glob(os.path.join(save_image_path, 'image*.png'))
    filenames = sorted(filenames)
    image = imageio.imread(filenames[-1])
    biggest_res = image.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeft = (biggest_res[0] // 4, biggest_res[1] // 4)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2
    for i, filename in enumerate(filenames):
        image = imageio.imread(filename)
        image = cv2.resize(image, (biggest_res[0], biggest_res[1]))
        cv2.putText(image, "{:05d}".format(i), bottomLeft, font, fontScale, fontColor, lineType)
        epoch_imgs.append(image)

    imageio.mimsave(anim_file, epoch_imgs, 'GIF', duration=0.2)

    import IPython
    if IPython.version_info > (6, 2, 0, ''):
        print("Python version high enough, but gif doesn't seem to show...")
        display.Image(filename=anim_file)
    else:
        print("Python version to old to display file directly. Trying html...")
    IPython.display.HTML('<img src="{}">'.format(anim_file))



def train_step(images, generator, discriminator, iteration, progbar):
    current_batch_size = images.shape[0]
    batch_logs = callbacks.duplicate_logs_for_models({'batch': iteration, 'size': current_batch_size})
    # print("Batch logs: {}".format(batch_logs))
    callbacks._call_batch_hook("train", 'begin', iteration, logs=batch_logs)
    progbar.on_batch_begin(iteration, batch_logs['dis'])  # as of now, the progress bar will only show the dis state
    gen_loss_tot = 0
    for gen_step in range(ratio_gen_dis):
        noise = tf.random.normal([current_batch_size, gconf['input_neurons']])
        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
            generated_images = generator(noise, training=True)
            fake_output = discriminator(generated_images, training=True)
            gen_loss = generator_loss(fake_output)
            gen_loss_tot += gen_loss
            # gen_loss = alternative_gen_loss(generated_images, images)

            if gen_step == ratio_gen_dis-1:
                real_output = discriminator(images, training=True)
                dis_loss = discriminator_loss(real_output, fake_output, images, generated_images, discriminator)
                batch_logs['dis']['dis_loss'] = dis_loss

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    batch_logs['gen']['gen_loss'] = gen_loss_tot/ratio_gen_dis
    gradients_of_discriminator = disc_tape.gradient(dis_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


    callbacks._call_batch_hook('train', 'end', iteration, batch_logs)
    progbar.on_batch_end(iteration, {**batch_logs['gen'], **batch_logs['dis']})
    return gen_loss, dis_loss


load_dataset()
checkpoint_dir = os.path.join(gan_dir, "checkpoints")
checkpoint_dir = os.path.join(checkpoint_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_path = os.path.join(checkpoint_dir, "summary")
if not os.path.exists(tb_path):
    os.makedirs(tb_path)

tb_path_gen = os.path.join(tb_path, "gen")
tb_path_dis = os.path.join(tb_path, "dis")
tb_callback_gen = tf.keras.callbacks.TensorBoard(log_dir=tb_path_gen)
tb_callback_dis = tf.keras.callbacks.TensorBoard(log_dir=tb_path_dis)
callbacks = CallbackList([tb_callback_dis, tb_callback_gen])

save_image_path = os.path.join(checkpoint_dir, "outputs")
if not os.path.exists(save_image_path):
    os.makedirs(save_image_path)

myModels = Models(conf)
discriminator = myModels.get_discriminator_model()
generator = myModels.get_generator_model()
callbacks.set_model([discriminator, generator])
callbacks.set_params(1)
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
progbar = training_utils.get_progbar(generator, 'steps')
progbar.params = callbacks.get_params()
progbar.params['verbose'] = 1

discriminator.summary()
generator.summary()

# Saving important files
with open(os.path.join(checkpoint_dir, 'dis_summary.txt'), 'w') as file:
    discriminator.summary(print_fn=lambda mylambda: file.write(mylambda + '\n'))
with open(os.path.join(checkpoint_dir, 'gen_summary.txt'), 'w') as file:
    generator.summary(print_fn=lambda mylambda: file.write(mylambda + '\n'))
cp_command = 'cp {} {}'.format(os.path.join(gan_dir, "{}"), checkpoint_dir)
os.system(cp_command.format("gan.py"))
os.system(cp_command.format("config.yaml"))

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([batch_size, gconf['input_neurons']])

# callbacks.set_params(batch_size, num_epochs, num_iterations, train_len, 1, do_validation, None,)  # not sure bout metrics=None
callbacks.stop_training(False)
callbacks._call_begin_hook('train')
progbar.on_train_begin()
# callbacks.print_models()
# print("Graph: {}".format(tf.keras.backend.get_session().graph))

if do_validation:
    val_cp_path = os.path.join(classifier_cp_dir, conf['val_model_cp'])
    val_model_path = os.path.join("/".join(val_cp_path.split("/")[:-1]), "model_config.json")
    with open(val_model_path) as json_file:
        json_config = json_file.read()
    custom_objects = {'Padder': Padder, 'FactorLayer': FactorLayer, 'SigmoidLayer': SigmoidLayer}
    val_model = tf.keras.models.model_from_json(json_config, custom_objects=custom_objects)
    val_model.load_weights(val_cp_path)
    val_model.summary()

for epoch in range(num_epochs):
    gen_losses = 0
    dis_losses = 0
    generator.reset_metrics()
    discriminator.reset_metrics()
    epoch_logs = callbacks.duplicate_logs_for_models({})
    callbacks.on_epoch_begin(epoch, epoch_logs)
    progbar.on_epoch_begin(epoch, epoch_logs)
    # shuffle indices pls
    np.random.shuffle(train_images)
    for iteration in range(num_train_it):
        x_ = train_images[iteration * batch_size:min((iteration + 1) * batch_size, train_len)]
        for i in range(2):
            if randint(0, 1):
                x_ = np.flip(x_, axis=i + 1)  # randomly flip x- and y-axis
        if False:
            img_np3 = x_[0, :, :, 0]
            plt.imshow(img_np3, cmap='gray', vmin=0, vmax=1)
            plt.show()
        gen_loss, dis_loss = train_step(x_, generator, discriminator, iteration, progbar)
        gen_losses += gen_loss
        dis_losses += dis_loss

    epoch_logs = {'gen': {'gen_loss': gen_losses / num_train_it},
                  'dis': {'dis_loss': dis_losses / num_train_it}}
    # only test dis on one randomly drawn batch of test data per epoch
    if do_validation:
        callbacks._call_begin_hook('test')
        dis_val_loss = 0
        gen_val_loss = 0
        for iteration in range(num_test_it):
            x_ = test_images[iteration*batch_size:min(test_len, (iteration+1)*batch_size)]
            real_output = discriminator(x_, training=False)
            noise = tf.random.normal([batch_size, gconf['input_neurons']])
            generated_images = generator(noise, training=False)
            fake_output = discriminator(generated_images, training=False)
            val_output = val_model(generated_images, training=False)/8
            gen_val_loss += generator_loss(val_output)
            dis_val_loss += discriminator_loss(real_output, fake_output, x_, generated_images, discriminator)
        callbacks._call_end_hook('test')
        epoch_logs['dis']['dis_val_loss'] = dis_val_loss/num_test_it
        epoch_logs['gen']['gen_val_loss'] = gen_val_loss/num_test_it

    # Save the model every few epochs
    if (epoch + 1) % conf['period_to_save_cp'] == 0:

        try:
            checkpoint_path = os.path.join(checkpoint_dir, "cp_epoch{}".format(epoch + 1))
            checkpoint.save(checkpoint_path)
            print("Checkpoint saved as: {}".format(checkpoint_path))
        except:
            print("Something went wrong with saving the checkpoint")

    # print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    callbacks.on_epoch_end(epoch, epoch_logs)
    progbar.on_epoch_end(epoch, {**epoch_logs['gen'], **epoch_logs['dis']})

    # Generate after every epoch
    generate_and_save_images(generator, epoch, seed)
callbacks._call_end_hook('train')
