"""
This is the main module of this project. It trains a conditional or unconditional DCGAN on either the (clustered)
star patches or the whole images. No arguments are required. Edit the necessary configuration in config.yaml.
Following methods are implemented:
    #discriminator_loss
    #generator_loss
    #train_step
    #train_gan
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import os
import yaml
import argparse
from datetime import datetime
from tensorflow.python.keras.engine import training_utils
from Models import Models, get_custom_objects
from create_complete_images import create_complete_images
from img_scorer import load_km_with_conf, load_rf_with_conf, score_tensor_with_rf, \
    score_tensor_with_keras_model, km_transform
from CustomCallbacks import CallbackList
from gan_utils import load_dataset, generate_and_save_images, detransform_norm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # disables some annoying tensorflow warnings

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, default=None, help="Complete path to directory containing images or "
                                                                  "image class folders with corresponding images")

def discriminator_loss(real_logits, fake_logits, conf):
    """
    :param real_logits: output of discriminator given a real input
    :param fake_logits: output of discriminator given a fake generated input
    :param conf: GAN configuration
    :return: discriminator loss where the loss kind is chosen from the configuration
    """

    label_zero = tf.zeros([real_logits.shape[0], 1])
    label_one = tf.ones([real_logits.shape[0], 1])

    if conf['gan_loss'] == 'lsgan':
        loss = tf.reduce_mean(tf.square(real_logits - label_one))
        if fake_logits is not None:
            loss = (loss + tf.reduce_mean(tf.square(fake_logits - label_zero))) / 2
        return loss
    else:
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(label_one, real_logits)
        if fake_logits is not None:
            loss = (loss + tf.keras.losses.BinaryCrossentropy(from_logits=True)(label_zero, fake_logits)) / 2
        return loss


def generator_loss(fake_logits, conf):
    """
    :param fake_logits: discriminator output given a fake generated input
    :param conf: GAN configuration
    :return: generator loss where the loss kind is chosen from the configuration
    """

    label_one = tf.ones([fake_logits.shape[0], 1])
    if conf['gan_loss'] == 'lsgan':
        return tf.reduce_mean(tf.square(fake_logits - label_one))
    else:
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)(label_one, fake_logits)


def train_step(images, labels, generator, discriminator, iteration, progbar, conf, callbacks, generator_optimizer,
               discriminator_optimizer):
    """
    Performs one training step, which consists of conf['ratio_gen_dis'] number of generator iterations and one
    discriminator iteration
    :param images: tensor of input input images ([b x image_size x image_size x 1])
    :param labels: tensor including one vector for each image if conditional else None
    :param generator: model
    :param discriminator: model
    :param iteration: the current iteration step
    :param progbar: progress bar object
    :param conf: GAN configuration
    :param callbacks: CustomCallback object in which generator and discriminator model are saved
    :param generator_optimizer:
    :param discriminator_optimizer:
    :return: generator and discriminator loss
    """

    current_batch_size = images.shape[0]
    batch_logs = callbacks.duplicate_logs_for_models({'batch': iteration, 'size': current_batch_size})
    # print("Batch logs: {}".format(batch_logs))
    callbacks.call_batch_hook("train", 'begin', iteration, logs=batch_logs)
    progbar.on_batch_begin(iteration, batch_logs['dis'])  # as of now, the progress bar will only show the dis state
    gen_loss_tot = 0
    for gen_step in range(conf['ratio_gen_dis']):
        noise = tf.random.normal([current_batch_size, conf['gen']['input_neurons']])
        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
            generated_images = generator([noise, labels], training=True) if conf['conditional'] else \
                generator(noise, training=True)
            fake_output = discriminator([generated_images, labels], training=True) if conf['conditional'] else \
                discriminator(generated_images, training=True)
            gen_loss = generator_loss(fake_output, conf)
            gen_loss_tot += gen_loss
            # gen_loss = alternative_gen_loss(generated_images, images)

            if gen_step == conf['ratio_gen_dis']-1:
                real_output = discriminator([images, labels], training=True) if conf['conditional'] else \
                    discriminator(images, training=True)
                dis_loss = discriminator_loss(real_output, fake_output, conf)
                batch_logs['dis']['dis_loss'] = dis_loss

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    batch_logs['gen']['gen_loss'] = gen_loss_tot/conf['ratio_gen_dis']
    gradients_of_discriminator = disc_tape.gradient(dis_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    callbacks.call_batch_hook('train', 'end', iteration, batch_logs)
    progbar.on_batch_end(iteration, {**batch_logs['gen'], **batch_logs['dis']})
    return gen_loss, dis_loss


def train_gan(args):
    """
    Main method that controls the whole training procedure. Validation uses RF and NN classifier models (only for
    patches). There are a total of conf['num_epochs'] number of epochs. Validation is performed every
    conf['period_for_val']. A checkpoint is saved every conf['period_to_save_cp'] and when a new max_gen_val_score has
    been achieved. A summary for tensorboard is kept in the summary folder inside the checkpoint directory.
    :param args: arguments passed to module
    """

    with open("config.yaml", 'r') as stream:
        conf = yaml.full_load(stream)
    image_size = conf['image_size']
    cil_dir = os.path.dirname(os.path.dirname(__file__))
    gan_dir = os.path.join(cil_dir, "cDCGAN")
    classifier_dir = os.path.join(cil_dir, "Classifier")
    if args.dataset_dir is None:
        if conf['conditional']:
            image_directory = os.path.join(cil_dir, "AE_plus_KMeans/clustered_images/labeled1_and_scoredover3_5cats")
        else:
            image_directory = os.path.join(cil_dir, "extracted_stars/labeled1_and_scoredover3")
    else:
        image_directory = args.dataset_dir

    do_validation = conf['percentage_train'] < 1
    models = ["dis", "gen"]
    batch_size = conf['batch_size']

    generator_optimizer = tf.keras.optimizers.Adam(conf['lr'], decay=conf['lr_decay'])
    discriminator_optimizer = tf.keras.optimizers.Adam(conf['lr'], decay=conf['lr_decay'])

    train_images, test_images, train_labels, test_labels = load_dataset(conf, image_directory, image_size)
    num_train_it = int(np.ceil(len(train_images) / batch_size))
    num_test_it = int(np.ceil(len(test_images) / batch_size))
    checkpoint_dir = os.path.join(os.path.join(gan_dir, "checkpoints"), datetime.now().strftime("%Y%m%d-%H%M%S"))
    tb_path = os.path.join(checkpoint_dir, "summary")
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)

    tb_path_gen = os.path.join(tb_path, "gen")
    tb_path_dis = os.path.join(tb_path, "dis")
    tb_callback_gen = tf.keras.callbacks.TensorBoard(log_dir=tb_path_gen)
    tb_callback_dis = tf.keras.callbacks.TensorBoard(log_dir=tb_path_dis)
    callbacks = CallbackList([tb_callback_dis, tb_callback_gen], models)

    save_image_path = os.path.join(checkpoint_dir, "outputs")
    if not os.path.exists(save_image_path):
        os.makedirs(save_image_path)

    my_models = Models(conf)
    discriminator = my_models.get_discriminator_model()
    generator = my_models.get_generator_model()
    callbacks.set_model([discriminator, generator])
    callbacks.set_params(do_validation, batch_size, conf['num_epochs'], num_train_it, len(train_images), 1)
    progbar = training_utils.get_progbar(generator, 'steps')
    progbar.params = callbacks.get_params()
    progbar.params['verbose'] = 1

    discriminator.summary()
    generator.summary()

    # Saving important files
    with open(os.path.join(checkpoint_dir, 'dis_summary.txt'), 'w') as file_dis:
        discriminator.summary(print_fn=lambda mylambda: file_dis.write(mylambda + '\n'))
    with open(os.path.join(checkpoint_dir, 'gen_summary.txt'), 'w') as file_gen:
        generator.summary(print_fn=lambda mylambda: file_gen.write(mylambda + '\n'))
    os.system('cp {} {}'.format(os.path.join(gan_dir, "config.yaml"), checkpoint_dir))

    # We will reuse this seed overtime to visualize progress in the animated GIF)
    seed = tf.random.normal([conf['num_classes'] ** 2, conf['gen']['input_neurons']]) if conf['conditional'] \
        else tf.random.normal([25, conf['gen']['input_neurons']])

    callbacks.stop_training(False)
    callbacks.call_begin_hook('train')
    progbar.on_train_begin()

    if do_validation:
        min_dis_val_loss = 10000
        max_gen_val_score = 0
        nn_valmodel, nn_valconf = load_km_with_conf(os.path.join(classifier_dir, conf['nn_val_model_path']))
        rf_model, rf_conf = load_rf_with_conf(os.path.join(os.path.join(cil_dir, "RandomForest"),
                                                           conf['rf_val_model_path']))

    json_config = generator.to_json()
    with open(os.path.join(checkpoint_dir, 'gen_config.json'), 'w') as json_file:
        json_file.write(json_config)
    json_config = discriminator.to_json()
    with open(os.path.join(checkpoint_dir, 'dis_config.json'), 'w') as json_file:
        json_file.write(json_config)
    for epoch in range(conf['num_epochs']):
        gen_losses = 0
        dis_losses = 0
        generator.reset_metrics()
        discriminator.reset_metrics()
        epoch_logs = callbacks.duplicate_logs_for_models({})
        callbacks.on_epoch_begin(epoch, epoch_logs)
        progbar.on_epoch_begin(epoch, epoch_logs)
        # shuffle indices pls
        indices = np.arange(len(train_images))
        np.random.shuffle(indices)
        for iteration in range(num_train_it):
            x_ = train_images[indices[iteration * batch_size:min((iteration + 1) * batch_size, len(train_images))]]
            labels = train_labels[indices[iteration * batch_size:min((iteration + 1) * batch_size,
                                                                     len(train_images))]] if conf[
                'conditional'] else None
            gen_loss, dis_loss = train_step(x_, labels, generator, discriminator, iteration, progbar, conf, callbacks,
                                            generator_optimizer, discriminator_optimizer)
            gen_losses += gen_loss
            dis_losses += dis_loss

        epoch_logs = {'gen': {'gen_loss': gen_losses / num_train_it},
                      'dis': {'dis_loss': dis_losses / num_train_it}}
        save_new_cp = ((epoch + 1) % conf['period_to_save_cp'] == 0)

        # do validation every specified number of epochs
        if do_validation and (epoch + 1) % conf['period_for_val'] == 0:
            callbacks.call_begin_hook('test')
            dis_val_loss = 0
            if image_size == 28:
                np_img_tensor, _, _ = create_complete_images(generator, conf['vmin'], conf['num_val_images'],
                                                             conf['num_classes'])
                np_img_tensor = detransform_norm(np_img_tensor, conf)
                rf_score = score_tensor_with_rf(np_img_tensor, rf_model, rf_conf)
                nn_score = score_tensor_with_keras_model(km_transform(np_img_tensor, nn_valconf['use_fft']),
                                                         nn_valmodel, nn_valconf['batch_size'])
                rf_score = np.expand_dims(rf_score, axis=1)
                score = np.concatenate((rf_score, nn_score))
                gen_val_score = tf.reduce_mean(score)
                save_new_cp = save_new_cp or gen_val_score > max_gen_val_score
                max_gen_val_score = max(max_gen_val_score, gen_val_score)
                epoch_logs['gen']['gen_val_score'] = gen_val_score
                epoch_logs['gen']['max_gen_val_score'] = max_gen_val_score

            for iteration in range(num_test_it):
                x_ = test_images[iteration * batch_size:min(len(test_images), (iteration + 1) * batch_size)]
                labels = test_labels[iteration * batch_size:min(len(test_images), (iteration + 1) * batch_size)] if \
                    conf['conditional'] else None
                real_output = discriminator([x_, labels], training=False) if conf['conditional'] else \
                    discriminator(x_, training=False)
                dis_val_loss += discriminator_loss(real_output, None, conf)

            callbacks.call_end_hook('test')
            dis_val_loss /= num_test_it
            min_dis_val_loss = min(min_dis_val_loss, dis_val_loss)

            epoch_logs['dis']['dis_val_loss'] = dis_val_loss
            epoch_logs['dis']['min_dis_val_loss'] = min_dis_val_loss

        # Save the model every few epochs
        if save_new_cp:
            checkpoint_path = os.path.join(checkpoint_dir, "cp_{}_epoch{}".format("{}", epoch + 1))
            generator.save_weights(checkpoint_path.format("gen"))
            discriminator.save_weights(checkpoint_path.format("dis"))
            print("Checkpoint saved as: {}".format(checkpoint_path))

        callbacks.on_epoch_end(epoch, epoch_logs)
        progbar.on_epoch_end(epoch, {**epoch_logs['gen'], **epoch_logs['dis']})

        # Generate after every epoch
        generate_and_save_images(generator, epoch, seed, conf, save_image_path)
    callbacks.call_end_hook('train')


if __name__ == '__main__':
    train_gan(parser.parse_args())
