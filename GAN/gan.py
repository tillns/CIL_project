from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import PIL
import os
from PIL import Image
import math
import yaml
from datetime import datetime
from tensorflow.python.keras.engine import training_utils
import cv2
from Models import Models, get_custom_objects
from create_complete_images import create_complete_images
from img_scorer import load_km_with_conf, load_rf_with_conf, score_tensor_with_rf, \
    score_tensor_with_keras_model, km_transform
from CustomCallbacks import CallbackList
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
gan_dir = os.path.join(home_dir, "CIL_project/GAN")
classifier_cp_dir = os.path.join(home_dir, "CIL_project/Classifier/checkpoints")
labeled_directory = os.path.join(home_dir, "dataset/cil-cosmology-2018/cosmology_aux_data_170429/labeled")
label_path = os.path.join(home_dir, "dataset/cil-cosmology-2018/cosmology_aux_data_170429/labeled.csv")
star_directory = os.path.join(home_dir, "CIL_project/AE_plus_KMeans/clustered_images/labeled1_and_scoredover3_5cats") \
    if conf['conditional'] else os.path.join(home_dir, "CIL_project/extracted_stars_Hannes")
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
conditional = conf['conditional']


def transform_norm(numpy_image_array):
    return numpy_image_array / 255.0 * (vmax-vmin) + vmin


def detransform_norm(numpy_image_array):
    return (numpy_image_array - vmin) / (vmax-vmin) * 255.0


def load_dataset():
    global train_images, test_images, train_labels, test_labels, dataset_len, train_len, test_len, \
        num_train_it, num_test_it
    images = []
    if not conditional:
        for filename in os.listdir(image_directory):
            if filename.endswith(".png") and not filename.startswith("._"):
                img = Image.open(os.path.join(image_directory, filename)).resize((image_size, image_size))
                img_np = transform_norm(np.array(img, dtype=np.float32).reshape((image_size, image_size, image_channels)))
                images.append(img_np)
        dataset = np.stack(images)
    else:
        label_list = []
        num_classes = len(os.listdir(image_directory))
        conf['num_classes'] = num_classes
        for label, folder_name in enumerate(sorted(os.listdir(image_directory))):
            folder_path = os.path.join(image_directory, folder_name)
            label_vec = np.zeros(num_classes)
            label_vec[label] += 1
            for img_name in sorted(os.listdir(folder_path)):
                img = Image.open(os.path.join(folder_path, img_name)).resize((image_size, image_size))
                img_np = transform_norm(np.array(img, dtype=np.float32).reshape((image_size, image_size, image_channels)))
                images.append(img_np)
                label_list.append(label_vec)
        dataset = np.stack(images)
        labels = np.stack(label_list)
        # dataset and labels are shuffled together, s.t. training and test set have same distribution
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
        dataset = dataset[indices]
        labels = labels[indices]

    dataset_len = len(dataset)
    train_len = round(percentage_train * dataset_len)
    test_len = dataset_len - train_len
    num_train_it = math.ceil(train_len/batch_size)
    num_test_it = math.ceil(test_len/batch_size)

    print("Loaded all {} images".format(dataset_len))
    train_images = dataset[:train_len]
    train_labels = labels[:train_len] if conditional else None
    if do_validation:
        test_images = dataset[train_len:]
        test_labels = labels[train_len:] if conditional else None
    else:
        test_images = None
        test_labels = None


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(D_real_logits, D_fake_logits):
    label_zero = tf.zeros([D_real_logits.shape[0], 1])
    label_one = tf.ones([D_real_logits.shape[0], 1])

    if gan_loss == 'lsgan':
        loss = tf.reduce_mean(tf.square(D_real_logits - label_one))
        if D_fake_logits is not None:
            loss = (loss + tf.reduce_mean(tf.square(D_fake_logits - label_zero)))/2
        return loss
    else:
        loss = cross_entropy(label_one, D_real_logits)
        if D_fake_logits is not None:
            loss = (loss + cross_entropy(label_zero, D_fake_logits))/2
        return loss


def generator_loss(D_fake_logits):
    label_one = tf.ones([D_fake_logits.shape[0], 1])
    if gan_loss == 'lsgan':
        return tf.reduce_mean(tf.square(D_fake_logits - label_one))
    else:
        return cross_entropy(label_one, D_fake_logits)


generator_optimizer = tf.keras.optimizers.Adam(conf['lr'], decay=conf['lr_decay'])
discriminator_optimizer = tf.keras.optimizers.Adam(conf['lr'], decay=conf['lr_decay'])


def one_hot(batch_y, num_classes):
    y_ = np.zeros((batch_y.shape[0], num_classes))
    y_[np.arange(batch_y.shape[0]), batch_y] = 1
    return y_


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).

    tot = test_input.shape[0]
    per_row = int(math.sqrt(tot)) if not conditional else conf['num_classes']
    if conditional:
        labels = one_hot(np.array(list(range(conf['num_classes']))*(tot//conf['num_classes'])), conf['num_classes'])
        predictions = model([test_input, labels], training=False)
    else:
        predictions = model(test_input, training=False)
    predictions = detransform_norm(predictions)
    res = predictions.shape[1]
    depth = predictions.shape[3]

    dist = int(0.25 * res)
    outres = per_row * res + dist * (per_row + 1)
    output_image = np.zeros((outres, outres, depth)) + 127.5
    for k in range(tot):
        i = k // per_row
        j = k % per_row
        output_image[dist * (i + 1) + i * res:dist * (i + 1) + res * (i + 1),
        dist * (j + 1) + j * res:dist * (j + 1) + res * (j + 1)] = predictions[k]

    cv2.imwrite(os.path.join(save_image_path, 'image_at_epoch_{:05d}.png'.format(epoch)), output_image)



def train_step(images, labels, generator, discriminator, iteration, progbar):
    current_batch_size = images.shape[0]
    batch_logs = callbacks.duplicate_logs_for_models({'batch': iteration, 'size': current_batch_size})
    # print("Batch logs: {}".format(batch_logs))
    callbacks._call_batch_hook("train", 'begin', iteration, logs=batch_logs)
    progbar.on_batch_begin(iteration, batch_logs['dis'])  # as of now, the progress bar will only show the dis state
    gen_loss_tot = 0
    for gen_step in range(ratio_gen_dis):
        noise = tf.random.normal([current_batch_size, gconf['input_neurons']])
        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
            generated_images = generator([noise, labels], training=True) if conditional else \
                generator(noise, training=True)
            fake_output = discriminator([generated_images, labels], training=True) if conditional else \
                discriminator(generated_images, training=True)
            gen_loss = generator_loss(fake_output)
            gen_loss_tot += gen_loss
            # gen_loss = alternative_gen_loss(generated_images, images)

            if gen_step == ratio_gen_dis-1:
                real_output = discriminator([images, labels], training=True) if conditional else \
                    discriminator(images, training=True)
                dis_loss = discriminator_loss(real_output, fake_output)
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
callbacks = CallbackList([tb_callback_dis, tb_callback_gen], models)

save_image_path = os.path.join(checkpoint_dir, "outputs")
if not os.path.exists(save_image_path):
    os.makedirs(save_image_path)

myModels = Models(conf)
discriminator = myModels.get_discriminator_model()
generator = myModels.get_generator_model()
callbacks.set_model([discriminator, generator])
callbacks.set_params(do_validation, batch_size, num_epochs, num_train_it, train_len, 1)
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
seed = tf.random.normal([conf['num_classes']**2, gconf['input_neurons']]) if conditional \
    else tf.random.normal([25, gconf['input_neurons']])

# callbacks.set_params(batch_size, num_epochs, num_iterations, train_len, 1, do_validation, None,)  # not sure bout metrics=None
callbacks.stop_training(False)
callbacks._call_begin_hook('train')
progbar.on_train_begin()
# callbacks.print_models()
# print("Graph: {}".format(tf.keras.backend.get_session().graph))

if do_validation:
    min_dis_val_loss = 10000
    max_gen_val_score = 0
    nn_valmodel, nn_valconf = load_km_with_conf(os.path.join(classifier_cp_dir, conf['nn_val_model_path']))
    rf_model, rf_conf = load_rf_with_conf(os.path.join(os.path.join(home_dir, "CIL_project/RandomForest"),
                                                       conf['rf_val_model_path']))

json_config = generator.to_json()
with open(os.path.join(checkpoint_dir, 'gen_config.json'), 'w') as json_file:
    json_file.write(json_config)
json_config = discriminator.to_json()
with open(os.path.join(checkpoint_dir, 'dis_config.json'), 'w') as json_file:
    json_file.write(json_config)
for epoch in range(num_epochs):
    gen_losses = 0
    dis_losses = 0
    generator.reset_metrics()
    discriminator.reset_metrics()
    epoch_logs = callbacks.duplicate_logs_for_models({})
    callbacks.on_epoch_begin(epoch, epoch_logs)
    progbar.on_epoch_begin(epoch, epoch_logs)
    # shuffle indices pls
    indices = np.arange(train_len)
    np.random.shuffle(indices)
    for iteration in range(num_train_it):
        x_ = train_images[indices[iteration * batch_size:min((iteration + 1) * batch_size, train_len)]]
        labels = train_labels[indices[iteration * batch_size:min((iteration + 1) * batch_size, train_len)]] if \
            conditional else None
        if False:
            img_np3 = x_[0, :, :, 0]
            plt.imshow(img_np3, cmap='gray', vmin=conf['vmin'], vmax=conf['vmax'])
            plt.xlabel(np.argmax(labels[0]))
            plt.show()
        gen_loss, dis_loss = train_step(x_, labels, generator, discriminator, iteration, progbar)
        gen_losses += gen_loss
        dis_losses += dis_loss

    epoch_logs = {'gen': {'gen_loss': gen_losses / num_train_it},
                  'dis': {'dis_loss': dis_losses / num_train_it}}
    # only test dis on one randomly drawn batch of test data per epoch
    if do_validation and (epoch+1)%conf['period_for_val'] == 0:
        callbacks._call_begin_hook('test')
        dis_val_loss = 0
        gen_val_score = 0
        if image_size == 28:
            np_img_tensor = detransform_norm(create_complete_images(generator, conf['vmin'], conf['num_val_images'],
                                                                    conf['num_classes']))
            rf_score = score_tensor_with_rf(np_img_tensor, rf_model, rf_conf)
            nn_score = score_tensor_with_keras_model(km_transform(np_img_tensor, nn_valconf['use_fft']),
                                                             nn_valmodel, nn_valconf['batch_size'])
            rf_score = np.expand_dims(rf_score, axis=1)
            score = np.concatenate((rf_score, nn_score))
            gen_val_score = tf.reduce_mean(score)

        for iteration in range(num_test_it):
            x_ = test_images[iteration*batch_size:min(test_len, (iteration+1)*batch_size)]
            labels = test_labels[iteration*batch_size:min(test_len, (iteration+1)*batch_size)] if conditional else None
            real_output = discriminator([x_, labels], training=False) if conditional else \
                discriminator(x_, training=False)
            dis_val_loss += discriminator_loss(real_output, None)

        callbacks._call_end_hook('test')
        dis_val_loss /= num_test_it
        min_dis_val_loss = min(min_dis_val_loss, dis_val_loss)
        save_new_cp = max_gen_val_score > gen_val_score
        max_gen_val_score = min(max_gen_val_score, gen_val_score)
        epoch_logs['dis']['dis_val_loss'] = dis_val_loss
        epoch_logs['dis']['min_dis_val_loss'] = min_dis_val_loss
        epoch_logs['gen']['gen_val_score'] = gen_val_score
        epoch_logs['gen']['max_gen_val_score'] = max_gen_val_score



    # Save the model every few epochs
    if ((epoch + 1) % conf['period_to_save_cp'] == 0) or (do_validation and save_new_cp ):

        try:
            checkpoint_path = os.path.join(checkpoint_dir, "cp_{}_epoch{}".format("{}", epoch + 1))
            generator.save_weights(checkpoint_path.format("gen"))
            discriminator.save_weights(checkpoint_path.format("dis"))
            print("Checkpoint saved as: {}".format(checkpoint_path))
        except:
            print("Something went wrong with saving the checkpoint")

    # print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    callbacks.on_epoch_end(epoch, epoch_logs)
    progbar.on_epoch_end(epoch, {**epoch_logs['gen'], **epoch_logs['dis']})

    # Generate after every epoch
    generate_and_save_images(generator, epoch, seed)
callbacks._call_end_hook('train')
