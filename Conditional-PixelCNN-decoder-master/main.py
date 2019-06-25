import tensorflow as tf
import numpy as np
import argparse
from models import PixelCNN
from autoencoder import *
from utils import *
import os

def train(conf, data, labels=None):
    X = tf.placeholder(tf.float32, shape=[None, conf.img_height, conf.img_width, conf.channel])
    model = PixelCNN(X, conf)

    trainer = tf.train.RMSPropOptimizer(1e-3)
    gradients = trainer.compute_gradients(model.loss)

    clipped_gradients = [(tf.clip_by_value(_[0], -conf.grad_clip, conf.grad_clip), _[1]) for _ in gradients]
    optimizer = trainer.apply_gradients(clipped_gradients)

    saver = tf.train.Saver(tf.trainable_variables())

    with tf.Session() as sess: 
        sess.run(tf.initialize_all_variables())
        if os.path.exists(conf.ckpt_file):
            saver.restore(sess, conf.ckpt_file)
            print("Model Restored")
       
        if conf.epochs > 0:
            print("Started Model Training...")
        pointer = 0
        for i in range(conf.epochs):
            indices = np.arange(len(data))
            np.random.shuffle(indices)
            for j in range(conf.num_batches):
                batch_x = data[indices[j*conf.batch_size:(j+1)*conf.batch_size]]
                batch_y = None if not conf.conditional else labels[indices[j*conf.batch_size:(j+1)*conf.batch_size]]

                data_dict = {X: batch_x}
                if conf.conditional is True:
                    data_dict[model.h] = batch_y
                _, cost = sess.run([optimizer, model.loss], feed_dict=data_dict)
            print("Epoch: %d, Cost: %f"%(i, cost))
            if (i+1)%10 == 0:
                saver.save(sess, conf.ckpt_file)
                generate_samples(sess, X, model.h, model.pred, conf, "")

        generate_samples(sess, X, model.h, model.pred, conf, "")


def transform(numpy_image_array, vmin=0, vmax=1):
    return numpy_image_array / 255.0 * (vmax-vmin) + vmin

def load_dataset(path, image_size, image_channels, vmin=0, vmax=1, conditional=False):
    images = []
    if not conditional:
        for img_name in sorted(os.listdir(path)):
            img = Image.open(os.path.join(path, img_name)).resize((image_size, image_size))
            img_np = transform(np.array(img, dtype=np.float32).reshape((image_size, image_size, image_channels)),
                               vmin, vmax)

            images.append(img_np)
        return np.stack(images), None
    else:
        labels = []
        num_classes = len(os.listdir(path))
        for label, folder_name in enumerate(sorted(os.listdir(path))):
            folder_path = os.path.join(path, folder_name)
            label_vec = np.zeros(num_classes)
            label_vec[label] += 1
            for img_name in sorted(os.listdir(folder_path)):
                img = Image.open(os.path.join(folder_path, img_name)).resize((image_size, image_size))
                img_np = transform(np.array(img, dtype=np.float32).reshape((image_size, image_size, image_channels)),
                                   vmin, vmax)
                np_to_look_at = img_np[:, :, 0]
                images.append(img_np)
                labels.append(label_vec)
        return np.stack(images), np.stack(labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--f_map', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--grad_clip', type=int, default=1)
    parser.add_argument('--model', type=str, default='', help="'', 'conditional' or 'autoencoder'")
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default='ckpts')
    parser.add_argument('--samples_path', type=str, default='samples')
    parser.add_argument('--summary_path', type=str, default='logs')
    conf = parser.parse_args()

    conf.conditional = conf.model != ''
    if conf.data_path is None:
        if conf.conditional:
            path = os.path.join(os.path.expanduser("~"),
                                "CIL_project/AE_plus_KMeans/clustered_images/labeled1_and_scoredover3_5cats")
        else:
            path = os.path.join(os.path.expanduser("~"),
                                "dataset/cil-cosmology-2018/cosmology_aux_data_170429/labeled1_and_scoredover3")
    conf.img_height = 28
    conf.img_width = 28
    conf.channel = 1
    data, labels = load_dataset(path, conf.img_height, conf.channel, conditional=conf.conditional)
    conf.num_batches = data.shape[0] // conf.batch_size
    conf.num_classes = np.argmax(labels[-1])+1 if labels is not None else 5

    conf = makepaths(conf) 
    if conf.model == '':
        train(conf, data)
    elif conf.model.lower() == 'conditional':
        train(conf, data, labels)
    elif conf.model.lower() == 'autoencoder':
        trainAE(conf, data)


