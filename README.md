# CIL_project
Project for the ETH Computational Intelligence Lab.

Authors: Sven Kellenberger, Hannes Pfammatter, Till Schnabel, Michelle Woon

Group: Galaxy Crusaders

## Table of Contents

1. [Requirements](#requirements)
2. [Image Generation Task](#image_generation)

   [stars_extractor](#stars_extractor)
   
   [Adhoc_generator](#adhoc)
   
   [DCGAN](#dcgan)
   
   [cDCGAN](#cdcgan)
   
   [Generate Complete Images](#generate_complete_images)
   
   [VAE_stars](#vae)
   
   [AE_plus_KMeans](#ae_plus_kmeans)
   
   [Image Scorer](#image_scorer)
   
3. [Similarity Scorer Task](#similarity_task)

   [Classifier](#classifier)
   
   [RandomForest](#random_forest)


<a name="requirements"/>

## Requirements

Install all requirements with

    pip install -r requirements.txt

After installing all requirements, head into the `utils/` folder and run `python setup.py build_ext -i`
to compile the cython files there.

<a name="image_generation"/>

## Image Generation Task

<a name="stars_extractor"/>

### stars_extractor

This project only contains some scripts to extract stars from the original images (stars_extractor.py), filter the
original star images (create_dir_for_labeled_star_images.py) and also for measuring and approximating an unsigned integer
gaussian distribution of all kinds of stars in the images (stars_clustered_distribution.py). The files can be
run without additional arguments (e.g. `python stars_extractor.py`). Please refer to their individual documentation for
adjustment of the default arguments.

<a name="adhoc"/>

### Adhoc_generator

This adhoc method randomly places stars that it has detected from the given labelled images and
places them randomly onto a black image.

    python Adhoc.py --data_path=/path/to/data

<a name="dcgan"/>

### DCGAN

**Sources**

https://www.tensorflow.org/alpha/tutorials/generative/dcgan (TensorFlow DCGAN tutorial, code is based on this)  
https://arxiv.org/abs/1511.06434 (DCGAN paper, contains values for most hyperparameters)  
https://github.com/Newmu/dcgan_code/blob/master/imagenet/load_pretrained.py (code from the authors of the DCGAN paper)  
https://github.com/carpedm20/DCGAN-tensorflow/blob/master/ops.py (referenced public TensorFlow implementation)  

<a name="cdcgan"/>

### cDCGAN

    python gan.py --dataset-dir=/path/to/dataset

To train a conditional model on the 28x28 star patches, adjust the config.yaml s.t. the variable `conditional`is set to `True` and `model_kind`to `4`. Also make sure that the
provided data set contains a folder for each category with the corresponding images inside. For unconditional training on
the 28x28 star patches, set `conditional`is set to `False` and `model_kind`to `3`, and make sure that the provided path
to the dat set directly contains the images. Th results will be saved in a new folder inside the `checkpoints` directory.

<a name="generate_complete_images"/>

### Generate Complete Images

Use `create_complete_images.py` for one to generate and score images using the save distribution and to find an even better
distribution. Provide the path to a cDCGAN checkpoint as argument `--checkpoint_path` if you wish to use another than the default
one. Set `--find_good_latents` to `False` if you wish to simply create and score some images. If not specified, the module
will loop infinitely to find a better distribution.   

<a name="vae"/>

### VAE_stars

A variational autoencoder model for star image generation.The path to the folder containing the labeled images and the path to the CSV file containing the image labels have to be set inside `star_vae_train.py` which can than be executed with
    
    python star_vae_train.py

The weights of the generative model (decoder) are subsequently saved inside `/ckpt_generative`.

To generate small star images run:

    python generate_star_images.py

The generated 28x28 star images are subsequently saved inside `/generated`.

To create complete star images, run then:

    python generate_complete_images.py

The generated 1000x1000 galaxy images are subsequently saved inside `/generated`.

To evaluate the generated image, run:

    python evaluate_complete_images.py

The generated 1000x1000 galaxy images inside `/generated` are evaluated. The similarity scores are stored inside `scorefile.csv`.

<a name="ae_plus_kmeans"/>

### AE_plus_KMeans

The purpose of this project is to find a compact representation for a dataset in order to cluster the stars using lower-dimensional data.
First, an autoencoder is trained to find said compact representation. Afterwards, k-means is applied to the encoder's latent
code of the images to cluster them.

For the autoencoder, run

    python ae.py --image_dir DIR/WITH/STAR/PATCHES

where `DIR/WITH/STAR/PATCHES` is the directory containing the star patches of size 28x28 directly. A trained model of the
encoder is saved in a separate directory inside the `checkpoints` folder. Provide the path to this encoder model to the
k-means script as argument. Run

    python kmeans.py --econder_path PATH/TO/encoder_config.json

The clustered images are saved to a separate directory inside `images/clustered_stars` if not specified otherwise via
the `--target_dir` argument.

<a name="image_scorer"/>

### Image Scorer

Use the file `cDCGAN/img_scorer.py` to score an arbitrary image of size 1000x1000 or a folder containing images of size
1000x1000. Provide the path to either the image or the folder via the argument `--path`. You will get as output a score
approximated by both the CNN and RF and additionally their mean. Run

    python img_scorer.py --path=/path/to/images

<a name="similarity_task"/>

## Similarity Scorer Task

<a name="classifier"/>

### Classifier

A similarity scorer using a Convolutional Neural Network.

Training:

    python classifier.py --dataset_dir=/path/to/cosmology_aux_data_170429

Prediction:

    python classifier.py --test_on_query=True --dataset_dir=/path/to/cosmology_aux_data_170429 --ckpt_path=/path/to/checkpoint/cp####.ckpt.data-00000-of-00001

Where `/path/to/checkpoint/cp####.ckpt.data-00000-of-00001`
is a valid path to the checkpoint and `####` is replaced with the checkpoint number.

<a name="random_forest"/>

### RandomForest

This model takes two command line arguments which are required:

- `--data-directory`: The directory where the dataset is stored.
- `--dump-direcotry`: The directory where all generated data should be stored. This directory
will be created if it doesn't exist yet.

All other options can be set in the config file.

To train the model, head into the `RandomForest/` subdirectory and run:

    python random_forest.py --data-directory=/path/to/the/data/ --dump-directory=/path/to/dump/directory/
