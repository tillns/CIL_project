# CIL_project
Project for the ETH Computational Intelligence Lab.

Authors: Sven Kellenberger, Hannes Pfammatter, Till Schnabel, Michelle Woon

Group: Galaxy Crusaders

## Requirements

- jupyter
- sklearn
- joblib
- pyyaml
- numpy
- PIL
- cv2
- pywt
- cython
- matplotlib
- gc
- albumentations

After installing all requirements, head into the `utils/` folder and run `python setup.py build_ext -i`
to compile the cython files there.

## Image generation task

### AE_plus_KMeans

### Adhoc_generator

This adhoc method randomly places stars that it has detected from the given labelled images and
places them randomly onto a black image. Open the notebook by heading into the `Adhoc_generator/`
subdirectory, running `jupyter notebook` and opening `Adhoc.ipynb`.

### DCGAN

### GAN

### VAE_stars

### stars_extractor

### Autoencoder plus KMeans

The purpose of this project is to find a compact representation for a dataset in order to cluster it with kmeans.
Use ae.py to train an autoencoder on an image directory containg 28x28 images
and use kmeans.py to then cluster the same images with the help of the saved encoder. Please refer to the classes'
individual documentation for detailed use.

TODO: Further explanation needed

## Similarity scorer task

### Classifier

A similarity scorer using a Convolutional Neural Network.

Training:

    python classifier.py --dataset_dir=/path/to/cosmology_aux_data_170429

Prediction:

    python classifier.py --test_on_query=True --restore_ckpt=True --ckpt_path=/path/to/checkpoint/cp####.ckpt.data-00000-of-00001

Where `/path/to/checkpoint/cp####.ckpt.data-00000-of-00001`
is a valid path to the checkpoint and `####` is replaced with the checkpoint number. 

### RandomForest

This model takes two command line arguments which are required:

- `--data-directory`: The directory where the dataset is stored.
- `--dump-direcotry`: The directory where all generated data should be stored. This directory
will be created if it doesn't exist yet.

All other options can be set in the config file.

To train the model, head into the `RandomForest/` subdirectory and run:

    python random_forest.py --data-directory=/path/to/the/data/ --dump-directory=/path/to/dump/directory/
