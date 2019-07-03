# CIL_project
Project for the ETH Computational Intelligence Lab.

Authors: Sven Kellenberger, Hannes Pfammatter, Till Schnabel, Michelle Woon

Group: Galaxy Crusaders

## TODO

### Delete unnecessary files

- Generated images are at the moment in many different locations and even have different names. Either delete them or make a seperate directory with only images and remove images everywhere else. Also remove `.zip` files.
- `RandomForest/final_model/` if this is needed for another model, save the `.pkl` file there (only this file, no config, losses etc.).
- `papers/` I don't think that we should have the papers we cite on Github
- `cDCGAN/` I would remove the `reference_run` because it is reproducible. I would also remove the `.pkl` files but we could also leave those.
- `Classifier/` Same for `reference_run`
- `DCGAN` remove checkpoints and the `.csv` files

### Code

- All open points in the code are marked with a `TODO` comment.
- Documentation is not always consistent (at times `Paramters`, other times `params:`). However this is consistent if you only look into at single files so this doens't have to be done.
- Some methods lack documentation. However they are mostly short so this isn't really urgent.
- I am not sure but I am under the impression that some things are not needed any more. The person who created it should be able to decide this but don't be afraid to delete things! (after all that's why we use git)

## Requirements

- jupyter
- sklearn
- joblib
- pyyaml
- numpy
- Pillow
- cv2
- pywt
- cython
- matplotlib
- gc
- albumentations
- opencv-python

After installing all requirements, head into the `utils/` folder and run `python setup.py build_ext -i`
to compile the cython files there.

## Image generation task

## Stars Extractor

This project only contains some scripts to extract stars from the original images (stars_extractor.py), filter the original star images (create_dir_for_labeled_star_images.py) and also for measuring and approximating an integer gaussian distribution of all kinds of stars in the images (stars_clustered_distribution.py). The files can be run without additional arguments (e.g. python stars_extractor.py). Please refer to their individual documentation for adjustment of the default arguments.

### AE_plus_KMeans

### Adhoc_generator

This adhoc method randomly places stars that it has detected from the given labelled images and
places them randomly onto a black image.

##### Generation
`python ./Adhoc_generator/Adhoc.py --data_path=/path/to/data`


### DCGAN

**Sources**

https://www.tensorflow.org/alpha/tutorials/generative/dcgan (TensorFlow DCGAN tutorial, code is based on this)  
https://arxiv.org/abs/1511.06434 (DCGAN paper, contains values for most hyperparameters)  
https://github.com/Newmu/dcgan_code/blob/master/imagenet/load_pretrained.py (code from the authors of the DCGAN paper)  
https://github.com/carpedm20/DCGAN-tensorflow/blob/master/ops.py (referenced public TensorFlow implementation)  

### GAN

### VAE_stars

A variational autoencoder model for star image generation.

#### Paths
- The path to the folder containing the labeled images and the path to the CSV file containing the image labels have to be set inside `star_vae_train.py`

#### Execution
##### Training

`python star_vae_train.py`

Or on the cluster:

`bsub -n 8 -W 4:00 -R "rusage[mem=2048, ngpus_excl_p=1]" "python star_vae_train.py"`

The weights of the generative model (decoder) are subsequently saved inside `/ckpt_generative`.

##### Star image generation

`python generate_star_images.py`

Or on the cluster:

`bsub -n 8 -W 4:00 -R "rusage[mem=2048, ngpus_excl_p=1]" "python generate_star_images.py`

The generated 28x28 star images are subsequently saved inside `/generated`.

##### Complete image generation

`python generate_complete_images.py`

Or on the cluster:

`bsub -n 8 -W 4:00 -R "rusage[mem=2048, ngpus_excl_p=1]" "python generate_complete_images.py`

The generated 1000x1000 galaxy images are subsequently saved inside `/generated`.

##### Complete image evaluation

`python evaluate_complete_images.py`

Or on the cluster:

`bsub -n 8 -W 4:00 -R "rusage[mem=2048, ngpus_excl_p=1]" "python evaluate_complete_images.py`

The generated 1000x1000 galaxy images inside `/generated` are evaluated. The similarity scores are stored inside `scorefile.csv`.

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
