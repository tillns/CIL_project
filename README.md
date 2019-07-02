# CIL_project
Project for the ETH Computational Intelligence Lab.

Authors: Sven Kellenberger, Hannes Pfammatter, Till Schnabel, Michelle Woon

Group: Galaxy Crusaders

## Requirements

- sklearn
- joblib
- pyyaml
- numpy
- PIL
- cv2
- pywt
- cython

After installing all requirements, head into the `utils/` folder and run `python setup.py build_ext -i`
to compile the cython files there.

## Image generation task

### AE_plus_KMeans
### Adhoc_generator
### Conditional-PixelCNN-decoder-master
### DCGAN
### GAN
### VAE_stars
### stars_extractor

## Similarity scorer task

### Classifier

### RandomForest

This model takes two command line arguments which are required:

- `--data-directory`: The directory where the dataset is stored.
- `--dump-direcotry`: The directory where all generated data should be stored. This directory
will be created if it doesn't exist yet.

All other options can be set in the config file.

To train the model, head into the `RandomForest` directory and run:

    python random_forest.py --data-directory=/path/to/the/data/ --dump-directory=/path/to/dump/directory/
