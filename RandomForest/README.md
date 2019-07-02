# RandomForest Similarity Scorer

Similarity Scorer based on an ensemble of RandomForests

## Building the CPP code
To run the main file, the cpp code has to be built. To do this, use the following command in this folder:

```python setup.py build_ext -i```

## Training and prediction of kaggle query

```python random_forest.py --data-directory=path/to/cosmology_aux_data_170429 --numpy-directory=./np_out```

Where ```path/to/cosmology_aux_data_170429``` is a valid path to the data folder.

## Requirements
- sklearn
- joblib
- pyyaml
- numpy
- PIL
- cv2
- pywt
- cython

# run this in this directory:
#
# pip install Cython
# python setup.py build_ext -i
