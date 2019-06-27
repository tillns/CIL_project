# RandomForest Similarity Scorer

Similarity Scorer based on an ensemble of RandomForests

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