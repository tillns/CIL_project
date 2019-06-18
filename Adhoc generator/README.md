# Adhoc image generator

This adhoc method randomly places stars that it has detected from the given labelled images and places them randomly onto a black image. This image has some low noise on it, so that it is not a solid plack colour.

## Execution

Make sure that the folder containing the images (from kaggle) is located in the same folder as the git directory, i.e. 

```data_path = '../../cosmology_aux_data_170429/labeled'```
from this directory.

Open up ```Adhoc.ipynb``` in the jupyter notebook and run the code.

## Requirements

- cv2
- matplotlib
- numpy
- PIL

