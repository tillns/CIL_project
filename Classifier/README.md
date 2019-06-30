# CNN Similarity Scorer
A similarity scorer using a Convolutional Neural Network.

## Paths
- Make sure that the folder containing the images (from kaggle) is located in the path

```'~/dataset/cil-cosmology-2018/cosmology_aux_data_170429/'```

- Also, make sure that the root directory of this project is located in your home directory. Else ```classifier.py``` will not run properly.

- Create folders named ```numpy_data``` and ```checkpoints``` in this folder.


## Execution
### Training

```python classifier.py```

Or on the cluster:

```bsub -n 20 -W 24:00 -o log_file -R "rusage[mem=5000, ngpus_excl_p=1]" -R "select[gpu_model0==TeslaV100_SXM2_32GB]" python classifier.py --is_cluster=True```

To replicate the reference run, the config file should be left as is and classifier.py should be run with either of the two commands above depending on the situation.

### Prediction

```python classifier.py --test_on_query=True --restore_ckpt=True --ckpt_path=/path/to/checkpoint/cp####.ckpt.data-00000-of-00001```

Or on the cluster:

```bsub -n 20 -W 1:00 -o log_file -R "rusage[mem=5000, ngpus_excl_p=1]" -R "select[gpu_model0==TeslaV100_SXM2_32GB]" python classifier.py --is_cluster=True --test_on_query=True --restore_ckpt=True --ckpt_path=/path/to/checkpoints/cp####.ckpt.data-00000-of-00001```

Where in both cases ```/path/to/checkpoint/cp####.ckpt.data-00000-of-00001``` is a valid path to the checkpoint and ```####``` is replaced with the checkpoint number. To replicate the score for the reference run, the path should be ```~/CIL_project/Classifier/reference_run/cp0140.ckpt.data-00000-of-00001```

## Requirements

- tensorflow
- cv2
- matplotlib
- numpy
- PIL
- sklearn
- yaml
- gc
- albumentations