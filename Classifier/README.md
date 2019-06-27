# CNN Similarity Scorer
A similarity scorer using a Convolutional Neural Network.

## Dataset path
Make sure that the folder containing the images (from kaggle) is located in the path

```'~/dataset/cil-cosmology-2018/cosmology_aux_data_170429/'```

## Execution
### Training

```python classifier.py```

Or on the cluster:

```bsub -n 20 -W 24:00 -o log_file -R "rusage[mem=5000, ngpus_excl_p=1]" -R "select[gpu_model0==TeslaV100_SXM2_32GB]" python classifier.py --is_cluster=True```

### Prediction

```python classifier.py --test_on_query=True --restore_ckpt=True --ckpt_path=/path/to/checkpoint/cp####.ckpt.data-00000-of-00001```

Or on the cluster:

```bsub -n 20 -W 1:00 -o log_file -R "rusage[mem=5000, ngpus_excl_p=1]" -R "select[gpu_model0==TeslaV100_SXM2_32GB]" python classifier.py --is_cluster=True --test_on_query=True --restore_ckpt=True --ckpt_path=/path/to/checkpoints/cp####.ckpt.data-00000-of-00001```

Where in both cases ```/path/to/checkpoint/cp####.ckpt.data-00000-of-00001``` is a valid path to the checkpoint and ```####``` is replaced with the checkpoint number.

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