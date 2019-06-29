# VAE for star image generation
A variational autoencoder model for star image generation.

## Paths
- All paths have to be set in ...


## Execution
### Training

```python classifier.py```

Or on the cluster:

```bsub -n 20 -W 24:00 -o log_file -R "rusage[mem=5000, ngpus_excl_p=1]" -R "select[gpu_model0==TeslaV100_SXM2_32GB]" python classifier.py --is_cluster=True```

### Image Geneneration

```python classifier.py --test_on_query=True --restore_ckpt=True --ckpt_path=/path/to/checkpoint/cp####.ckpt.data-00000-of-00001```

Or on the cluster:

```bsub -n 20 -W 1:00 -o log_file -R "rusage[mem=5000, ngpus_excl_p=1]" -R "select[gpu_model0==TeslaV100_SXM2_32GB]" python classifier.py --is_cluster=True --test_on_query=True --restore_ckpt=True --ckpt_path=/path/to/checkpoints/cp####.ckpt.data-00000-of-00001```

Where in both cases ```/path/to/checkpoint/cp####.ckpt.data-00000-of-00001``` is a valid path to the checkpoint and ```####``` is replaced with the checkpoint number.

## Requirements

- tensorflow
- cv2
- numpy
