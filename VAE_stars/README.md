# VAE for star image generation
A variational autoencoder model for star image generation.

## Paths
- The path to the folder containing the labeled images and the path to the CSV file containing the image labels have to be set inside ```star_vae_train.py```

## Execution
### Training

```python star_vae_train.py```

Or on the cluster:

```bsub -n 8 -W 4:00 -R "rusage[mem=2048, ngpus_excl_p=1]" "python star_vae_train.py"```

The weights of the generative model (decoder) are subsequently saved inside ```/ckpt_generative```.

### Star image generation

```python generate_star_images.py```

Or on the cluster:

```bsub -n 8 -W 4:00 -R "rusage[mem=2048, ngpus_excl_p=1]" "python generate_star_images.py```

The generated 28x28 star images are subsequently saved inside ```/generated```.

### Complete image generation

```python generate_complete_images.py```

Or on the cluster:

```bsub -n 8 -W 4:00 -R "rusage[mem=2048, ngpus_excl_p=1]" "python generate_complete_images.py```

The generated 1000x1000 galaxy images are subsequently saved inside ```/generated```.

### Complete image evaluation

```python evaluate_complete_images.py```

Or on the cluster:

```bsub -n 8 -W 4:00 -R "rusage[mem=2048, ngpus_excl_p=1]" "python evaluate_complete_images.py```

The generated 1000x1000 galaxy images inside ```/generated``` are evaluated. The similarity scores are stored inside ```scorefile.csv```.

## Requirements

- tensorflow-gpu==2.0.0-alpha0
- opencv-python==4.1.0.25
- numpy==1.16.3
- joblib==0.13.2
- scikit-learn==0.20.3
