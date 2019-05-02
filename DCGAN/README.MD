# DCGAN

**Sources**

https://www.tensorflow.org/alpha/tutorials/generative/dcgan (TensorFlow DCGAN tutorial, code is based on this)  
https://arxiv.org/abs/1511.06434 (DCGAN paper, contains values for most hyperparameters)  
https://github.com/Newmu/dcgan_code/blob/master/imagenet/load_pretrained.py (code from the authors of the DCGAN paper)  
https://github.com/carpedm20/DCGAN-tensorflow/blob/master/ops.py (referenced public TensorFlow implementation)  

**Virtual environment setup**

*need modules python, CUDA, cuDNN*    
module load gcc/4.8.5 python_gpu/3.6.4 hdf5 eth_proxy  
module load cudnn/7.4  
module load cuda/10.0.130  

*install virtualenvwrapper*  
pip install --user virtualenvwrapper   
source $HOME/.local/bin/virtualenvwrapper.sh  

*create virtual environment*  
mkvirtualenv "galaxy-image-tf2"

*disable / enable virtual environment*  
deactivate   
workon "galaxy-image-tf2"  

*install required dependencies*  
pip install tensorflow-gpu==2.0.0-alpha0  
pip install Pillow  

*add to ~/.bashrc*  
module load gcc/4.8.5 python_gpu/3.6.4 hdf5 eth_proxy  
module load cudnn/7.4  
module load cuda/10.0.130  
source $HOME/.local/bin/virtualenvwrapper.sh  
workon "env-name"  

