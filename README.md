# machine-learning-notes
This repository was created for self studying.

## Setup Python 3.6.5 environment for Ubuntu 18.04:
1. Download and install [Miniconda](https://conda.io/miniconda.html)
```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
bash Miniconda3-latest-Linux-x86_64.sh
```
2. Create environment:
```
 conda create -n env_name python=3.7
```
3. Activate environment:
```
source activate env_name
```
4. Install dependency:
```
conda install scipy numpy matplotlib scikit-learn scikit-image csvkit jupyter \
nomkl numexpr cloudpickle pickleshare h5py CFFI requests beautifulsoup4 line_profiler \
memory_profiler pillow tqdm lxml opencv
```
5. [optional] [Installing TensorFlow on Ubuntu](https://www.tensorflow.org/install/install_linux)
6. [optional] [Install PyCharm](https://www.jetbrains.com/pycharm/download/#section=linux)  
sudo snap install pycharm-community --classic
