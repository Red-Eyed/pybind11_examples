# Virtual env with conda
This repository was created for self studying.

## Setup Python 3.6.5 environment for Ubuntu 18.04:
1. Download and install [Miniconda](https://conda.io/miniconda.html)
```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
bash Miniconda3-latest-Linux-x86_64.sh
```
2. Create environment:
```
 conda create -n py36 python=3.6.5
```
3. Activate environment:
```
source activate py36
```
4. Install dependency:
```
conda install scipy numpy matplotlib scikit-learn scikit-image csvkit jupyter \
nomkl numexpr cloudpickle pickleshare h5py CFFI requests beautifulsoup4 line_profiler \
memory_profiler pillow tqdm lxml opencv
```

