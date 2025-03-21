# Data-driven Global Ocean Modeling for Seasonal to Decadal Prediction

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2405.15412-b31b1b.svg)](https://arxiv.org/abs/2405.15412)

</div>

**This repository contains the official implementation of ORCA-DL**
---------------------------------------------------------------

<!-- ## 📌 Overview

**Brief description of your implementation (1-2 paragraphs). Mention:**

* **Key contributions of the paper**
* **Main features of this implementation**
* **Framework/libraries used (PyTorch/TensorFlow/JAX etc.)** -->

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/OpenEarthLab/ORCA-DL.git
cd ORCA-DL
conda create -n orca python=3.9.17
conda activate orca
pip install -r requirements.txt
```

### Resources Download

All the model weights and data can be found in  https://1drv.ms/f/c/49d761d10f0b201d/Emi9scIyaWBCrNTgRo6t12oBLnF2qGDRGj0M7-g0ekRM1A

### Quick Demo

See [demo.ipynb](https://github.com/OpenEarthLab/ORCA-DL/blob/main/demo.ipynb)

Please note that ORCA-DL initially uses [GODAS](https://psl.noaa.gov/data/gridded/data.godas.html) data as its starting point. Initialization with other data is also feasible, but it is necessary to ensure that the data is interpolated to the correct longitude and latitude range and resolution, as in [example_data](https://github.com/OpenEarthLab/ORCA-DL/blob/main/example_data). Here is an example of how to interpolate the original data using [CDO](https://code.mpimet.mpg.de/projects/cdo) (recommended):

```bash
wget https://downloads.psl.noaa.gov/Datasets/godas/sshg.1980.nc -O sshg-1980.nc   # download a 2D data from GODAS
cdo -b f64 remapbil,grid sshg-1980.nc sshg-1980-processed.nc

wget https://downloads.psl.noaa.gov/Datasets/godas/salt.1980.nc -O salt-1980.nc   # download a 3D data from GODAS
cdo -b f64 remapbil,grid salt-1980.nc tmp1.nc
cdo intlevel,10,15,30,50,75,100,125,150,200,250,300,400,500,600,800,1000 tmp1.nc tmp2.nc
cdo setzaxis,zaxis.txt tmp2.nc salt-1980-processed.nc
rm tmp1.nc tmp2.nc
```

After the data interpolation is completed, you can refer to the [demo.ipynb](https://github.com/OpenEarthLab/ORCA-DL/blob/main/demo.ipynb) to run ORCA-DL.

## 📋 Updates

- Training data and code are coming soon.
- **2025-03-21:** Data preprocessing processes are released.
- **2025-03-04:** Model weights and demo code are released.

## 📄 Citation

**If you find this work useful, please cite our paper:**

```
@article{guo2024data,
  title={Data-driven Global Ocean Modeling for Seasonal to Decadal Prediction},
  author={Guo, Zijie and Lyu, Pumeng and Ling, Fenghua and Bai, Lei and Luo, Jing-Jia and Boers, Niklas and Yamagata, Toshio and Izumo, Takeshi and Cravatte, Sophie and Capotondi, Antonietta and Ouyang, Wanli},
  journal={arXiv preprint arXiv:2405.15412},
  year={2024}
}
```
