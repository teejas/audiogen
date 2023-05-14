#!/bin/bash

sudo apt-get update -y && sudo apt-get install -y python3 python3-pip ffmpeg # ffmpeg necessary for passing mp3 to audioLDM
alias python=python3
alias pip=pip3

# install python dependencies
pip install -r requirements.txt

# next 2 lines install the m-full ckpt file, for the medium sized pre-trained model. Paths to other ckpt files can be found in the readme
mkdir ./ckpt
wget https://zenodo.org/record/7813012/files/audioldm-m-full.ckpt -O ./ckpt/audioldm-m-full.ckpt

# next 2 lines are GCP-specific, installs nvidia drivers to leverage GPU
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
python install_gpu_driver.py
