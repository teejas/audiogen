#!/bin/bash

sudo apt-get update -y && sudo apt-get install -y python3 python3-pip ffmpeg
alias python=python3
alias pip=pip3
pip install -r requirements.txt
mkdir ./ckpt
wget https://zenodo.org/record/7813012/files/audioldm-m-full.ckpt -O ./ckpt/audioldm-m-full.ckpt
