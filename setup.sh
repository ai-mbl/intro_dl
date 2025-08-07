#!/usr/bin/env bash

# Create environment
conda create -y -n 01_intro_dl python=3.11

# Install dependencies
conda install -y matplotlib jupyter tqdm --name 01_intro_dl

# Install PyTorch
ENV_PATH=$(conda info --envs | grep 01_intro_dl | awk '{print $NF}')
$ENV_PATH/bin/pip install "torch"
$ENV_PATH/bin/pip install "torchvision"

