#!/usr/bin/env bash

ENV_NAME="01_intro_dl"

# Create environment
conda create -y -n $ENV_NAME python=3.11

# Install dependencies
conda install -y matplotlib jupyter tqdm --name 01_intro_dl

# Install PyTorch
ENV_PATH=$(conda info --base)/envs/$ENV_NAME
$ENV_PATH/bin/pip install "torch"
$ENV_PATH/bin/pip install "torchvision"

