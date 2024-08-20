#!/usr/bin/env bash

# Create environment
conda create -y -n 01_intro_dl python=3.11

# Activate environment
conda activate 01_intro_dl

# Install dependencies
conda install -y matplotlib jupyter tqdm
conda install -y pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

conda activate base
