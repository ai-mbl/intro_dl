#!/usr/bin/env bash

# Create environment
conda create -y -n 02_intro_dl python=3.9

# Activate environment
conda activate 02_intro_dl

# Install dependencies
conda install -y matplotlib jupyter tqdm
conda install -y pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

conda activate base
