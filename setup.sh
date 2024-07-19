#!/usr/bin/env bash

# Create environment
mamba create -y -n 02_intro_dl python=3.9

# Activate environment
mamba activate 02_intro_dl

# Install dependencies
mamba install -y matplotlib jupyter tqdm
mamba install -y pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

mamba deactivate

