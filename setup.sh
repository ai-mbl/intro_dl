#!/usr/bin/env bash

# Create environment
mamba create -y -n 02_intro_dl python=3.9

# Activate environment
mamba activate 02_intro_dl

# Install dependencies
mamba install -y tensorflow tensorflow-gpu keras matplotlib jupyter

mamba deactivate

