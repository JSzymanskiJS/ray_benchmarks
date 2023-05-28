#!/bin/bash

conda update conda -y
conda info
conda create --name ray_benchmarks python=3.10 -y
conda init bash
conda activate ray_benchmarks
conda install jupyter -y
pip install torch torchvision torchaudio
pip install numpy pandas boto3 tqdm tabulate
pip install -U "ray[default]"
