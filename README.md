# PD-INR: Prior-Driven Implicit Neural Representations for Time-of-Flight PET Reconstruction

## Introduction
PD-INR is a self-supervised reconstruction framework for TOF-PET imaging.  
It integrates priors into an implicit neural representation (INR) model, providing high-quality image reconstruction.

## Installation

```bash
git clone https://github.com/zyl123300/PD-INR.git
cd PD-INR
conda env create -f environment.yml
conda activate PD_INR
```

## Usage 
Data Generation:
The training data is simulated using `data/generate_sino.py`, based on the `pytomography` library.

To start training,run:
```bash
python train.py --config config/basic.yaml
```