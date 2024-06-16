# Aerial segment buildings

[![lint](https://github.com/AmarskiyArtem/aerial-segment-buildings/actions/workflows/ci.yml/badge.svg)](https://github.com/AmarskiyArtem/aerial-segment-buildings/actions/workflows/ci.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/AmarskiyArtem/aerial-segment-buildings)

This repository provides an interface for validating building segmentation methods: Segment-anything, YOLOv8seg and [rgb-footprint-extract](https://github.com/aatifjiwani/rgb-footprint-extract).

## Installation
Create conda env.
```
conda create -n aerial-segment python=3.9
```
Clone this repo.
```
git clone https://github.com/AmarskiyArtem/aerial-segment-buildings.git
cd aerial-segment-buildings
```
Install requirements.
```
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install -r requirements.txt 
```
## Running
```
python3 eval.py [args]
```
args:
| Parameter | Description |
| --------  |--------------|
| --model_name | sam, yolo or drn_c42 (rgbfootprint)
| --checkpoint_path | Path to model checkpoint
| --device | cuda or cpu
| --data_dir | path to the directory with images. Should contains images/ and masks/ dirs
| --batch_size | batch size (default: 20)
| --model_type | model type for SAM model (default: None)
