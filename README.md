# MLVICX: Multi-Level Variance-Covariance Exploration for Chest X-ray Self-Supervised Representation Learning

The code repository of IEEE JBHI 2024 paper [MLVICX: Multi-Level Variance-Covariance Exploration for Chest X-ray Self-Supervised Representation Learning](https://ieeexplore.ieee.org/document/10666966)

## Structure of this repository
This repository is organized as:
- [config](/config/) YAML files
- [data](/data/) preprocessed data and dataloaders
- [models](/models/) MLVICX model architecture
- [optimizers](/optimizers/) LARS for SGD
- [trainer](/trainer/) trainer for MLVICX
- [utils](/utils/) utility functions
- [main.py](/train.py) training MLVICX

## Usage Guide
### Dataset Preparation
#### NIH-CXR 14 Dataset
NIH data is available [here](https://www.kaggle.com/datasets/nih-chest-xrays/data)

We resize images to 224 x 224 for training. Preprocessed CSV files are provided in /data/. Please use the provided dataloaders in /data/ to load data.

Please don't forget to update paths in /config/mlvicx.yaml

The process is similar to training on any other dataset.

### Running
#### Training MLVICX
Once the dataset is set up, update the necessary hyperparameters according to the paper to reproduce any experiments and run,
```
python main.py --mode ssl --init rand --bs 64 --epoch 300 --dataset nih --seed 42 --gpu 0 --resume False 
```
After training, the checkpoints will be stored in ```/ckpt``` as assigned. Check the trainer for making any changes. 

If you want to try different models, use ```--model```. For fine-tuning, change mode ```--mode sl```.

### Citation
If you find our work useful please cite as,
```bibtex
@article{mlvicx,
  author={Singh, Azad and Gorade, Vandan and Mishra, Deepak},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={MLVICX: Multi-Level Variance-Covariance Exploration for Chest X-Ray Self-Supervised Representation Learning}, 
  year={2024},
  volume={28},
  number={12},
  pages={7480-7490}}
```
