# Practical Work in AI: Personalizing item relevance with psychology-based interest for repeat-aware music recommender systems

## Pre-requisites

- miniconda installed
- GPU available
- CUDA 11.5 or higher
- Python 3.8 or higher installed

In order to get started with the code, follow these steps:

## 1. Clone GitHub repository

Clone this repo and initialize the submodules via this command:

`git clone --recurse-submodules https://github.com/Laura-Legat/Practical-Work-AI`

## 2. Create miniconda environment such that the libraries work

Via this command (executed from the root folder):

`conda env create -f environment.yml`

And activate:

`conda activate pr_ai`

## 3. Run the setup to install location-independent custom code

Via this command (executed from the root folder):

`python setup.py install`

## 4. Create dataset files 

Via this command (executed from the root folder):

`python preprocess.py -sl 50 -st 1`

The `-sl` flag is the sequence length, `-st` the stride. Feel free to adapt these to your liking.

## Hyperparameter optimization

Perform hyperparameter optimization for any number of trials for either of the models with the `optuna_paropt.py` script

## Training

Train/finaltrain Ex2Vec by running the `train.py` script. Similarly, train GRU4Rec by running the `run.py` script located in the `GRU4Rec_Fork` submodule folder.

## Dataset
The dataset is provided [here](https://zenodo.org/record/8316236).

## References

```
@inproceedings{sguerra2023ex2vec,
  title={Ex2Vec: Characterizing Users and Items from the Mere Exposure Effect},
  author={Sguerra, Bruno and Tran, Viet-Anh and Hennequin, Romain},
  booktitle = {Proceedings of the 17th ACM Conference on Recommender Systems},
  year = {2023}
}
```
