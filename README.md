# Practical Work in AI: Personalizing item relevance with psychology-based interest for repeat-aware music recommender systems

## Pre-requisites

- miniconda installed
- GPU available

In order to get started with the code, follow these steps:

## 1. Clone GitHub repository

Clone this repo and initialize the submodules via this command:

`git clone --recurse-submodules https://github.com/Laura-Legat/Practical-Work-AI`

## 2. Create miniconda environment such that the libraries work

Via this command (executed from the root folder):

`conda env create -f environment.yml`

## 3. Run the setup to install location-independent custom code

Via this command (executed from the root folder):

`python setup.py install`

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
