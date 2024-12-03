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

The `-sl` flag is the sequence length, `-st` the stride. Feel free to adapt these to your liking. The output files will be stores in the `/data/` subdirectory.

## Hyperparameter optimization

Perform hyperparameter optimization for any number of trials for either of the models with the `optuna_paropt.py` script. Please refer to the argparse documentation within the script for further information on the flags. The best parameters as well as trial information will be stored in the `/optim/` subdirectory. For this step to work, please create `ex2vec_search_space.csv` and `gru4rec_search_space.csv` files beforehand. Search space, optuna visualization as well as best trained model visualization information will eb stored in the `/results/` subdirectory. To modify the search spaces, modify the files in the `/paramspaces/` subdirectory located in the `GRU4Rec_Fork` submodule folder.

## Training

Train/finaltrain Ex2Vec by running the `train.py` script. Similarly, train GRU4Rec by running the `run.py` script located in the `GRU4Rec_Fork` submodule folder. Please refer to the argparse documentation within the script for further information on the flags. Resulting models will either be stored as checkpoints in their respective `chckpts` subdirectories (any checkpoint models during training), or in the `models` subdirectory (final model). For Ex2Vec, tensorboard information will be stored in the `/runs/` subdirectory.

## Combination ideas

Combination idea 1 and 3 can be run via their respetive scripts located in the root folder. Combination idea 2 can be run by adding the `-ep` flag when executing `train.py` to specify a GRU4Rec model path. Please refer to the argparse documentation within the script for further information on the flags.

## Dataset
The dataset (new_release_stream.csv) is provided [here](https://zenodo.org/record/8316236).

## References

```
@inproceedings{sguerra2023ex2vec,
  title={Ex2Vec: Characterizing Users and Items from the Mere Exposure Effect},
  author={Sguerra, Bruno and Tran, Viet-Anh and Hennequin, Romain},
  booktitle = {Proceedings of the 17th ACM Conference on Recommender Systems},
  year = {2023}
}

Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, Domonkos Tikk: Session-based Recommendations with Recurrent Neural Networks. arXiv preprint arXiv:1511.06939, 2015. https://arxiv.org/abs/1511.06939 Presented at the 4th International Conference on Learning Representations, ICLR 2016.

Balázs Hidasi, Alexandros Karatzoglou: Recurrent Neural Networks with Top-k Gains for Session-based Recommendations. arXiv preprint arXiv:1706.03847, 2017. https://arxiv.org/abs/1706.03847
```
