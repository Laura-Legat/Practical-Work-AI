import os
import torch
import sys
import pandas as pd
import json
import numpy as np
from ast import literal_eval
from collections import OrderedDict
from GRU4Rec_Fork.gru4rec_utils import convert_to_param_str
from ex2vec import Ex2VecEngine

# load ex2vec best parameters and use them as config
ex2vec_best_param_str = convert_to_param_str('./optim/best_params_ex2vec.json')
ex2vec_config = OrderedDict([x.split('=') for x in ex2vec_best_param_str.split(',') if "=" in x])

config = config = {
    "alias": 'ex2vec_baseline_finaltrain_comb2_FULL',
    "num_epoch": int(ex2vec_config['num_epoch']),
    "batch_size": int(ex2vec_config['batch_size']),
    "optimizer": 'adam',
    "lr": float(ex2vec_config['learning_rate']),
    "rmsprop_alpha": float(ex2vec_config['rmsprop_alpha']),
    "momentum": float(ex2vec_config['momentum']),
    "n_users": 3623,
    "n_items": 879,
    "latent_dim": 64,
    "num_negative": 0,
    "l2_regularization": float(ex2vec_config['l2_regularization']),
    "use_cuda": True,
    "device_id": 0,
    "pretrain": True,
    "pretrain_dir": "./models/ex2vec_baseline_FULL_finaltrain__BS512LR0.00014455048679195258L_DIM64N_EP50_Epoch49_f10.6326.pt",
    "model_dir": "./models/{}_Epoch{}_f1{:.4f}.pt",
    "chckpt_dir":"./chckpts/{}_Epoch{}_f1{:.4f}.pt",
}

ex2vec_pre = Ex2VecEngine(config, './')
ex2vec_pre.model.eval()

gru4rec_path = './models/GRU4Rec_FULL_finaltrain_bestparams_trial19.pt'
gru = torch.load(gru4rec_path)
gru.model.eval()

test_data_path = './data/seq_test.csv'
test_data = pd.read_csv(test_data_path, sep=',', converters={"relational_interval": literal_eval})
n_unique_users = test_data['userId'].nunique()

# prepare variables for results tables and experiments
ex2vec_model_name = os.path.basename(config['pretrain_dir']).split('.pt')[0]
gru4rec_model_name = os.path.basename(gru4rec_path).split('.pt')[0]

# comparison parameters
batch_size = 4096
topk_list = [20, 50, 100] # set different k for top-k experiments
alpha_list = [0.01, 0.03, 0.2] # set different alpha's for experiment (alpha = tradeoff between the models)
cutoff = [5, 10, 20] # cutoffs for mrr and recall

from GRU4Rec_Fork.evaluation import store_only

# eval on both models and store the scores for all items in test set
store_only(gru=gru, test_data=test_data, batch_size=batch_size, ex2vec=ex2vec_pre.model, score_store_pth='./results')
