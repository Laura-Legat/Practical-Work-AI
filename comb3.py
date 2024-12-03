import os
import torch
import sys
import json
import numpy as np
from GRU4Rec_Fork.gru4rec_utils import convert_to_param_str

gru4rec_best_params = convert_to_param_str('./optim/best_params_gru4rec.json')
combination_mode = 'weighted'
cutoffs = '1 5 10 20'
alpha = '0.2'

command = f'python ./GRU4Rec_Fork/run.py ./data/seq_combined.csv -ps {gru4rec_best_params} -t ./data/seq_test.csv -s ./models/GRU4Rec_bestparams_ex2vec.pt -m {cutoffs} -a {alpha} -ik "itemId" -tk "timestamp" -pm recall -lpm -c {combination_mode} -ex ./models/ex2vec_baseline_FULL_finaltrain__BS512LR0.00014455048679195258L_DIM64N_EP50_Epoch49_f10.6326.pt -pth ./'
get_ipython().system(command)
