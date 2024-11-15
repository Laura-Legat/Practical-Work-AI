import os
import torch
import sys
import shutil
import pandas as pd
import json
import numpy as np
from ast import literal_eval
from collections import OrderedDict
import argparse
from GRU4Rec_Fork.gru4rec_utils import convert_to_param_str
from ex2vec import Ex2VecEngine

class MyHelpFormatter(argparse.HelpFormatter):
    def __init__(self, *args, **kwargs):
        super(MyHelpFormatter, self).__init__(*args, **kwargs)
        self._width = shutil.get_terminal_size().columns

parser = argparse.ArgumentParser(formatter_class=MyHelpFormatter, description='Train an Ex2Vec model.')
parser.add_argument('-bp', '--best_params_pth', type=str, default='./optim/best_params_ex2vec.json', help='Path to best ex2vec parameters.')
parser.add_argument('-ex', '--pretrain_dir_ex2vec', type=str, default=None, help='Path to pretrain dir for ex2vec')
parser.add_argument('-gru', '--pretrain_dir_gru4rec', type=str, default=None, help='Path to pretrain dir for GRU4Rec')
parser.add_argument('-sp', '--scores_path', type=str, default=None, help='Path for storing and retrieving the .h5 file containing the Ex2Vec and GRU4Rec scores')
parser.add_argument('-pth', '--base_path', type=str, default=None, help='Project base path.')
parser.add_argument('-mp', '--metrics_path', type=str, default=None, help='Path where to store the metrics csv.')
parser.add_argument('--store_only', action='store_true', default=False, help='Run the function for storing the parameters, instead of the visualization.')
parser.add_argument('-a', '--alpha', type=float, nargs='+', default=[0.1], help='Alpha value used for combination.')
parser.add_argument('-co', '--combination_modes', type=str, nargs='+', default=['direct', 'weighted', 'boosted', 'mult'], help='The combination mode for combining Ex2Vec and GRU4Rec scores (options: weighted, boosted, direct, mult).')
parser.add_argument('-e', '--eval_mode', type=str, default='conservative', help='The evaluation mode for calculating MRR and Recall (options: standard, conservative, median).')
parser.add_argument('-m', '--measure', metavar='AT', type=int, nargs='+', default=[20], help='Measure recall & MRR at the defined recommendation list length(s). Multiple values can be provided. (Default: 20)')
parser.add_argument('-k', '--topk', type=int, default=100, help='The k for the topk evaluation')
parser.add_argument('-bs', '--batch_size', type=int, default=4096, help='Batch size used for evaluation.')

args = parser.parse_args() # store command line args into args variable

# load ex2vec best parameters and use them as config
ex2vec_best_param_str = convert_to_param_str(args.best_params_pth)
ex2vec_config = OrderedDict([x.split('=') for x in ex2vec_best_param_str.split(',') if "=" in x])

config = config = {
    "alias": 'ex2vec_baseline_finaltrain_comb2_FULL',
    "num_epoch": int(ex2vec_config['num_epoch']),
    "batch_size": int(ex2vec_config['batch_size']),
    "optimizer": 'adam',
    "lr": float(ex2vec_config['learning_rate']),
    "rmsprop_alpha": float(ex2vec_config['rmsprop_alpha']),
    "momentum": float(ex2vec_config['momentum']),
    "n_users": 100, # 3623
    "n_items": 728, # 879
    "latent_dim": 64,
    "num_negative": 0,
    "l2_regularization": float(ex2vec_config['l2_regularization']),
    "use_cuda": True,
    "device_id": 0,
    "pretrain": True,
    "pretrain_dir": args.pretrain_dir_ex2vec,
    "model_dir": "./models/{}_Epoch{}_f1{:.4f}.pt",
    "chckpt_dir":"./chckpts/{}_Epoch{}_f1{:.4f}.pt",
}

ex2vec_pre = Ex2VecEngine(config, args.base_path)
ex2vec_pre.model.eval()

gru4rec_path = args.pretrain_dir_gru4rec
gru = torch.load(gru4rec_path)
gru.model.eval()

test_data_path = args.base_path + 'data/seq_test.csv'
test_data = pd.read_csv(test_data_path, sep=',', converters={"relational_interval": literal_eval})
n_unique_users = test_data['userId'].nunique()

# prepare variables for results tables and experiments
ex2vec_model_name = os.path.basename(config['pretrain_dir']).split('.pt')[0]
gru4rec_model_name = os.path.basename(gru4rec_path).split('.pt')[0]

# comparison parameters
batch_size = args.batch_size
k = args.topk
alpha_list = args.alpha
scores_path = args.scores_path
measure = args.measure

if args.store_only:
    from GRU4Rec_Fork.evaluation import store_only

    # eval on both models and store the scores for all items in test set
    store_only(gru=gru, test_data=test_data, batch_size=batch_size, k=k, ex2vec=ex2vec_pre.model, score_store_pth=scores_path)
else:
    from GRU4Rec_Fork.evaluation import calc_metrics_from_scores

    for alpha in alpha_list:
        for combination_mode in args.combination_modes:
            eval_res = calc_metrics_from_scores(eval_ds_path=scores_path, alpha=[alpha], combination_mode=combination_mode, eval_mode=args.eval_mode, cutoffs=measure)

            recalls = ''
            recalls_comb = ''
            mrrs = ''
            mrrs_comb = ''
            print(['Recall@{} for k={}: {:.6f} MRR@{}: {:.6f}'.format(c, k, eval_res[0][c], c, eval_res[1][c]) for c in measure])
            print(['Combined Recall@{} for alpha={}, combination_mode={}, k={}: {:.6f} Combined MRR@{}: {:.6f}'.format(c, alpha, combination_mode, k, eval_res[2][c], c, eval_res[3][c]) for c in measure])

            recalls += ','.join(['Recall@{}={:.6f}'.format(c, eval_res[0][c]) for c in measure])
            recalls_comb += ','.join(['Recall@{}={:.6f}'.format(c, eval_res[2][c]) for c in measure])
            mrrs += ','.join(['MRR@{}={:.6f}'.format(c, eval_res[1][c]) for c in measure])
            mrrs_comb += ','.join(['MRR@{}={:.6f}'.format(c, eval_res[3][c]) for c in measure])

            # log current run of combination idea 2
            combination_row = {
                'gru4rec_model': gru4rec_model_name,
                'ex2vec_model': ex2vec_model_name,
                'seq_len': 50,
                'batch_size': batch_size,
                'combination_mode':combination_mode,
                'alpha': alpha,
                'k':k,
                'n_user_histories': str(n_unique_users),
                'recalls_comb': recalls_comb,
                'mrrs_comb': mrrs_comb,
                'baseline_recalls': recalls,
                'baseline_mrrs': mrrs
            }

            combination_df = pd.DataFrame([combination_row])

            if not os.path.isfile(args.metrics_path):
                combination_df.to_csv(args.metrics_path, mode='w', index=False, header=True)
            else:
                combination_df.to_csv(args.metrics_path, mode='a', index=False, header=False)