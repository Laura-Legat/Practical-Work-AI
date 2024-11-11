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

"""
# inference run without combination
eval_res = batch_eval(gru=gru, test_data=test_data, cutoff=cutoff, ex2vec=ex2vec_pre.model)

baseline_recalls = ''
baseline_mrrs = ''
for i, c in enumerate(cutoff):
  print('\nRecall@{}: {:.6f} MRR@{}: {:.6f}'.format(c, eval_res[0][c], c, eval_res[1][c]))
  baseline_recalls += 'Recall@{}={:.6f}'.format(c, eval_res[0][c])
  baseline_mrrs += 'MRR@{}={:.6f}'.format(c, eval_res[1][c])

  if i != (len(cutoff) -1):
    baseline_recalls += ','
    baseline_mrrs += ','

#store_only(gru=gru, test_data=test_data, ex2vec=ex2vec_pre.model, score_store_pth='./results')
combination_path = './results/score_combination.csv'
combination_modes = ['direct', 'weighted', 'boosted', 'mult']

for topk in topk_list:
    for combination_mode in combination_modes:
        eval_res = batch_eval(gru=gru, test_data=test_data, cutoff=cutoff, ex2vec=ex2vec_pre.model, combination=combination_mode, k=topk, alpha_list=alpha_list)

        for i, alpha in enumerate(alpha_list):
            recalls = ''
            mrrs = ''
            print(['Recall@{} for alpha={}, combination_mode={}, k={}: {:.6f} MRR@{}: {:.6f}'.format(c, alpha, combination_mode, topk, eval_res[0][c][i], c, eval_res[1][c][i]) for c in cutoff])

            recalls += ','.join(['Recall@{}={:.6f}'.format(c, eval_res[0][c][i]) for c in cutoff])
            mrrs += ','.join(['MRR@{}={:.6f}'.format(c, eval_res[1][c][i]) for c in cutoff])

            # log current run of combination idea 2
            combination_row = {
                'gru4rec_model': gru4rec_model_name,
                'ex2vec_model': ex2vec_model_name,
                'seq_len': 50,
                'batch_size': batch_size,
                'combination_mode':combination_mode,
                'alpha': alpha,
                'k':topk,
                'n_user_histories': str(n_unique_users),
                'recalls': recalls,
                'mrrs': mrrs,
                'baseline_recalls': baseline_recalls,
                'baseline_mrrs': baseline_mrrs
            }

            combination_df = pd.DataFrame([combination_row])

            if not os.path.isfile(combination_path):
                combination_df.to_csv(combination_path, mode='w', index=False, header=True)
            else:
                combination_df.to_csv(combination_path, mode='a', index=False, header=False)
"""
