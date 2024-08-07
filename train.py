# imports modules for preparing data and for training/evaluating the ex2vec model
import argparse # lib for parsing command-line args
import data_sampler
from ex2vec import Ex2VecEngine
import os
import shutil
import torch
import sys
sys.path.append('/content/drive/MyDrive/JKU/practical_work/Practical-Work-AI/GRU4Rec_Fork')
import gru4rec_pytorch
import json
import optuna
from collections import OrderedDict

class MyHelpFormatter(argparse.HelpFormatter):
    def __init__(self, *args, **kwargs):
        super(MyHelpFormatter, self).__init__(*args, **kwargs)
        self._width = shutil.get_terminal_size().columns

parser = argparse.ArgumentParser(formatter_class=MyHelpFormatter, description='Train an Ex2Vec model.')
parser.add_argument('-ep', '--embds_path', type=str, default='', help='Path to the GRU4Rec trained model')
parser.add_argument('-ps', '--param_str', type=str, default=None, help='Parameters to optimize')
parser.add_argument('-t', '--tuning', type=str, default="N", help='Set whether this is a run with or without hyperparameter tuning.')
parser.add_argument('-n', '--name', type=str, default='ex2vec', help='Set the alias for the model.')
parser.add_argument('-ut', '--use_test', type=str, default='N', help='Whether or not to use the test set for validation.')

args = parser.parse_args() # store command line args into args variable

ex2vec_params = None
if args.param_str: # if parameter string is provided, parse it and create an ordered dict of params
    ex2vec_params = OrderedDict([x.split('=') for x in args.param_str.split(',') if "=" in x]) # splits e.g. "loss=bpr" to {"loss":"bpr"}

n_user, n_item = data_sampler.get_n_users_items() # get number of unique users and number of unique items

# hyperparams - batch size, learning rate, latent/embedding dim
BS = int(ex2vec_params['batch_size']) if ex2vec_params else 512  # , 1024, 2048]
LR = float(ex2vec_params['learning_rate']) if ex2vec_params else 5e-5  # [5e-5, 1e-4, 5e-3, 0.0002, 0.00075, 0.001]
L_DIM = 64
NUM_EPOCH = int(ex2vec_params['num_epoch']) if ex2vec_params else 3
L2_REG = float(ex2vec_params['l2_regularization']) if ex2vec_params else 0.001
OPTIM = 'adam'
model_name = args.name
RMSPROP_ALPHA = float(ex2vec_params['rmsprop_alpha']) if ex2vec_params else 0.99
MOMENTUM = float(ex2vec_params['momentum']) if ex2vec_params else 0

# construct unique training configuration
alias = model_name + "__BS" + str(BS) + "LR" + str(LR) + "L_DIM" + str(L_DIM) + "N_EP" + str(NUM_EPOCH)

# config for training ex2vec model
config = {
    "alias": alias,
    "num_epoch": NUM_EPOCH,
    "batch_size": BS,
    "optimizer": OPTIM,
    "lr": LR, # can be used for adam, sgd, rmsprop
    "rmsprop_alpha": RMSPROP_ALPHA,
    "momentum": MOMENTUM, # can be used for sgd_momentum and rmsprop_momentum
    "n_users": n_user,
    "n_items": n_item,
    "latent_dim": L_DIM,
    "num_negative": 0,
    "l2_regularization": L2_REG,
    "use_cuda": True,
    "device_id": 0,
    "pretrain": False,
    "pretrain_dir": "/content/drive/MyDrive/JKU/practical_work/Practical-Work-AI/models/Ex2Vec_pretrained.pt",
    "model_dir": "/content/drive/MyDrive/JKU/practical_work/Practical-Work-AI/models/{}_Epoch{}_f1{:.4f}.pt",
    "chckpt_dir":"/content/drive/MyDrive/JKU/practical_work/Practical-Work-AI/chckpts/{}_Epoch{}_f1{:.4f}.pt",
}

print("Ex2Vec model is created with the following parameters for this run:\n")
for k,v in config.items():
  print(f'{k}:{v}')

# initialize ex2vec engine with above configuration
engine = Ex2VecEngine(config)

train_loader = data_sampler.instance_a_train_loader(BS, args.use_test)

# change setting to using testing vs validation set for evaluation, 0 = val, 1 = test
if args.use_test == "N":
    use_test = 0
else:
    use_test = 1

eval_data = data_sampler.evaluate_data(use_test)

# indicate start of training + current configuration
print("Started training model: ", config["alias"])

best_f1 = -torch.inf
for epoch in range(config["num_epoch"]): # loop over epochs in config
    print("Epoch {} starts...".format(epoch))
    engine.train_an_epoch(train_loader, epoch_id=epoch, embds_path=args.embds_path) # train 1 epoch
    acc, recall, f1, bacc = engine.evaluate(eval_data, epoch_id=epoch, embds_path=args.embds_path) # calculate metrics

    curr_metric = f1

    if args.tuning == "N":
        engine.save(config["alias"], epoch, curr_metric) # save model chkpt
    if curr_metric > best_f1:
         best_f1 = curr_metric

res_str = f"PRIMARY METRIC: {best_f1}"
print(res_str)