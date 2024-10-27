# imports modules for preparing data and for training/evaluating the ex2vec model
import argparse # lib for parsing command-line args
import data_sampler
from ex2vec import Ex2VecEngine
import os
import shutil
import torch
from GRU4Rec_Fork import gru4rec_pytorch
import json
import optuna
from collections import OrderedDict
from GRU4Rec_Fork.gru4rec_utils import convert_to_param_str

class MyHelpFormatter(argparse.HelpFormatter):
    def __init__(self, *args, **kwargs):
        super(MyHelpFormatter, self).__init__(*args, **kwargs)
        self._width = shutil.get_terminal_size().columns

parser = argparse.ArgumentParser(formatter_class=MyHelpFormatter, description='Train an Ex2Vec model.')
parser.add_argument('-ep', '--embds_path', type=str, default=None, help='Path to the GRU4Rec trained model')
parser.add_argument('-ps', '--param_str', type=str, default=None, help='Parameters to optimize, or to train with.')
parser.add_argument('-pf', '--param_file', type=str, default=None, help='File where the parameters are stored.')
parser.add_argument('-t', '--tuning', type=str, default="N", help='Set whether this is a run with or without hyperparameter tuning.')
parser.add_argument('-n', '--name', type=str, default='ex2vec', help='Set the alias for the model.')
parser.add_argument('-ud', '--use_dataset', type=int, default=0, help='Which set to use for validation for the current run. Modes: 0 = Validation, 1 = Test, 2 = Custom (Default = 0).')
parser.add_argument('-pth', '--base_path', type=str, default='./', help='The base directory where everything related to the PR (runs, chckpts, final models, results etc.) will be stored.')
parser.add_argument('--use_cuda', action='store_true', help='Sets the flag for training Ex2Vec on the GPU')
parser.add_argument('--pretrain', action='store_true', help='Sets the flag for using a pretrained Ex2vec stored under pretrain_dir.')
parser.add_argument('-pp', '--pretrain_path', type=str, default='./models/Ex2Vec_pretrained.pt', help='The filename of the the pretrained Ex2Vec model.')


args = parser.parse_args() # store command line args into args variable

ex2vec_params = None
if args.param_str: # if parameter string is provided, parse it and create an ordered dict of params
    ex2vec_params = OrderedDict([x.split('=') for x in args.param_str.split(',') if "=" in x]) # splits e.g. "loss=bpr" to {"loss":"bpr"}
elif args.param_file:
    param_str = convert_to_param_str(args.param_file)
    ex2vec_params = OrderedDict([x.split('=') for x in param_str.split(',') if "=" in x])

n_user, n_item = data_sampler.get_n_users_items() # get number of unique users and number of unique items

# hyperparams - batch size, learning rate, latent/embedding dim
BS = int(ex2vec_params['batch_size']) if ex2vec_params else 512  # , 1024, 2048]
LR = float(ex2vec_params['learning_rate']) if ex2vec_params else 5e-5  # [5e-5, 1e-4, 5e-3, 0.0002, 0.00075, 0.001]
L_DIM = 64
NUM_EPOCH = int(ex2vec_params['num_epoch']) if ex2vec_params else 100
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
    "use_cuda": args.use_cuda,
    "device_id": 0,
    "pretrain": args.pretrain,
    "pretrain_dir": args.pretrain_path,
    "model_dir": args.base_path + 'models/{}_Epoch{}_f1{:.4f}.pt',
    "chckpt_dir": args.base_path + 'chckpts/{}_Epoch{}_f1{:.4f}.pt',
    "results_dir": args.base_path + 'results/best_models.csv'
}

print("Ex2Vec model is created with the following parameters for this run:\n")
for k,v in config.items():
  print(f'{k}:{v}')

# initialize ex2vec engine with above configuration
engine = Ex2VecEngine(config, args.base_path)

# prepare data
train_loader = data_sampler.instance_a_train_loader(BS, args.use_dataset)
eval_data = data_sampler.evaluate_data(args.use_dataset)

# if -ep flag is set, load GRU4Rec weights and pass to ex2vec training
if args.embds_path:
    gru4rec_loaded = torch.load(args.embds_path, weights_only=False)
    item_embds = gru4rec_loaded.model.Wy.weight.data

# indicate start of training + current configuration
print("Started training model: ", config["alias"])

for epoch in range(config["num_epoch"]): # loop over epochs in config
    print("Epoch {} starts...".format(epoch))
    engine.train_an_epoch(train_loader, epoch_id=epoch, item_embds=item_embds) # train 1 epoch
    acc, recall, f1, bacc = engine.evaluate(eval_data, epoch_id=epoch, item_embds=args.embds_path) # calculate metrics

    if args.tuning == "N":
        engine.save(config["alias"], epoch, f1, args.param_str, f"acc={acc}, recall={recall}, f1={f1}, bacc={bacc}", args.embds_path) # save model chkpt


# logging all metrics + primary metric at the end of training run
all_metrics_str = f"FINAL METRICS: ACC: {acc}, RECALL: {recall}, F1: {f1}, BACC: {bacc}"
print(all_metrics_str)

res_str = f"PRIMARY METRIC: {f1}"
print(res_str)