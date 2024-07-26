# imports modules for preparing data and for training/evaluating the ex2vec model
import argparse # lib for parsing command-line args
import data_sampler
from ex2vec import Ex2VecEngine
import os
import shutil
import torch
import sys
sys.path.append('/content/drive/MyDrive/JKU/practical_work/Practical-Work-AI/GRU4Rec_PyTorch_Fork')
import gru4rec_pytorch
import json
import optuna

class MyHelpFormatter(argparse.HelpFormatter):
    def __init__(self, *args, **kwargs):
        super(MyHelpFormatter, self).__init__(*args, **kwargs)
        self._width = shutil.get_terminal_size().columns

parser = argparse.ArgumentParser(formatter_class=MyHelpFormatter, description='Train an Ex2Vec model.')
parser.add_argument('-ep', '--embds_path', type=str, default=None, help='Path to the GRU4Rec trained model')
parser.add_argument('-o', '--optim', type=str, default=None, help='Type Y for activating hyperparameter optimization after the training loop')
args = parser.parse_args() # store command line args into args variable

n_user, n_item = data_sampler.get_n_users_items() # get number of unique users and number of unique items

# hyperparams - batch size, learning rate, latent/embedding dim
BS = 512  # , 1024, 2048]
LR = 5e-5  # [5e-5, 1e-4, 5e-3, 0.0002, 0.00075, 0.001]
L_DIM = 64

# construct unique training configuration
alias = "ex2vec_baseline_" + "BS" + str(BS) + "LR" + str(LR) + "L_DIM" + str(L_DIM)

# config for training ex2vec model
config = {
    "alias": alias,
    "num_epoch": 5,
    "batch_size": BS,
    "optimizer": "adam",
    "adam_lr": LR,
    "n_users": n_user,
    "n_items": n_item,
    "latent_dim": L_DIM,
    "num_negative": 0,
    "l2_regularization": 0.001,
    "use_cuda": True,
    "device_id": 0,
    "pretrain": False,
    "pretrain_dir": "/content/drive/MyDrive/JKU/practical_work/Practical-Work-AI/models/Ex2Vec_pretrained.pt",
    "model_dir": "/content/drive/MyDrive/JKU/practical_work/Practical-Work-AI/models/{}_Epoch{}_f1{:.4f}.pt",
    "chckpt_dir":"/content/drive/MyDrive/JKU/practical_work/Practical-Work-AI/chckpts/{}_Epoch{}_f1{:.4f}.pt",
}

def train_and_eval(configuration, batch_size, args, tuning = False):
    # initialize ex2vec engine with above configuration
    engine = Ex2VecEngine(configuration)

    # obtain data loader for training set
    train_loader = data_sampler.instance_a_train_loader(batch_size)

    # change setting to using testing vs validation set for model evaluation, 0 = val, 1 = test
    use_test = 0

    eval_data = data_sampler.evaluate_data(use_test)

    # indicate start of training + current configuration
    print("Started training of model: ", configuration["alias"])

    best_f1 = -torch.inf
    for epoch in range(configuration["num_epoch"]): # loop over epochs in config
        print("Epoch {} starts !".format(epoch))
        engine.train_an_epoch(train_loader, epoch_id=epoch, embds_path=args.embds_path)
        acc, recall, f1, bacc = engine.evaluate(eval_data, epoch_id=epoch, embds_path=args.embds_path)
        if tuning == False:
            engine.save(configuration["alias"], epoch, f1) # save model chkpt
        
        if f1 > best_f1:
            best_f1 = f1
    return best_f1

def objective(trial):
    # hyperparameters to tune
    BS_optim = trial.suggest_categorical('BS', [512, 1024, 2048])
    LR_optim = trial.suggest_loguniform('LR', 1e-5, 1e-3)
    #L_DIM_optim = trial.suggest_categorical('L_DIM', [32, 64, 128])
    num_epoch_optim = trial.suggest_int('num_epoch', 2,3) # 50, 100, 200
    l2_regularization_optim = trial.suggest_loguniform('l2_regularization', 1e-5, 1e-2)

    """# optimizer specific tuning
    optimizer_type_optim = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])
    sgd_momentum = None
    rmsprop_alpha = None
    rmsprop_momentum = None

    if optimizer_type_optim == "sgd":
        sgd_momentum = trial.suggest_uniform('sgd_momentum', 0.0, 0.9)
    elif optimizer_type_optim == "rmsprop":
        rmsprop_alpha = trial.suggest_uniform('rmsprop_alpha', 0.0, 0.99)
        rmsprop_momentum = trial.suggest_uniform('rmsprop_momentum', 0.0, 0.9) """

    # construct unique tuning configuration
    alias_optim = "ex2vec_optim_" + "BS" + str(BS_optim) + "LR" + str(LR_optim) + "L_DIM" + str(L_DIM)

    # config for training ex2vec model
    config_optim = {
    "alias": alias_optim,
    "num_epoch": num_epoch_optim,
    "batch_size": BS_optim,
    "optimizer": "adam",
    "adam_lr": LR_optim,
    "n_users": n_user,
    "n_items": n_item, 
    "latent_dim": L_DIM,
    "num_negative": 0,
    "l2_regularization": l2_regularization_optim,
    "use_cuda": True,
    "device_id": 0,
    "pretrain": False,
    "pretrain_dir": "/content/drive/MyDrive/JKU/practical_work/Practical-Work-AI/models/Ex2Vec_pretrained.pt",
    "model_dir": "/content/drive/MyDrive/JKU/practical_work/Practical-Work-AI/models/{}_Epoch{}_f1{:.4f}.pt",
    "chckpt_dir":"/content/drive/MyDrive/JKU/practical_work/Practical-Work-AI/chckpts/{}_Epoch{}_f1{:.4f}.pt",
    }

    return train_and_eval(config_optim, BS_optim, args, tuning=True)


if args.optim == 'Y':
    print('Starting hyperparameter optimization with Optuna...')
    
    # hyperparameter tuning
    N_TRIALS = 3 #change to 50

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials = N_TRIALS)

    with open('/content/drive/MyDrive/JKU/practical_work/Practical-Work-AI/best_params_Ex2Vec.json', 'w') as f:
        json.dump(study.best_params, f, indent=4)

else:
    # initial training of Ex2Vec
    train_and_eval(config, BS, args)