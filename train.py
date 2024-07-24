# imports modules for preparing data and for training/evaluating the ex2vec model
import argparse # lib for parsing command-line args
import data_sampler
from ex2vec import Ex2VecEngine
import os
import shutil
import torch

class MyHelpFormatter(argparse.HelpFormatter):
    def __init__(self, *args, **kwargs):
        super(MyHelpFormatter, self).__init__(*args, **kwargs)
        self._width = shutil.get_terminal_size().columns

parser = argparse.ArgumentParser(formatter_class=MyHelpFormatter, description='Train an Ex2Vec model.')
parser.add_argument('-ep', '--embds_path', type=str, default=None, help='Path to the GRU4Rec trained model')
args = parser.parse_args() # store command line args into args variable

n_user, n_item = data_sampler.get_n_users_items() # get number of unique users and number of unique items

# hyperparams - batch size, learning rate, latent/embedding dim
BS = 512  # , 1024, 2048]
LR = 5e-5  # [5e-5, 1e-4, 5e-3, 0.0002, 0.00075, 0.001]
L_DIM = 64

# construct unique training configuration
alias = "ex2vec_" + "BS" + str(BS) + "LR" + str(LR) + "L_DIM" + str(L_DIM)

# config for training ex2vec model
config = {
    "alias": alias,
    "num_epoch": 20, #100
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
    "pretrain_dir": "/content/drive/MyDrive/JKU/practical_work/Practical-Work-AI/checkpoints/{}".format("pretrain_Ex2vec.pt"),
    "model_dir": "/content/drive/MyDrive/JKU/practical_work/Practical-Work-AI/checkpoints/{}_Epoch{}_f1{:.4f}.pt",
}

# initialize ex2vec engine with above configuration
engine = Ex2VecEngine(config)

train_loader = data_sampler.instance_a_train_loader(BS)

# change setting to using testing vs validation set for evaluation, 0 = val, 1 = test
use_test = 0

eval_data = data_sampler.evaluate_data(use_test)

# indicate start of training + current configuration
print("started training model: ", config["alias"])


for epoch in range(config["num_epoch"]): # loop over epochs in config
    print("Epoch {} starts !".format(epoch))
    engine.train_an_epoch(train_loader, epoch_id=epoch, embds_path=args.embds_path) # train 1 epoch
    acc, recall, f1, bacc = engine.evaluate(eval_data, epoch_id=epoch, embds_path=args.embds_path) # calculate metrics
    engine.save(config["alias"], epoch, f1) # save model chkpt
