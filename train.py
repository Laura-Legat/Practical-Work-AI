# TRAIN THE MODEL

# imports modules for preparing data and for training/evaluating the ex2vec model
import data_sampler
from ex2vec import Ex2VecEngine

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
    "num_epoch": 100,
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
    "pretrain_dir": "checkpoints/{}".format("pretrain_Ex2vec.model"),
    "model_dir": "checkpoints/{}_Epoch{}_f1{:.4f}.model",
}

# initialize ex2vec engine with above configuration
engine = Ex2VecEngine(config)

train_loader = data_sampler.instance_a_train_loader(BS)
eval_data = data_sampler.evaluate_data()

# indicate start of training + current configuration
print("started training model: ", config["alias"])



for epoch in range(config["num_epoch"]): # loop over epochs in config
    print("Epoch {} starts !".format(epoch))
    engine.train_an_epoch(train_loader, epoch_id=epoch) # train 1 epoch
    acc, recall, f1 = engine.evaluate(eval_data, epoch_id=epoch) # calculate metrics
    engine.save(config["alias"], epoch, f1) # save model chkpt
