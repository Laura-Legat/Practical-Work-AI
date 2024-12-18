# IMPLEMENTS THE TWO FORMULAS IN THE EX2VEC PAPER

import torch

from engine import Engine
from utils import resume_checkpoint, use_cuda
from data_sampler import get_userId_from_mapping, get_itemId_from_mapping
import sys
#sys.path.append('/content/drive/MyDrive/JKU/practical_work/Practical-Work-AI/GRU4Rec_Fork/')
sys.path.append('./GRU4Rec_Fork/')

# Ex2Vec Class
class Ex2Vec(torch.nn.Module): # Ex2Vec neural network model 
    def __init__(self, config):
        super(Ex2Vec, self).__init__()
        # set up ex2vec model with config gained from train.py
        self.config = config
        self.n_users = config["n_users"]
        self.n_items = config["n_items"]
        self.latent_d = config["latent_dim"]

        # lambda parameter to move the embedding
        self.global_lamb = torch.nn.Parameter(torch.tensor(1.0)) # global lambda

        self.user_lamb = torch.nn.Embedding(self.n_users, 1)  #user lambda
        # self.item_lamb = torch.nn.Embedding(self.n_items, 1)

        # SECOND FORMULA PARAMS
        self.user_bias = torch.nn.Embedding(self.n_users, 1)
        self.item_bias = torch.nn.Embedding(self.n_items, 1)

        # quadratic function parameters/coefficients
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))
        self.beta = torch.nn.Parameter(torch.tensor(-0.065))
        self.gamma = torch.nn.Parameter(torch.tensor(0.5))

        # the cutoff value (c in paper)
        self.cutoff = torch.nn.Parameter(torch.tensor(3.0))

        # user and item embedding vectors (u and v in paper)
        # initialize embedding layer structures
        # nn.Embedding = PyTorch modules designed to map integer indices
        # (user and item id's in our case) to dence vectors in a high-dimensional space 
        # n_users = number of unique users
        # n_items = number of unique items
        # latent_d = 64 according to config = dimensionality of latent space
        # this tells the function that it should expect n_users-indices and allocate memory for eaxch embedding
        # THESE LINES a) SET UP THE MODEL ARCHITECTURE AND b) ALREADY CALCULATE THE EMBEDDING MATRIX, AKA AN EMBEDDING VECTOR FOR EACH USERID
        # embedding matrix is stored within torch.nn.Embedding layer as learnable parameter
        # it does not need to know the actual IDs, this is abstracted away into giving each item a unique index which is mapped to a weight matrix/embedding matrix
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_d
        )
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_d
        )

        self.logistic = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh() # define alternative activation fct for comb3

    def forward(self, user_indices, item_indices, r_interval, item_embds=None, use_tanh=False):
        # calculate u and v for first formula
        # retrieve embedding vectors for each idx in user_indices -> fetch corresponding rows from embedding matrix, e.g. for user_idx=3, it fetches the 3rd user embedding
        user_embeddings = self.embedding_user(user_indices)

        if item_embds != None: # if there is a GRU4Rec model given, use the GRU4Rec item embeddings
            item_embeddings = item_embds[item_indices]
        else:
            item_embeddings = self.embedding_item(item_indices)

        # calculate b for second formula
        u_bias = self.user_bias(user_indices).squeeze(-1)
        i_bias = self.item_bias(item_indices).squeeze(-1)

        # d(u,v) -> Euclidian Distance formula
        difference = torch.sub(item_embeddings, user_embeddings)
        base_distance = torch.sqrt((difference**2)).sum(axis=1) 

        # compute the base_level activation
        # get only time gaps superior to zero
        # r_interval = t - t_j
        mask = (r_interval > 0).float() #  indicator function = 1 when t - t_j > 0 = t > t_j, see paper p 973
        delta_t = r_interval * mask # t - t_j * indicator_func

        delta_t = delta_t + self.cutoff.clamp(min=0.1, max=100) # t - t_j * indicator_func + c
        decay = 0.5  # self.decay.clamp(min = 0.01, max = 10)
        pow_values = torch.pow(delta_t, -decay) * mask # indicator_func * (t-t_j + c)^-d -> why 2x mask?
        base_level = torch.sum(pow_values, -1) # whole sum from formula
        
        # compute how much to move the user embedding
        # calculate lambda and clamp it to specific space
        lamb = self.global_lamb.clamp(0.01, 10) + self.user_lamb(user_indices).squeeze(-1).clamp(0.1, 10)

        base_activation = torch.mul(base_level, lamb)

        # select the smaller part, if d(u,v) is smaller, then we have
        # d(u,v) - d(u,v) = 0, and it activation is smaller, we have a number
        # > 0, which is similar to say: we pick either 0 or the larger than
        # 0 number
        activation = torch.minimum(base_activation, base_distance) 
        # move the user embedding in the direction of the item given a factor lambda
        # d(u,v) - activation
        distance = base_distance - activation  # self.lamb*distance*base_level -> why not distance = activation?
        
        # formula 2 from paper: alpha * d_u,i + beta * d_u,i^2 + gamma + bias
        I = self.alpha * distance  + self.beta * torch.pow(distance, 2) + self.gamma + u_bias + i_bias

        if use_tanh:
            interest = self.tanh(I)
        else:
            interest = self.logistic(I)
        return interest, distance

    def load_pretrain_weights(self):
        """Loading weights from trained GMF model"""
        config = self.config # get current config

        ex2vec_pre = Ex2Vec(config) # set up model with that config
        if config["use_cuda"] is True:
            ex2vec_pre.cuda() # move model to GPU
            ex2vec_pre.load_state_dict(torch.load(config['pretrain_dir']))

        # copy state dict of pretrained ex2vec model to currently initiated ex2vec model
        self.embedding_user.weight.data = ex2vec_pre.embedding_user.weight.data
        self.embedding_item.weight.data = ex2vec_pre.embedding_item.weight.data
        self.item_bias.weight.data = ex2vec_pre.item_bias.weight.data
        self.user_bias.weight.data = ex2vec_pre.user_bias.weight.data
        self.user_lamb.weight.data = ex2vec_pre.user_lamb.weight.data
        self.cutoff.data = ex2vec_pre.cutoff.data
        self.gamma.data = ex2vec_pre.gamma.data
        self.beta.data = ex2vec_pre.beta.data
        self.alpha.data = ex2vec_pre.alpha.data
        self.global_lamb.data = ex2vec_pre.global_lamb.data

class Ex2VecEngine(Engine):
    """Engine for training & evaluating MEE model"""

    def __init__(self, config, ex2vec_path):
        self.model = Ex2Vec(config) # create new ex2vec model
        if config["use_cuda"] is True: # move it to GPU
            use_cuda(True, config["device_id"])
            self.model.cuda()
        super(Ex2VecEngine, self).__init__(config, ex2vec_path) # initialize model

        if config["pretrain"]:
            self.model.load_pretrain_weights() # switch to pre-train mode
