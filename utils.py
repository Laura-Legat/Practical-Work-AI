# Provides utility functions for saving/loading checkpoints, managing CUDA usage, and initializing optimizers for neural network training

# import pytorch library - a tensor library for deep learning using GPUs and CPUs
import torch
import json


# Checkpoints - save the model state dict to the specified directory "model_dir"
def save_checkpoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)

# loads state dict found in "model_dir" directory into a given model
# map_location remaps tensors to appropriate device (useful for loading a model checkpoint saved on a GPU to a CPU or vice versa)
# 
def resume_checkpoint(model, model_dir, device_id):
    state_dict = torch.load(
        model_dir, map_location=lambda storage, loc: storage.cuda(device=device_id) # maps storage (data tensor being loaded) to cuda device with given device_id (aka always on GPU)
    ) 
    model.load_state_dict(state_dict) # load state dict into provided model

# Hyper params
def use_cuda(enabled, device_id=0):
    if enabled: # if enabled, check CUDA support -> if that is available, set device to specified cuda device
        assert torch.cuda.is_available(), "CUDA is not available"
        torch.cuda.set_device(device_id)

# takes in network (neural network model) and params (dict containing optimizer parameters)
# based on value of "optimizer" key, it initializes and returns an optimizer object
def use_optimizer(network, params):
    if params["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            network.parameters(),
            lr=params["lr"],
            momentum=params["momentum"],
            weight_decay=params["l2_regularization"],
        )
    elif params["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=params["lr"],
            weight_decay=params["l2_regularization"],
        )
    elif params["optimizer"] == "rmsprop":
        optimizer = torch.optim.RMSprop(
            network.parameters(),
            lr=params["lr"],
            alpha=params["rmsprop_alpha"],
            momentum=params["momentum"],
        )
    else:
        raise ValueError(
            f'{params.get("optimizer", None)} is not allowed as optimizer value'
        )
    return optimizer


# convert best parameters to parameter string
def convert_to_param_str(best_param_path):
  """
  Helper function that converts parameters from a JSON file into a string used for training models.

  Args:
    best_param_path: The path of the JSON file containing parameters for model training.

  Returns:
    String of parameters of the form "param1=value1,param2=value2,..."
  """
  with open(best_param_path, 'r') as f:
    data = json.load(f)

  params = data['best_params'] # extract only the parameter part, not the optuna n_trials
  return ','.join([f'{key}={value}' for key,value in params.items()])
