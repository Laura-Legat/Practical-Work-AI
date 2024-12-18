import random
from ast import literal_eval

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# set random seed for reproducibility
random.seed(123)

# implements a custom dataset class which inherits from pytorch's Dataset class
# wrap/convert user, item and relational interval tensors into pytorch datasets
class InteractionDataset(Dataset):
    """Wrapper, convert <user, item, rel_int, neg_item> Tensor into Pythorch Dataset"""

    def __init__(self, user_tensor, item_tensor, rel_int_tensor, target_tensor):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.rel_int_tensor = rel_int_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return (
            self.user_tensor[index],
            self.item_tensor[index],
            self.rel_int_tensor[index],
            self.target_tensor[index],
        )

    def __len__(self):
        return self.user_tensor.size(0)

DATA_PATH = './data/'
orig_data = DATA_PATH + 'processed.csv'

# read processed.csv into pandas dataframe while converting  relational_interval from string to python list bc some are saved as strings
df = pd.read_csv(orig_data, converters={"relational_interval": literal_eval})

# array of unique users and items 
unique_users = df["userId"].unique()
unique_items = df["itemId"].unique()

# total amount of unique users and items
n_users = len(unique_users)
n_items = len(unique_items)

# determines mapping for users and items to avoid embedding errors since ID can have higher numbers than there are unique ID's
#e.g. maps user/item ID's 45, 3, 29 -> 0, 1, 2
user_mapping = pd.Series(data=np.arange(n_users, dtype='int32'), index=unique_users, name='userIdx')
item_mapping = pd.Series(data=np.arange(n_items, dtype='int32'), index=unique_items, name='itemIdx')

# apply mapping to dataset
df['userId'] = df['userId'].map(user_mapping)
df['itemId'] = df['itemId'].map(item_mapping)

# split dataset into train, test and val based 
df_test = df[df["set"] == "test"].copy()
df_val = df[df["set"] == "val"].copy()
df_train = df[(df["set"] == "train")].copy()
df_combined = df[df["set"].isin(["train", "val"])].copy()

print("The size of the training set is: {}".format(len(df_train)))
print("The size of the validation set is: {}".format(len(df_val)))
print("The size of the test set is: {}".format(len(df_test)))
print("The size of the combined (train+val) set is: {}".format(len(df_combined)))

# create relational interval dict to later use for ex2vec and gru4rec combination 
df_combined.sort_values(by='timestamp')
rel_int_dict = {}
for row in df_combined.itertuples():
    # create new key for each unique user-item combination
    rel_int_dict[(row.userId, row.itemId)] = row.relational_interval

def get_mappings():
    """
    Helper function which returns the userId->userIdx and itemId->itemIdx mappings.
    """
    return user_mapping, item_mapping

def get_userId_from_mapping(idx_list):
  """
  Helpfer funtion which returns the corresponding userId to a userIdx.

  Args:
      idx_list: List of ints of userIdxs that one wants to convert
  """
  return user_mapping.index[idx_list]

def get_itemId_from_mapping(idx_list):
  """
  Helpfer funtion which returns the corresponding userId to a userIdx.

  Args:
      idx_list: List of ints of userIdxs that one wants to convert
  """
  return item_mapping.index[idx_list]

def get_rel_int_dict():
    """
    Helper function that gets the current dictionary containing the relational intervals for each user-item interaction.
    """
    return rel_int_dict

def update_rel_int_dict(userid, itemid, relational_interval):
    """
    Helper function that updates the relational interval entry for a specific user-item interaction.
    """
    key = (userid, itemid)
    rel_int_dict[key] = relational_interval

# function that returns the train, val and test set
def get_train_test_val_comb():
    return df_test, df_train, df_val, df_combined

# function that returns the number of users and items
def get_n_users_items():
    return df.userId.nunique(), df.itemId.nunique()

# build the training set in batches
def instance_a_train_loader(batch_size, dataset_mode=0):
    users, items, rel_int, interests = [], [], [], []

    if dataset_mode == 1: # aka test set is used for eval, thus combine val + train for training loop
        # combine val and train as train
        print("Using combined training set.")
        train_stream = (
          df_combined.copy()
        )
    else:
        # make copy of training set
        train_stream = (
            df_train.copy()
        )  # merge(df_negative[["userId", "negative_items"]], on="userId")
    for row in train_stream.itertuples(): # loop over each df row as itertuples, aka named tuples, for readability
        # values are extracted from csv and appended to respective lists
        users.append(int(row.userId))
        items.append(int(row.itemId))
        interests.append(int(row.y))

        # add -1 to the rel_int until arriving at the max(50 reps)
        ri = row.relational_interval
        # pad ri with -1 until it reaches length of 50
        ri = np.pad(ri, (0, 50 - len(ri)), constant_values=-1)
        # rel_int = [[10, 14, 11, -1, -1, -1,..., -1], [18, 2, 112, 1019, -1, -,1 ..., -1], ...]
        rel_int.append(ri)

    dataset = InteractionDataset( # create dataset
        user_tensor=torch.LongTensor(users),
        item_tensor=torch.LongTensor(items),
        rel_int_tensor=torch.FloatTensor(np.array(rel_int)),
        target_tensor=torch.LongTensor(interests),
    )
    # create and return dataloader which gives out batches of InteractionDataset rows, where data is shuffles before each epoch (one pass through the entire dataset)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# create the evaluation dataset (user x item consumption sequences)
# preoare evaluation dataset by extracting relevant information from df_val and formatting it into tensors for evaluation by the model
def evaluate_data(dataset_mode=0, custom_eval_data=None):
    """
    Args:
        dataset_mode: Which set to use for validation for the current run. Modes: 0 = Validation, 1 = Test, 2 = Custom -> int
        custom_eval_data: If dataset mode == 2, then custom dataset is used, which is relayed to the function via this parameter -> Pandas DataFrame
    """
    test_users, test_items, test_rel_int, test_listen = [], [], [], []

    # change whether validation or test set is used for evaluation of the model
    if dataset_mode == 0:
        print("Using validation set for evaluation\n")
        df_eval = df_val
    elif dataset_mode == 1:
        print("Using test set for evaluation\n")
        df_eval = df_test
    else:
        print('Using custom data set')
        df_eval = custom_eval_data

    for row in df_eval.itertuples():
        ri = row.relational_interval
        ri = np.pad(ri, (0, 50 - len(ri)), constant_values=-1)

        test_rel_int.append(ri)
        test_users.append(int(row.userId))
        test_items.append(int(row.itemId))
        test_listen.append(int(row.y))

    return [
        torch.LongTensor(test_users),
        torch.LongTensor(test_items),
        torch.FloatTensor(np.array(test_rel_int)),
        torch.FloatTensor(test_listen),
    ]
