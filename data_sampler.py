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


DATA_PATH = '/content/drive/MyDrive/JKU/practical_work/Practical-Work-AI/data/'
orig_data = DATA_PATH + 'processed.csv'

# read processed.csv into pandas dataframe while converting  relational_interval from string to python list
df = pd.read_csv(orig_data, converters={"relational_interval": literal_eval})

# determines unique users and items in dataset
user_pool = set(df["userId"].unique())
item_pool = set(df["itemId"].unique())

# split dataset into train, test and val based 
df_test = df[df["set"] == "test"].copy()
df_val = df[df["set"] == "val"].copy()
df_train = df[(df["set"] == "train")].copy()

print("The size of the training set is: {}".format(len(df_train)))
print("The size of the validation set is: {}".format(len(df_val)))
print("The size of the test set is: {}".format(len(df_test)))


# function that returns the train, val and test set
def get_train_test_val():
    return df_test, df_train, df_val


# function that returns the number of users and items
def get_n_users_items():
    return df.userId.nunique(), df.itemId.nunique()

# build the training set in batches
def instance_a_train_loader(batch_size):
    users, items, rel_int, interests = [], [], [], []
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
def evaluate_data(val_or_test=0):
    """
    Args:
        val_or_test: Boolean variable whether the validation set (val_or_test=0), or the test set (val_or_test=1) should be used for evaluation
    """
    test_users, test_items, test_rel_int, test_listen = [], [], [], []

    # change whether validation or test set is used for evaluation of the model
    if val_or_test == 0:
        print("Using validation set for evaluation\n")
        df_eval = df_val
    else:
        print("Using test set for evaluation\n")
        df_eval = df_test

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
