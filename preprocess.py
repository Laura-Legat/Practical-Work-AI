# CREATES PROCESSED.CSV OUT OF NEW_RELEASE_STREAM.CSV

import random
import numpy as np
import pandas as pd


# function that computes delta_t, i.e., the time interval between consumptions (not considered when y = 0 )
def get_delta_t(row):
    ts = row["timestamp"]
    activations = row["activations"]
    act = np.array(activations)
    act = act[act != 99] # removes timestamps of songs whioch were not listened to over 80%
    act = ts - act # for each timestamp, calculate timestamp_i - timestamp_j
    return act


# defines path for raw deezer dataset
data_path = "data/new_release_stream.csv"

# read deezer dataset into pandas dataframe and sort dataframe by timestamp column (from smallest to largest timestamp)
df = pd.read_csv(data_path)
df = df.sort_values(by="timestamp", ascending=True)


# collect each user's listening history
df["activations"] = df["timestamp"] # create new column "activations" and set to the timestamps
df.loc[df.y == 0, "activations"] = 99  # if the item was not consumed over 80%, the timestamp does not enter the history (activation column is set to 99 in that row)
df["activations"] = df["activations"].apply(lambda x: [int(x)]) # makes each value to a list, e.g. if "activations" = 99 at a point, this function converts it to "activations" = [99]
# groups whole table according to same user-item pairs and sums up timestamps/activations for each user-item pair
df["activations"] = df.groupby(["userId", "itemId"], group_keys=False)["activations"].apply(lambda x: x.cumsum()) # cumsum to measure how user interest evolves over iterations

#create new column relational_interval 
df["relational_interval"] = df.apply(get_delta_t, axis=1)
df["relational_interval"] = df["relational_interval"] / (60.0 * 60)  # time_scalar -> scales seconds to hours
df["relational_interval"] = df["relational_interval"].map(list) # maps each element to its own list containing this element, similar to line 26


# sample 2 user-items pairs for test set
temp = df[["userId", "itemId"]].drop_duplicates() # select unique user-item combinations and store them in temp df
temp = temp.groupby("userId").sample(n=2).reset_index(drop=True) # for each unique user, sample 2 user-item pairs
temp["set"] = "test" # make this slice of original df the testset-df

df = df.merge(temp, on=["userId", "itemId"], how="left") # merge testset-df back into general df
df["set"] = df["set"].fillna("train") # fill all other cells in set-col with train

# sample 2 user-items pairs for evaluation
temp = df[df.set == "train"].copy() # filters df only by training data
temp = temp[["userId", "itemId"]].drop_duplicates()
temp = temp.groupby("userId").sample(n=2).reset_index(drop=True)
temp["set_val"] = "val" # creates specific validation set from training set
#print(temp.columns)
df = df.merge(temp, on=["userId", "itemId"], how="left") # merges val set back
df.loc[df["set_val"]=="val","set"]="val"
#print(df.columns)

# save everything to new, processed.csv
save_path = "data/processed.csv"
df[["userId", "itemId", "timestamp", "y", "relational_interval", "set"]].to_csv(
    save_path, index=False
)
