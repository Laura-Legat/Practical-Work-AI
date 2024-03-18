import random
import numpy as np
import pandas as pd


# function that computes delta_t, i.e., the time interval between consumptions (not considered when y = 0  )
def get_delta_t(row):
    ts = row["timestamp"]
    activations = row["activations"]
    act = np.array(activations)
    act = act[act != 99]
    act = ts - act
    return act


# defines path for raw deezer dataset
data_path = "data/new_release_stream.csv"

# read deezer dataset into pandas datafra,e and sort dataframe by timestamp column (smallest timestamp on top)
df = pd.read_csv(data_path)
df = df.sort_values(by="timestamp", ascending=True)


# collect each user's listening history
df["activations"] = df["timestamp"] # create new column "activations" and set to the timestamps
df.loc[df.y == 0, "activations"] = 99  # if the item was not consumed, the timestamp does not enter the history (activation column is set to 99 in that row)
df["activations"] = df["activations"].apply(lambda x: [int(x)]) # makes each value to a list, e.g. if "activations" = 99 at a point, this function converts it to "activations" = [99]
df["activations"] = df.groupby(["userId", "itemId"], group_keys=False)["activations"].apply(lambda x: x.cumsum())

df["relational_interval"] = df.apply(get_delta_t, axis=1)
df["relational_interval"] = df["relational_interval"] / (60.0 * 60)  # time_scalar
df["relational_interval"] = df["relational_interval"].map(list)


# sample 2 user-items pairs for evaluation
temp = df[["userId", "itemId"]].drop_duplicates()
temp = temp.groupby("userId").sample(n=2).reset_index(drop=True)
temp["set"] = "test"

df = df.merge(temp, on=["userId", "itemId"], how="left")
df["set"] = df["set"].fillna("train")

# sample 2 user-items pairs for the test set
temp = df[df.set == "train"].copy()
temp = temp[["userId", "itemId"]].drop_duplicates()
temp = temp.groupby("userId").sample(n=2).reset_index(drop=True)
temp["set_val"] = "val"
print(temp.columns)
df = df.merge(temp, on=["userId", "itemId"], how="left")
df.loc[df["set_val"]=="val","set"]="val"
print(df.columns)

save_path = "data/processed.csv"
df[["userId", "itemId", "timestamp", "y", "relational_interval", "set"]].to_csv(
    save_path, index=False
)
