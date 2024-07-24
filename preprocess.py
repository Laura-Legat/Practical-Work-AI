# CREATES PROCESSED.CSV OUT OF NEW_RELEASE_STREAM.CSV

import random
import numpy as np
import pandas as pd

print('Pre-processing dataset for Ex2Vec...')


# function that computes delta_t, i.e., the time interval between consumptions (not considered when y = 0 )
def get_delta_t(row):
    ts = row["timestamp"]
    activations = row["activations"]
    act = np.array(activations)
    act = act[act != 99] # removes timestamps of songs whioch were not listened to over 80%
    act = ts - act # for each timestamp, calculate timestamp_i - timestamp_j
    return act


# defines path for raw deezer dataset
DATA_PATH = '/content/drive/MyDrive/JKU/practical_work/Practical-Work-AI/data/'

orig_dataset = DATA_PATH + 'new_release_stream.csv'

# read deezer dataset into pandas dataframe and sort dataframe by timestamp column (from smallest to largest timestamp)
df = pd.read_csv(orig_dataset)
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

"""
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
"""

"""
New data-splitting technique:
    1) Group by user histories: group by userId and sort each user history by timestamp
    2) Split each user history into: 70% for training ("train"), 10 % for validation ("val") and 20% test ("test")
    3) Sort the whole thing according to timestamp again
"""

# 5-core filtering, filter out users and items with interactions under a certain threshold
FILTERING_THRESHOLD = 5

user_interactions_cnt = df.groupby('userId').size()
valid_user_ids = user_interactions_cnt[user_interactions_cnt >= FILTERING_THRESHOLD].index
filtered_df = df[df['userId'].isin(valid_user_ids)]

item_interaction_cnt = filtered_df.groupby('itemId').size()
valid_idem_ids = item_interaction_cnt[item_interaction_cnt >= FILTERING_THRESHOLD].index
filtered_df = filtered_df[filtered_df['itemId'].isin(valid_idem_ids)]

# SPLIT EACH USER HISTORY INTO TRAIN-VAL-TEST 70-10-20 %
# get user histories and group each of them by timestamp
df_user_histories = filtered_df.groupby('userId', group_keys=False).apply(lambda x: x.sort_values('timestamp'))

# create structures for storing the rows belonging to the splits
train_rows = []
val_rows = []
test_rows = []

# split each user history 70-10-20 into train-val-test
for _, user_history in df_user_histories.groupby('userId'):
    # calculate row numbers for 70-10-20 split
    history_n_rows = len(user_history)
    train_size = int(history_n_rows * 0.7)
    val_size = int(history_n_rows * 0.1)

    train_rows.append(user_history[:train_size])
    val_rows.append(user_history[train_size:train_size + val_size])
    test_rows.append(user_history[train_size + val_size:])

train_df = pd.concat(train_rows).assign(set='train')
val_df = pd.concat(val_rows).assign(set='val')
test_df = pd.concat(test_rows).assign(set='test')


# concatinate everything to new df
final_df = (
    pd.concat([train_df, val_df, test_df]).sort_values(by=['userId', 'timestamp'])
)

# save full split-up dataset as processed.csv
final_df[["userId", "itemId", "timestamp", "y", "relational_interval", "set"]].to_csv(
    DATA_PATH + 'processed.csv', index=False
)

print('Saved processed.csv')
print('Pre-processing dataset for GRU4Rec...')

# PREPARE DATA FOR GRU4REC
#seq length 50 split training
SEQ_LEN = 50

def split_into_seqs(whole_df, seq_length):
    seq_list = []
    whole_df['SessionId'] = '' # initialize sessionid col

    # group whole training set per user
    for user_id, user_df in whole_df.groupby('userId'):
        user_df = user_df.sort_values(by="timestamp") # order user history 
        n_seqs = len(user_df) // seq_length # how many (whole number) sequences will fit into the user training history

        for i in range(n_seqs):
            # calculate start and end indices
            start = i * seq_length
            end = start + seq_length

            seq_df = user_df.iloc[start:end] # slice out corresponding rows
            seq_df.loc[:, 'SessionId'] = f'{user_id}_{i}' # give global sequence ID
            seq_list.append(seq_df)

        # if last slice of set is < 50, include it as partial seq (as GRU4Rec can handle sequences of different length)
        start_remainding = n_seqs * SEQ_LEN
        if start_remainding < len(user_df):
            rem_seq_df = user_df.iloc[start_remainding:] # add partial seq rows
            rem_seq_df.loc[:, 'SessionId'] = f'{user_id}_{n_seqs}' # remainder gets last n_seq number as indexing starts with 0 for the other rows
            seq_list.append(rem_seq_df)

    return pd.concat(seq_list).sort_values(by=['userId', 'timestamp']) # create pandas df out of sequences and return

# create train,val, and test df's split into sequenecs
train_df_seq = split_into_seqs(train_df, SEQ_LEN)
val_df_seq = split_into_seqs(val_df, SEQ_LEN)
test_df_seq = split_into_seqs(test_df, SEQ_LEN)

# filtering out irrelevant columns and saving as separate csv files for GRU4Rec
train_df_seq[['itemId', 'timestamp', 'SessionId']].to_csv(DATA_PATH + 'seq_train.csv', index=False)
val_df_seq[['itemId', 'timestamp', 'SessionId']].to_csv(DATA_PATH + 'seq_val.csv', index=False)
test_df_seq[['itemId', 'timestamp', 'SessionId']].to_csv(DATA_PATH + 'seq_test.csv', index=False)

print('Saved sequenced files for GRU4Rec')

""" # concatinate everything to new df
final_df_seq = (
    pd.concat([train_df_seq, val_df_seq, test_df_seq]).sort_values(by=['userId', 'timestamp'])
)

# filter out irrelevant columns
final_seq = final_df_seq[['itemId', 'timestamp', 'set', 'SessionId']]

final_seq.to_csv(save_path + 'sequenced.csv', index=False)
print('Saved sequenced.csv') """
