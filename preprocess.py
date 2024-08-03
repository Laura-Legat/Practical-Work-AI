# CREATES PROCESSED.CSV OUT OF NEW_RELEASE_STREAM.CSV
import argparse # lib for parsing command-line args
import random
import shutil
import numpy as np
import pandas as pd

class MyHelpFormatter(argparse.HelpFormatter):
    def __init__(self, *args, **kwargs):
        super(MyHelpFormatter, self).__init__(*args, **kwargs)
        self._width = shutil.get_terminal_size().columns

parser = argparse.ArgumentParser(formatter_class=MyHelpFormatter, description='Preprocessing for Ex2Vec and GRU4Rec models.')
parser.add_argument('-sl', '--seq_len', type=str, default=50, help='Sequence length for data-splitting for GRU4Rec model. Default = 50.')
parser.add_argument('-st', '--stride', type=str, default=1, help='Stride for overlap during data-splitting for GRU4Rec. Default = 1.')
parser.add_argument('-sm', '--small_version', type=str, default="N", help='If a small version of the dataset should be used for ex2vec and GRU4Rec preprocessing. Type Y for yes, N for no. Default = N.')
args = parser.parse_args() # store command line args into args variable


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
df = pd.read_csv(orig_dataset, index_col=False)
df = df.sort_values(by="timestamp", ascending=True)

# generate a smaller version of the preprocessed dataset for testing purposes
if args.small_version == 'Y':
    # sample first 1000 unique userIDs
    selected_user_ids = np.arange(1750)
    # filter full dataset to only contain interactions from the first 1000 selected users
    df_sm = df[df['userId'].isin(selected_user_ids)]
    # resort new dataframe
    df_sm = df_sm.sort_values(by='timestamp', ascending=True)
    df = df_sm # replace old, full dataset

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
for _, user_history in filtered_df.groupby('userId'):
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
SEQ_LEN = int(args.seq_len)
STRIDE = int(args.stride)

def generate_window(last_start_idx, df, user_id, seq_id, seq_length, stride) -> tuple[list, int]:
    """
    Helper function for slicing a dataframe.

    Args:
        last_start_idx: The last possible starting index for a window of length SEQ_LEN within len(df)
        df: The current dataframe to split up
        user_id: The current user the df belongs to
        seq_id: Increasing sequence ID
        seq_length: The length of one window
        stride: How many shifting positions each window has

    Returns:
        seqs: List containing all windows that df was split into
        seq_id: The current seq_id as int
    """
    seqs = []
    for idx in range(0, last_start_idx, stride):
        seq_df = df.iloc[idx:idx+seq_length].copy() # slide out window of size SEQ_LEN
        seq_df.loc[:, 'SessionId'] = f'{user_id}_{seq_id}' # assign global (wrt to user) sequence ID
        seq_df[seq_df.columns[seq_df.columns.get_loc('tempSessionId')]] = seq_id
        seqs.append(seq_df)
        seq_id += 1
    return seqs, seq_id


def split_into_seqs(whole_df, seq_length, stride) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Function for splitting user histories into overlapping windows of length SEQ_LEN.

    Args:
        whole_df: A whole dataset as pandas DataFrame
        seq_length: The length of one window
        stride: How many shifting positions each window has

    Returns:
        Tuple of pandas DataFrames where each DataFrame contains the sequences per set
    """
    whole_df['SessionId'] = '' # initialize sessionid col
    whole_df['tempSessionId'] = None
    train_seqs, val_seqs, test_seqs = [], [], []

    # group whole training set per user
    for user_id, user_df in whole_df.groupby('userId'):
        seq_id = 0 # create sequence ids across sets per user
        user_df = user_df.sort_values(by="timestamp") # order user history by consumption timestamp

        # separate user history based on set column
        train_df = user_df[user_df['set'] == 'train']
        val_df = user_df[user_df['set'] == 'val']
        test_df = user_df[user_df['set'] == 'test']

        last_start_idx = (len(train_df) - seq_length) + 1 # +1 because of 0 indexing, calculates the last possible starting indices of a window in a df rows which lead to complete windows (no partial sequences)

        train_seqs_partial, seq_id = generate_window(last_start_idx, train_df, user_id, seq_id, seq_length, stride)
        train_seqs.extend(train_seqs_partial)

        # get last SEQ_LEN-1 items from training set and concat for validation set
        combined_val_df = pd.concat([train_df.iloc[-seq_length+1:], val_df])
        last_start_idx = (len(combined_val_df) - seq_length) + 1

        val_seqs_partial, seq_id = generate_window(last_start_idx, combined_val_df, user_id, seq_id, seq_length, stride)
        val_seqs.extend(val_seqs_partial)

        combined_test_df = pd.concat([val_df.iloc[-seq_length+1:], test_df])
        if len(val_df.iloc[-seq_length+1:]) + 1 < seq_length: # check if there are too little validation set items and we need to thus add training items
            diff = seq_length - (len(val_df.iloc[-seq_length+1:]) + 1)
            combined_test_df = pd.concat([train_df.iloc[-diff:], combined_test_df])
        last_start_idx = (len(combined_test_df) - seq_length) + 1

        test_seqs_partial, seq_id = generate_window(last_start_idx, combined_test_df, user_id, seq_id, seq_length, stride)
        test_seqs.extend(test_seqs_partial)

    return pd.concat(train_seqs).sort_values(by=['userId', 'tempSessionId']), pd.concat(val_seqs).sort_values(by=['userId', 'tempSessionId']), pd.concat(test_seqs).sort_values(by=['userId', 'tempSessionId'])

# split dataset into sequences
train_df_seq, val_df_seq, test_df_seq = split_into_seqs(final_df, SEQ_LEN, STRIDE)

# filtering out irrelevant columns and saving as separate csv files for GRU4Rec
train_df_seq[['itemId', 'timestamp', 'SessionId', 'relational_interval']].to_csv(DATA_PATH + 'seq_train.csv', index=False)
val_df_seq[['itemId', 'timestamp', 'SessionId', 'relational_interval']].to_csv(DATA_PATH + 'seq_val.csv', index=False)
test_df_seq[['itemId', 'timestamp', 'SessionId', 'relational_interval']].to_csv(DATA_PATH + 'seq_test.csv', index=False)

print('Saved sequenced files for GRU4Rec')