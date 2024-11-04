import shutil
import argparse
import pandas as pd

class MyHelpFormatter(argparse.HelpFormatter):
    def __init__(self, *args, **kwargs):
        super(MyHelpFormatter, self).__init__(*args, **kwargs)
        self._width = shutil.get_terminal_size().columns

parser = argparse.ArgumentParser(formatter_class=MyHelpFormatter, description='Create stats about the Deezer datasets used.')
parser.add_argument('-p', '--data_path', type=str, default=None, help='Path to dataset to analyse')
args = parser.parse_args()

df = pd.read_csv(args.data_path)

print("Number of user-item interactions in total: ", len(df))
print("Number of unique users: ", df.userId.nunique())
print("Number of unique songs: ", df.itemId.nunique())

if 'SessionId' in df:
	print('Unique sessions across all users: ', df['SessionId'].nunique())

	seq_len_cnt = df.groupby('SessionId').size()
	print('Avg length of one session: ', round(seq_len_cnt.mean(), 1))

	unique_n_sess_per_user = df.groupby('userId')['SessionId'].nunique()
	print('Avg unique windows per user: ', round(unique_n_sess_per_user.mean(), 1))
else:
	user_interaction_counts = df.groupby('userId').size()
	avg_user_counts = user_interaction_counts.mean()

	print('Avg interactions per users: ', round(avg_user_counts, 1))

	item_interaction_counts = df.groupby('itemId').size()
	avg_item_counts = item_interaction_counts.mean()

	print('Avg interactions per item: ', round(avg_item_counts, 1))

	reps_cnt = df.groupby(['userId', 'itemId']).size().reset_index(name='n_reps')
	reps_cnt_mean_per_user = reps_cnt.groupby('userId')['n_reps'].mean()

	print('Number of reps per user: ', round(reps_cnt_mean_per_user.mean(), 1))
