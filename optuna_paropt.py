import argparse
import shutil
import optuna
import json
import pexpect
import re
import os
import importlib
from collections import OrderedDict
import pandas as pd
import pickle
import csv

class MyHelpFormatter(argparse.HelpFormatter):
    def __init__(self, *args, **kwargs):
        super(MyHelpFormatter, self).__init__(*args, **kwargs)
        self._width = shutil.get_terminal_size().columns

parser = argparse.ArgumentParser(formatter_class=MyHelpFormatter, description='Perform hyperparameter optimization on a model.')
parser.add_argument('-opf', '--optuna_parameter_file', metavar='PATH', type=str, help='Path to JSON file describing the parameter space for optuna.')
parser.add_argument('-nt', '--ntrials', metavar='NT', type=int, nargs='?', default=50, help='Number of optimization trials to perform (Default: 50)')
parser.add_argument('-o', '--output_path', type=str, help='Path of file where to save best parameters of study.')
parser.add_argument('-mo', '--model', type=str, default='ex2vec', help='The model on which to optimize (type ex2vec or gru4rec).')
parser.add_argument('-ovc', '--optuna_vis_csv', type=str, help='Path to store optuna study dataframe')
parser.add_argument('-ovp', '--optuna_vis_pkl', type=str, help='Path to store optuna study object')
parser.add_argument('-sp', '--storage_path', type=str, help='Path where to store Optuna study/where to resume it from.')
parser.add_argument('-sn', '--study_name', type=str, help='Unique study name to be associated with current study.')
parser.add_argument('-s', '--save_path', type=str, help='Path to save the model to (optional).')
parser.add_argument('-pth', '--base_path', type=str, default='./', help='The base directory where everything related to the PR (runs, chckpts, final models, results etc.) will be stored.')

#Ex2Vec specific args
parser.add_argument('-ep', '--embds_path', type=str, default='', help='Path to the pretrained GRU4Rec trained')
parser.add_argument('-a', '--alias', type=str, default='ex2vec_tuning', help='The alias of the model. Used primarily for tensorboard logging.')
parser.add_argument('--use_cuda', action='store_true', help='Sets the flag for training Ex2Vec on the GPU')

#GRU4Rec specific args
parser.add_argument('path', metavar='PATH', type=str, help='Path to the training data (TAB separated file (.tsv or .txt) or pickled pandas.DataFrame object (.pickle)) (if the --load_model parameter is NOT provided) or to the serialized model (if the --load_model parameter is provided).')
parser.add_argument('-t', '--test', metavar='TEST_PATH', type=str, help='Path to the test data set(s) located at TEST_PATH. Multiple test sets can be provided (separate with spaces). (Default: don\'t evaluate the model)')
parser.add_argument('-g', '--gru4rec_model', metavar='GRFILE', type=str, default='gru4rec_pytorch', help='Name of the file containing the GRU4Rec class. Can be used to select different varaiants. (Default: gru4rec_pytorch)')
parser.add_argument('-m', '--measure', metavar='AT', type=str, default=[20], help='Measure recall & MRR at the defined recommendation list length(s). Multiple values can be provided. (Default: 20)')
parser.add_argument('-pm', '--primary_metric', metavar='METRIC', choices=['recall', 'mrr'], default='recall', help='Set primary metric, recall or mrr (e.g. for paropt). (Default: recall)')
parser.add_argument('-e', '--eval_type', metavar='EVAL_TYPE', choices=['standard', 'conservative', 'median'], default='standard', help='Sets how to handle if multiple items in the ranked list have the same prediction score (which is usually due to saturation or an error). See the documentation of batch_eval() in evaluation.py for further details. (Default: standard)')
parser.add_argument('-d', '--device', metavar='D', type=str, default='cuda:0', help='Device used for computations (default: cuda:0).')
parser.add_argument('-ik', '--item_key', metavar='IK', type=str, default='ItemId', help='Column name corresponding to the item IDs (detault: ItemId).')
parser.add_argument('-sk', '--session_key', metavar='SK', type=str, default='SessionId', help='Column name corresponding to the session IDs (default: SessionId).')
parser.add_argument('-tk', '--time_key', metavar='TK', type=str, default='Time', help='Column name corresponding to the timestamp (default: Time).')
parser.add_argument('-l', '--load_model', action='store_true', help='Load an already trained model instead of training a model. Mutually exclusive with the -ps (--parameter_string) and the -pf (--parameter_file) arguments and one of the three must be provided.')
parser.add_argument('--optim', action='store_true', help='Sets the flag that this is hyperparameter tuning instead of normal training.')

args = parser.parse_args() # store command line args into args variable

class Parameter:
    def __init__(self, name, dtype, values, step=None, log=False):
        assert dtype in ['int', 'float', 'categorical']
        assert type(values)==list
        assert len(values)==2 or dtype=='categorical'
        self.name = name
        self.dtype = dtype
        self.values = values
        self.step = step
        if self.step is None and self.dtype=='int':
            self.step = 1
        self.log = log
    @classmethod
    def fromjson(cls, json_string):
        obj = json.loads(json_string)
        return Parameter(obj['name'], obj['dtype'], obj['values'], obj['step'] if 'step' in obj else None, obj['log'] if 'log' in obj else False)
    def __call__(self, trial):
        if self.dtype == 'int':
            return trial.suggest_int(self.name, int(self.values[0]), int(self.values[1]), step=self.step, log=self.log)
        if self.dtype == 'float':
            return trial.suggest_float(self.name, float(self.values[0]), float(self.values[1]), step=self.step, log=self.log)
        if self.dtype == 'categorical':
            return trial.suggest_categorical(self.name, self.values)
    def __str__(self):
        desc = 'PARAMETER name={} \t type={}'.format(self.name, self.dtype)
        if self.dtype == 'int' or self.dtype == 'float':
            desc += ' \t range=[{}..{}] (step={}) \t {} scale'.format(self.values[0], self.values[1], self.step if self.step is not None else 'N/A', 'UNIFORM' if not self.log else 'LOG')
        if self.dtype == 'categorical':
            desc += ' \t options: [{}]'.format(','.join([str(x) for x in self.values]))
        return desc
    
def generate_command(optimized_param_str) -> str:
    """
    Generate a command as a string for executing a Python script run.py with several parameters.

    Args:
        optimized_param_str: String containing the parameters which to optimize and their sampled value for the current run -> str
    Returns:
        command: The command line string which to execute -> str
    """
    command = ''
    if args.model == 'gru4rec':
        command = 'python "{}" "{}" -t "{}" -ps {} -pm {} -lpm -e {} -ik {} -sk {} -tk {} -d {} -m {} -s {} {}'.format(args.base_path + 'GRU4Rec_Fork/run.py', args.path, args.test, optimized_param_str, args.primary_metric, args.eval_type, args.item_key, args.session_key, args.time_key, args.device, args.measure, args.save_path, '--optim' if args.optim else '')
    elif args.model == 'ex2vec':
        command = 'python "{}" -ep "{}" -ps {} -t {} -n {} -pth {}'.format(args.base_path + 'train.py', args.embds_path, optimized_param_str, "Y", args.alias, args.base_path)
        if args.use_cuda is True:
            command += ' --use_cuda'
    return command

def train_and_eval(optimized_param_str):
    """
    Execute training script and report results.

    Args:
        optimized_param_str: String containing the parameters which to optimize and their sampled value for the current run -> str
    Returns:
        val: The result of the chosen metric for the current run -> float
        all_metrics: The results of all other metrics for the current run -> List
    """
    all_metrics = []

    command = generate_command(optimized_param_str) # get execution command
    cmd = pexpect.spawnu(command, timeout=None, maxread=1) # run command in a spawned subprocess
    line = cmd.readline() # read in first line that the model outputs
    val = 0
    while line:
        line = line.strip() # remove leading and trailing whitespaces
        print(line)

        if args.model == 'gru4rec':
            # match all metrics (recall, mrr...) from gru4rec besides primary metrics
            matches = re.match('Recall@(\\d+): (-*\\d\\.\\d+e*-*\\d*) MRR@(\\d+): (-*\\d\\.\\d+e*-*\\d*)', line)
        elif args.model == 'ex2vec':
            # match all metrics (bacc, acc, recall, ...) from ex2vec besides primary metric
            matches = re.match('FINAL METRICS: ACC: (-*\\d\\.\\d+e*-*\\d*), RECALL: (-*\\d\\.\\d+e*-*\\d*), F1: (-*\\d\\.\\d+e*-*\\d*), BACC: (-*\\d\\.\\d+e*-*\\d*)', line)

        # append all results
        if matches:
          all_metrics.append(matches.group(0))

        if re.match('PRIMARY METRIC: -*\\d\\.\\d+e*-*\\d*', line): # matches lines of the form 'PRIMARY METRIC: [value]'
            t = line.split(':')[1].lstrip() # splits off the '[value]' part
            val = float(t) # converts value to float
            break
        line = cmd.readline()
    return val, all_metrics
    
def objective(trial, par_space):
    """
    Defining the Optuna optimization process.

    Args:
        trial: Represents a single optimization run -> Optuna Trial object
        par_space: List of parameters to be optimized -> List
    Returns:
        metric: The result of the chosen metric for the current run  -> float
    """
    # create whole parameter string
    optimized_param_str = []
    for par in par_space: # for each parameter
        val = par(trial) # sampled value from specified 'values' field
        optimized_param_str.append('{}={}'.format(par.name,val)) # e.g. loss=bpr-max

    param_dict = {par.name: par(trial) for par in par_space}

    # Enforce constraint: if constrained_embedding is True, embedding must be 0
    if param_dict.get('constrained_embedding') and param_dict.get('embedding') != 0:
        param_dict['embedding'] = 0
        optimized_param_str.append('embedding=0')

    optimized_param_str = ','.join(optimized_param_str) # e.g. loss=bpr-max,embedding=0,...
    primary_metric, all_metrics = train_and_eval(optimized_param_str)

    # log all metrics as csv and store them in a temporary csv
    metrics_df = pd.DataFrame()
    if args.model == 'ex2vec':
      matches = re.match('FINAL METRICS: ACC: (-*\\d\\.\\d+e*-*\\d*), RECALL: (-*\\d\\.\\d+e*-*\\d*), F1: (-*\\d\\.\\d+e*-*\\d*), BACC: (-*\\d\\.\\d+e*-*\\d*)', all_metrics[0])
      if matches:
        acc, recall, f1, bacc = matches.group(1), matches.group(2), matches.group(3), matches.group(4) # extract metrics values

        # construct new row for temp csv
        all_metrics_log_dict = {
          'trial_id': trial.number,
          'acc': acc,
          'recall': recall,
          'f1': f1,
          'bacc': bacc
        }
        metrics_df = pd.DataFrame([all_metrics_log_dict])
        temp_path = args.base_path + 'temp_metrics_ex2vec.csv'
    
        metrics_df.to_csv(temp_path, mode='a', header=not os.path.exists(temp_path), index=False)

    elif args.model == 'gru4rec':
        if len(all_metrics) > 0:
          log_dict = {
            'trial_id': trial.number
          }
          trial_id_df = pd.DataFrame([log_dict])

          # Initialize an empty dictionary to hold all metrics
          all_metrics_combined = {}
          for metrics in all_metrics: # since all metrics is a list of recalls/mrr's for each cutoff
            matches = re.match('Recall@(\\d+): (-*\\d\\.\\d+e*-*\\d*) MRR@(\\d+): (-*\\d\\.\\d+e*-*\\d*)', metrics)
            # extract recall & mrr values for one cutoff
            recall = matches.group(2)
            mrr = matches.group(4)

            all_metrics_combined[f'Recall@{matches.group(1)}'] = recall
            all_metrics_combined[f'MRR@{matches.group(3)}'] = mrr

          metrics_df = pd.DataFrame([all_metrics_combined])
          
          metrics_df = pd.concat([trial_id_df, metrics_df], axis=1)
    
          # save temporary metrics file as csv
          temp_path = args.base_path + 'temp_metrics_gru4rec.csv'
    
          metrics_df.to_csv(temp_path, mode='a', header=not os.path.exists(temp_path), index=False)

    return primary_metric # return metric to optimize study for

par_space = []
with open(args.optuna_parameter_file, 'rt') as f: # open json file containing parameters to optimize in read text mode
    print('-'*80)
    print('PARAMETER SPACE:')
    for line in f: # for each line in the corresponding parameter file
        par = Parameter.fromjson(line)
        print('\t' + str(par))
        par_space.append(par)
    print('-'*80)


# SEARCH SPACE LOGGING
# define file where to log search space
if args.model == 'gru4rec':
    par_space_log_path = args.base_path + 'results/gru4rec_search_space.csv'
elif args.model == 'ex2vec' and args.embds_path == '':
    par_space_log_path = args.base_path + 'results/ex2vec_search_space.csv'
elif args.model == 'ex2vec' and args.embds_path != '':
    par_space_log_path = args.base_path + 'results/ex2vec_gruembds_search_space.csv'

# log currently used search space
with open(par_space_log_path, 'r') as file:
    n_rows = list(csv.reader(file))

    if len(n_rows) <= 1: # only header or empty file
        search_space_id = 1
    else:
        last_logged_row = n_rows[-1]
        search_space_id = int(last_logged_row[0]) + 1

with open(par_space_log_path, mode='a', newline='') as file:
    writer = csv.writer(file)

    if file.tell() == 0: # if file is empty
        writer.writerow(['search_space_id', 'param', 'search_space']) # write header row

    # log each parameter
    for par in par_space:
        writer.writerow([search_space_id, par.name, json.dumps(par.values)])


study = optuna.create_study(study_name=args.study_name, storage = args.storage_path, direction='maximize', load_if_exists=True) # goal is to maximize val which is returned from the objective function
study.optimize(lambda trial: objective(trial, par_space), n_trials=args.ntrials) # run objective function for a numer of ntrials iterations

# append results of this study to previous results
new_res = {
    "n_trials": args.ntrials,
    "best_params": study.best_params
}

# store the current best params
with open(args.output_path, 'w') as f:
    f.write(json.dumps(new_res, indent=4) + '\n')

# get current trials df
trials_df = study.trials_dataframe()
trials_df_copy = trials_df.copy() 

optuna_vis_csv_path = args.optuna_vis_csv # path for the optuna vis file containing trial information from previous runs

if os.path.exists(optuna_vis_csv_path): # if ths trials csv already exists, aka if this is not the first run of the study
    optuna_vis_csv = pd.read_csv(optuna_vis_csv_path) # store information from previous runs
    new_trials = trials_df[~trials_df['number'].isin(optuna_vis_csv['number'])] # filter out new rows from this trial that are not yet part of the optuna vis csv
    trials_df_copy = pd.concat([optuna_vis_csv, new_trials], ignore_index=True) # concatinate new rows to new trails df

#read out temp file all metrics and based on trial number, assign the col values
temp_path = args.base_path + 'temp_metrics_gru4rec.csv' if args.model == 'gru4rec' else args.base_path + 'temp_metrics_ex2vec.csv'
if os.path.exists(temp_path) and os.path.getsize(temp_path) >0:
    metrics_temp_df = pd.read_csv(temp_path)
    metrics_temp_df['search_space_id'] = search_space_id # add the used search space to the current trial

    if args.embds_path != '': # add col to log from which gru4rec model the item embeddings were extracted
        model_name = os.path.basename(args.embds_path).split('.')[0] # get model name without file extension
        metrics_temp_df['gru_item_embds_model'] = model_name

# merge new column data to trial information where number col = trial_id col, keeping all rows from trials_df_copy and adding additional info from metrics_temp_df
trials_df_copy = trials_df_copy.merge(metrics_temp_df, left_on='number', right_on='trial_id', how='left')
trials_df_copy = trials_df_copy.drop(columns=['trial_id']) # drop redundant trial info

if os.path.exists(optuna_vis_csv_path):
    trials_df_copy = trials_df_copy.drop(columns=['Unnamed: 0']) # drop redundant trial info after merge of new and old

    # combine new and old columns into the same columns
    if args.model == 'ex2vec':
      if args.embds_path == '':
          for col in ['acc', 'recall', 'f1', 'bacc', 'search_space_id']:
              trials_df_copy[col] = trials_df_copy[f'{col}_x'].combine_first(trials_df_copy[f'{col}_y'])
              trials_df_copy.drop(columns=[f'{col}_x', f'{col}_y'], inplace=True)
      else:
          for col in ['acc', 'recall', 'f1', 'bacc', 'search_space_id', 'gru_item_embds_model']:
              trials_df_copy[col] = trials_df_copy[f'{col}_x'].combine_first(trials_df_copy[f'{col}_y'])
              trials_df_copy.drop(columns=[f'{col}_x', f'{col}_y'], inplace=True)
    elif args.model == 'gru4rec':
      cols = []
      col_patterns = [r'Recall@', r'MRR@', r'search_space_id'] # define match patterns since we can have different cutoffs 
      for pattern in col_patterns:
        cols.extend([col for col in trials_df_copy.columns if re.search(pattern, col)]) # match all columns which contain recall or mrr scores for varying cutoffs

      for col in cols:
          # transform to base cols
          if col.endswith('_x'):
            base_col = col[:-2] # remove the _x from the string to get base col name
            trials_df_copy[base_col] = trials_df_copy[f'{base_col}_x'].combine_first(trials_df_copy[f'{base_col}_y'])
            trials_df_copy.drop(columns=[f'{base_col}_x', f'{base_col}_y'], inplace=True)
    

# save updates trial info csv
trials_df_copy.to_csv(args.optuna_vis_csv)

# empty temporary file
metrics_temp_df = metrics_temp_df.head(0)
# drop previous cols
metrics_temp_df.to_csv(temp_path, index=False)
metrics_temp_df_cleaned = pd.read_csv(temp_path, index_col=False)
metrics_temp_df_cleaned = metrics_temp_df_cleaned.drop('search_space_id', axis=1)

metrics_temp_df_cleaned.to_csv(temp_path, index=False)

# save current study for visualizations
with open(args.optuna_vis_pkl, 'wb') as f:
    pickle.dump(study, f)