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

class MyHelpFormatter(argparse.HelpFormatter):
    def __init__(self, *args, **kwargs):
        super(MyHelpFormatter, self).__init__(*args, **kwargs)
        self._width = shutil.get_terminal_size().columns

parser = argparse.ArgumentParser(formatter_class=MyHelpFormatter, description='Perform hyperparameter optimization on a model.')
parser.add_argument('-opf', '--optuna_parameter_file', metavar='PATH', type=str, help='Path to JSON file describing the parameter space for optuna.')
parser.add_argument('-nt', '--ntrials', metavar='NT', type=int, nargs='?', default=50, help='Number of optimization trials to perform (Default: 50)')
parser.add_argument('-p', '--prog_path', type=str, help='Python training script path.')
parser.add_argument('-o', '--output_path', type=str, help='Path of file where to save best parameters of study.')
parser.add_argument('-mo', '--model', type=str, default='ex2vec', help='The model on which to optimize (type ex2vec or gru4rec).')
parser.add_argument('-f', '--final_run', type=str, default='N', help='Whether (Y) or not (N) to re-train the model on the best parameters (Default: N).')
parser.add_argument('-ovc', '--optuna_vis_csv', type=str, help='Path to store optuna study dataframe')
parser.add_argument('-ovp', '--optuna_vis_pkl', type=str, help='Path to store optuna study object')

#Ex2Vec specific args
parser.add_argument('-ep', '--embds_path', type=str, default='', help='Path to the pretrained GRU4Rec trained')

#GRU4Rec specific args
parser.add_argument('path', metavar='PATH', type=str, help='Path to the training data (TAB separated file (.tsv or .txt) or pickled pandas.DataFrame object (.pickle)) (if the --load_model parameter is NOT provided) or to the serialized model (if the --load_model parameter is provided).')
parser.add_argument('-t', '--test', metavar='TEST_PATH', type=str, help='Path to the test data set(s) located at TEST_PATH. Multiple test sets can be provided (separate with spaces). (Default: don\'t evaluate the model)')
parser.add_argument('-g', '--gru4rec_model', metavar='GRFILE', type=str, default='gru4rec_pytorch', help='Name of the file containing the GRU4Rec class. Can be used to select different varaiants. (Default: gru4rec_pytorch)')
parser.add_argument('-pf', '--parameter_file', metavar='PARAM_PATH', type=str, help='Alternatively, training parameters can be set using a config file specified in this argument. The config file must contain a single OrderedDict named `gru4rec_params`. The parameters must have the appropriate type (e.g. layers = [100]). Mutually exclusive with the -ps (--parameter_string) and the -l (--load_model) arguments and one of the three must be provided.')
parser.add_argument('-m', '--measure', metavar='AT', type=str, default=[20], help='Measure recall & MRR at the defined recommendation list length(s). Multiple values can be provided. (Default: 20)')
parser.add_argument('-pm', '--primary_metric', metavar='METRIC', choices=['recall', 'mrr'], default='recall', help='Set primary metric, recall or mrr (e.g. for paropt). (Default: recall)')
parser.add_argument('-e', '--eval_type', metavar='EVAL_TYPE', choices=['standard', 'conservative', 'median'], default='standard', help='Sets how to handle if multiple items in the ranked list have the same prediction score (which is usually due to saturation or an error). See the documentation of batch_eval() in evaluation.py for further details. (Default: standard)')
parser.add_argument('-d', '--device', metavar='D', type=str, default='cuda:0', help='Device used for computations (default: cuda:0).')
parser.add_argument('-ik', '--item_key', metavar='IK', type=str, default='ItemId', help='Column name corresponding to the item IDs (detault: ItemId).')
parser.add_argument('-sk', '--session_key', metavar='SK', type=str, default='SessionId', help='Column name corresponding to the session IDs (default: SessionId).')
parser.add_argument('-tk', '--time_key', metavar='TK', type=str, default='Time', help='Column name corresponding to the timestamp (default: Time).')
parser.add_argument('-l', '--load_model', action='store_true', help='Load an already trained model instead of training a model. Mutually exclusive with the -ps (--parameter_string) and the -pf (--parameter_file) arguments and one of the three must be provided.')

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
    
def generate_command(optimized_param_str, tuning=True) -> str:
    """
    Generate a command as a string for executing a Python script run.py with several parameters.

    Args:
        optimized_param_str: String containing the parameters which to optimize and their sampled value for the current run -> str
    Returns:
        command: The command line string which to execute -> str
    """
    command = ''
    if args.model == 'gru4rec':
        command = 'python "{}" "{}" -t "{}" -ps {} -pm {} -lpm -e {} -ik {} -sk {} -tk {} -d {} -m {}'.format(args.prog_path, args.path, args.test, optimized_param_str, args.primary_metric, args.eval_type, args.item_key, args.session_key, args.time_key, args.device, args.measure)
    elif args.model == 'ex2vec':
        command = 'python "{}" -ep "{}" -ps {}, -pm {}, -t {}'.format(args.prog_path, args.embds_path, optimized_param_str, args.primary_metric, tuning)
    return command

def train_and_eval(optimized_param_str):
    """
    Execute training script and report results.

    Args:
        optimized_param_str: String containing the parameters which to optimize and their sampled value for the current run -> str
    Returns:
        val: The result of the chosen metric for the current run -> float
    """
    command = generate_command(optimized_param_str) # get execution command
    cmd = pexpect.spawnu(command, timeout=None, maxread=1) # run command in a spawned subprocess
    line = cmd.readline() # read in first line that the model outputs
    val = 0
    while line:
        line = line.strip() # remove leading and trailing whitespaces
        print(line)
        if re.match('PRIMARY METRIC: -*\\d\\.\\d+e*-*\\d*', line): # matches lines of the form 'PRIMARY METRIC: [value]'
            t = line.split(':')[1].lstrip() # splits off the '[value]' part
            val = float(t) # converts value to float
            break
        line = cmd.readline()
    return val # return primary metric's value
    
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
    optimized_param_str = ','.join(optimized_param_str) # e.g. loss=bpr-max,embedding=0,...
    metric = train_and_eval(optimized_param_str)
    return metric

par_space = []
with open(args.optuna_parameter_file, 'rt') as f: # open json file containing parameters to optimize in read text mode
    print('-'*80)
    print('PARAMETER SPACE:')
    for line in f: # for each line in the corresponding parameter file
        par = Parameter.fromjson(line)
        print('\t' + str(par))
        par_space.append(par)
    print('-'*80)

study = optuna.create_study(direction='maximize') # goal is to maximize val which is returned from the objective function
study.optimize(lambda trial: objective(trial, par_space), n_trials=args.ntrials) # run objective function for a numer of ntrials iterations

# append results of this study to previous results
new_res = {
    "n_trials": args.ntrials,
    "best_params": study.best_params
}

# Open the file in append mode and write the new entry
with open(args.output_path, 'a') as f:
    f.write(json.dumps(new_res, indent=4) + '\n')

study.trials_dataframe().to_csv(args.optuna_vis_csv)

with open(args.optuna_vis_pkl, 'wb') as f:
    pickle.dump(study, f)

# retrain model using best parameters
if args.final_run == 'Y':
  print('Preparing for final training...')
  optimized_param_str = ','.join(['{}={}'.format(k,v) for k,v in study.best_params.items()])
  # generate final command with tuning flag set to false such that final model is saved
  command = generate_command(optimized_param_str, tuning=False)
  print('Start final training...')
  cmd = pexpect.spawnu(command, timeout=None, maxread=1) # run command in a spawned subprocess
  line = cmd.readline()
  while line:
      line = line.strip() # remove leading and trailing whitespaces
      print(line)
      line = cmd.readline()