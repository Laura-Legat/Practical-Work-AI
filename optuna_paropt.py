import argparse
import shutil
import optuna
import json # since optimization files are in JSON format

class MyHelpFormatter(argparse.HelpFormatter):
    def __init__(self, *args, **kwargs):
        super(MyHelpFormatter, self).__init__(*args, **kwargs)
        self._width = shutil.get_terminal_size().columns

parser = argparse.ArgumentParser(formatter_class=MyHelpFormatter, description='Perform hyperparameter optimization on a model.')
parser.add_argument('-opf', '--optuna_parameter_file', metavar='PATH', type=str, help='Path to JSON file describing the parameter space for optuna.')
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

par_space = []
with open(args.optuna_parameter_file, 'rt') as f: # open json file containing parameters to optimize in read text mode
    print('-'*80)
    print('PARAMETER SPACE:')
    for line in f: # for each line in the corresponding parameter file
        par = Parameter.fromjson(line)
        print('\t' + str(par))
        par_space.append(par)
    print('-'*80)