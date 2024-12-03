from setuptools import setup

# to have location-independent file packages
setup(
    name="pr_ext",
    version="0.1",
    py_modules=['data_sampler', 'utils', 'metrics', 'engine', 'ex2vec', 'GRU4Rec_Fork.gru4rec_pytorch', 'GRU4Rec_Fork.gru4rec_utils', 'GRU4Rec_Fork.evaluation'],
    install_requires=[
        'numpy',    
        'pandas',   
        'torch',    
        'tqdm'      
    ],
)
