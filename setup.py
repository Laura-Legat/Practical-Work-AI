from setuptools import setup

setup(
    name="pr_ext",
    version="0.1",
    py_modules=['data_sampler', 'ex2vec', 'GRU4Rec_Fork.gru4rec_pytorch', 'GRU4Rec_Fork.gru4rec_utils', 'GRU4Rec_Fork.evaluation'],  # List your standalone .py files
    install_requires=[
        'numpy',    
        'pandas',   
        'torch',    
        'tqdm'      
    ],
)
