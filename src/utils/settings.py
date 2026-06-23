import os
from pathlib import Path

HOME_DIR = Path(os.getcwd()) / 'scenarIA'
CONFIG_DIR = HOME_DIR / 'configs'

SCRATCH_DIR = Path('/scratch/globc/garcia/scenarIA/')
RAW_DATA_DIR = SCRATCH_DIR / 'rawdata'
DATASET_DIR = SCRATCH_DIR / 'datasets'
RUNS_DIR = SCRATCH_DIR / 'runs'
GRAPHS_DIR = SCRATCH_DIR / 'graphs'
PREDICTIONS_DIR = SCRATCH_DIR / 'predictions'

SIMUS_COLORS_DICT = {'historical':'k',
               'hist-aer': 'darkorange',
               'hist-GHG': 'brown',
               'ssp119': 'y',
              'ssp126': 'green', 
              'ssp245': 'b', 
              'ssp370':'r', 
              'ssp585': 'mediumorchid'}