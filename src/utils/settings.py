import os
from pathlib import Path

HOME_DIR = Path(os.getcwd()) / 'scenarIA'
CONFIG_DIR = HOME_DIR / 'configs'

SCRATCH_DIR = Path('/scratch/globc/garcia/scenarIA/')
RAW_DATA_DIR = SCRATCH_DIR / 'rawdata'
DATASET_DIR = SCRATCH_DIR / 'datasets'
RUNS_DIR = SCRATCH_DIR / 'runs'