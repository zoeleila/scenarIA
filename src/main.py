import itertools, sys, yaml
import argparse
from scenarIA.src.utils.config import load_config
from scenarIA.src.utils.settings import CONFIG_DIR
from scenarIA.src.train import run

def run_sensitivity(name):
    spec = yaml.safe_load(open(CONFIG_DIR / name))
    base, grid = spec.get('base_overrides', {}), spec['grid']
    keys = list(grid)
    for values in itertools.product(*grid.values()):
        config = load_config({**base, **dict(zip(keys, values))})
        run(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='sensitivity.yaml')
    args = parser.parse_args()
    run_sensitivity(args.file)