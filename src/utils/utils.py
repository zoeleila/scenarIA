from pathlib import Path
import yaml

from scenarIA.src.utils.settings import RUNS_DIR

def save_infos_from_config(config: dict) -> dict:
    runs_dir = RUNS_DIR / config['train']['runs_dir']
    exp_infos = runs_dir.parent / 'experiment_infos.yaml'
    print(f'Experiment infos will be saved to {exp_infos}')
    
    if exp_infos.exists():
        infos = yaml.safe_load(open(exp_infos, 'r'))
    else:
        infos = {}
    exp = config['data']['exp']
    infos[exp] = {
        'simus_train': config['train']['simus_train'],
        'inputs': config['train']['inputs']}
    
    with open(exp_infos, 'w') as file:
        yaml.dump(infos, file)

