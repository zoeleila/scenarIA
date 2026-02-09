from pathlib import Path
import yaml

from scenarIA.src.utils.settings import RUNS_DIR

def save_infos_from_config(config: dict) -> dict:
    runs_dir = RUNS_DIR / config['train']['runs_dir']
    exp_infos = runs_dir.parent.parent / 'experiment_infos.yaml' # data and training settings root
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

def test_name_from_config(config:dict) -> str:
    arch = config['train']['arch']
    seed = config['train']['seed']
    seq_length = config['data']['seq_length']
    nb_member_per_subsets = config['data']['nb_member_per_subsets']
    nb_subsets = config['data']['nb_subsets'] if bool(config['data']['one_to_many']) else 1
    test_name = f'{arch}_seed{seed}_seq{seq_length}_mem{nb_member_per_subsets}_sub{nb_subsets}'
    return test_name
