import gc
import torch
import yaml, copy, itertools

from scenarIA.src.utils.settings import CONFIG_DIR
from scenarIA.src.train import run

def deep_set(d, dotted_key, value):
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value

with open(CONFIG_DIR / "config.yaml") as f:
    base = yaml.safe_load(f)

with open(CONFIG_DIR / "sensitivity.yaml") as f:
    sens = yaml.safe_load(f)

keys = list(sens["grid"].keys())
values = list(sens["grid"].values())

for combo in itertools.product(*values):
    cfg = copy.deepcopy(base)          # copie propre à chaque run
    for k, v in zip(keys, combo):
        deep_set(cfg, k, v)
    run(cfg)
    gc.collect()
    torch.cuda.empty_cache()