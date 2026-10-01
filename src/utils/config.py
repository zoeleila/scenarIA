import copy
import yaml
from scenarIA.src.utils.settings import CONFIG_DIR

CONFIG_VERSION = 1

# Valeurs implicites des anciens runs (clé absente = cette valeur).
# Figé pour toujours : à ne pas confondre avec default.yaml.
LEGACY_DEFAULTS = {
    'data': {
        'smoothed_outputs': False,
        'lonshift': False,
        'add_clim_to_predictors': False,
    },
    'train': {
        'arch': 'cnn-lstm',
        'unet_features': 32,
        'lstm_units': 25,
        'links': 5,
        'encoder': 'resnet18',
        'simus_val': None,
        'alpha': 5,
        'monitor_metric': 'val_rmse',
        'climax': {},
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _fill_missing(cfg: dict, defaults: dict) -> dict:
    for k, v in defaults.items():
        if isinstance(v, dict):
            cfg[k] = _fill_missing(cfg.get(k) or {}, v)
        else:
            cfg.setdefault(k, v)
    return cfg


def _read(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _has_dotted(cfg, dotted_key):
    d = cfg
    for p in dotted_key.split('.'):
        if not isinstance(d, dict) or p not in d:
            return False
        d = d[p]
    return True


def _set_dotted(cfg, dotted_key, value):
    *parents, last = dotted_key.split('.')
    d = cfg
    for p in parents:
        d = d.setdefault(p, {})
    d[last] = value


def load_config(overrides: dict | None = None) -> dict:
    """default.yaml < archs/<arch>.yaml < overrides (clés pointées)."""
    overrides = overrides or {}
    cfg = _read(CONFIG_DIR / 'default.yaml')
    arch = overrides.get('train.arch', cfg['train']['arch'])
    arch_file = CONFIG_DIR / 'archs' / f'{arch}.yaml'
    if not arch_file.exists():
        raise FileNotFoundError(f"Pas de config pour l'archi '{arch}' : {arch_file}")
    cfg = deep_merge(cfg, _read(arch_file))
    cfg['train']['arch'] = arch

    for k, v in overrides.items():
        if k != 'train.arch' and not _has_dotted(cfg, k):
            raise KeyError(f"Override inconnu : '{k}' (absent de default.yaml et de archs/{arch}.yaml)")
        _set_dotted(cfg, k, v)

    cfg['config_version'] = CONFIG_VERSION
    return cfg


# --- Rétrocompatibilité ---------------------------------------------------
def _v0_to_v1(cfg):
    # place pour les futurs renommages / déplacements de clés
    return cfg

MIGRATIONS = {0: _v0_to_v1}


def migrate_config(config: dict) -> dict:
    """Idempotente : sans effet sur une config à jour, complète une ancienne."""
    cfg = copy.deepcopy(config)
    v = cfg.get('config_version', 0)
    while v < CONFIG_VERSION:
        cfg = MIGRATIONS[v](cfg)
        v += 1
    cfg = _fill_missing(cfg, LEGACY_DEFAULTS)
    cfg['config_version'] = CONFIG_VERSION
    return cfg