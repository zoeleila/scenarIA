import copy
import pytest

def _deep_update(d, u):
    """Merge nested dict u into d (in-place), similar to dict.update but recursive."""
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = _deep_update(d.get(k, {}) if isinstance(d.get(k, {}), dict) else {}, v)
        else:
            d[k] = v
    return d

@pytest.mark.parametrize("params", [
    # Exemple 1 : regrouper 50 membres par sous-ensemble
    ({'data': {'nb_member_per_subsets': 50, 'nb_subsets': 20}}),
    # Exemple 2 : ne pas regrouper (1 membre par sous-ensemble)
    ({'data': {'nb_member_per_subsets': 1}}),
    # Exemple 3 : regrouper 10 membres par sous-ensemble
    ({'data': {'nb_member_per_subsets': 10}}),
])
def test_dataset_various_configs(dataset, config, params):
    """
    Tester la sortie de la classe dataset en faisant varier des champs du dict `config`.

    - `dataset` et `config` sont supposés être des fixtures fournies par le projet.
    - `params` contient les modifications à appliquer à `config['data']`.
    Les assertions vérifient que pour chaque configuration :
      * on obtient bien des sorties (y non vides),
      * toutes les simulations ont le même nombre de sous-ensembles (invariance attendue ici).
    Adaptez les assertions si votre validité attendue est différente.
    """
    cfg = copy.deepcopy(config)
    _deep_update(cfg, params)

    ds = scenarIA(transform=None, config=cfg, data_type='train')

    # Vérifications basiques et invariantes
    assert hasattr(ds, 'y'), f"Dataset missing attribute 'y' for params {params}"
    values = list(ds.y.values)
    assert len(values) > 0, f"No simulations found in dataset.y for params {params}"

    lengths = [len(y_simu) for y_simu in values]
    # Ici on s'attend (comme dans votre test initial) à ce que toutes les simulations aient le même nombre de sous-ensembles
    assert all(l == lengths[0] for l in lengths), f"Nombre de sous-ensembles inégal entre simulations for params {params}: {lengths}"

    # Optionnel : vous pouvez vérifier des propriétés additionnelles, par exemple >= 1
    assert lengths[0] >= 1, f"Expected at least one subset per simulation for params {params}"