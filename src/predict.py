import yaml

from scenarIA.src.utils.settings import CONFIG_DIR
from scenarIA.src.evaluation.compare import predict

if __name__=='__main__':
    with open(CONFIG_DIR / 'runs.yaml') as file:
        runs = yaml.safe_load(file)
    runs_dict = runs['predict']
    y_all, y_hat_dict, t_all, infos = predict(runs_dict, seeds_mean=True)
    # y_all on s'en fout, xr.dataset dircetment ? pas de loop dans la fonction predict ? 
    # une fonction compare qui inclut une fonction predict ???