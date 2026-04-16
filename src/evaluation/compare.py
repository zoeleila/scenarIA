from ast import arg
from os import times
import yaml
import glob
import torch
from tqdm import tqdm
import numpy as np
from datetime import datetime
import argparse
import pandas as pd

from scenarIA.src.utils.settings import CONFIG_DIR, GRAPHS_DIR, RUNS_DIR, DATASET_DIR
from scenarIA.src.data.dataloader import get_dataloaders, get_climatology
from scenarIA.src.data.lightning_module import scenarIALightningModule
from scenarIA.src.utils.evalutils import EvaluationPlots, compare_metric_maps2, compare_metric_maps,compare_metrics, compare_temporal_profiles
from scenarIA.src.predict import predict

def compare(runs_dict, 
            simus_test=None,    
            seeds_mean=False,
            var_name=None, 
            eval_func=None, 
            plot_save_dir=None):
    # TODO adapt to multi variate ??
    y_hat_dict = {}

    for i, test_name in enumerate(runs_dict.keys()):
        runs_list = runs_dict[test_name]
        y_hat_all_seeds = [] # if different seeds
        for seed_ixd, run_dir in enumerate(runs_list):
            print(run_dir)
            if i == 0 and seed_ixd == 0: # only for the first run, to get the true values and time
                y_hat_all, y_all, t_all = predict(run_dir, data_type='test', simus_to_predict=simus_test)
            else:
                y_hat_all, _, _ = predict(run_dir, data_type='test', simus_to_predict=simus_test)
            y_hat_all_seeds.append(y_hat_all)
        if eval_func is not None:
            eval_func.plot_diff_maps(y_all[-21:,:,:], 
                                    np.stack(y_hat_all_seeds, axis=0).mean(axis=0)[-21:,:,:], 
                                    np.stack(y_hat_all_seeds, axis=0).std(axis=0)[-21:,:,:], 
                                    title=f'{test_name} (2080-2100)',
                                    save_path=plot_save_dir/f'diff_maps_{var_name}_{test_name}.png')
            eval_func.plot_all_maps(y_true=y_all[-21:,:,:],
                                    y_pred_list=[y_hat_all_seeds[i][-21:,:,:] for i in range(len(y_hat_all_seeds))],
                                    title=f'{test_name} (2080-2100)',
                                    save_path=plot_save_dir/f'seeds_maps_{var_name}_{test_name}.png')
        if seeds_mean:
            y_hat_all_seeds = np.stack(y_hat_all_seeds, axis=0)
            y_hat_all_seeds = y_hat_all_seeds.mean(axis=0)
        y_hat_dict[test_name] = y_hat_all_seeds

    return y_all, y_hat_dict, t_all

#def load_xr_to_dict(): TODO
    #return y, y_hat_dict, t

#def get_plot_path ????


if __name__=='__main__':
    argparser = argparse.ArgumentParser(description="Compare different runs")
    argparser.add_argument("--simus_test", type=str, default='ssp245')
    argparser.add_argument("--var_name", type=str, default='pr')
    args = argparser.parse_args()
    var_name = args.var_name


    with open(CONFIG_DIR / 'runs.yaml') as file:
        runs = yaml.safe_load(file)
    with open(CONFIG_DIR / 'plots.yaml') as file:
        config_plots = yaml.safe_load(file)

    model_name = runs['model_name']
    timescale = runs['timescale']
    exp = runs['exp']

    graph_dir = GRAPHS_DIR / 'tests'
    # graph_dir = GRAPHS_DIR/f'runs/{model_name}/{timescale}/{exp}'
    title = args.simus_test + ' (' + runs['model_name'] + ' ' + runs['timescale'] + ' ' + runs['exp'] + ')'
    
    runs_dict = runs['compare']['runs_to_compare']
    eval_func = EvaluationPlots(config_plots=config_plots, 
                                simulation_name=args.simus_test, 
                                var_name=var_name)
    y, y_hat_dict, t = compare(runs_dict, 
                               simus_test=args.simus_test,
                               var_name=var_name,
                               eval_func=eval_func,
                               plot_save_dir=graph_dir)

    lats = dict(np.load(DATASET_DIR / model_name / timescale / 'coords.npz', allow_pickle=True))['lat']
    #compare_metrics(y, y_hat_dict, lats=lats, var_name=var_name, title=None, save_dir=graph_dir)
    compare_temporal_profiles(y, y_hat_dict, t, var_name=var_name, lats=lats, config_plots=config_plots,
                              title=title, 
                              save_dir=graph_dir)
    compare_metric_maps2(y, y_hat_dict, t, var_name=var_name, config_plots=config_plots, 
                         title=title, save_dir=graph_dir)

