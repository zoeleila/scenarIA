import yaml
import glob
import torch
from tqdm import tqdm
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
import pandas as pd

from scenarIA.src.utils.settings import CONFIG_DIR, GRAPHS_DIR, RUNS_DIR, DATASET_DIR
from scenarIA.src.utils.evalutils import EvaluationPlots, compare_metric_maps2, compare_metrics2, compare_temporal_profiles
from scenarIA.src.predict import predict
from scenarIA.src.utils.metrics import NRMSE_ClimateBench, NRMSE_g_ClimateBench, NRMSE_s_ClimateBench, LatWeightedRMSEMetric

def compare_tests(runs_dict, 
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
                y_hat_all, y_all, t_all, _ = predict(run_dir, data_type='test', simus_to_predict=simus_test)
            else:
                y_hat_all, _, _, _ = predict(run_dir, data_type='test', simus_to_predict=simus_test)
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

def compare_runs(runs_list, params_list=['learning_rate'], data_type='test', save_dir=None):
    """Fonction à supprimer"""
    
    rows = []  # Liste de dicts, une entrée par run
    
    for run_dir in runs_list:
        version = run_dir.split('/')[-1]
        print(version)
        
        try:
            y_hat_all, y_all, _, hparams = predict(run_dir, data_type='test', best_checkpoint=True)
            print(hparams)
        except Exception as e:
            print(f"Error occurred while predicting: {e}")
            continue
        
        lats = dict(np.load(DATASET_DIR / hparams['data']['dataset_path'] / 'coords.npz', allow_pickle=True))['lat']
        y_hat_all, y_all = y_hat_all[-21:,:,:], y_all[-21:,:,:]
        
        row = {
            'version': version,  # string
            **{param: float(hparams['train'][param]) for param in params_list},  # float
            'nrmse': NRMSE_ClimateBench(torch.tensor(y_hat_all), torch.tensor(y_all), torch.tensor(lats)).item(),
            'srmse': NRMSE_s_ClimateBench(torch.tensor(y_hat_all), torch.tensor(y_all), torch.tensor(lats),
                        normalize=False, weights_normalization='sum').item(),
            'grmse': NRMSE_g_ClimateBench(torch.tensor(y_hat_all), torch.tensor(y_all), torch.tensor(lats),
                        normalize=False, weights_normalization='sum').item(),
            'nsrmse': NRMSE_s_ClimateBench(torch.tensor(y_hat_all), torch.tensor(y_all), torch.tensor(lats),
                        normalize=True, weights_normalization='sum').item(),
            'ngrmse': NRMSE_g_ClimateBench(torch.tensor(y_hat_all), torch.tensor(y_all), torch.tensor(lats),
                        normalize=True, weights_normalization='sum').item()
        }
        print(row)
        rows.append(row)
    
    df = pd.DataFrame(rows)  # Une ligne par run, types corrects
    print(df.dtypes)
    
    if save_dir is not None:
        df.to_csv(save_dir / 'compare_versions_metrics_last_epoch.csv', index=False)  # index=False évite la colonne 'Unnamed: 0'
    
    return df

def compare_hp(runs_list, params_list=['learning_rate'], save_dir=None):
    """ à implémenter directement dans le lightning module ???"""
    rows = []  # Liste de dicts, une entrée par run
    val_rmse = LatWeightedRMSEMetric()

    # Prepare CSV path and load existing versions if any
    csv_path = None
    existing_versions = set()
    if save_dir is not None:
        try:
            csv_path = save_dir / 'compare_versions_val_metrics_best_checkpoint.csv'
            if csv_path.exists():
                df_existing = pd.read_csv(csv_path)
                if 'version' in df_existing.columns:
                    existing_versions = set(df_existing['version'].astype(str).tolist())
        except Exception as e:
            print(f"Could not read existing CSV: {e}")
            existing_versions = set()

    for run_dir in runs_list:
        version = run_dir.split('/')[-1]
        if version in existing_versions:
            print(f"{version} already computed, skipping.")
            continue

        print(version)
        
        try:
            y_hat_all, y_all, _, hparams = predict(run_dir, data_type='val', best_checkpoint=True)
            if not all(param in hparams['train'] for param in params_list):
                print(f"Skipping {version}: missing one of params {params_list} in hparams['train']")
                continue
            lats = dict(np.load(DATASET_DIR / hparams['data']['dataset_path'] / 'coords.npz', allow_pickle=True))['lat']
            for time in range(y_hat_all.shape[0]):
                val_rmse.update(torch.tensor(y_hat_all[time]), torch.tensor(y_all[time]), torch.tensor(lats))
            print(f"Validation RMSE for {version}: {val_rmse.compute().item()}")
        except Exception as e:
            print(f"Error occurred while predicting {version}: {e}")
            continue
        
        row = {
            'version': version,  # string
            **{param: float(hparams['train'][param]) for param in params_list},  # float
            'val_rmse': val_rmse.compute().item()
        }
        val_rmse.reset()  # reset pour le prochain run
        print(row)
        rows.append(row)

    # Build dataframe and persist, merging with existing file if present
    df_new = pd.DataFrame(rows)
    if csv_path is not None:
        try:
            if csv_path.exists():
                df_existing = pd.read_csv(csv_path)
                # Avoid duplicates just in case
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined = df_combined.drop_duplicates(subset=['version'], keep='first')
            else:
                df_combined = df_new
            df_combined.to_csv(csv_path, index=False)
            return df_combined
        except Exception as e:
            print(f"Could not write CSV to {csv_path}: {e}")
            return df_new
    else:
        return df_new


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
    #exp = runs['exp']

    graph_dir = GRAPHS_DIR/'runs/MPI-ESM1-2-LR/annual/'
    title = runs['model_name'] + '_' + runs['timescale'] + '_ssp245_configtest_lstm64_epoch=100'

    runs_dict = runs['compare']['runs_to_compare']
    eval_func = EvaluationPlots(config_plots=config_plots,
                                simulation_name=None,
                                var_name=var_name)
    
    y, y_hat_dict, t = compare_tests(runs_dict,
                               simus_test=None,
                               var_name=var_name,
                               eval_func=None,
                               plot_save_dir=None)
    lats = dict(np.load(DATASET_DIR / model_name / timescale / 'coords.npz', allow_pickle=True))['lat']
    compare_metrics2(y, y_hat_dict, lats=lats, var_name=var_name, title=title, save_dir=graph_dir, 
     ensemble_scoring='mean_of_scores')

    