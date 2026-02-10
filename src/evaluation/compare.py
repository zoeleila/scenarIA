import yaml
import glob
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd


from scenarIA.src.utils.settings import CONFIG_DIR, GRAPHS_DIR, RUNS_DIR
from scenarIA.src.data.dataloader import get_dataloaders
from scenarIA.src.data.lightning_module import scenarIALightningModule
from scenarIA.src.utils.datautils import weighted_global_mean
from scenarIA.src.utils.metrics import NRMSE_ClimateBench

def test(runs_dict):
    # TODO adapt to multi variate ??
    y_all = []
    t_all = []
    y_hat_dict = {}
    for i, test_name in enumerate(runs_dict.keys()):
        runs_list = runs_dict[test_name]
        y_hat_all_seeds = [] # if different seeds
        for seed, run_dir in enumerate(runs_list):
            checkpoint_dir = glob.glob(str(RUNS_DIR / run_dir / 'checkpoints/best-checkpoint*.ckpt'))[0]
            model = scenarIALightningModule.load_from_checkpoint(checkpoint_dir, map_location='cpu')
            model.eval()
            hparams = model.hparams['config']
            test_dataloader = get_dataloaders('test', config=hparams)

            y_hat_all = []
            for batch in tqdm(test_dataloader, desc="Computing stats from dataloader"):
                if i == 0 and seed == 0:
                    x, y, t = batch
                    y_all.append(y)
                    t_all.append(t)
                else:
                    x, _, _ = batch
                x = x.float().to('cpu')
                with torch.no_grad():
                    y_hat = model(x).cpu()
                y_hat_all.append(y_hat)
            y_hat_all = torch.stack(y_hat_all, axis=0).view(-1, hparams['train']['img_size'][0], hparams['train']['img_size'][1])
            y_hat_all_seeds.append(y_hat_all)
        y_hat_dict[test_name] = y_hat_all_seeds

    y_all = torch.cat(y_all, dim=0).squeeze().numpy()
    t_all = torch.cat(t_all, dim=0).numpy()
    t_all = np.array([np.datetime64(datetime(year, month, day)) for year, month, day, *_ in t_all])
    t_all = pd.to_datetime(t_all, format="%Y-%m-%d")
    infos = {'outputs': hparams['train']['outputs'],
             'simus_test': hparams['train']['simus_test']}
    return y_all, y_hat_dict, t_all, infos

# function plot : y [t, lat, lon], y_hat {'test_name' : [[t, lat, lon], ...]}, t [t]

def compare_temporal_profiles(y, y_hat_dict, t, var_name, config_plots=None, title=None, save_dir=None):

    lats = np.linspace(-90, 90, y.shape[-2])
    unit = config_plots[var_name]['unit'] if config_plots else ''

    y= weighted_global_mean(y, lats=lats)
    vmin = y.min()*0.9
    vmax = y.max()*1.1

    plt.figure(figsize=(6,4))
    for test, y_hat in y_hat_dict.items():
        if len(y_hat) > 0:
            y_hat = torch.stack(y_hat, dim=0)
            y_hat_mean = y_hat.mean(axis=0)
            y_hat_std = y_hat.std(axis=0)
        else:
            y_hat_mean = y_hat[0]
            y_hat_std = torch.zeros_like(y_hat_mean)

        y_hat_mean = weighted_global_mean(y_hat_mean, lats=lats)
        y_hat_std = weighted_global_mean(y_hat_std, lats=lats)
        line, = plt.plot(t, y_hat_mean, label=test)
        color = line.get_color()
        plt.fill_between(t, y_hat_mean - y_hat_std, y_hat_mean + y_hat_std, color=color, alpha=0.1)
    plt.plot(t, y, label='true', color='k')
    plt.ylim(vmin, vmax)
    plt.xlabel('Time')
    plt.ylabel(f'{var_name} {unit}')
    plt.legend()
    plt.title(title)
    plt.savefig(save_dir)

def compare_metrics(y, y_hat_dict, var_name, config_plots=None, title=None, save_dir=None):
    lats = np.linspace(-90, 90, y.shape[-2])
    
    metric_dict = {'nrmse':{}}
    for test, y_hat in y_hat_dict.items():  
        y_hat = torch.stack(y_hat, dim=0)
        y_hat_mean = y_hat.mean(axis=0)
        metric_dict['nrmse'][test] = NRMSE_ClimateBench(torch.tensor(y_hat_mean), 
                                                        torch.tensor(y), 
                                                        torch.tensor(lats)) # TODO : add metrics ?
        print(f'NRMSE {test} = ', metric_dict['nrmse'][test])

    for metric in metric_dict:
        plt.figure(figsize=(8,4))
        plt.bar(metric_dict[metric].keys(), metric_dict[metric].values())
        plt.grid(axis='y', alpha=0.5)
        plt.title(title)
        plt.ylabel(f'{var_name} {metric}') # TODO add unit if unit
        plt.savefig(save_dir / f'{metric}_bar_plot_{var_name}.png')



if __name__=='__main__':
    with open(CONFIG_DIR / 'runs.yaml') as file:
        runs = yaml.safe_load(file)

    runs_dict = runs['compare']['runs_to_compare']
    y, y_hat_dict, t, infos = test(runs_dict)
    compare_metrics(y, y_hat_dict, var_name='tas', 
                    title='ssp245 (MPI-ESM1-2-LR annual)', 
                    save_dir=GRAPHS_DIR/'runs/MPI-ESM1-2-LR/annual/exp1/')