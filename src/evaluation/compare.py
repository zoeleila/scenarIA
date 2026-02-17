from logging import config
import yaml
import glob
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from scenarIA.src.utils.settings import CONFIG_DIR, GRAPHS_DIR, RUNS_DIR
from scenarIA.src.data.dataloader import get_dataloaders
from scenarIA.src.data.lightning_module import scenarIALightningModule
from scenarIA.src.utils.datautils import weighted_global_mean
from scenarIA.src.utils.metrics import NRMSE_ClimateBench, NRMSE_g_ClimateBench, NRMSE_s_ClimateBench
from scenarIA.src.utils.plotutils import EvaluationPlots
from utils.plotutils import plot_test

def test(runs_dict, eval_func=None, save_dir=None):
    # TODO adapt to multi variate ??
    y_all = []
    t_all = []
    y_hat_dict = {}

    for i, test_name in enumerate(runs_dict.keys()):
        runs_list = runs_dict[test_name]
        y_hat_all_seeds = [] # if different seeds
        for seed, run_dir in enumerate(runs_list):
            print(run_dir)
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
            y_hat_all = torch.stack(y_hat_all, axis=0).view(-1, hparams['train']['img_size'][0], hparams['train']['img_size'][1]).numpy()
            y_hat_all_seeds.append(y_hat_all)
        if eval_func is not None:
            eval_func.plot_diff_maps(torch.cat(y_all, dim=0).squeeze().numpy()[-21:,:,:], 
                                    np.stack(y_hat_all_seeds, axis=0).mean(axis=0)[-21:,:,:], 
                                    np.stack(y_hat_all_seeds, axis=0).std(axis=0)[-21:,:,:], 
                                    title=f'{test_name}', 
                                    save_path=save_dir/f'diff_maps_{test_name}.png')
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
            y_hat = np.stack(y_hat, axis=0)
            y_hat = weighted_global_mean(y_hat, lats)
            y_hat_mean = y_hat.mean(axis=0)
            y_hat_std = y_hat.std(axis=0)
        else:
            y_hat_mean = y_hat[0]
            y_hat_mean = weighted_global_mean(y_hat_mean, lats)
            y_hat_std = np.zeros_like(y_hat_mean)

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
    y = y[-21:,:,:]
    metric_dict = {'nrmse':{},
                   'srmse' : {},
                   'grmse': {}}
    for test, y_hat in y_hat_dict.items():  
        y_hat = np.stack(y_hat, axis=0)
        y_hat_mean = y_hat.mean(axis=0)[-21:,:,:]
        metric_dict['nrmse'][test] = NRMSE_ClimateBench(torch.tensor(y_hat_mean), 
                                                        torch.tensor(y), 
                                                        torch.tensor(lats))
        metric_dict['srmse'][test] = NRMSE_s_ClimateBench(torch.tensor(y_hat_mean), 
                                                        torch.tensor(y), 
                                                        torch.tensor(lats),
                                                        normalize=False)
        metric_dict['grmse'][test] = NRMSE_g_ClimateBench(torch.tensor(y_hat_mean), 
                                                        torch.tensor(y), 
                                                        torch.tensor(lats),
                                                        normalize = False)
        ## add ACC

    for metric in metric_dict:
        test_names = metric_dict[metric].keys()
        metric_values = metric_dict[metric].values()
        print(f'{test_names} = {metric_values}')
        plt.figure(figsize=(8,4))
        plt.bar(test_names, metric_values)
        plt.grid(axis='y', alpha=0.5)
        plt.title(title)
        plt.ylabel(f'{var_name} {metric}') # TODO add unit if unit
        plt.savefig(save_dir / f'{metric}_bar_plot_{var_name}_notnorm.png')

def compare_rmse_maps(y, y_hat_dict, t, var_name, 
                      periods=[['2015', '2040'],['2041', '2070'], ['2071', '2100']],
                      save_dir=None):
    tests = list(y_hat_dict.keys())
    n_tests = len(tests)
    n_periods = len(periods)

    for test, y_hat in y_hat_dict.items():
        if len(y_hat) > 1:
            y_hat_dict[test] = np.stack(y_hat, axis=0).mean(axis=0)
        else:
            y_hat_dict[test] = y_hat[0]

    vmin = 0
    vmax = None

    #levels = np.linspace(vmin, vmax, 11)

    fig, axes = plt.subplots(n_tests, n_periods,
                             figsize=(3*n_periods, 2*n_tests),
                             subplot_kw={'projection': ccrs.Robinson()},
                             squeeze=False)

    for i, test in enumerate(tests):
        y_hat = y_hat_dict[test]

        for j, (start, end) in enumerate(periods):

            start_date = np.datetime64(f"{start}-01-01")
            end_date   = np.datetime64(f"{end}-12-31")

            mask = (t >= start_date) & (t <= end_date)
            y_p = y[mask]
            y_hat_p = y_hat[mask]

            rmse_map = np.sqrt(np.mean((y_p - y_hat_p)**2, axis=0))

            ax = axes[i, j]
            cs = ax.imshow(np.flip(rmse_map,axis=0),
                   transform=ccrs.PlateCarree(),
                   cmap="Reds",
                   #levels=levels,
                   vmin=vmin,
                   vmax=vmax,
                   extent=[0., 360., -90., 90.],
                   #extend='both'
                   )

            ax.add_feature(cfeature.COASTLINE, linewidth=0.8, alpha=0.7)

            if i == 0:
                ax.set_title(f"{start}-{end}", fontsize=11)
            if j == 0:
                ax.text(-0.10, 0.5, test,
                        transform=ax.transAxes,
                        rotation=90,
                        va='center',
                        ha='right',
                        fontsize=11,
                        fontweight='bold')

    cbar_ax = fig.add_axes([0.90, 0.2, 0.02, 0.6])
    cbar = fig.colorbar(cs, cax=cbar_ax)
    #cbar.set_ticks(np.linspace(vmin, vmax, 5))
    #cbar.set_ticklabels([str(round(float(i), 1)) for i in np.linspace(vmin, vmax, 5)])

    cbar.set_label(f"RMSE {var_name}")
    plt.subplots_adjust(hspace=0, wspace=0.05, right=0.88)
    plt.savefig(save_dir)


if __name__=='__main__':
    with open(CONFIG_DIR / 'runs.yaml') as file:
        runs = yaml.safe_load(file)
    with open(CONFIG_DIR / 'plots.yaml') as file:
        config_plots = yaml.safe_load(file)

    #eval_func = EvaluationPlots(simulation_name='ssp245', var_name='tas', config_plots=config_plots)
    runs_dict = runs['compare']['runs_to_compare']
    y, y_hat_dict, t, infos = test(runs_dict)

    compare_rmse_maps(y, y_hat_dict, t, var_name='pr', 
                      periods=[['2015', '2040'],['2041', '2070'], ['2071', '2100']],
                      save_dir=GRAPHS_DIR/'runs/MPI-ESM1-2-LR/annual/exp1/rmse_maps_pr_seq_lengthim.png')

    