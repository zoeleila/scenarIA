import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd

from scenarIA.src.utils.datautils import dataset_xr_formatting, weighted_global_mean

def plot_mean_std_simulations(files_list,
                                var_name,
                                labels,
                                colors=None,
                                vmin=None,
                                vmax=None,
                                title=None,
                                save_path=None):


    plt.figure(figsize=(6, 4)) 
    for i, file in enumerate(files_list):
        ds = xr.open_dataset(file)
        try:
            ds = ds.groupby('time.year').mean().rename({'year':'time'})
        except:
            pass
        print('timeok')
        ds = dataset_xr_formatting(ds)
        unit = ds[var_name].attrs["units"]
        ds = weighted_global_mean(ds)
        print('meanok')
        # TODO : add climatology file
        time = ds.time.values
        if 'member' in ds.dims:
            mean = ds[var_name].mean(['member']).values
            std = ds[var_name].std('member').values
        else:
            mean = ds[var_name].values
            std = 0
        ds.close()
        if colors:
            plt.plot(time, mean, label=labels[i], color=colors[i])
            plt.fill_between(time, mean - std, mean + std, alpha=0.2, color=colors[i])
        else:
            plt.plot(time, mean, label=labels[i])
            plt.fill_between(time, mean - std, mean + std, alpha=0.2)
        
    plt.grid(axis='y', alpha = 0.5)
    plt.ylabel(f'{var_name} {unit}')
    plt.ylim(vmin, vmax)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path / f'{var_name}_temporal_evolution.png')