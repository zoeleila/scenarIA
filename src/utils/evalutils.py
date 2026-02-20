from logging import config
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import matplotlib.colors as colors
import cartopy.feature as cfeature
import torch

from scenarIA.src.utils.datautils import weighted_global_mean
from scenarIA.src.utils.metrics import NRMSE_ClimateBench, NRMSE_g_ClimateBench, NRMSE_s_ClimateBench
import csv


class EvaluationPlots():
    """Class for plotting evaluation metrics for 2D times series data.
    Data is assumed to be in the shape (time, lat, lon) or (time, y, x)."""
    def __init__(self,
                 simulation_name:str=None,
                 var_name:str=None,
                 domain=[0., 360., -90., 90.],
                 projection=ccrs.PlateCarree(),
                 config_plots:dict=None,
                 unit:str=None):
        self.simulation_name = simulation_name
        self.var_name = var_name
        self.domain = domain
        self.projection = projection
        self.config_plots = config_plots[var_name]
        if self.config_plots:
            self.unit = self.config_plots['unit']
        else:
            self.unit = unit
        print("evalllll")

    def plot_time_series(self, y_true, y_pred, title=None, save_path=None):
        """Plot time series of true vs predicted values."""
        _, ax = plt.subplots(figsize=(8,5))
        time = np.arange(y_true.shape[0])
        ax.plot(time, np.mean(y_true, axis=(1,2)), label='True', color='blue')
        ax.plot(time, np.mean(y_pred, axis=(1,2)), label='Predicted', color='orange')
        ax.set_xlabel('Time')
        ax.set_ylabel(f'{self.var_name} ({self.unit})')
        ax.legend()
        ax.set_title(title)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def plot_spatial_map(self, 
                         y_true, 
                         y_pred, 
                         time_index=None,
                         title=None,
                         save_path=None,
                         cmap='viridis',
                         no_limits=False):
        """Plot spatial map of true vs predicted values at a specific time index."""
        if time_index is None:
            y_true = np.mean(y_true, axis=0)
            y_pred = np.mean(y_pred, axis=0)
        else:
            y_true = y_true[time_index]
            y_pred = y_pred[time_index]
        fig, axes = plt.subplots(1, 2, figsize=(8,4), subplot_kw={'projection': ccrs.Robinson() })
        #vmin, vmax = y_true.min(), y_true.max()

        if self.config_plots:
            cmap = self.config_plots['colors']['values']
            if no_limits is False:
                lim = self.config_plots['lim']['values']
                levels = np.linspace(lim[0], lim[1], 8)
        else: 
            lim = [None, None]
            levels=None

        for i, data in enumerate([y_true, y_pred]):
            data = np.flip(data, axis=0)
            ax = axes[i]
            cs = ax.contourf(data,
                     cmap=cmap,
                     levels=levels,
                     extent=self.domain,
                     transform=self.projection
                    )

            axes[i].add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1, zorder=10)
            axes[i].add_feature(cfeature.BORDERS, linestyle='--', linewidth=1, edgecolor='gray', zorder=10)
            var_name = self.var_name
            cbar = fig.colorbar(cs, ax=ax, shrink=0.7,
                            orientation='horizontal',
                            location='bottom',
                            pad=0.05,
                            aspect=30, 
                            label=f'{var_name} ({self.unit}) at t={time_index}' if time_index is not None else f'{var_name} ({self.unit})')
            
            if self.config_plots and no_limits is False:
                cbar.set_ticks(np.linspace(lim[0], lim[1], 5))
                cbar.set_ticklabels([str(round(float(i), 1)) for i in np.linspace(lim[0], lim[1], 5)])
            
        plt.tight_layout()
    
        axes[0].set_title('True Values')
        axes[1].set_title('Predicted Values')
        fig.suptitle(title)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def plot_error_maps(self,
                        y_true,
                        y_pred,
                        title=None,
                        save_path=None,
                        no_limits=False):
        """
        y [time, lat, lon]
        Plot spatial maps of error metrics (MAE, RMSE, R2, temporal correlation) between true and predicted values."""
        me_map = np.mean(y_pred - y_true, axis=0)
        mae_map = np.mean(np.abs(y_pred - y_true), axis=0)
        rmse_map = np.sqrt(np.mean((y_pred - y_true)**2, axis=0))
        temporal_corr_map = np.array([[np.corrcoef(y_true[:, i, j], y_pred[:, i, j])[0,1] 
                                      for j in range(y_true.shape[2])]
                                      for i in range(y_true.shape[1])])
        metrics = {'mean_error' : me_map,
                   'mae': mae_map, 'rmse': rmse_map, 
                   'temporal_correlation': temporal_corr_map}
        cmap_dict = {'mean_error': 'coolwarm', 
                'mae': 'viridis', 
                'rmse': 'viridis', 
                'temporal_correlation': 'coolwarm'}
        names = {'mean_error': 'Mean Error', 
                 'mae': 'Mean Absolute Error', 
                 'rmse': 'Root Mean Squared Error', 
                 'temporal_correlation': 'Temporal Correlation'}
        fig, axes = plt.subplots(2, 2, figsize=(8,6), subplot_kw={'projection': ccrs.Robinson() })
        for ax, (metric_name, data) in zip(axes.ravel(), metrics.items()):
            cmap = cmap_dict[metric_name]
            if self.config_plots and no_limits is False:
                lim = self.config_plots['lim'][metric_name]
                levels = np.linspace(lim[0], lim[1], 8)
            else: 
                lim = [None, None]
                levels=None
            cs = ax.contourf(data,
                     cmap=cmap,
                     levels=levels,
                     extent=self.domain,
                     transform=self.projection
                    )
            ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1, zorder=10)
            ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=1, edgecolor='gray', zorder=10)
            name = names[metric_name]
            cbar = fig.colorbar(cs, ax=ax, shrink=0.7,
                         orientation='horizontal',
                         location='bottom',
                         pad=0.05,
                         aspect=30, 
                         label=f'{self.var_name} {name}' if metric_name=='temporal_correlation' else f'{self.var_name} {name} ({self.unit})')
            if self.config_plots and no_limits is False:
                cbar.set_ticks(np.linspace(lim[0], lim[1], 5))
                cbar.set_ticklabels([str(round(float(i), 1)) for i in np.linspace(lim[0], lim[1], 5)])
        plt.tight_layout(pad=1.5)
        fig.suptitle(title)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def plot_diff_maps(self,
                        y_true,
                        y_pred_mean,
                        y_pred_std,
                        title=None,
                        save_path=None,
                        no_limits=False):
        """Plot spatial maps of true and predicted values (mean + std)."""

        if y_true.ndim == 3:
            y_true = y_true.mean(axis=0)
            y_pred_mean = y_pred_mean.mean(axis=0)
            y_pred_std = y_pred_std.mean(axis=0)

        if self.config_plots:
            cmap_dict = {'Prediction mean' : self.config_plots['cmap']['values'], 
                 'True' : self.config_plots['cmap']['values'],
                 'Prediction std' : self.config_plots['cmap']['std'], 
                 'Prediction - True': self.config_plots['cmap']['mean_error']}
            lims = {'Prediction mean' : self.config_plots['lim']['values'], 
                 'True' : self.config_plots['lim']['values'],
                 'Prediction std' : [y_pred_std.min(), y_pred_std.max()], 
                 'Prediction - True': self.config_plots['lim']['mean_error']}
        values = {'Prediction mean' : y_pred_mean, 
                 'True' : y_true,
                 'Prediction std' : y_pred_std, 
                 'Prediction - True': y_pred_mean - y_true}
        fig, axes = plt.subplots(2, 2, figsize=(8,6), subplot_kw={'projection': ccrs.Robinson() })
        for ax, (name, map) in zip(axes.ravel(), values.items()):
            cmap = cmap_dict[name]
            if self.config_plots and no_limits is False:
                lim = lims[name]
                if lim[0] is not None:
                    levels = np.linspace(lim[0], lim[1], 11)
            else: 
                lim = [None, None]
                levels=None
            print(map.shape)
            cs = ax.contourf(map,
                     cmap=cmap,
                     levels=levels,
                     extent=self.domain,
                     transform=self.projection,
                     extend='both'
                    )
            ax.set_title(name)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8, alpha=0.7)
            cbar = fig.colorbar(cs, ax=ax, shrink=0.7,
                         orientation='horizontal',
                         location='bottom',
                         pad=0.05,
                         aspect=30, 
                         label= f'{self.var_name} ({self.unit})')
            if self.config_plots and no_limits is False:
                cbar.set_ticks(np.linspace(lim[0], lim[1], 5))
                cbar.set_ticklabels([str(round(float(i), 1)) for i in np.linspace(lim[0], lim[1], 5)])
        plt.tight_layout(pad=1.5)
        fig.suptitle(title)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
def compare_temporal_profiles(y, y_hat_dict, t, var_name, point=None, config_plots=None, title=None, save_dir=None):

    lats = np.linspace(-90, 90, y.shape[-2])
    unit = config_plots[var_name]['unit'] if config_plots else ''

    if point:
        y = y[..., point[0], point[1]]
    else:
        y = weighted_global_mean(y, lats=lats)
    vmin = y.min()*0.8 if (y.min() > 0) else y.min()*1.2
    vmax = y.max()*0.8 if (y.max() < 0) else y.max()*1.2

    plt.figure(figsize=(6,4))
    for test, y_hat in y_hat_dict.items():
        if len(y_hat) > 1:
            y_hat = np.stack(y_hat, axis=0)
            if point:
                y_hat = y_hat[..., point[0], point[1]]
            else:
                y_hat = weighted_global_mean(y_hat, lats)
            y_hat_mean = y_hat.mean(axis=0)
            y_hat_std = y_hat.std(axis=0)
        else:
            y_hat_mean = y_hat[0]
            if point:
                y_hat_mean = y_hat_mean[..., point[0], point[1]]
            else:
                y_hat_mean = weighted_global_mean(y_hat_mean, lats)
            y_hat_std = np.zeros_like(y_hat_mean)

        line, = plt.plot(t, y_hat_mean, label=f'pred {test}')
        color = line.get_color()
        plt.fill_between(t, y_hat_mean - 2*y_hat_std, y_hat_mean + 2*y_hat_std, color=color, alpha=0.1)
        
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
        y_hat_mean = y_hat[-21:,:,:]
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


def compare_metric_maps(y, y_hat_dict, t, var_name,
                      periods=[['2020', '2050'],['2070', '2100']],
                      config_plots=None,
                      title=None,
                      save_dir=None):
    tests = list(y_hat_dict.keys())
    n_tests = len(tests)
    n_periods = len(periods)

    for test, y_hat in y_hat_dict.items():
        if len(y_hat) > 1:
            y_hat_dict[test] = np.stack(y_hat, axis=0).mean(axis=0)
        else:
            y_hat_dict[test] = y_hat[0]
    config_plots = config_plots[var_name] if config_plots else None
    metrics = ['rmse', 'mean_error', 'error']
    
    for metric in metrics:
        cnorm = None
        if config_plots:          
            vmin = config_plots['lim'][metric][0]
            vmax = config_plots['lim'][metric][1]
            levels = np.linspace(vmin, vmax, 11)
            cmap = config_plots['cmap'][metric]
            if cmap == 'coolwarm':
                levels=np.linspace(vmin, vmax, 8)
        else:
            vmin, vmax = None, None
            levels = None
            cmap = 'viridis'

        fig, axes = plt.subplots(n_periods, n_tests,
                                figsize=(3*n_tests, 2*n_periods),
                                subplot_kw={'projection': ccrs.Robinson()},
                                squeeze=False)
        plt.suptitle(title)

        for i, (start, end) in enumerate(periods):
            start_date = np.datetime64(f"{start}-01-01")
            end_date   = np.datetime64(f"{end}-12-31")
            mask = (t >= start_date) & (t <= end_date)

            for j, test in enumerate(tests):
                y_hat = y_hat_dict[test]
            
                y_p = y[mask]
                y_hat_p = y_hat[mask]

                index = None
                if metric == 'rmse':
                    map = np.sqrt(np.mean((y_p - y_hat_p)**2, axis=0))
                    if var_name == 'pr':
                        index = np.where(map > 0.5)
                    elif var_name == 'tas':
                        index = np.where(map > 0.6)
                elif metric == 'mean_error':
                    map = np.mean((y_hat_p - y_p), axis=0)
                    if var_name == 'pr':
                        index = np.where(np.abs(map) > 0.4)
                    elif var_name == 'tas':
                        index = np.where(np.abs(map) > 0.5)
                elif metric == 'error':
                    map = np.mean(y_hat_p, axis=0) - np.mean(y_p, axis=0)

                # Write index list to CSV for the last period (i) and last test (j)
                if (index is not None) and (i == n_periods - 1) and (j == n_tests - 1):
                    # Prepare path
                    fname = save_dir / f"{metric}_indices_{var_name}_{start}-{end}.csv"
                    # Zip indices and write rows
                    lat_idxs = index[0].tolist()
                    lon_idxs = index[1].tolist()
                    with open(str(fname), 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(['metric', 'lat_idx', 'lon_idx', 'value'])
                        for r, c in zip(lat_idxs, lon_idxs):
                            writer.writerow([metric, int(r), int(c), map[int(r), int(c)]])
                
                ax = axes[i, j]
                cs = ax.contourf(map,
                    transform=ccrs.PlateCarree(),
                    cmap=cmap,
                    norm=cnorm,
                    levels=levels,
                    vmin=vmin,
                    vmax=vmax,
                    extent=[0., 360., -90., 90.],
                    extend='both'
                    )

                ax.add_feature(cfeature.COASTLINE, linewidth=0.8, alpha=0.7)

                if i == 0:
                    ax.set_title(test, fontsize=11)
                if j == 0:
                    ax.text(-0.10, 0.5, f"{start}-{end}",
                            transform=ax.transAxes,
                            rotation=90,
                            va='center',
                            ha='right',
                            fontsize=11,
                            fontweight='bold')

        cbar_ax = fig.add_axes([0.90, 0.2, 0.02, 0.6])
        cbar = fig.colorbar(cs, cax=cbar_ax)
        if levels is not None:
            cbar.set_ticks(np.linspace(vmin, vmax, 5))
            cbar.set_ticklabels([str(round(float(i), 1)) for i in np.linspace(vmin, vmax, 5)])

        cbar.set_label(f"{metric} {var_name}")
        plt.subplots_adjust(hspace=0, wspace=0.05, right=0.88)
        tests_str = "_".join(tests)
        plt.savefig(save_dir / f'{metric}_maps_{var_name}_{tests_str}.png')