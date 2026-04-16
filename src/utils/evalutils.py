from logging import config
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import matplotlib.colors as colors
import cartopy.feature as cfeature
import torch

from scenarIA.src.utils.datautils import weighted_global_mean, apply_moving_average, get_statistics_from_bootstrap
from scenarIA.src.utils.metrics import NRMSE_ClimateBench, NRMSE_g_ClimateBench, NRMSE_s_ClimateBench
from scenarIA.src.utils.settings import CONFIG_DIR, GRAPHS_DIR, RUNS_DIR, DATASET_DIR
from scenarIA.src.utils.plotutils import plot_map_image, plot_map_contour

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

    def plot_all_maps(self,
                      y_true,
                      y_pred_list,
                      title=None,
                      save_path=None,
                      max_cols: int = 4):
        """ Plot spatial maps with y_true and all individual predictions and the mean prediction.
        Arranges maps in a grid with up to max_cols columns so many predictions don't appear in a single row.
        """
        # protect against empty predictions list
        if len(y_pred_list) == 0:
            maps = [y_true]
            names = ['True']
        else:
            pred_mean = np.mean(y_pred_list, axis=0)
            maps = [y_true] + y_pred_list + [pred_mean]
            print(len(maps))
            names = ['True'] + [f'Pred {i}' for i in range(len(y_pred_list))] + ['Pred mean']

        n = len(maps)
        ncols = min(max_cols, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4 * ncols, 4 * nrows),
                                 subplot_kw={'projection': ccrs.Robinson()},
                                 constrained_layout=True)
        # flatten axes array for easy indexing
        if isinstance(axes, np.ndarray):
            axes_flat = axes.ravel()
        else:
            axes_flat = np.array([axes])

        for i, (map_obj, name) in enumerate(zip(maps, names)):
            ax = axes_flat[i]
            map2d = map_obj.mean(axis=0) if map_obj.ndim == 3 else map_obj

            if name == 'True':
                cmap = self.config_plots['cmap']['values'] if self.config_plots else 'viridis'
                data_to_plot = map2d
                lim = self.config_plots['lim']['values'] if self.config_plots else [None, None]
            else:
                cmap = self.config_plots['cmap']['mean_error'] if self.config_plots else 'coolwarm'
                # show difference to true mean for predictions
                true_mean = y_true.mean(axis=0) if y_true.ndim == 3 else y_true
                data_to_plot = map2d - true_mean
                lim = self.config_plots['lim']['mean_error'] if self.config_plots else [None, None]

            levels = np.linspace(lim[0], lim[1], 11) if lim[0] is not None else None
            cs = ax.contourf(data_to_plot,
                             cmap=cmap,
                             levels=levels,
                             extent=self.domain,
                             transform=self.projection,
                             extend='both')

            ax.set_title(name)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8, alpha=0.7)
            cbar = fig.colorbar(cs, ax=ax, shrink=0.7,
                                orientation='horizontal',
                                location='bottom',
                                pad=0.05,
                                aspect=30,
                                label=f'{self.var_name} ({self.unit})')
            if self.config_plots:
                cbar.set_ticks(np.linspace(lim[0], lim[1], 5))
                cbar.set_ticklabels([str(round(float(i), 1)) for i in np.linspace(lim[0], lim[1], 5)])

        # turn off any unused subplots
        for j in range(n, len(axes_flat)):
            try:
                axes_flat[j].axis('off')
            except Exception:
                pass

        fig.suptitle(title)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


def compare_temporal_profiles(y, y_hat_dict, t, var_name, lats, point=None, window_size:int=None, config_plots=None, title=None, save_dir=None):

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

    if window_size:
        y_window = apply_moving_average(y, window_size)
        t_window = t[window_size//2 : -window_size//2 + 1] if window_size % 2 == 0 else t[window_size//2 : -(window_size//2)]
        plt.plot(t_window, y_window, label=f'true {window_size}-yr avg.', color='k')
        plt.plot(t, y, label='true', color='k', alpha=0.1)
    else:
        plt.plot(t, y, label='true', color='k')
    plt.ylim(vmin, vmax)
    plt.xlabel('Time')
    plt.ylabel(f'{var_name} {unit}')
    plt.legend()
    plt.title(title)
    plt.savefig(save_dir)

def compare_metrics(y, y_hat_dict, var_name, lats=None, title=None, save_dir=None):
    '''
    Plot bar charts comparing metrics (NRMSE, sRMSE, gRMSE) for different tests. with confidence intervals if multiple runs are provided.

    y: np.array of shape (time, lat, lon)
    y_hat_dict: dict of {test_name: list of np.array of shape (time, lat, lon)}. The list can contain one or more arrays (e.g. for ensemble members or bootstrap samples).
    var_name: name of the variable (e.g. 'tas')
    lats: np.array of shape (lat,) containing the latitudes corresponding to the lat dimension of y and y_hat. If None, it is assumed to be equally spaced from -90 to 90.
    title: title of the plot
    save_dir: directory to save the plot
    '''
    lats = np.linspace(-90, 90, y.shape[-2]) if lats is None else lats
    y = y[-21:,:,:]

    metric_dict = {'nrmse':{},
                   'srmse' : {},
                   'grmse': {}}

    for test, y_hat in y_hat_dict.items(): 
        if isinstance(y_hat, list) and len(y_hat) > 1:
            y_hat_stack = np.stack(y_hat, axis=0)[:,-21:,:,:]
            res = get_statistics_from_bootstrap(y_hat_stack, n_bootstrap=1000).bootstrap_distribution # shape (time, lat, lon, n_bootstrap)
            res = np.transpose(res, (3, 0, 1, 2))
            metric_dict['nrmse'][test] = [NRMSE_ClimateBench(torch.tensor(res[i]),
                                                    torch.tensor(y),
                                                    torch.tensor(lats)) for i in range(res.shape[0])]
            metric_dict['srmse'][test] = [NRMSE_s_ClimateBench(torch.tensor(res[i]), 
                                                            torch.tensor(y), 
                                                            torch.tensor(lats),
                                                            normalize=False,
                                                            weights_normalization='mean') for i in range(res.shape[0])]
            metric_dict['grmse'][test] = [NRMSE_g_ClimateBench(torch.tensor(res[i]), 
                                                            torch.tensor(y), 
                                                            torch.tensor(lats),
                                                            normalize = False,
                                                            weights_normalization='mean') for i in range(res.shape[0])]
        
        else: 
            y_hat = y_hat[0] if isinstance(y_hat, list) else y_hat
            y_hat_mean = y_hat[-21:,:,:]
            metric_dict['nrmse'][test] = NRMSE_ClimateBench(torch.tensor(y_hat_mean), 
                                                            torch.tensor(y), 
                                                            torch.tensor(lats))
            metric_dict['srmse'][test] = NRMSE_s_ClimateBench(torch.tensor(y_hat_mean), 
                                                            torch.tensor(y), 
                                                            torch.tensor(lats),
                                                            normalize=False,
                                                            weights_normalization='mean')
            metric_dict['grmse'][test] = NRMSE_g_ClimateBench(torch.tensor(y_hat_mean), 
                                                            torch.tensor(y), 
                                                            torch.tensor(lats),
                                                            normalize = False,
                                                            weights_normalization='mean')
        ## add ACC

    for metric in metric_dict:
        test_names = list(metric_dict[metric].keys())
        if isinstance(metric_dict[metric][test_names[0]], list):
            metrics_values_mean = [torch.tensor(metric_dict[metric][test]).mean().item() for test in test_names]
            metrics_values_lower = [torch.tensor(metric_dict[metric][test]).quantile(0.025).item() for test in test_names]
            metrics_values_upper = [torch.tensor(metric_dict[metric][test]).quantile(0.975).item() for test in test_names]        
            print(f'{test_names} = {metrics_values_mean} with 95% CI [{metrics_values_lower}, {metrics_values_upper}]')
        else:
            metrics_values_mean = [metric_dict[metric][test].item() for test in test_names]
        print(f'{test_names} = {metrics_values_mean}')
        plt.figure(figsize=(8,4))
        if isinstance(metric_dict[metric][test_names[0]], list):
            plt.bar(test_names, metrics_values_mean, yerr=[np.array(metrics_values_mean) - np.array(metrics_values_lower), 
                                                        np.array(metrics_values_upper) - np.array(metrics_values_mean)],
                    capsize=5)
        else:
            plt.bar(test_names, metrics_values_mean)
        plt.grid(axis='y', alpha=0.5)
        plt.title(title)
        plt.ylabel(f'{var_name} {metric}') # TODO add unit if unit
        plt.savefig(save_dir / f'{metric}_bar_plot_{var_name}.png')

def compare_metrics_single_runs(y, y_hat_dict, var_name, lats=None, title=None, save_dir=None):
    lats = np.linspace(-90, 90, y.shape[-2]) if lats is None else lats
    y = y[-21:,:,:]

    metric_dict = {'nrmse':{},
                   'srmse' : {},
                   'grmse': {}}

    for test, y_hat in y_hat_dict.items():  
        print(y_hat[0].shape)
        metric_dict['nrmse'][test] = [NRMSE_ClimateBench(torch.tensor(y_hat[i][-21:,:,:]), 
                                                        torch.tensor(y), 
                                                        torch.tensor(lats)) for i in range(len(y_hat))]
        metric_dict['srmse'][test] = [NRMSE_s_ClimateBench(torch.tensor(y_hat[i][-21:,:,:]), 
                                                        torch.tensor(y), 
                                                        torch.tensor(lats),
                                                        normalize=False,
                                                        weights_normalization='mean') for i in range(len(y_hat))]
        metric_dict['grmse'][test] = [NRMSE_g_ClimateBench(torch.tensor(y_hat[i][-21:,:,:]), 
                                                        torch.tensor(y), 
                                                        torch.tensor(lats),
                                                        normalize = False,
                                                        weights_normalization='mean') for i in range(len(y_hat))]
    
    for metric in metric_dict:
        test_names = metric_dict[metric].keys()
        metric_values = metric_dict[metric].values()
        print(f'{test_names} = {metric_values}')
        plt.figure(figsize=(8,4))
        plt.boxplot(metric_values, labels=test_names)
        plt.grid(axis='y', alpha=0.5)
        plt.title(title)
        plt.ylabel(f'{var_name} {metric}') # TODO add unit if unit
        plt.savefig(save_dir / f'{metric}_box_plot_{var_name}.png')


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
            y_hat = np.stack(y_hat, axis=0)
            y_hat_dict[test] = y_hat.mean(axis=0)
            
        else:
            y_hat_dict[test] = y_hat[0]
    config_plots = config_plots[var_name] if config_plots else None
    metrics = ['rmse', 'mean_error']
    
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

                if metric == 'rmse':
                    map = np.sqrt(np.mean((y_p - y_hat_p)**2, axis=0))

                elif metric == 'mean_error':
                    map = np.mean(y_hat_p - y_p, axis=0)

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
        plt.savefig(save_dir / 'test.png')
        #plt.savefig(save_dir / f'{metric}_maps_{var_name}_{tests_str}.png')

def compare_metric_maps2(y, y_hat_dict, t, var_name,
                        periods=[['2020', '2050'], ['2070', '2100']],
                        config_plots=None,
                        title=None,
                        save_dir=None):
    tests = list(y_hat_dict.keys())
    n_tests = len(tests)
    n_periods = len(periods)
    '''
    for test, y_hat in y_hat_dict.items():
        if len(y_hat) > 1:
            y_hat = np.stack(y_hat, axis=0)
            y_hat_dict[test] = y_hat.mean(axis=0)
            y_hat_std = y_hat.std(axis=0)
        else:
            y_hat_dict[test] = y_hat[0]
            y_hat_std = np.zeros_like(y_hat[0])
    '''

    config_plots = config_plots[var_name] if config_plots else None
    metrics = ['rmse', 'mean_error']

    for metric in metrics:
        cnorm = None
        if config_plots:
            vmin = config_plots['lim'][metric][0]
            vmax = config_plots['lim'][metric][1]
            levels = np.linspace(vmin, vmax, 11)
            cmap = config_plots['cmap'][metric]
            if cmap == 'coolwarm':
                levels = np.linspace(vmin, vmax, 8)
        else:
            vmin, vmax = None, None
            levels = None
            cmap = 'viridis'

        fig, axes = plt.subplots(n_periods, n_tests,
                                figsize=(5 * n_tests, 3.5 * n_periods),
                                subplot_kw={'projection': ccrs.Robinson()},
                                squeeze=False)
        plt.suptitle(title, fontsize=18)

        for i, (start, end) in enumerate(periods):
            start_date = np.datetime64(f"{start}-01-01")
            end_date = np.datetime64(f"{end}-12-31")
            mask = (t >= start_date) & (t <= end_date)

            for j, test in enumerate(tests):

                y_p = y[mask]

                y_hat = y_hat_dict[test]
                if len(y_hat) > 1:
                    y_hat = np.stack(y_hat, axis=0)
                    y_hat[:, :, 0, :] = y[:, 0, :] # to remove after
                    y_hat[:, :, -1, :] = y[:, -1, :] # to remove after
                    y_hat_p = y_hat[:, mask, :, :]
                    
                    ####
                    # calculate robustness mask based on 95% confidence interval from bootstrap
                    if metric == 'mean_error':
                        # bootstrap on the mean prediction across ensemble members
                        y_hat_p_t_mean = y_hat_p.mean(axis=1)
                        res = get_statistics_from_bootstrap(y_hat_p_t_mean, n_bootstrap=1000)
                        ci_lower_bootstrap = res.confidence_interval.low
                        ci_upper_bootstrap = res.confidence_interval.high

                        plot_map_image(ci_lower_bootstrap, 
                                    var_desc=var_name,
                                    cmap=config_plots['cmap']['values'] if config_plots else 'viridis',
                                    vmin=config_plots['lim']['values'][0] if config_plots else None,
                                    vmax=config_plots['lim']['values'][1] if config_plots else None,
                                    title=f'{var_name} Bootstrap quantile 0.025 \n ({test} {start}-{end})', 
                                    save_dir=save_dir / f'bootstrap_lower_{test}_{start}_{end}.png')
                        plot_map_image(ci_upper_bootstrap,
                                    var_desc=var_name,
                                        cmap=config_plots['cmap']['values'] if config_plots else 'viridis',
                                        vmin=config_plots['lim']['values'][0] if config_plots else None,
                                        vmax=config_plots['lim']['values'][1] if config_plots else None,
                                        title=f'{var_name} Bootstrap quantile 0.975 \n ({test} {start}-{end})', 
                                        save_dir=save_dir / f'bootstrap_upper_{test}_{start}_{end}.png')


                        condition = (ci_lower_bootstrap <= y_p.mean(axis=0)) & (y_p.mean(axis=0) <= ci_upper_bootstrap)
                        mask_robust = np.where(condition, np.nan, 1) # to plot hatching only where condition is not met (i.e. where prediction is not robust)
                        print(mask_robust.size, mask_robust[~np.isnan(mask_robust)].sum())

                    y_hat_p = y_hat_p.mean(axis=0)
                else:
                    y_hat = y_hat[0]
                    y_hat_p = y_hat[mask]

                if metric == 'rmse':
                    map = np.sqrt(np.mean((y_p - y_hat_p) ** 2, axis=0))

                elif metric == 'mean_error':
                    map = np.mean(y_hat_p, axis=0) - y_p.mean(axis=0)

                ax = axes[i, j]

                cs = ax.contourf(map,
                                transform=ccrs.PlateCarree(),
                                cmap=cmap,
                                norm=cnorm,
                                levels=levels,
                                vmin=vmin,
                                vmax=vmax,
                                extent=[0., 360., -90., 90.],
                                extend='both')
                if metric == 'mean_error':
                    # Hachurer les points de grille en dehors de l'intervalle de confiance à 95%
                    ax.contourf(mask_robust, colors='none', hatches=['///'], transform=ccrs.PlateCarree(), extent=[0., 360., -90., 90.])
                
                ax.add_feature(cfeature.COASTLINE, linewidth=0.8, alpha=0.7)

                if i == 0:
                    ax.set_title(test, fontsize=16)
                if j == 0:
                    ax.text(-0.10, 0.5, f"{start}-{end}",
                            transform=ax.transAxes,
                            rotation=90,
                            va='center',
                            ha='right',
                            fontsize=16,
                            fontweight='bold')

        cbar_ax = fig.add_axes([0.90, 0.2, 0.02, 0.6])
        cbar = fig.colorbar(cs, cax=cbar_ax)
        if levels is not None:
            cbar.set_ticks(np.linspace(vmin, vmax, 5))
            cbar.set_ticklabels([str(round(float(i), 1)) for i in np.linspace(vmin, vmax, 5)], fontsize=14)

        cbar.set_label(f"{metric} {var_name}", fontsize=14)
        plt.subplots_adjust(hspace=0, wspace=0.05, right=0.88)
        tests_str = "_".join(tests)
        plt.savefig(save_dir / f'{metric}_maps_{var_name}_{tests_str}.png')

if __name__ == "__main__":
    y = np.random.rand(100, 96, 192)  # (time, lat, lon)
    y_hat_dict = {
        'test1': [np.random.rand(100, 96, 192) for _ in range(6)],  # 6 ensemble members
        'test2': [np.random.rand(100, 96, 192) for _ in range(6)],
        'test3': [np.random.rand(100, 96, 192) for _ in range(6)]
    }
    t = np.array([np.datetime64(f"{2000+i}-01-01") for i in range(100)])
    compare_metric_maps2(y, y_hat_dict, t, var_name='tas', title='Test Metric Maps', 
                        save_dir=GRAPHS_DIR)