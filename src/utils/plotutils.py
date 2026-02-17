import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from pathlib import Path
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import gridspec
import yaml


from scenarIA.src.utils.settings import GRAPHS_DIR, CONFIG_DIR
from scenarIA.src.utils.datautils import dataset_xr_formatting, weighted_global_mean

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
                     transform=self.projection
                    )
            ax.set_title(name)
            ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1, zorder=10)
            ax.add_feature(cfeature.BORDERS, linestyle='--', linewidth=1, edgecolor='gray', zorder=10)
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

def plot_mean_std_simulations(files_dict,
                                var_name,
                                colors_dict=None,
                                vmin=None,
                                vmax=None,
                                time_res = 'Y',
                                title=None,
                                save_path=None):

    
    plt.figure(figsize=(6, 4)) 
    for simu, files in files_dict.items():
        if isinstance(files, str) or isinstance(files, Path):
            files = [files]
        color = colors_dict[simu] if colors_dict else None
        time = []
        mean = []
        std = []
        for file in files:
            if time_res == 'm':
                ds = xr.open_dataset(file, chunks= {'time':12})
                ds = dataset_xr_formatting(ds, original_time_format = "%Y%m")
                ds = ds[var_name].groupby('time.year').mean('time').rename({'year': 'time'})
        
            else:
                ds = xr.open_dataset(file)
                ds = dataset_xr_formatting(ds, original_time_format = "%Y")
                ds = ds[var_name]
            unit = ds.attrs["units"]
            ds = weighted_global_mean(ds)

            # TODO : add climatology file
            time.append(ds.time.values)
            if 'member' in ds.dims:
                mean.append(ds.mean('member').values)
                std.append(ds.std('member').values)
            else:
                mean.append(ds.values)
                std.append(np.zeros_like(ds.values))
            ds.close()
        time = np.concatenate(time)
        mean = np.concatenate(mean)
        std = np.concatenate(std)
        plt.plot(time, mean, label=simu, color=color)
        plt.fill_between(time, mean - std, mean + std, alpha=0.2, color=color)

    plt.grid(axis='y', alpha = 0.5)
    plt.ylabel(f'{var_name} {unit}')
    plt.ylim(vmin, vmax)
    plt.legend(loc ='upper left')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)

def plot_hist(data, label=None, title=None, save_dir=None, color='blue',**kwargs):
    _ = plt.figure(figsize=(4, 4))
    plt.hist(data, 
             histtype='stepfilled', 
             density=True, 
             bins=30, 
             linewidth=2, 
             color=color, 
             label=label)
    plt.axvline(np.mean(data), linestyle='--', linewidth=2, color=color)
    plt.xlabel(label)
    plt.title(title)
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir)

def plot_map_image(var,
                    ax = None,
                    var_desc: str = None,
                    cmap: str = 'OrRd',
                    vmin: float = None,
                    vmax: float = None,
                    domain: list = [0, 360, -90, 90],
                    fig_projection = ccrs.PlateCarree(),
                    data_projection = ccrs.PlateCarree(),
                    title: str = None,
                    save_dir: str = None
                ):
    """
    Plots a 2D map image using the provided data and configurations.
    Args:
        var (Any): The 2D array-like data to be plotted.
        var_desc (str, optional): Description of the variable to be used as the colorbar label. Defaults to None.
        cmap (str, optional): Colormap to be used for the plot. Defaults to 'OrRd'.
        vmin (float, optional): Minimum value for the color scale. Defaults to None.
        vmax (float, optional): Maximum value for the color scale. Defaults to None.
        domain (list, optional): List defining the spatial extent of the plot in the format [min_lon, max_lon, min_lat, max_lat]. Defaults to None.
        fig_projection (Any, optional): Cartopy projection for the figure. Defaults to ccrs.PlateCarree().
        data_projection (Any, optional): Cartopy projection for the data. Defaults to ccrs.PlateCarree().
        title (str, optional): Title of the plot. Defaults to None.
        save_dir (str, optional): Path to save the plot as an image file. If None, the function returns the figure and axis objects. Defaults to None.
    Returns:
        tuple: A tuple containing the figure and axis objects if `save_dir` is None. Otherwise, saves the plot to the specified directory.
    """

    if ax is None:
        fig, axi = plt.subplots(
            figsize=(6,5),
            subplot_kw={"projection": fig_projection}
        )
    else:
        axi=ax

    img = axi.imshow(
        var,
        extent=domain,
        transform=data_projection,
        origin='lower',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    axi.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1, zorder=10, alpha=0.7)

    if ax is not None:
        return axi, img

    cbar = plt.colorbar(img, ax=axi, pad=0.05, shrink=0.8)
    cbar.set_label(label=var_desc, size=14, labelpad=10)
    cbar.ax.tick_params(labelsize=14)
    plt.tight_layout()
    plt.title(title, fontsize=16, pad=10)
    if save_dir is None:
        return fig, axi
    else:
        print('coucou')
        plt.savefig(save_dir)

def plot_multi_samples(data, 
                       n_rows=2, 
                       n_cols=3,
                       labels=None,
                       title=None,
                       unit=None,
                       cmap='OrRd',
                       var_name=None,
                       save_path=None):
    """
    data : [n_examples, lat, lon] or [n_examples, time, lat, lon]
    lebels : [n examples] or [n_examples, time]
    """

    n_examples = data.shape[0]
    is_time_series = (data.ndim == 4)

    vmin = np.nanmin(data)
    vmax = np.nanmax(data)

    if is_time_series:
        n_time = data.shape[1]
        n_cols = n_time
        indices = np.random.choice(n_examples, n_rows, replace=False)
        total_plots = n_rows * n_cols
    else:
        total_plots = n_rows * n_cols
        indices = np.random.choice(n_examples, total_plots, replace=False)
    
    data_selected = data[indices]
    if labels is not None:
        labels_selected = labels[indices] 

    # --- GridSpec AVEC colonne colorbar ---
    fig = plt.figure(figsize=(4*(n_cols+1), 3*n_rows))
    gs = gridspec.GridSpec(
        n_rows, n_cols + 1,
        width_ratios=[1]*n_cols + [0.1],
        wspace=0.05,
        hspace=0.15
    )

    imgs = []

    if is_time_series:
        for i, sample in enumerate(data_selected):
            for t in range(n_cols):
                ax = fig.add_subplot(
                    gs[i, t],
                    projection=ccrs.PlateCarree()
                )

                ax, img = plot_map_image(
                    sample[t],
                    ax=ax,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax
                )
                if labels is not None:
                    ax.set_title(str(labels_selected[i, t]))
                imgs.append(img)

    else:
        for i, image in enumerate(data_selected):
            row = i // n_cols
            col = i % n_cols

            ax = fig.add_subplot(
                gs[row, col],
                projection=ccrs.PlateCarree()
            )

            ax, img = plot_map_image(
                image,
                ax=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax
            )
            if labels is not None:
                ax.set_title(str(labels_selected[i]))
            imgs.append(img)

        for j in range(i + 1, total_plots):
            row = j // n_cols
            col = j % n_cols
            fig.add_subplot(gs[row, col]).axis("off")

    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(
        imgs[0],
        cax=cax,
        orientation='vertical',
        pad=0.1
    )
    cbar.set_label(f'{var_name} {unit}', fontsize=22)
    cbar.ax.tick_params(labelsize=18)

    plt.suptitle(title, fontsize=20)
    plt.tight_layout(rect=[0, 0, 0.97, 0.95])
    if save_path:
        plt.savefig(save_path)

if __name__=='__main__':
    y = np.random.rand(70, 96, 192)
    yhat = np.random.rand(70, 96, 192)
    yhat_std = np.random.rand(70, 96, 192)
    with open(CONFIG_DIR / 'plots.yaml') as file:
        config_plots = yaml.safe_load(file)
    eval = EvaluationPlots(simulation_name='ssp245',
                           var_name='tas',
                           config_plots=config_plots)
    eval.plot_diff_maps(y, yhat, yhat_std, title='ggggg', save_path=GRAPHS_DIR/'tests/test.png')