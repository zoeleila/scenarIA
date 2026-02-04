import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from pathlib import Path
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import gridspec


from scenarIA.src.utils.settings import GRAPHS_DIR
from scenarIA.src.utils.datautils import dataset_xr_formatting, weighted_global_mean

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
        fig, ax = plt.subplots(
            figsize=(6,5),
            subplot_kw={"projection": fig_projection}
        )

    img = ax.imshow(
        var,
        extent=domain,
        transform=data_projection,
        origin='lower',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1, zorder=10, alpha=0.7)

    if ax:
        return ax, img

    cbar = plt.colorbar(img, ax=ax, pad=0.05, shrink=0.8)
    cbar.set_label(label=var_desc, size=14, labelpad=10)
    cbar.ax.tick_params(labelsize=14)
    plt.tight_layout()
    plt.title(title, fontsize=16, pad=10)
    if save_dir is None:
        return fig, ax
    else:
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
    data = np.random.rand(100, 5, 96, 192)
    plot_multi_samples(data, save_path=GRAPHS_DIR/'tests/test.png',
                       n_rows=5)