from re import A
import re
from scenarIA.src.utils.settings import GRAPHS_DIR
import xarray as xr
import torch
import pandas as pd
import numpy as np
import glob
from scipy.stats import bootstrap



def standardize_dims_and_coords(ds) :
    # Camille Le Gloannec inspired script
    # GCM models have inconsistent names of dimensions and coordinates, 
    # this function fix that at the dataset level by naming dimensions (x,y) and coordinates (lon,lat).

    dim_mapping = {
        "lon": ["i", "ni", "xh", "x", "nlon"],
        "lat": ["j", "nj", "yh", "y", "nlat"],
        "lev": ["olevel"],
    }

    for standard_name, possible_names in dim_mapping.items():
        for name in possible_names:
            if name in ds.dims and name != standard_name:
                ds = ds.rename({name: standard_name})
                break

    coord_mapping = {
        "lon": ["longitude", "nav_lon"],
        "lat": ["latitude", "nav_lat"],
    }

    for standard_name, possible_names in coord_mapping.items():
        for name in possible_names:
            if name in ds.coords and name != standard_name:
                ds = ds.rename({name: standard_name})
                break

    return ds


def standardize_latlon(ds):
    # Camille Le Gloannec inspired script

    # ---------- LONGITUDE ----------
    if "lon" in ds.coords:
        lon = ds["lon"]
        ds = ds.assign_coords(lon=lon % 360)

        if lon.ndim == 1:
            ds = ds.sortby("lon")
        else:
            for dim in lon.dims:
                ds = ds.sortby(dim)

    elif "x" in ds.coords:
        x = ds["x"]
        ds = ds.assign_coords(x=x % 360)
        ds = ds.sortby("x")

    # ---------- LATITUDE ----------
    if "lat" in ds.coords:
        lat = ds["lat"]

        ds = ds.assign_coords(lat=lat.clip(-90, 90))

        if lat.ndim == 1:
            ds = ds.sortby("lat")
        else:
            for dim in lat.dims:
                ds = ds.sortby(dim)

    elif "y" in ds.coords:
        y = ds["y"]
        ds = ds.assign_coords(y=y.clip(-90, 90))
        ds = ds.sortby("y")

    return ds

def standardize_dim_order(ds, priority_dims = ['time', 'lat', 'lon', 'member']):
    existing_dims = list(ds.dims)
    ordered_dims = [d for d in priority_dims if d in existing_dims]
    remaining_dims = [d for d in existing_dims if d not in ordered_dims]
    return ds.transpose(*(ordered_dims + remaining_dims))


def standardize_time(ds, original_time_format="%Y"):
    time_values = ds.time.values

    if np.issubdtype(time_values.dtype, np.datetime64):
        return ds

    if original_time_format is not None:
        parsed = pd.to_datetime(time_values.astype(str), format=original_time_format)
        return ds.assign_coords(time=parsed.values)


def standardize_units(ds):

    for var in ds.data_vars:
        units = ds[var].attrs.get("units", None)
        if var == "tas":
            if units in ["°C", "degC", "C"]:
                ds[var] = ds[var] + 273.15
                ds[var].attrs["units"] = "K"

            elif units is None and np.nanmean(ds[var]) < 100:
                ds[var] = ds[var] + 273.15
                ds[var].attrs["units"] = "K"
            elif units is None and np.nanmean(ds[var]) > 100:
                ds[var].attrs["units"] = "K"
        
        elif var == 'pr':
            if units in ['kg m-2 s-1', 'kg/m2/s']:
                ds[var] = ds[var] * 86400
                ds[var].attrs["units"] = "mm/day"
            elif units is None and np.nanmean(ds[var]) < 0.01 :
                ds[var] = ds[var] * 86400
                ds[var].attrs["units"] = "mm/day"
            elif units is None and np.nanmean(ds[var]) > 0.1:
                ds[var].attrs["units"] = "mm/day"
            elif units == 'mm/day' and np.nanmean(ds[var]) < 0.01:
                ds[var] = ds[var] * 86400
    return ds
            
def drops_useless_vars(ds, 
                       vars_to_drop = ['heights', 'lat_bnds', 'lon_bnds', 'time_bnds', 'time_bounds']):
    if isinstance(vars_to_drop, str):
        vars_to_drop = [vars_to_drop]
    for var in vars_to_drop:
        try:
            ds = ds.drop_vars(var)
        except:
            pass
    return ds
    
def dataset_xr_formatting(ds, original_time_format = "%Y", priority_dims = ['time', 'lat', 'lon', 'member']):
    ds = standardize_dims_and_coords(ds)
    ds = standardize_latlon(ds)
    ds = standardize_time(ds, original_time_format)
    ds = standardize_units(ds)
    ds = drops_useless_vars(ds)
    ds = standardize_dim_order(ds, priority_dims=priority_dims)
    return ds

def compute_annual_means(ds, resample_time='YS'): # year-01-01 and not year-12-31 to avoid issues with leap years and different calendar types, but can be changed if needed
    # Compute annual means (timestamp set to year-end)
    if np.issubdtype(ds["time"].dtype, np.datetime64):
        ds = ds.resample(time=resample_time).mean(dim='time')

    elif np.issubdtype(ds["time"].dtype, np.int64):
        ds = ds.groupby('time.year').mean('time')
    return ds



def compute_weights_from_lats(lats, deg2rad=True, weights_normalization='sum'):
    if isinstance(lats, np.ndarray):
        if deg2rad:
            weights = np.cos((np.pi * lats) / 180)
        else:
            weights = np.cos(lats)
    elif isinstance(lats, torch.Tensor):
        if deg2rad:
            weights = torch.cos((torch.pi * lats) / 180)
        else:
            weights = torch.cos(lats)
    elif isinstance(lats, xr.DataArray):
        if deg2rad:
            weights = np.cos(np.deg2rad(lats))
        else:
            weights = np.cos(lats)
        
    if weights_normalization == 'sum':
        weights = weights / weights.sum() # attention, NRMSE_ClimateBench pas de normalisation des po
    elif weights_normalization == 'mean':
        weights = weights / weights.mean() # attention, NRMSE_ClimateBench pas de normalisation des po
    return weights

def weighted_global_mean(data, lats=None, deg2rad=True, weights_normalization='sum'):
    if isinstance(data, np.ndarray):
        assert lats.shape[0] == data.shape[-2], "Latitude dimension does not match data shape."
        weights = compute_weights_from_lats(lats, deg2rad=deg2rad, weights_normalization=weights_normalization)
        data = np.average(data, axis=-2, weights=weights) # division par la somme des poids, pas besoin de normalisation des poids pour la moyenne globale pondérée
        data = np.mean(data, axis=-1)

    elif isinstance(data, torch.Tensor):
        assert lats.shape[0] == data.shape[-2], "Latitude dimension does not match data shape."
        weights = compute_weights_from_lats(lats, deg2rad=deg2rad, weights_normalization=weights_normalization)
        weights = weights.unsqueeze(-1)
        data = torch.sum(data * weights, dim=-2)
        data = torch.mean(data, dim=-1)

    elif isinstance(data, xr.Dataset) or isinstance(data, xr.DataArray):
        data = standardize_dims_and_coords(data)
        weights = compute_weights_from_lats(data.lat, deg2rad=deg2rad, weights_normalization=weights_normalization)
        if 'lon' in data.coords:
            data = data.weighted(weights).mean(['lat', 'lon'])
        else:
            data = data.weighted(weights).mean(['lat'])
    return data
    
def apply_moving_average(x, window_size):
    """Apply sliding mean to a 1D array."""
    return np.convolve(x, np.ones(window_size)/window_size, mode='valid')

def save_coords_from_ds(ds, save_path):
    coords = {'lon': ds.lon.values, 'lat': ds.lat.values}
    np.savez(save_path, **coords)

def get_statistics_from_bootstrap(data, n_bootstrap=1000):
    ''' data [n_samples, ...] '''
    res = bootstrap((data,), 
                    np.mean,
                    axis=0, 
                    n_resamples=n_bootstrap, 
                    method='basic', 
                    confidence_level=0.975, 
                    random_state=42)
    return res
    

if __name__ == "__main__":
    data = np.random.rand(100, 96, 192)
    res = get_statistics_from_bootstrap(data, n_bootstrap=1000)
    print("Bootstrap confidence interval:", res.confidence_interval)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    print(res.bootstrap_distribution.shape)
    ax.hist(res.bootstrap_distribution.mean(axis=(1,2)), bins=25)
    ax.set_title('Bootstrap Distribution')
    ax.set_xlabel('mean value of runs')
    ax.set_ylabel('frequency')
    plt.savefig(GRAPHS_DIR/ 'test.png')
    plt.close()
