import xarray as xr
import numpy as np

def compute_annual_means(ds, resample_time='Y'):
    # Compute annual means (timestamp set to year-end)
    if np.issubdtype(ds["time"].dtype, np.datetime64):
        ds = ds.resample(time=resample_time).mean(dim='time')

    elif np.issubdtype(ds["time"].dtype, np.int64):
        ds = ds.groupby('time.year').mean('time')
    return ds

def time_format_conversion(ds, time_format="datetime64[ns]"):
    # Ensure the time coordinate is in the desired format
    ds["time"] = ds["time"].astype(time_format)
    return ds

def dataset_xr_formatting(ds, 
                        resample_time=None, 
                        time_format=None,
                        dim_to_add_dict=None):

    ds = ds.load()
    
    if resample_time:
        ds = compute_annual_means(ds, resample_time=resample_time)
    if time_format:
        ds = time_format_conversion(ds, time_format=time_format)
    if dim_to_add_dict:
        try:
            ds = ds.expand_dims(dim_to_add_dict)
        except:
            pass
    try:
        ds = ds.drop_vars(['lat_bnds', 'lon_bnds'])
    except:
        pass
    try:
        data = data.drop_vars(['height'])
    except:
        pass
    try:
        ds = ds.rename({'longitude':'lon', 'latitude': 'lat'})
    except:
        pass
    
    return ds
