import glob
import xarray as xr
from scenarIA.funcs.datautils import compute_annual_means, time_format_conversion, dataset_xr_formatting

def mpi_esm1_2_lr_annual_outputs_formatting(ds):
    # Data retrived from Lutjens et al. (2025) 10.48550/arXiv.2408.05288
    ds = dataset_xr_formatting(ds,
                               dim_to_add_dict={'member': [1]})
    ds = ds.transpose('time', 'lat', 'lon', 'member')
    ds = ds[['tas', 'pr']]
    #ds['time'] = ds['time.year']
    ds['pr'].attrs['units'] = 'mm/day'
    ds['pr'].attrs['title'] = 'Ensemble Precipitation in mm/day'
    if ds['tas'].attrs['units'] == 'K':
        # Near-surface temperature
        #ds['tas'] -= 273.15 # convert from Kelvin to Celsius
        ds['tas'].attrs['units'] = '°C'
        ds['tas'].attrs['title'] = 'Ensemble Near-Surface Air Temperature in °C'
    return ds

def mpi_esm1_2_lr_annual_inputs_formatting(ds):
    # Data retrieved from Watson-Parris et al. (2023) 10.1029/2021MS002954
    ds = dataset_xr_formatting(ds)
    ds = ds.transpose('time', 'lat', 'lon')
    ds = ds[['CO2', 'SO2', 'CH4', 'BC']]
    return ds

if __name__ == "__main__":
    DATASET_PATH = '/scratch/globc/garcia/scenarIA/datasets/MPI-ESM1-2-LR/annual/'
    files = glob.glob(DATASET_PATH + 'outputs*.nc')
    for file in files:
        ds = xr.open_dataset(file)
        ds = mpi_esm1_2_lr_annual_outputs_formatting(ds)
        ds.to_netcdf(file.replace('.nc', '_2.nc'))