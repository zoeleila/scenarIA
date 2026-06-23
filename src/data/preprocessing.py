import glob
from pathlib import Path
from scenarIA.src.utils.settings import DATASET_DIR, RAW_DATA_DIR
import xesmf as xe
import xarray as xr
import numpy as np
from scenarIA.src.utils.datautils import compute_annual_means, dataset_xr_formatting
import os

# TODO : Check data class to create dataset with same time format, unit, lat (-90,90), lon (0, 360)
# class input and output ?

class Outputs:
    def __init__(self,
                 files_dict,
                 dataset_path):
        self.files_dict = files_dict
        self.dataset_path = Path(dataset_path)

        # on suppose que les données sont déjà concatenées de dimension (time, lat, lon, member) ... avec toutes les varibales dedans,
        # il peut y avoir différents fichiers si il y a un membre différent de membre par simus, exemple CNRM-CM6-1

    def build_dataset(self, original_time_format="%Y"):
        for simu, files in self.files_dict.items():
            if isinstance(files, str) or isinstance(files, Path):
                ds = xr.open_dataset(files, chunks= {'time': 10})
                ds = dataset_xr_formatting(ds, original_time_format)
                ds.to_netcdf(self.dataset_path / f'outputs_{simu}.nc')
            else:
                for file in files:
                    ds = xr.open_dataset(file, chunks= {'time': 10})
                    ds = dataset_xr_formatting(ds, original_time_format)
                    start = ds.time.dt.year.values[0]
                    end = ds.time.dt.year.values[-1]
                    print(ds)
                    ds.to_netcdf(self.dataset_path / f'outputs_{simu}_{start}-{end}.nc')

class Inputs:
    def __init__(self,
                 files_dict,
                 dataset_path):
        self.files_dict = files_dict
        self.dataset_path = Path(dataset_path)

    def build_dataset(self, original_time_format="%Y"):
        for simu, files in self.files_dict.items():
            ds = xr.open_dataset(files)
            print(ds.time.values)
            ds = dataset_xr_formatting(ds, original_time_format, priority_dims=['time', 'lat', 'lon'])
            print(ds.time.values)
            return ds
            #ds.to_netcdf(self.dataset_path / f'inputs_{simu}_regrid2.nc')
 

# TODO : static fonction for var concatenation ? member concatenation, reggrid ?? not the same nb of member per simus


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

def regrid_inputs_to_outputs(ds, ds_target):
    regridder = xe.Regridder(ds, ds_target, 'conservative')
    co2 = ds['CO2']
    ch4 = ds['CH4']
    ds = ds.drop_vars(['CO2','CH4'])
    ds = regridder(ds)
    ds['CO2'] = co2
    ds['CH4'] = ch4
    return ds
        
def concatenate_files_time_dim(files):
    """ Concatenate all time steps """
    # use chunks
    ds_list = []
    for file in files:
        ds = xr.open_dataset(file, chunks={'time': 5})
        ds_list.append(ds)
    ds_concat = xr.concat(ds_list, dim='time')
    return ds_concat

def build_outputs_dataset(rawdata_dir, simu, member_list, vars_list, dataset_dir, annual_mean=True):
    """ From ESGF standard files, carefull data are already concatenate in time dimension """
    """ TODO new func for raw ESGF downloaded files classification """
    ds_list = []
    for var in vars_list:
        ds_var_list = []
        for member in member_list:
            file = np.sort(glob.glob(str(rawdata_dir /f'{var}*{member}*.nc')))[0]
            print(file)
            ds = xr.open_dataset(file, chunks={'time': 5})
            if annual_mean:
                ds = compute_annual_means(ds)
            ds_var_list.append(ds)
        ds_concat = xr.concat(ds_var_list, dim='member')
        ds_list.append(ds_concat)
    ds_final = xr.merge(ds_list)
    ds_final = dataset_xr_formatting(ds_final)
    ds_final.to_netcdf(dataset_dir / f'outputs_{simu}.nc')
    return ds_final

def build_inputs_dataset(rawdata_dir, simu, vars_list, dataset_dir, annual_mean=True):
    for var in vars_list:
        file = np.sort(glob.glob(str(rawdata_dir /f'{var}*{simu}*.nc')))[0]
        ds = xr.open_dataset(file).sum('sector')
        if annual_mean:
            ds = compute_annual_means(ds)
        

    

if __name__ == "__main__":
    file = '/gpfs-calypso/scratch/globc/garcia/scenarIA/datasets/MPI-ESM1-2-LR/annual/inputs_ssp119_regrid2.nc'
    input = Inputs(files_dict={'ssp119': file}, dataset_path=DATASET_DIR/'MPI-ESM1-2-LR/annual')
    ds = input.build_dataset(original_time_format="%Y")
    ds.to_netcdf(DATASET_DIR/'MPI-ESM1-2-LR/annual'/'inputs_ssp119_regrid3.nc')
    
