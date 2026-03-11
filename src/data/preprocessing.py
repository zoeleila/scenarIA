import glob
from pathlib import Path
import xesmf as xe
import xarray as xr
from scenarIA.src.utils.datautils import compute_annual_means, dataset_xr_formatting

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
                    ds.to_netcdf(self.dataset_path / f'outputs_{simu}_{start}-{end}.nc')
                

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
        

if __name__ == "__main__":
    # TODO : add argparse for dataset_path or list path et output ou inputs !!!

    DATASET_PATH = Path('/scratch/globc/garcia/scenarIA/datasets/MPI-ESM1-2-LR/annual/')

    ds_target = xr.open_dataset('/gpfs-calypso/scratch/globc/garcia/scenarIA/datasets/MPI-ESM1-2-LR/annual/outputs_ssp126.nc')
    #baseline_pr = baseline_pr.rename({'lon':'longitude', 'lat':'latitude'})
    exps = ['hist-GHG', 'hist-aer','ssp126', 'ssp245', 'ssp370', 'ssp585', 'historical']
    print(ds_target.lat)
    for exp in exps:
        file = DATASET_PATH / f'inputs_{exp}.nc'
        ds = xr.open_dataset(file)
        print(exp)
        ds = regrid_inputs_to_outputs(ds, ds_target)
        ds.to_netcdf(DATASET_PATH / f'inputs_{exp}_regrid.nc')