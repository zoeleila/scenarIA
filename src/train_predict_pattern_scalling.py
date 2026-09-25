'''
Inspired by https://github.com/blutjens/climate-emulator/blob/public/emcli2/models/pattern_scaling/model.py
'''

from cProfile import label
import xarray as xr
import numpy as np
from sklearn.linear_model import LinearRegression
import yaml
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import torch
from matplotlib.lines import Line2D

from datetime import datetime


from scenarIA.src.utils.evalutils import EvaluationPlots
from scenarIA.src.utils.datautils import weighted_global_mean
from scenarIA.src.data.dataloader import get_dataset, get_dataloaders, get_climatology
from scenarIA.src.utils.datautils import standardize_units
from scenarIA.src.utils.settings import CONFIG_DIR, DATASET_DIR, PREDICTIONS_DIR, RUNS_DIR, GRAPHS_DIR, SIMUS_COLORS_DICT

class PatternScaling(object):
    """
    Does pattern scaling. Here we fit one linear model per 
    grid point. The linear model maps a global variable, 
    e.g., cum. CO2 emissions or temperature, to 
    the grid point's local value. 
    This model captures temporal patterns in each grid cell.
    The model is local, i.e., it will be independent of 
    neighboring grid points. The model is linear in time, 
    i.e., it assumes no non-linearly amplifying feedbacks 
    between the global and local variable.
    """
    def __init__(self, deg=1):
        """
        Args:
            deg int: degree of polynomial fit. Default is 1 
                for linear fit.
        """
        self.deg = deg
        self.coeffs = None

    def train(self, in_global, out_local):
        """
        Fits polynomial with degree self.deg from in_global to 
        every location in out_local. Choose deg=1 for linear fit.

        Args:
            in_global np.array((n_t,)): The model input is a 
                global variable, e.g., annual global mean surface 
                temperature anomalies of in °C
            out_local np.array((n_t,n_lat,n_lon)): The model 
                output is a locally-resolved variable. E.g., annual 
                mean surface temperature anomalies at every lat,lon
                in °C
        Sets:
            coeffs np.array((deg+1, n_lat, n_lon))
        """
        n_t, n_lat, n_lon = out_local.shape
        
        # Preprocess data, by flattening in space
        out_local = out_local.reshape(n_t,-1) # (n_t, n_lat*n_lon)

        # Fit linear regression coefficients to every grid point
        self.coeffs = np.polyfit(in_global, out_local, deg=self.deg) # (2, n_lat*n_lon)

        # Reshape coefficients onto locally-resolved grid
        self.coeffs = self.coeffs.reshape(-1, n_lat, n_lon) # (2, n_lat, n_lon)

    def predict(self, in_global):
        """
        Args:
            in_global np.array((n_t,))
        Returns:
            preds np.array((n_t, n_lat, n_lon))
        """
        n_lat = self.coeffs.shape[1]
        n_lon = self.coeffs.shape[2]

        # Predict by applying pattern scaling coefficients on locally-resolved grid
        in_global = np.tile(in_global[:,np.newaxis, np.newaxis], reps=(1,n_lat,n_lon)) # repeat onto local grid to get shape (n_t, n_lat, n_lon)
        preds = np.polyval(self.coeffs, in_global) # (n_t, n_lat, n_lon)

        return preds
    
def prepare_dataset_for_global_fit(data):
    data_global = np.stack([weighted_global_mean(data[..., i], lats=lat) for i in range(data.shape[-1])], 
                           axis=0).transpose()
    return data_global

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Compare different runs")
    argparser.add_argument("--var_name", type=str, default='tas')
    argparser.add_argument("--simu_to_predict", type=str, default='ssp245')
    argparser.add_argument("--data_type", type=str, default='test')
    args = argparser.parse_args()
    var_name = args.var_name
    simu_test = args.simu_to_predict
    data_type = args.data_type

    with open(CONFIG_DIR / 'config.yaml') as file:
        config = yaml.safe_load(file)

    with open(CONFIG_DIR / 'plots.yaml') as file:
        config_plots = yaml.safe_load(file)
    
    config['data']['smoothed_outputs'] = False
    config['data']['seq_length'] = 1
    config['train']['batch_size'] = 1
    #config['train']['simus_train'] = ['historical', 'ssp119', 'ssp126', 'ssp585']
    config['data']['add_clim_to_predictors'] = False
    lat = dict(np.load(DATASET_DIR / config['data']['dataset_path'] / 'coords.npz', allow_pickle=True))['lat']
    lon = dict(np.load(DATASET_DIR / config['data']['dataset_path'] / 'coords.npz', allow_pickle=True))['lon']
    
    # carefull to train for the 10 seeds

    # Fit global emissions to global tas
    config['train']['inputs'] = ['CO2']
    config['train']['outputs'] = ['tas']

    train_dataloader = get_dataloaders(config=config, data_type='train', transforms=False)
    train_in = []
    train_out = []
    simus_all = []
    for batch in train_dataloader:
        x, y, _, simu = batch
        train_in.append(x)
        train_out.append(y)
        simus_all.append(simu[0])
    train_in = np.concatenate(train_in, axis=0).squeeze(1) # remove time=1 dimension shape (n, n_lat, n_lon, channels)
    train_out = np.concatenate(train_out, axis=0).squeeze(1)[..., np.newaxis] # remove time=1 dimension shape (n, n_lat, n_lon, channels)
    train_in_global = prepare_dataset_for_global_fit(train_in)
    train_out_global = prepare_dataset_for_global_fit(train_out)
    print("train_in_global shape:", train_in_global.shape)
    print("train_out_global shape:", train_out_global.shape)
    

    linear = LinearRegression()
    linear.fit(train_in_global,
               train_out_global)
    a = np.ravel(linear.coef_)[0]          # pente
    b = np.ravel(linear.intercept_)[0] 
    r2 = linear.score(train_in_global.reshape(-1, 1), train_out_global)
    x = np.linspace(train_in_global.min(), train_in_global.max(), 100).reshape(-1, 1)
    y = linear.predict(x)

    
    fig, ax = plt.subplots()

    colors = np.array([SIMUS_COLORS_DICT[s] for s in simus_all])   # une couleur par point

    ax.scatter(train_in_global.squeeze(), 
                train_out_global.squeeze(), 
                c=colors)
    ax.plot(x, y, color='black', linestyle='--', linewidth=2)


    handles = [Line2D([], [], marker='o', linestyle='', color=SIMUS_COLORS_DICT[s], label=s)
            for s in np.unique(simus_all)]
    handles.append(Line2D([], [], color='black', linestyle='--',
                      label=f'y = {a:.3g}·x + {b:.3g}'))
    ax.text(0.05, 0.95, f'$R^2$ = {r2:.3f}',
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=12,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))
    ax.legend(handles=handles, loc='lower right')
    ax.set_xlabel('Cumulative CO2 anthropique emissions')
    ax.set_ylabel('Global mean temperature')
    plt.savefig(GRAPHS_DIR/f'runs/MPI-ESM1-2-LR/annual/exp9/global_forcings_to_global_{simu_test}_tas_and_CO2_pattern_scaling.png')

    
    
    config['train'][f'simus_{data_type}'] = [simu_test]
    test_dataloader = get_dataloaders(config=config, data_type=data_type, transforms=False)

    # to predict but shuffle feels weird
    test_in = []
    test_out = []
    t_all = []
    for batch in test_dataloader:
        if data_type == 'test':
            x, y, t, _ = batch
            test_out.append(y)
        else:
            x, t, _ = batch
        test_in.append(x)
        t_all.append(t)
    t_all = torch.cat(t_all, dim=0).numpy()
    t_all = np.array([np.datetime64(datetime(year, month, day)) for year, month, day, *_ in t_all])
    t_all = pd.to_datetime(t_all, format="%Y-%m-%d")
    test_in = np.concatenate(test_in, axis=0).squeeze(1) # remove time=1 dimension shape (n, n_lat, n_lon, channels)
    test_in_global = prepare_dataset_for_global_fit(test_in)
    

    pred_out_global = linear.predict(test_in_global)
    print("y_hat shape:", pred_out_global.shape)
    print("test_in_global shape:", test_in_global.shape)

    
    if data_type == 'test':
        test_out = np.concatenate(test_out, axis=0).squeeze(1)[..., np.newaxis] # remove time=1 dimension shape (n, n_lat, n_lon, channels)
        test_out_global = prepare_dataset_for_global_fit(test_out)
        print("test_out_global shape:", test_out_global.shape)

    plt.figure()
    plt.plot(pred_out_global, label='Linear Regression', color='royalblue')
    if data_type == 'test':
        plt.plot(test_out_global, label='True', color='k')
    plt.xlabel('Time')
    plt.ylabel('Temperature Anomalies (°C)')
    plt.title(f'Global mean Predictions ({simu_test})')
    plt.legend()
    plt.savefig(GRAPHS_DIR/ f'runs/MPI-ESM1-2-LR/annual/exp9/global_forcings_to_global_{simu_test}_tas_pattern_scaling.png')

    # Fit global tas to local var (univariate)
    if var_name == 'tas':
        train_out_local = train_out.squeeze()
        test_out_local = test_out.squeeze() if data_type == 'test' else []
    else:
        config['train']['outputs'] = [var_name]
        train_dataloader_var = get_dataloaders(config=config, data_type='train', transforms=False)
        train_out_var = []
        for batch in train_dataloader_var:
            _, y, _, _ = batch
            train_out_var.append(y)
        train_out_local = np.concatenate(train_out_var, axis=0).squeeze() # (n, lat, lon) 1 var

        if data_type == 'test':
            test_dataloader_var = get_dataloaders(config=config, data_type='test', transforms=False)
            test_out_var = []
            for batch in test_dataloader_var:
                _, y, _, _ = batch
                test_out_var.append(y)
            test_out_local = np.concatenate(test_out_var, axis=0).squeeze() # (n, lat, lon) 1 var
      
    ps = PatternScaling(deg=1)
    ps.train(train_out_global.squeeze(), train_out_local)

    pred_out_local = ps.predict(pred_out_global.squeeze()).squeeze()

    plt.figure()
    plt.plot(weighted_global_mean(pred_out_local, lats=lat), label='Pattern scaling', color='royalblue')
    if data_type == 'test':
        plt.plot(weighted_global_mean(test_out_local, lats=lat), label='True', color='k')
    plt.xlabel('Time')
    plt.ylabel(f'{var_name} Anomalies')
    plt.title(f'Global mean predictions ({simu_test})')
    plt.legend()
    plt.savefig(GRAPHS_DIR/ f'runs/MPI-ESM1-2-LR/annual/exp9/global_tas_to_local_{simu_test}_{var_name}_pattern_scaling.png')


    pred_out_local = pred_out_local
    
    ds = xr.Dataset(
        data_vars={
            var_name: (('run', 'time', 'lat', 'lon'), pred_out_local[np.newaxis,...])
        },
        coords={
            'run': [42],
            'time': t_all,
            'lat': lat,
            'lon': lon
        }
    )
    # add units
    ds = standardize_units(ds)
    print(ds)
    ds.to_netcdf(
        PREDICTIONS_DIR / f'MPI-ESM1-2-LR/annual/exp9/MPI-ESM1-2-LR_annual_exp9_{simu_test}_{var_name}_pattern-scaling_seq1_mem30.nc')
