import os
import yaml
import torch
from tqdm import tqdm
import glob
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
import xarray as xr

from scenarIA.src.utils.settings import CONFIG_DIR, GRAPHS_DIR, RUNS_DIR, DATASET_DIR, PREDICTIONS_DIR
from scenarIA.src.data.dataloader import get_dataloaders, get_climatology
from scenarIA.src.data.lightning_module import scenarIALightningModule
from scenarIA.src.utils.datautils import standardize_units

def predict(run_dir,
            data_type='test',
            simus_to_predict=None,
            best_checkpoint=True):
    if best_checkpoint:
        checkpoint_dir = glob.glob(str(RUNS_DIR / run_dir / 'checkpoints/best-checkpoint*.ckpt'))[0]
    else:
        checkpoint_dir = glob.glob(str(RUNS_DIR / run_dir / 'checkpoints/last.ckpt'))[0]
    model = scenarIALightningModule.load_from_checkpoint(checkpoint_dir, map_location='cpu')
    model.eval()
    hparams = model.hparams['config']
    if simus_to_predict is not None:
        hparams['train'][f'simus_{data_type}'] = [simus_to_predict]
    dataloader = get_dataloaders(data_type, config=hparams)
    y_hat_all = []
    y_all = None if data_type == 'inference' else []
    t_all = None if data_type == 'inference' else []
    for batch in tqdm(dataloader, desc="Computing stats from dataloader", disable=True):
        if data_type == 'inference':
            x, _, _, _ = batch
        else:
            x, y, t, _ = batch
            y_all.append(y)
            t_all.append(t)
        x = x.float().to('cpu')
        with torch.no_grad():
            y_hat = model(x).cpu()
        y_hat_all.append(y_hat)
    y_hat_all = torch.stack(y_hat_all, axis=0).view(-1, hparams['train']['img_size'][0], hparams['train']['img_size'][1]).numpy()
    if data_type == 'inference':
        return y_hat_all, None, t_all, hparams
    else:
        y_all = torch.cat(y_all, dim=0).squeeze().numpy()
        t_all = torch.cat(t_all, dim=0).numpy()
        t_all = np.array([np.datetime64(datetime(year, month, day)) for year, month, day, *_ in t_all])
        t_all = pd.to_datetime(t_all, format="%Y-%m-%d")
        return y_hat_all, y_all, t_all, hparams

def save_predictions_as_netcdf(runs_to_predict, data_type='test', simus_to_predict=None, best_checkpoint=True):
    runs_to_predict = [runs_to_predict] if isinstance(runs_to_predict, str) else runs_to_predict
    y_hat_all_list = []
    seed_list = []
    for run_dir in runs_to_predict:
        y_hat_all, _, t_all, hparams = predict(run_dir, data_type=data_type, simus_to_predict=simus_to_predict, best_checkpoint=best_checkpoint)
        simu_test = hparams['train']['simus_test'][0]
        seed = hparams['train']['seed']
        output = hparams['train']['outputs'][0] # à modifier quand multivarié
        test_name = hparams['train']['test_name']
        test_name = test_name.replace(f'_seed{seed}_', '_')
        lat = dict(np.load(DATASET_DIR / hparams['data']['dataset_path'] / 'coords.npz', allow_pickle=True))['lat']
        lon = dict(np.load(DATASET_DIR / hparams['data']['dataset_path'] / 'coords.npz', allow_pickle=True))['lon']
        climatology = get_climatology(hparams)
        y_hat_all = y_hat_all + climatology.squeeze() # à modifier quand multivarié
        y_hat_all_list.append(y_hat_all)
        seed_list.append(seed)
        
        

    y_hat_all_array = np.stack(y_hat_all_list, axis=0)
    ds = xr.Dataset(
        data_vars={
            output: (('run', 'time', 'lat', 'lon'), y_hat_all_array)
        },
        coords={
            'run': seed_list,
            'time': t_all,
            'lat': lat,
            'lon': lon
        }
    )
    # add units
    ds = standardize_units(ds)
    print(ds)
    model_name = hparams['data']['model_name']
    timescale = hparams['data']['timescale']
    exp = hparams['data']['exp']
    simu_test = hparams['train']['simus_test'][0]
    max_epochs = hparams['train']['max_epochs']
    filename = f'{model_name}_{timescale}_{exp}_{simu_test}_{test_name}_epoch{max_epochs}.nc'
    os.makedirs(PREDICTIONS_DIR / model_name / timescale / exp, exist_ok=True)
    ds.to_netcdf(PREDICTIONS_DIR / model_name / timescale / exp / filename) # create directories if they don't exist
    

if __name__=='__main__':
    with open(CONFIG_DIR / 'runs.yaml') as file:
        runs = yaml.safe_load(file)
    runs_to_predict_dict = runs['predict']
    for exp_name in runs_to_predict_dict.keys():
        runs_to_predict = runs_to_predict_dict[exp_name]
        save_predictions_as_netcdf(runs_to_predict, 
                               data_type='test', 
                               simus_to_predict='ssp245', 
                               best_checkpoint=True)
