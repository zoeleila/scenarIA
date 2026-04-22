from webbrowser import get
import yaml
import torch
from tqdm import tqdm
import glob
import numpy as np
import pandas as pd
import argparse
from datetime import datetime

from scenarIA.src.utils.settings import CONFIG_DIR, RUNS_DIR, DATASET_DIR, PREDICTIONS_DIR
from scenarIA.src.data.dataloader import get_dataloaders, get_climatology
from scenarIA.src.data.lightning_module import scenarIALightningModule

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
    y_all = [] if data_type == 'test' else None
    t_all = [] if data_type == 'test' else None
    for batch in tqdm(dataloader, desc="Computing stats from dataloader", disable=True):
        if data_type == 'inference':
            x, _, _ = batch
        elif data_type == 'test':
            x, y, t = batch
            y_all.append(y)
            t_all.append(t)
        x = x.float().to('cpu')
        with torch.no_grad():
            y_hat = model(x).cpu()
        y_hat_all.append(y_hat)
    y_hat_all = torch.stack(y_hat_all, axis=0).view(-1, hparams['train']['img_size'][0], hparams['train']['img_size'][1]).numpy()
    if data_type == 'test':
        y_all = torch.cat(y_all, dim=0).squeeze().numpy()
        t_all = torch.cat(t_all, dim=0).numpy()
        t_all = np.array([np.datetime64(datetime(year, month, day)) for year, month, day, *_ in t_all])
        t_all = pd.to_datetime(t_all, format="%Y-%m-%d")
        return y_hat_all, y_all, t_all, hparams
    elif data_type == 'inference':
        return y_hat_all, None, t_all, hparams

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Predict with trained models")
    parser.add_argument("simus_inference", type=str, default=None)
    args = parser.parse_args()

    simus_inference = args.simus_inference

    with open(CONFIG_DIR / 'runs.yaml') as file:
        runs = yaml.safe_load(file)
    runs_dict = runs['predict']
    for file_name, runs_dir in runs_dict.items():
        if isinstance(runs_dir, list):
            y_hat_seeds = []
            for run_dir in runs_dir:
                y_hat = predict(run_dir, simus_inference)
                y_hat_seeds.append(y_hat)
            y_hat_all = np.stack(y_hat_seeds, axis=0).mean(axis=0) # moyenne ou alors on peut aussi garder les différentes seeds pour faire une analyse de la variance
        else:
            y_hat_all = predict(runs_dir, simus_inference)

        np.save(PREDICTIONS_DIR / f'{file_name}_predictions.npy', y_hat_all)