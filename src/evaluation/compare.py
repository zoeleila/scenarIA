import yaml
import glob
import torch
from tqdm import tqdm
import numpy as np
from datetime import datetime
import pandas as pd

from scenarIA.src.utils.settings import CONFIG_DIR, GRAPHS_DIR, RUNS_DIR
from scenarIA.src.data.dataloader import get_dataloaders
from scenarIA.src.data.lightning_module import scenarIALightningModule
from scenarIA.src.utils.evalutils import EvaluationPlots, compare_metric_maps


def predict(runs_dict, seeds_mean = False, eval_func=None, plot_save_dir=None):
    # TODO adapt to multi variate ??
    y_all = []
    t_all = []
    y_hat_dict = {}
    
    for i, test_name in enumerate(runs_dict.keys()):
        runs_list = runs_dict[test_name]
        y_hat_all_seeds = [] # if different seeds
        for seed, run_dir in enumerate(runs_list):
            print(run_dir)
            checkpoint_dir = glob.glob(str(RUNS_DIR / run_dir / 'checkpoints/best-checkpoint*.ckpt'))[0]
            model = scenarIALightningModule.load_from_checkpoint(checkpoint_dir, map_location='cpu')
            model.eval()
            hparams = model.hparams['config']
            outputs = hparams['train']['outputs']
            test_dataloader = get_dataloaders('test', config=hparams)

            y_hat_all = []
            for batch in tqdm(test_dataloader, desc="Computing stats from dataloader"):
                if i == 0 and seed == 0:
                    x, y, t = batch
                    y_all.append(y)
                    t_all.append(t)
                else:
                    x, _, _ = batch
                x = x.float().to('cpu')
                with torch.no_grad():
                    y_hat = model(x).cpu()
                y_hat_all.append(y_hat)
            y_hat_all = torch.stack(y_hat_all, axis=0).view(-1, hparams['train']['img_size'][0], hparams['train']['img_size'][1]).numpy()
            y_hat_all_seeds.append(y_hat_all)
        if eval_func is not None:
            outputs_str = "_".join(outputs)
            eval_func.plot_diff_maps(torch.cat(y_all, dim=0).squeeze().numpy()[-21:,:,:], 
                                    np.stack(y_hat_all_seeds, axis=0).mean(axis=0)[-21:,:,:], 
                                    np.stack(y_hat_all_seeds, axis=0).std(axis=0)[-21:,:,:], 
                                    title=f'{test_name} (2080-2100)', 
                                    save_path=plot_save_dir/f'diff_maps_{outputs_str}_{test_name}.png')
        if seeds_mean:
            y_hat_all_seeds = np.stack(y_hat_all_seeds, axis=0)
            y_hat_all_seeds = y_hat_all_seeds.mean(axis=0)
        y_hat_dict[test_name] = y_hat_all_seeds

    y_all = torch.cat(y_all, dim=0).squeeze().numpy()
    t_all = torch.cat(t_all, dim=0).numpy()
    t_all = np.array([np.datetime64(datetime(year, month, day)) for year, month, day, *_ in t_all])
    t_all = pd.to_datetime(t_all, format="%Y-%m-%d")
    infos = {'outputs': outputs,
             'simus_test': hparams['train']['simus_test']}
    return y_all, y_hat_dict, t_all, infos

#def load_xr_to_dict(): TODO
    #return y, y_hat_dict, t

#def get_plot_path ????


if __name__=='__main__':
    with open(CONFIG_DIR / 'runs.yaml') as file:
        runs = yaml.safe_load(file)
    with open(CONFIG_DIR / 'plots.yaml') as file:
        config_plots = yaml.safe_load(file)

    eval_func = EvaluationPlots(simulation_name='ssp245', var_name='pr', config_plots=config_plots)
    runs_dict = runs['compare']['runs_to_compare']
    y, y_hat_dict, t, infos = predict(runs_dict, eval_func=eval_func,
                                      plot_save_dir=GRAPHS_DIR/'runs/MPI-ESM1-2-LR/annual/exp1/')
    

    