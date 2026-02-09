import yaml
import glob
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from scenarIA.src.utils.settings import CONFIG_DIR, GRAPHS_DIR, RUNS_DIR
from scenarIA.src.data.dataloader import get_dataloaders
from scenarIA.src.data.lightning_module import scenarIALightningModule
from scenarIA.src.utils.datautils import weighted_global_mean

with open(CONFIG_DIR / 'runs.yaml') as file:
    runs = yaml.safe_load(file)

device = 'cpu'

runs_list = runs['compare']['runs_to_compare']['1']
print(runs_list)
# TODO attention cross validation
# ne mettre que les metrics_test_sets ? voir rachid

y_all = []
y_hat_dict = {}
for test, runs_list in runs['compare']['runs_to_compare'].items():
    y_hat_all_seeds = [] # if different seeds
    for i, run_dir in enumerate(runs_list):
        checkpoint_dir = glob.glob(str(RUNS_DIR / run_dir / 'checkpoints/best-checkpoint*.ckpt'))[0]
        model = scenarIALightningModule.load_from_checkpoint(checkpoint_dir, map_location='cpu')
        model.eval()
        hparams = model.hparams['config']
        test_dataloader = get_dataloaders('test', config=hparams)

        y_hat_all = []
        for batch in tqdm(test_dataloader, desc="Computing stats from dataloader"):
            if i == 0:
                x, y, _ = batch
                y_all.append(y)
            else:
                x, _, _ = batch
            x = x.float().to(device)
            with torch.no_grad():
                y_hat = model(x).cpu()
            y_hat_all.append(y_hat)
        y_hat_all = torch.stack(y_hat_all, axis=0).view(-1, hparams['train']['img_size'][0], hparams['train']['img_size'][1])
        y_hat_all_seeds.append(y_hat_all)
    print(len(y_hat_all_seeds), y_hat_all_seeds[0].shape)
    y_hat_dict[test] = y_hat_all_seeds
# liste de 3 seeds, if list == 1


y_all = torch.stack(y_all, dim=0).numpy()
y_all = np.squeeze(y_all)
time = np.arange(y_all.shape[0])
lats = np.linspace(-90, 90, y_all.shape[-2])
y_all = weighted_global_mean(y_all, lats=lats)
plt.figure()
plt.plot(y_all)

for test, y_hat in y_hat_dict.items():
    if len(y_hat) > 0:
        y_hat = torch.stack(y_hat, dim=0)
        y_hat_mean = y_hat.mean(axis=0)
        print(y_hat_mean.shape)
        y_hat_std = y_hat.std(axis=0)
    else:
        y_hat_mean = y_hat[0]
        y_hat_std = torch.zeros_like(y_hat_mean)

    y_hat_mean = weighted_global_mean(y_hat_mean, lats=lats)
    y_hat_std = weighted_global_mean(y_hat_std, lats=lats)
    print(y_hat_mean.shape, y_hat_std.shape)

    plt.plot(y_hat_mean, label=test)
    print(time.shape, y_hat_mean.shape, y_hat_std.shape)
    plt.fill_between(time, y_hat_mean - y_hat_std, y_hat_mean + y_hat_std, alpha=0.2)

plt.savefig(GRAPHS_DIR/'tests/test.png')
