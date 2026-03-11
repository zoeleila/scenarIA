"""

"""

import sys
sys.path.append('.')

from pathlib import Path
import os
import time
import yaml
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import pandas as pd
import matplotlib.pyplot as plt
from torchmetrics import PearsonCorrCoef, MeanSquaredError, MeanAbsoluteError

from scenarIA.src.models.CNNLSTM import CNNLSTMModel
from scenarIA.src.utils.losses import LLweighted_MSELoss_Climax
from scenarIA.src.utils.metrics import NRMSE_ClimateBench
from scenarIA.src.utils.datautils import weighted_global_mean
from scenarIA.src.utils.settings import RUNS_DIR, CONFIG_DIR
from scenarIA.src.utils.evalutils import EvaluationPlots


layout = {
    "Check Overfit": {
        "loss": ["Multiline", ["loss/train", "loss/val"]],
    },
}

class scenarIALightningModule(pl.LightningModule):
    def __init__(self, config:dict, lats=None):
        super().__init__()
        self.seq_length = config['data']['seq_length']
        self.learning_rate = config['train']['learning_rate']
        self.runs_dir = RUNS_DIR / config['train']['runs_dir']
        self.outputs = config['train']['outputs']
        self.inputs = config['train']['inputs']
        self.img_size = config['train']['img_size']
        self.scheduler_step_size = config['train']['scheduler_step_size']
        self.scheduler_gamma = config['train']['scheduler_gamma']
        self.arch = config['train'].get('arch', 'cnn-lstm')
        self.predict_only_last_timestep = config['data']['predict_only_last_timestep']
        os.makedirs(self.runs_dir, exist_ok=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lats = lats.to(device) if lats is not None else torch.linspace(-90, 90, self.img_size[0]).to(device)
        self.loss = LLweighted_MSELoss_Climax(lats=self.lats)
        #self.loss = nn.MSELoss()
        self.metrics_dict = nn.ModuleDict({
                    "rmse": MeanSquaredError(squared=True),
                    "mae": MeanAbsoluteError()
                })
        self.spatial_corr_metric = PearsonCorrCoef()

        self.get_model()
        self.test_metrics = {}
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs_true = []
        self.test_step_outputs_hat = []
        self.test_step_times = []
        
        self.save_hyperparameters()
        self.epoch_start_time = None

        with open(CONFIG_DIR / 'plots.yaml') as file:
            config_plots = yaml.safe_load(file)
        self.config_plots = config_plots

    def get_model(self):
        if self.predict_only_last_timestep:
            output_seq_len = 1
        else:
            output_seq_len = self.seq_length
        match self.arch:
            case 'cnn-lstm':
                self.model = CNNLSTMModel(self.seq_length, height=self.img_size[0], width=self.img_size[1], channels=len(self.inputs),
                                          output_seq_len=output_seq_len).float()

    def forward(self, x):
        return self.model(x) 

    def on_train_start(self):
        self.logger.experiment.add_custom_scalars(layout)
        self.logger.log_hyperparams(vars(self.hparams))

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def common_step(self, x, y):
        y_hat = self(x)
        #if y.shape[-1] == 1:
            #y = y.squeeze(-1)
            #y_hat = y_hat.squeeze(-1)
        loss = self.loss(y_hat, y) # if len(var) == 1, squeeze, else : loop on variables
        return y_hat, loss

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat, loss = self.common_step(x, y)
        self.train_step_outputs.append(loss)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self):
        epoch_average = torch.stack(self.train_step_outputs).mean()
        self.logger.experiment.add_scalar("loss/train", epoch_average, self.current_epoch)
        self.train_step_outputs.clear()
        epoch_duration = time.time() - self.epoch_start_time
        self.log("epoch_time", epoch_duration, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat, loss = self.common_step(x, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.val_step_outputs).mean()
        self.logger.experiment.add_scalar("loss/val", epoch_average, self.current_epoch)
        self.val_step_outputs.clear()

        
    def test_step(self, batch, batch_idx):
        x, y, t = batch
        y_hat, loss = self.common_step(x, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            
        batch_dict = {"loss": loss}
        for metric_name, metric in self.metrics_dict.items():
            metric.update(y_hat, y)
            batch_dict[metric_name] = metric.compute()
            self.logger.experiment.add_scalar(metric_name, metric.compute(), batch_idx)
            metric.reset()
        self.test_metrics[batch_idx] = batch_dict

        y_flat = y.mean(dim=1).flatten() # [batch, lat, lon]
        y_hat_flat = y_hat.mean(dim=1).flatten()
        self.spatial_corr_metric.update(y_hat_flat, y_flat)

        self.test_step_outputs_true.append(y)
        self.test_step_outputs_hat.append(y_hat)
        self.test_step_times.append(t)

        if batch_idx == 0:

            fig, ax = plt.subplots()
            vmin, vmax = np.min(y.cpu().numpy()), np.max(y.cpu().numpy())
            levels = np.linspace(vmin, vmax, 11)
            cs = ax.contourf(y[batch_idx,0,:,:].cpu().numpy(), cmap='OrRd', levels=levels)
            plt.colorbar(cs, ax=ax, pad=0.05)
            self.logger.experiment.add_figure('Figure/test_y_0', fig) 
    
            fig, ax = plt.subplots()
            cs = ax.contourf(y_hat[batch_idx,0,:,:].cpu().numpy(), cmap='OrRd', levels=levels)
            plt.colorbar(cs, ax=ax, pad=0.05)
            self.logger.experiment.add_figure('Figure/test_yhat_0', fig)
 
            
    def build_metrics_dataframe(self):
        data = []
        first_sample = list(self.test_metrics.keys())[0]
        metrics = list(self.test_metrics[first_sample].keys())
        for name_sample, metrics_dict in self.test_metrics.items():
            data.append([name_sample] + [metrics_dict[m].item() for m in metrics])
        return pd.DataFrame(data, columns=["Name"] + metrics)

    def save_test_metrics_as_csv(self, df):
        path_csv = Path(self.logger.log_dir) / "metrics_test_set.csv"
        df.to_csv(path_csv, index=False)
    
    def on_test_epoch_end(self):
        df = self.build_metrics_dataframe()
        self.save_test_metrics_as_csv(df)
        df = df.drop("Name", axis=1)

        y_all = torch.stack(self.test_step_outputs_true, axis=0).view(-1, self.img_size[0], self.img_size[1])
        y_hat_all = torch.stack(self.test_step_outputs_hat, axis=0).view(-1, self.img_size[0], self.img_size[1])
        t_all = torch.cat(self.test_step_times).cpu().numpy()

        self.log('hp_metric', NRMSE_ClimateBench(y_hat_all, y_all, self.lats))
        self.log('loss', df['loss'].mean())

        spatial_corr = self.spatial_corr_metric.compute()
        self.log("hp_metric_corr", spatial_corr)
        
        fig, ax = plt.subplots()
        ax.plot(weighted_global_mean(y_all, self.lats).cpu().numpy(), label='True')
        ax.plot(weighted_global_mean(y_hat_all, self.lats).cpu().numpy(), label='Predicted')
        ax.set_xlabel('time') # TODO change according to time features !! monthly
        ax.set_ylabel(f'{self.outputs} value')
        ax.legend()
        self.logger.experiment.add_figure('Figure/test_true_vs_predicted', fig)

        
        eval = EvaluationPlots(simulation_name='ssp245',# TODO change by returning simu in dataloader get item
                               var_name=self.outputs[0], # TODO change for multivariate
                               config_plots=self.config_plots)
        start_year = t_all[0, 0]
        end_year = t_all[-1, 0]
        eval.plot_error_maps(y_all.cpu().numpy(), 
                             y_hat_all.cpu().numpy(), 
                             title=f'ssp245 {start_year}-{end_year}',
                             save_path=Path(self.logger.log_dir) / 'error_maps.png')

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma)
        return [optimizer], [scheduler]