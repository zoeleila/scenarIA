"""

"""
import sys

from models import CNN
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
import segmentation_models_pytorch as smp
from torchmetrics import PearsonCorrCoef, MeanSquaredError, MeanAbsoluteError

from scenarIA.src.models.CNN import CNNBase
from scenarIA.src.models.unet import UNet
from scenarIA.src.models.miniunet import MiniUNet
from scenarIA.src.models.time_unet import time_UNet
from scenarIA.src.models.CNNLSTM import CNNLSTMModel
from scenarIA.src.models.convlstm import ConvLSTM
from scenarIA.src.utils.losses import LLweighted_MSELoss_Climax
from scenarIA.src.utils.metrics import NRMSE_ClimateBench, LatWeightedRMSEMetric, NRMSE_g_ClimateBench, NRMSE_s_ClimateBench
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
        inputs = config['train']['inputs']
        self.inputs = inputs[:-1] if 'climatology' in inputs else inputs # for a specific run ...
        self.add_clim_to_predictors = config['data'].get('add_clim_to_predictors', False)
        if self.add_clim_to_predictors:
            self.inputs_len = len(self.inputs) + 1
        else:
            self.inputs_len = len(self.inputs)
        self.img_size = config['train']['img_size']
        self.simus_val = config['train'].get('simus_val', None) # ['ssp370'] or None for old runs
        self.simus_test = config['train']['simus_test']
        self.scheduler_step_size = config['train']['scheduler_step_size']
        self.scheduler_gamma = config['train']['scheduler_gamma']
        
        self.lstm_units = config['train'].get('lstm_units', 25) # for old runs
        self.arch = config['train'].get('arch', 'cnn-lstm')
        self.encoder = config['train'].get('encoder', 'resnet18')

        self.predict_only_last_timestep = config['data']['predict_only_last_timestep']
        os.makedirs(self.runs_dir, exist_ok=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lats = lats.to(device) if lats is not None else torch.linspace(-90, 90, self.img_size[0]).to(device)
        self.loss = LLweighted_MSELoss_Climax(lats=self.lats)
        #self.loss = nn.MSELoss()
        
        # Define metrics for test set evaluation
        self.metrics_dict = nn.ModuleDict({
                    "rmse": MeanSquaredError(squared=True),
                    "mae": MeanAbsoluteError()
                })
        self.spatial_corr_metric = PearsonCorrCoef()

        # hp_metric for hyperparameter optimization
        self.val_rmse = LatWeightedRMSEMetric()
        self.best_val_score = float('inf') 
        self.best_val_outputs_per_simu = {}
        self.val_outputs_per_simu = {} 
        self.valid_across_all_simus = True if self.simus_val is None else False
        self.monitor_metric = config['train'].get('monitor_metric', 'val_rmse') # for best checkpointing and hyperparameter optimization

        self.get_model()
        self.time_per_epoch = []
        self.test_metrics = {}
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs_true = []
        self.test_step_outputs_hat = []
        self.test_step_simus = [] 
        self.test_step_times = []

        self.save_hyperparameters(ignore=['lats'])
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
                #self.model = CNNLSTMModel(self.seq_length, height=self.img_size[0], width=self.img_size[1], channels=self.inputs_len,
                #                          output_seq_len=output_seq_len, lstm_units=self.lstm_units).float()
                self.model = CNNBase(slider=self.seq_length, height=self.img_size[0], width=self.img_size[1], channels=self.inputs_len,
                                     output_seq_len=output_seq_len, time_module_name='lstm', hidden_size=self.lstm_units).float()
            case 'cnn-gru':
                self.model = CNNBase(slider=self.seq_length, height=self.img_size[0], width=self.img_size[1], channels=self.inputs_len,
                                     output_seq_len=output_seq_len, time_module_name='gru', hidden_size=self.lstm_units).float()
            
            case 'unet':
                # variables and timesteps are concatenated in the channel dimension
                #self.model = UNet(in_channels=len(self.inputs)*self.seq_length, 
                                  #out_channels=len(self.outputs)*output_seq_len, init_features=32).float()
                
                self.model = smp.Unet(
                        encoder_name=self.encoder,
                        encoder_weights=None,
                        in_channels=self.inputs_len*self.seq_length,
                        classes=len(self.outputs)*output_seq_len,
                        encoder_depth = 4,
                        activation='identity',
                        decoder_channels = (128, 64, 32, 16)
                    )
                
            case 'miniunet':
                self.model = MiniUNet(in_channels=self.inputs_len*self.seq_length, 
                                      out_channels=len(self.outputs)*output_seq_len, init_features=32).float()
                
            case 'time-unet':
                self.model = time_UNet(
                    in_var_ids=self.inputs.append('climatology') if self.add_clim_to_predictors else self.inputs,
                    out_var_ids=self.outputs,
                    longitude=self.img_size[1],
                    latitude=self.img_size[0],
                    activation_function=None,
                    datamodule_config=None,
                    channels_last=True,
                    seq_to_seq=not self.predict_only_last_timestep,
                    seq_len=self.seq_length,
                ).float()

            case 'convlstm':
                hidden_dim = self.lstm_units
                if isinstance(hidden_dim, int):
                    hidden_dim = [hidden_dim]  # Convert to list if it's a single integer
                self.model = ConvLSTM(input_dim=self.inputs_len, 
                                    hidden_dim=hidden_dim, 
                                    kernel_size=(3, 3),
                                    num_layers=len(hidden_dim), 
                                    batch_first=True, 
                                    bias=True, 
                                    return_all_layers=False)    

    def forward(self, x):
        if self.arch == 'unet':
            #x = x.permute(0, 2, 3, 4, 1) # (B, lat, lon, C, T)
            #x = x.contiguous()
            #x = x.view(x.size(0), x.size(1), x.size(2), x.size(3)* x.size(4)) 
            # for smp.UNet
            x = x.permute(0, 4, 1, 2, 3) # (B, C, T, lat, lon)
            x = x.contiguous()
            x = x.view(x.size(0), x.size(1) * x.size(2), x.size(3), x.size(4)) # (B, C, T, lat, lon) --> (B, C*T, lat, lon)
        y_hat = self.model(x)
        #if self.arch == 'unet':
            #y_hat = y_hat.unsqueeze(1) # (B, lat, lon, C) --> (B, 1, lat, lon, C)
        if y_hat.size(-1) == 1:
            y_hat = y_hat.squeeze(-1)
        return y_hat

    def on_train_start(self):
        self.logger.experiment.add_custom_scalars(layout)

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def common_step(self, x, y):
        y_hat = self(x)
        loss = self.loss(y_hat, y) # if len(var) == 1, squeeze, else : loop on variables
        return y_hat, loss

    def training_step(self, batch, batch_idx):
        x, y, _, _ = batch
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
        self.time_per_epoch.append(epoch_duration)


    def validation_step(self, batch, batch_idx):
        x, y, _, simu = batch
        y_hat, loss = self.common_step(x, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_step_outputs.append(loss)
        self.val_rmse.update(y_hat, y, self.lats)

        if self.valid_across_all_simus is False:
            simu_name = simu[0]  # list of string
            if simu_name not in self.val_outputs_per_simu:
                self.val_outputs_per_simu[simu_name] = {'true': [], 'hat': []}
            self.val_outputs_per_simu[simu_name]['true'].append(y.detach().cpu())
            self.val_outputs_per_simu[simu_name]['hat'].append(y_hat.detach().cpu())
        return loss

    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.val_step_outputs).mean()
        self.logger.experiment.add_scalar("loss/val", epoch_average, self.current_epoch)
        
        # val_rmse
        rmse = self.val_rmse.compute()
        self.log('val_rmse', rmse, on_step=False, on_epoch=True)
        self.val_rmse.reset()
        self.val_step_outputs.clear()
        
        # val_nrmse
        if self.valid_across_all_simus is False:
            nrmse_per_simu = {}
            for simu_name, outputs in self.val_outputs_per_simu.items():
                y_all = torch.cat(outputs['true'], dim=0).view(-1, self.img_size[0], self.img_size[1])
                y_hat_all = torch.cat(outputs['hat'], dim=0).view(-1, self.img_size[0], self.img_size[1])
                nrmse_s = NRMSE_ClimateBench(y_hat_all, y_all, self.lats.cpu())
                nrmse_g_s = NRMSE_g_ClimateBench(y_hat_all, y_all, self.lats.cpu())
                nrmse_s_s = NRMSE_s_ClimateBench(y_hat_all, y_all, self.lats.cpu())
                nrmse_per_simu[simu_name] = nrmse_s
                self.log(f'val_nrmse_{simu_name}', nrmse_s, on_step=False, on_epoch=True)
                self.log(f'val_nrmse_g_{simu_name}', nrmse_g_s, on_step=False, on_epoch=True)
                self.log(f'val_nrmse_s_{simu_name}', nrmse_s_s, on_step=False, on_epoch=True)

            nrmse = torch.stack(list(nrmse_per_simu.values())).sum() 
            self.log('val_nrmse', nrmse, on_step=False, on_epoch=True)

            # Tracking du best score
            monitored = nrmse if self.monitor_metric == 'val_nrmse' else rmse
            if monitored < self.best_val_score:
                self.best_val_score = monitored
                self.best_val_outputs_per_simu = {
                    simu_name: {
                        'y_all': torch.cat(outputs['true'], dim=0).view(-1, self.img_size[0], self.img_size[1]),
                        'y_hat_all': torch.cat(outputs['hat'], dim=0).view(-1, self.img_size[0], self.img_size[1]),
                    }
                    for simu_name, outputs in self.val_outputs_per_simu.items()
                }

            self.val_outputs_per_simu.clear()

        else:
            monitored = rmse
            if monitored < self.best_val_score:
                self.best_val_score = monitored

    def on_fit_end(self):
        if self.valid_across_all_simus is False and self.best_val_outputs_per_simu:
            for simu_name, outputs in self.best_val_outputs_per_simu.items():
                y_all = outputs['y_all']
                y_hat_all = outputs['y_hat_all']
                fig, ax = plt.subplots()
                ax.plot(weighted_global_mean(y_all, self.lats.cpu()).cpu().numpy(), label='True')
                ax.plot(weighted_global_mean(y_hat_all, self.lats.cpu()).cpu().numpy(), label='Predicted')
                ax.set_xlabel('time')
                ax.set_ylabel(f'{self.outputs} value')
                ax.set_title(f'Val {simu_name} – best {self.monitor_metric}: {self.best_val_score:.4f}')
                ax.legend()
                self.logger.experiment.add_figure(f'Figure/val_best_true_vs_predicted_{simu_name}', fig)

        # Étapes pour afficher tous les hyperparamètres dans TensorBoard
        valid_types = (int, float, str, bool, list, tuple)
        
        flat_hparams = {}
        config = self.hparams['config']  # niveau 1 : data, train
        for section_k, section_v in config.items():
            if isinstance(section_v, dict):  # niveau 2 : les valeurs
                for k, v in section_v.items():
                    if isinstance(v, valid_types):
                        flat_hparams[f"{section_k}/{k}"] = v
            elif isinstance(section_v, valid_types):
                flat_hparams[section_k] = section_v

        self.logger.log_hyperparams(
            flat_hparams,
            {f"hp/{self.monitor_metric}": self.best_val_score,
             "hp/final_duration": sum(self.time_per_epoch),
             "hp/avg_epoch_duration": np.mean(self.time_per_epoch)
            }
        )
        self.logger.experiment.flush()

    def test_step(self, batch, batch_idx):
        x, y, t, simu = batch
        y_hat, loss = self.common_step(x, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            
        batch_dict = {"loss": loss}
        for metric_name, metric in self.metrics_dict.items():
            print(y_hat.shape, y.shape)
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
        self.test_step_simus.append(simu)

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

        test_nrmse = NRMSE_ClimateBench(y_hat_all, y_all, self.lats) # all time period
        self.log('test_nrmse', test_nrmse)
        self.log('loss', df['loss'].mean())
        spatial_corr = self.spatial_corr_metric.compute()
        self.log("test_corr", spatial_corr)

        outputs_per_simu = {}
        for y, y_hat, t, simu in zip(
            self.test_step_outputs_true,
            self.test_step_outputs_hat,
            self.test_step_times,
            self.test_step_simus 
        ):
            simu_name = simu[0]
            if simu_name not in outputs_per_simu:
                outputs_per_simu[simu_name] = {'true': [], 'hat': [], 'times': []}
            outputs_per_simu[simu_name]['true'].append(y)
            outputs_per_simu[simu_name]['hat'].append(y_hat)
            outputs_per_simu[simu_name]['times'].append(t)

        for simu_name, outputs in outputs_per_simu.items():
            y_s = torch.cat(outputs['true'], dim=0).view(-1, self.img_size[0], self.img_size[1])
            y_hat_s = torch.cat(outputs['hat'], dim=0).view(-1, self.img_size[0], self.img_size[1])
            t_s = torch.cat(outputs['times']).cpu().numpy()

            fig, ax = plt.subplots()
            ax.plot(weighted_global_mean(y_s, self.lats).cpu().numpy(), label='True')
            ax.plot(weighted_global_mean(y_hat_s, self.lats).cpu().numpy(), label='Predicted')
            ax.set_xlabel('time')
            ax.set_ylabel(f'{self.outputs} value')
            ax.set_title(f'Test {simu_name} {t_s[0,0]:.0f}-{t_s[-1,0]:.0f} – NRMSE : {test_nrmse:.4f}')
            ax.legend()
            self.logger.experiment.add_figure(f'Figure/test_true_vs_predicted_{simu_name}', fig)

            eval = EvaluationPlots(simulation_name=simu_name,
                                var_name=self.outputs[0],
                                config_plots=self.config_plots)
            eval.plot_error_maps(y_s.cpu().numpy(),
                                y_hat_s.cpu().numpy(),
                                title=f'{simu_name} {t_s[0,0]:.0f}-{t_s[-1,0]:.0f}',
                                save_path=Path(self.logger.log_dir) / f'error_maps_{simu_name}.png')

    def configure_optimizers(self):
        #optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "epoch",
            "monitor": self.monitor_metric
        }
    }
