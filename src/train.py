import random
import numpy as np
import torch
import yaml
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import pytorch_lightning as pl

from scenarIA.src.data.dataloader import get_dataloaders
from scenarIA.src.data.lightning_module import scenarIALightningModule
from scenarIA.src.utils.utils import save_infos_from_config, test_name_from_config
from scenarIA.src.utils.settings import CONFIG_DIR, RUNS_DIR, DATASET_DIR

torch.cuda.is_available()

def run(config):
    save_infos_from_config(config)
    test_name = test_name_from_config(config)
    config['train']['test_name'] = test_name
    config['train']['runs_dir'] = config['train']['runs_dir'] + test_name
    lats = dict(np.load(DATASET_DIR / config['data']['dataset_path'] / 'coords.npz', allow_pickle=True))['lat']

    seed = config['train'].get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Data
    train_dataloader = get_dataloaders('train', config)
    val_dataloader = get_dataloaders('val', config)
    test_dataloader = get_dataloaders('test', config)

    # Model
    model = scenarIALightningModule(config, lats=torch.tensor(lats))

    # Logger
    logger = TensorBoardLogger(save_dir=RUNS_DIR / config['train']['runs_dir'], 
                               name='lightning_logs',
                               default_hp_metric=False)
    
    # Callbacks
    monitor = config['train']['monitor_metric']

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        filename='best-checkpoint-{epoch:02d}-' + monitor + '-{' + monitor + ':.2f}',
        save_top_k=1,
        mode='min',
        save_last=True
    )

    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=config['train'].get('early_stopping_patience', 10),
        mode='min',
        verbose=True,
        check_finite=True,   # arrête si val_rmse devient NaN ou inf
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')


    torch.set_float32_matmul_precision('high') # For hybrid partition

    trainer = pl.Trainer(max_epochs=config['train']['max_epochs'], 
                        default_root_dir= RUNS_DIR / config['train']['runs_dir'],
                        log_every_n_steps=1,
                        accelerator="gpu",
                        devices="auto",
                        precision='16-mixed',
                        logger=logger,
                        callbacks=[checkpoint_callback, early_stopping, lr_monitor])

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, dataloaders=test_dataloader, ckpt_path='best')

if __name__ == "__main__":
    parser= argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    args = parser.parse_args()

    with open(CONFIG_DIR / args.config) as file:
        config = yaml.safe_load(file)
    run(config)