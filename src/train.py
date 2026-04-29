from ast import parse
import random
import numpy as np
from traitlets import default
import torch
import yaml
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
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

    train_dataloader = get_dataloaders('train', config)
    val_dataloader = get_dataloaders('val', config)
    test_dataloader = get_dataloaders('test', config)

    model = scenarIALightningModule(config, lats=torch.tensor(lats))

    logger = TensorBoardLogger(save_dir=RUNS_DIR / config['train']['runs_dir'], 
                               name='lightning_logs',
                               default_hp_metric=False)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", 
        filename='best-checkpoint-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )

    checkpoint_last = ModelCheckpoint(
    filename='last-{epoch:02d}',
    save_last=True,       # toujours écrase le précédent
)

    torch.set_float32_matmul_precision('high') # For hybrid partition

    trainer = pl.Trainer(max_epochs=config['train']['max_epochs'], 
                        default_root_dir= RUNS_DIR / config['train']['runs_dir'],
                        log_every_n_steps=1,
                        accelerator="gpu",
                        devices="auto",
                        precision='16-mixed',
                        logger=logger,
                        callbacks=[checkpoint_callback, checkpoint_last])

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, dataloaders=test_dataloader, ckpt_path='best')

if __name__ == "__main__":
    parser= argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    args = parser.parse_args()

    with open(CONFIG_DIR / args.config) as file:
        config = yaml.safe_load(file)
    run(config)