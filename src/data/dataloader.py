from logging import config
from statistics import mean
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import numpy as np
from typing import Optional
import torch
from torch import Tensor
import yaml
from tqdm import tqdm
from pathlib import Path
import xarray as xr
import json
import os
import copy

from scenarIA.src.utils.transforms import ToTensor, Normalize
from scenarIA.src.utils.settings import RUNS_DIR, DATASET_DIR
from scenarIA.src.utils.datautils import weighted_global_mean


# plot dataset ?

class scenarIA(Dataset):
    def __init__(self,
                 transform: Optional[v2.Compose],
                 config,
                 data_type: str = 'train'):
        self.transform = transform
        self.data_type = data_type
        self.seed = config['train']['seed']
        self.dataset_path = DATASET_DIR / config['data']['dataset_path']
        self.inputs_var_list = config['train']['inputs']
        self.outputs_var_list = config['train']['outputs']
        self.val_size = config['train']['val_size']
        self.seq_length = config['data']['seq_length']
        self.moving_window = config['data'].get('moving_window', 1)
        self.predict_only_last_timestep = bool(config['data']['predict_only_last_timestep'])
        self.nb_subsets = config['data']['nb_subsets']
        self.seed_subsets = config['data']['seed_subsets']
        self.nb_member_per_subsets = config['data']['nb_member_per_subsets']
        self.one_to_many = bool(config['data']['one_to_many'])
        
        if data_type == 'val':
            self.simus = config['train']['simus_train']
        else :
            self.simus = config['train'][f'simus_{data_type}']
        
        if isinstance(self.simus, str):
            self.simus = [self.simus]

        if data_type == 'test' or data_type == 'inference':
            self.one_to_many = False

        self.load_inputs()
        if self.data_type == 'inference':
            self.outputs = None
        else:
            self.load_outputs()

        # annual mean
        # pi control

        if data_type == 'train' or data_type == 'val':
            rng = np.random.default_rng(self.seed)
            perm = rng.permutation(self.inputs.shape[0])
            self.inputs = self.inputs[perm]
            self.outputs = self.outputs[perm]
            self.time = [self.time[i] for i in perm]
        split_index = int(self.inputs.shape[0] * (1 - self.val_size))
        if self.data_type == 'train':
            self.inputs = self.inputs[:split_index]
            self.outputs = self.outputs[:split_index]
        elif self.data_type == 'val': 
            self.inputs = self.inputs[split_index:]
            self.outputs = self.outputs[split_index:]
        
        print('final inputs shape shuffle', self.inputs.shape)
        print('final outputs shape shuffle', self.outputs.shape)
        print('time length shuffle', len(self.time), self.time[0])
   

    def load_inputs(self):
        """
        Loads input (inputs) data from NetCDF files for each simulation
        specified in self.simus.
        inputs = {'simu_name': ds}
        """
        inputs_all = []

        for simu in self.simus:
            inputs_xr = xr.open_dataset(self.dataset_path / f'inputs_{simu}.nc')[self.inputs_var_list]

            for _ in range(self.nb_subsets if self.one_to_many else 1):
                inputs, _ = scenarIA.build_sequence_array_from_xr(
                    inputs_xr,
                    seq_length=self.seq_length,
                    predict_only_last_timestep=False,
                    moving_window=self.moving_window
                )
                inputs_all.append(inputs)

        inputs_concat = np.concatenate(inputs_all, axis=0) if len(inputs_all) > 0 else np.array([])
        self.inputs = inputs_concat
    
    def load_outputs(self):
        """
        Loads output (outputs) data from NetCDF files for each simulation
        specified in self.simus. Unified handling for one-to-one and one-to-many:
        - if self.one_to_many is False, n_subsets == 1 and we either take the ensemble mean
          (when member_size <= nb_member_per_subsets) or sample once.
        - if self.one_to_many is True, we draw self.nb_subsets random subsets with different seeds.
        """
        outputs_all = []
        time_all = []

        for simu in self.simus:
            outputs_xr_ensemble = xr.open_dataset(self.dataset_path / f'outputs_{simu}.nc')[self.outputs_var_list]
            member_size = outputs_xr_ensemble.member.size

            if self.data_type == 'test':  # always compare with best estimation of forced response
                self.nb_member_per_subsets = member_size

            n_subsets = self.nb_subsets if self.one_to_many else 1

            for i in range(n_subsets):
                if n_subsets == 1 and member_size <= self.nb_member_per_subsets:
                    outputs_xr = outputs_xr_ensemble.mean('member')
                else:
                    outputs_xr = self.get_random_member_subset(
                        outputs_xr_ensemble,
                        self.seed_subsets + i, # if n_subsets > 1, different seed for each subset
                        self.nb_member_per_subsets,
                        mean=True
                    )

                outputs, time = scenarIA.build_sequence_array_from_xr(
                    outputs_xr,
                    seq_length=self.seq_length,
                    predict_only_last_timestep=self.predict_only_last_timestep,
                    moving_window=self.moving_window
                )
                outputs_all.append(outputs)
                time_all += time

        self.outputs = np.concatenate(outputs_all, axis=0) if len(outputs_all) > 0 else np.array([])
        self.outputs = np.squeeze(self.outputs, axis=-1) # remove for multi variate ???
        self.time = time_all
    
    @staticmethod
    def get_random_member_subset(
                                 ds: xr.Dataset, 
                                 seed: int, 
                                 nb_member_per_subsets: int,
                                 mean: bool = True
                                 ) -> xr.Dataset:
        np.random.seed(seed)
        members = np.random.choice(ds.member.values, size=nb_member_per_subsets, replace=True)
        if mean:
            return ds.sel(member=members).mean('member')
        return ds.sel(member=members)

    @staticmethod
    def build_sequence_array_from_xr(ds: xr.Dataset, 
                              seq_length: int,
                              predict_only_last_timestep: bool = False,
                              moving_window: int = 1):
        '''
        Builds sequence samples from an xarray Dataset.

        Returns:
            data_array: numpy array of shape
                - (num_samples, seq_length, lat, lon, variables) if predict_only_last_timestep is False
                - (num_samples, 1, lat, lon, variables) if predict_only_last_timestep is True (only last timestep kept)
            times: list of time entries corresponding to each sample. Each entry is either an array of length seq_length
                   (when predict_only_last_timestep is False) or a single time value (when True).
        '''
        data = ds.to_array().transpose('time', 'lat', 'lon', 'variable').to_numpy()
        time = ds.time.values

        T = data.shape[0]
        step = max(1, int(moving_window))
        data_list = []
        time_list = []

        if seq_length > T:
            print('seq_length greater than time dimension, returning empty arrays')
            return np.array(data_list), time_list

        if predict_only_last_timestep:
            for i in range(0, T - seq_length + 1, step):
                last = data[i + seq_length - 1, ...]
                data_list.append(last.reshape(1, *last.shape))
                time_list.append(time[i + seq_length - 1])
        else:
            for i in range(0, T - seq_length + 1, step):
                data_list.append(data[i:i + seq_length, ...])
                time_list.append(time[i:i + seq_length])
        return np.array(data_list), time_list    
    
    def get_stats(self):
        mean = np.mean(self.inputs, axis=(0,1,2,3))
        std = np.std(self.inputs, axis=(0,1,2,3))
        min = np.min(self.inputs, axis=(0,1,2,3))
        max = np.max(self.inputs, axis=(0,1,2,3))
        return mean, std, min, max

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return self.inputs.shape[0]
    
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple[Tensor, Tensor, str or List(str)]: Transformed input (x) and target (outputs) tensors.
            t is either one date value or a list of dates depending on predict_only_last_timestep
        """
        x = self.inputs[idx, ...]
        y = self.outputs[idx, ...]
        t = self.time[idx]
        if self.transform:
            x, y = self.transform((x, y))
            x.float(), y.float()
        return x, y, t # TODO : return scenario name

def get_dataloaders(data_type: str, config:dict, transforms:bool=True) -> DataLoader:
    """
    Creates and returns a PyTorch DataLoader for the specified data type.
    Args:
        data_type (str): The type of data to load. Expected values are 'train' or other types
                            (e.g., 'validation', 'test'). Determines the shuffle behavior and batch size.
    Returns:
        DataLoader: A PyTorch DataLoader object configured with the appropriate dataset,
                    transformations, batch size, and shuffle settings.
    """
    seed = config['train']['seed']
    if transforms:
        runs_dir = RUNS_DIR / config['train']['runs_dir']
        statistic_file =  runs_dir.parent / 'statistics.json' # only data settings
        if not statistic_file.exists():
            stats = compute_statistics(copy.deepcopy(config), seeds=seed)
        else:
           with open(statistic_file, 'r') as f:
               stats = json.load(f)
        if str(seed) not in stats:
           stats = compute_statistics(copy.deepcopy(config), seeds=seed)
        transforms = v2.Compose([ToTensor(),
                                 Normalize(stats = stats[str(seed)])])
    else:
        transforms = v2.Compose([ToTensor()])
    
    dataset = scenarIA(transform=transforms,
                            config=config,
                            data_type=data_type)
    
    if data_type == 'train':
        batch_size = config['train']['batch_size']
    else : 
        batch_size = 1

    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=False,
                            num_workers=1)
    return dataloader


def compute_statistics(config, seeds: int = 42):
    if isinstance(seeds, int):
        seeds = [seeds]
    runs_dir = RUNS_DIR / config['train']['runs_dir']
    stats_path = runs_dir.parent / 'statistics.json'
    config['data']['one_to_many'] = False
    config['data']['seq_length'] = 1

    if stats_path.exists():
        try:
            with open(stats_path, 'r') as f:
                stats = json.load(f)
        except Exception:
            stats = {}
    else:
        stats = {}

    for seed in seeds:
        config['train']['seed'] = seed
        dataset = scenarIA(transform=None,
                            config=config,
                            data_type='train')
        mean_arr, std_arr, min_arr, max_arr = dataset.get_stats()
        seed_key = str(seed)
        stats.setdefault(seed_key, {})
        for i, var in enumerate(config['train']['inputs']):
            stats[seed_key][var] = {
                'mean': float(mean_arr[i]),
                'std': float(std_arr[i]),
                'min': float(min_arr[i]),
                'max': float(max_arr[i])
            }

    os.makedirs(runs_dir, exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    return stats


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    config_path = Path('/gpfs-calypso/home/globc/garcia/scenarIA/configs/config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    dataset = scenarIA(config=config, transform=v2.Compose([ToTensor()]), data_type='test')
    outputs = dataset.outputs
    outputs = outputs.reshape(-1, outputs.shape[-2], outputs.shape[-1])

    plt.figure()
    plt.plot(weighted_global_mean(outputs, lats= np.linspace(-90, 90, outputs.shape[-2])), label='True')
    plt.plot(np.mean(outputs, axis=(1,2)), label='no weight')
    plt.legend()
    plt.savefig('utils/test1')
    