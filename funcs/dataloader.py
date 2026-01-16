from curses import window
from logging import config
from pickletools import int4
import time
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import numpy as np
import glob
from typing import Optional
import torch
from torch import Tensor
import argparse
import yaml
from tqdm import tqdm
from pathlib import Path
import re
import xarray as xr

from scenarIA.funcs.transforms import ToTensor, Normalize


class scenarIA(Dataset):
    def __init__(self,
                 transform: Optional[v2.Compose],
                 config,
                 data_type: str = 'train'):
        self.transform = transform
        self.data_type = data_type
        self.dataset_path = Path(config['data']['dataset_path'])
        self.inputs_var_list = config['train']['inputs']
        self.outputs_var_list = config['train']['outputs']
        self.val_size = config['train']['val_size']
        self.piControl_diff = config['data']['piControl_diff']
        self.seq_length = config['data']['seq_length']
        self.predict_only_last_timestep = bool(config['data']['predict_only_last_timestep'])
        self.nb_subsets = config['data']['nb_subsets']
        self.seed_subsets = config['data']['seed_subsets']
        self.nb_member_per_subsets = config['data']['nb_member_per_subsets']
        self.one_to_many = bool(config['data']['one_to_many'])
        
        if data_type == 'val':
            self.simus = config['train']['simus_train'] # The split between train and val is done after loading all the data.
        else :
            self.simus = config['train'][f'simus_{data_type}']

        if data_type == 'test' or data_type == 'inference':
            self.one_to_many = False

        self.load_inputs()
        if self.data_type == 'inference':
            self.outputs = None
        else:
            self.load_outputs()

        print('final inputs shape', self.inputs.shape, self.inputs.shape[0])
        print('final outputs shape', self.outputs.shape)
        print('time length', len(self.time), self.time[0], self.time)

        # annual mean
        # pi control

        '''
        split_index = int(len(list_samples) * (1 - stop))
        if self.data_type == 'train':
            self.samples = list_samples[:split_index]
        else: 
            self.samples = list_samples[split_index:]
        ''' 

    def load_inputs(self):
        """
        Loads input (inputs) data from NetCDF files for each simulation
        specified in self.simus.
        inputs = {'simu_name': ds}
        """

        inputs_all = []   
        time_all = []

        for simu in self.simus:
            print(simu)

            inputs_xr = xr.open_dataset(self.dataset_path / f'inputs_{simu}.nc')[self.inputs_var_list]

            if not self.one_to_many:
                print('one-to-one approach')
                inputs, time = self.build_sequence_samples_from_xr(inputs_xr, 'inputs')
                inputs_all.append(inputs)
                time_all += time
            else:
                print('one-to-many approach')
                for i in range(self.nb_subsets):
                    inputs, time = self.build_sequence_samples_from_xr(inputs_xr, 'inputs')
                    inputs_all.append(inputs)
                    time_all += time

        inputs_concat = np.concatenate(inputs_all, axis=0) if len(inputs_all) > 0 else np.array([])
        self.inputs = inputs_concat
        self.time = time_all
    
    def load_outputs(self):
        """
        Loads output (outputs) data from NetCDF files for each simulation
        specified in self.simus.
        outputs = {'simu_name':[ds1, ds2,...]} # list of forced responses from random subsets. 
        If one-to-one, list of one element.
        """
        outputs_all = []

        for simu in self.simus:
            print(simu)

            outputs_xr_ensemble = xr.open_dataset(self.dataset_path / f'outputs_{simu}.nc')[self.outputs_var_list]

            if self.data_type == 'test': # always compare with the best estimation of forced response
                self.nb_member_per_subsets = outputs_xr_ensemble.member.size

            if not self.one_to_many:
                print('one-to-one approach')
                if outputs_xr_ensemble.member.size == self.nb_member_per_subsets or outputs_xr_ensemble.member.size < self.nb_member_per_subsets:
                    outputs_xr = outputs_xr_ensemble.mean('member')
                else:
                    outputs_xr = self.get_random_member_subset(outputs_xr_ensemble, 
                                                                self.seed_subsets, 
                                                                self.nb_member_per_subsets,
                                                                mean=True)
                outputs, _ = self.build_sequence_samples_from_xr(outputs_xr, 'outputs')
                outputs_all.append(outputs)
            else:   
                print('one-to-many approach')
                for i in range(self.nb_subsets):
                    ds_subset = self.get_random_member_subset(outputs_xr_ensemble, 
                                                            self.seed_subsets + i, 
                                                            self.nb_member_per_subsets,
                                                            mean=True)
                    outputs, _ = self.build_sequence_samples_from_xr(ds_subset, 'outputs')
                    outputs_all.append(outputs)
        outputs_concat = np.concatenate(outputs_all, axis=0) if len(outputs_all) > 0 else np.array([])
        self.outputs = outputs_concat 
    
    def get_random_member_subset(self, 
                                 ds: xr.Dataset, 
                                 seed: int, 
                                 nb_member_per_subsets: int,
                                 mean: bool = True) -> xr.Dataset:
        print('subsets')
        np.random.seed(seed)
        members = np.random.choice(ds.member.values, size=nb_member_per_subsets, replace=True)
        if mean:
            return ds.sel(member=members).mean('member')
        return ds.sel(member=members)

    def build_sequence_samples_from_xr(self, 
                              ds, 
                              input_or_outputs:str):
        '''
        Builds sequence samples from an xarray Dataset. Returns expected predicted time series.
        '''
        data_list = []
        time_list = []

        data = ds.to_array().transpose('time', 'lat', 'lon', 'variable').to_numpy()
        time = ds.time.values
        print('data shape', data.shape)

        if self.predict_only_last_timestep:
            print('last time step of sequence')
            for i in range(data.shape[0] - self.seq_length + 1):
                if input_or_outputs == 'inputs':
                    data_list.append(data[i:i+self.seq_length, ...])
                else:
                    data_list.append(data[i+self.seq_length-1, ...].reshape(1, data.shape[1], data.shape[2], data.shape[3]))
                time_list.append(time[i+self.seq_length-1])
        else: # moving window = seq_length
            print('whole sequence')
            for i in range(0, data.shape[0] - self.seq_length + 1, self.seq_length):

                data_list.append(data[i:i+self.seq_length, ...])
                time_list.append(time[i:i+self.seq_length])
            # add moving window < seq_length at the end ?
        print('data_list shape', np.array(data_list).shape)
        return np.array(data_list), time_list

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
        return x, y, t

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
    if transforms:
        transforms = v2.Compose([ToTensor(),
                                 Normalize(dataset_dir=Path(config['data']['dataset_dir']))])
    else:
        transforms = v2.Compose([ToTensor()])
    
    dataset = scenarIA(transform=transforms,
                            config=config,
                            data_type=data_type)
    
    if data_type == 'train':
        batch_size = config['train']['batch_size']
    else : 
        batch_size = 1
    
    if data_type == 'train':
        shuffle = True
    else: 
        shuffle = False

    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle,
                            num_workers=1)
    return dataloader


if __name__ == "__main__":
    config_path = Path('/gpfs-calypso/home/globc/garcia/scenarIA/configs/config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    train_dataloader = get_dataloaders(data_type='train', config=config, transforms=False)
    for i, batch in enumerate(train_dataloader):
        x, y, t = batch
        print(t)
        print('x shape:', x.shape)
        print('y shape:', y.shape)


