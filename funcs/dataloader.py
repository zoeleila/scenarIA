from logging import config
from pickletools import int4
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


class scenarIA(Dataset):
    def __init__(self,
                 transform: Optional[v2.Compose],
                 config,
                 data_type: str = 'train'):
        self.transform = transform
        self.data_type = data_type
        self.dataset_path = Path(config['data']['dataset_path'])
        self.inputs = config['train']['inputs']
        self.outputs = config['train']['outputs']
        self.val_size = config['train']['val_size']
        self.piControl_diff = config['data']['piControl_diff']
        self.seq_length = config['data']['seq_length']
        self.predict_only_last_timestep = config['data']['predict_only_last_timestep']

        if self.data_type == 'train' or self.data_type == 'val':
            self.simus = config['train']['simus_train']
        else:
            self.simus = config['train']['simus_test']

        x, y = self.load_xr_data(Path(config['data']['dataset_path']),
                                 inputs_list=config['train']['inputs'],
                                 outputs_list=config['train']['outputs'],
                                 seed_subsets=config['data']['seed_subsets'],
                                 nb_member_per_subsets=config['data']['nb_member_per_subsets'],
                                 nb_subsets=config['data']['nb_subsets'])
        print('x', x)
        print('y', y)
        # annual mean
        # concatenate with historical is necessary (after mean of members)
        # pi control
        # define sequence length
        # return x = np.array(num_samples, seq_length, height, width, channels)
        # return y = np.array(num_samples, seq_length or last time step, height, width, channels)

        '''
        split_index = int(len(list_samples) * (1 - stop))
        if self.data_type == 'train':
            self.samples = list_samples[:split_index]
        else: 
            self.samples = list_samples[split_index:]
        ''' 
    def load_xr_data(self,
                     dataset_path,
                     inputs_list,
                     outputs_list,
                     seed_subsets: int,
                     nb_member_per_subsets: int,
                     nb_subsets: int
                     ) -> tuple[dict, dict]:
        """
        Loads input (x) and output (y) data from NetCDF files for each simulation
        specified in self.simus.
        x = {'simu_name': ds}
        y = {'simu_name':[ds1, ds2,...]} # list of forced responses from random subsets. 
        If one-to-one, list of one element.
        Returns:
            tuple[dict, dict]: Two dictionaries containing input and output data
            for each simulation.
        """
        x, y = {}, {}
        for simu in self.simus:
            print(simu)
            #### x
            x[simu] = xr.open_dataset(dataset_path / f'inputs_{simu}.nc')[inputs_list]
            # add historical if needed

            #### y (supposing all exp have the same number of member)
            y_subsets = []
            ds_ensemble = xr.open_dataset(dataset_path / f'outputs_{simu}.nc')[outputs_list]
            # individual members ??? nb_subsets = nb max membre et nb_member_per_subsets = 1
            if ds_ensemble.member.size == nb_member_per_subsets or ds_ensemble.member.size < nb_member_per_subsets:
                print('all members used or not enough members')
                y[simu] = [ds_ensemble.mean('member')]
            else: 
                if nb_subsets > 1: # one-to-many approach
                    print('one-to-many approach')
                    for i in range(nb_subsets):
                        ds_subset = self.get_random_member_subset(ds_ensemble, 
                                                                seed_subsets + i, 
                                                                nb_member_per_subsets,
                                                                mean=True)
                        y_subsets.append(ds_subset)
                    y[simu] = y_subsets
                else: # one-to-one approach
                    print('one-to-one approach')
                    ds_ensemble_mean = self.get_random_member_subset(ds_ensemble, 
                                                                seed_subsets, 
                                                                nb_member_per_subsets,
                                                                mean=True)
                    y[simu] = [ds_ensemble_mean]

            print('yshape', len(y[simu]))
            
        return x, y
    
    
    def get_random_member_subset(self, 
                                 ds: xr.Dataset, 
                                 seed: int, 
                                 nb_member_per_subsets: int,
                                 mean: bool = True) -> xr.Dataset:
        np.random.seed(seed)
        members = np.random.choice(ds.member.values,
                                   size=nb_member_per_subsets,
                                   replace=False)
        if mean:
            return ds.sel(member=members).mean('member')
        return ds.sel(member=members)


    def get_ivar_to_predict(self) -> int:
        """
        Returns the index of the variable to predict based on its name.
        """
        return self.vars_to_predict.index(self.var_to_predict)


    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple[Tensor, Tensor]: Transformed input (x) and target (y) tensors.
        """
        data = dict(np.load(self.samples[idx], allow_pickle=True))
        x, y = data['x'], data['y'][:,:,:,self.get_ivar_to_predict()]
        if self.transform:
            x, y = self.transform((x, y))
            x.float(), y.float()
        return x, y 

if __name__ == "__main__":
    config_path = Path('/gpfs-calypso/home/globc/garcia/scenarIA/configs/config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    dataset = scenarIA(transform=None, config=config, data_type='train')

