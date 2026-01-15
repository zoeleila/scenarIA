from curses import window
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

        inputs, outputs = self.load_xr_data(Path(config['data']['dataset_path']),
                                 inputs_list=config['train']['inputs'],
                                 outputs_list=config['train']['outputs'],
                                 seed_subsets=config['data']['seed_subsets'],
                                 nb_member_per_subsets=config['data']['nb_member_per_subsets'],
                                 nb_subsets=config['data']['nb_subsets'])
        print('inputs', inputs)
        print('outputs', outputs)

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
                     nb_subsets: int,
                     ) -> tuple[dict, dict]:
        """
        Loads input (inputs) and output (outputs) data from NetCDF files for each simulation
        specified in self.simus.
        inputs = {'simu_name': ds}
        outputs = {'simu_name':[ds1, ds2,...]} # list of forced responses from random subsets. 
        If one-to-one, list of one element.
        Returns:
            tuple[dict, dict]: Two dictionaries containing input and output data
            for each simulation.
        """

        inputs, outputs = {}, {}
        for simu in self.simus:
            print(simu)
            
            inputs[simu] = xr.open_dataset(dataset_path / f'inputs_{simu}.nc')[inputs_list]
            
            ds_ensemble = xr.open_dataset(dataset_path / f'outputs_{simu}.nc')[outputs_list]

            # case with no subsets. Ex : all members used or not enough members
            if ds_ensemble.member.size == nb_member_per_subsets or ds_ensemble.member.size < nb_member_per_subsets:
                print('all members used or not enough members')
                ds_ensemble_mean = ds_ensemble.mean('member')
                outputs[simu] = [ds_ensemble_mean]

            # case with subsets. Ex : not all members used or one-to-many approach
            else:
                outputs_subsets = []
                if nb_subsets > 1: # one-to-many approach
                    print('one-to-many approach')
                    for i in range(nb_subsets):
                        ds_subset = self.get_random_member_subset(ds_ensemble, 
                                                                seed_subsets + i, 
                                                                nb_member_per_subsets,
                                                                mean=True)
                        outputs_subsets.append(ds_subset)
                    outputs[simu] = outputs_subsets
                else: # one-to-one approach
                    print('one-to-one approach')
                    ds_ensemble_mean = self.get_random_member_subset(ds_ensemble, 
                                                                seed_subsets, 
                                                                nb_member_per_subsets,
                                                                mean=True)
                    outputs[simu] = [ds_ensemble_mean]

            print(f'nombre de sous ensemble pour {simu}', len(outputs[simu]))
            
        return inputs, outputs
    
    
    def get_random_member_subset(self, 
                                 ds: xr.Dataset, 
                                 seed: int, 
                                 nb_member_per_subsets: int,
                                 mean: bool = True) -> xr.Dataset:
        np.random.seed(seed)
        members = np.random.choice(ds.member.values, size=nb_member_per_subsets, replace=True)
        if mean:
            return ds.sel(member=members).mean('member')
        return ds.sel(member=members)

    def build_samples_from_xr(self, 
                              inputs_dict, 
                              outputs_dict,
                              seq_length: int,
                              predict_only_last_timestep: bool):
        inputs_list = []
        outputs_list = []
        for simu in self.simus:
            inputs = inputs_dict[simu].to_array().transpose('time', 'lat', 'lon', 'variable').to_numpy()
            for ds_output in outputs_dict[simu]:
                outputs = ds_output.to_array().transpose('time', 'lat', 'lon', 'variable').to_numpy()
                if predict_only_last_timestep:
                    for i in range(inputs.shape[0] - seq_length + 1):
                        inputs_list.append(inputs[i:i+seq_length, ...])
                        outputs_list.append(outputs[i+seq_length-1, ...])
                else: # moving window = seq_length
                    for i in range(0, inputs.shape[0] + 1, seq_length):
                        inputs_list.append(inputs[i:i+seq_length, ...])
                        outputs_list.append(outputs[i:i+seq_length, ...])
                # add moving window < seq_length at the end ?
                        
        return np.array(inputs_list), np.array(outputs_list)

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
            tuple[Tensor, Tensor]: Transformed input (x) and target (outputs) tensors.
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

    # test 1 : tout les membres sont moyennés les 50
    config['data']['nb_member_per_subsets'] = 30
    config['data']['nb_subsets'] = 4 # pas utilisé en principe
    dataset = scenarIA(transform=None, config=config, data_type='train')

