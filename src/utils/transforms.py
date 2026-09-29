import torch
from torch import Tensor
import numpy as np
import json

from scenarIA.src.utils.datautils import shift_longitudes
import matplotlib.pyplot as plt
from scenarIA.src.utils.settings import RUNS_DIR, DATASET_DIR, CONFIG_DIR, GRAPHS_DIR

class ToTensor:
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample: tuple[np.ndarray, np.ndarray]) -> tuple[Tensor, Tensor]:
        x, y = sample
        if y is None:
            return torch.tensor(x, dtype=torch.float32), y
        else:
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        
class Normalize:
    """Normalize a tensor sample with mean and standard deviation."""
    def __init__(self, stats):
        self.mean = torch.tensor([stats[c]['mean'] for c in stats])
        self.std = torch.tensor([stats[c]['std'] for c in stats])

    def __call__(self, sample: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        x, y = sample
        x = [(x[..., c] - self.mean[c]) / self.std[c] for c in range(x.shape[-1])]
        x = torch.stack(x, dim=-1)
        return x, y
    
class DeNormlize:
    def __init__(self, dataset_dir):
        stats_path = dataset_dir / 'statistics.json'
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        self.mean = torch.tensor([stats[c]['mean'] for c in stats])
        self.std = torch.tensor([stats[c]['std'] for c in stats])

    def __call__(self, sample: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        x, y = sample
        x = [(x[..., c] * self.std[c]) + self.mean[c] for c in range(x.shape[-1])]
        x = torch.stack(x, dim=-1)
        return x, y
    
class DiffClimatology: # a modifier
    """Subtract climatology from tensor sample."""
    def __init__(self, climatology=None, add_clim_to_predictors=False):
        # climatology: Tensor of shape (lat, lon, channels)

        self.climatology = climatology # TODO change for multivariate
        self.add_clim_to_predictors = add_clim_to_predictors

    def __call__(self, sample: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        # y shape: (time, lat, lon, channels)
        x, y = sample
        if self.climatology is None:
            return x, y
        else:
            if y is not None: # inference mode
                y = y - self.climatology
            if self.add_clim_to_predictors:
                mean = self.climatology.mean()
                std = self.climatology.std()
                self.clim_normalized = (self.climatology - mean) / (std + 1e-8)
                clim = self.clim_normalized[None, :, :, None].expand(x.shape[0], -1, -1, -1)
                x = torch.cat((x, clim), dim=-1)
            return x, y

class LonShift:
    """Convertit les longitudes de [0, 360] vers [-180, 180].

    Les données x et y sont réordonnées le long de la dimension -2 (longitude)
    pour que les longitudes soient croissantes dans [-180, 180].
    """

    def __init__(self, lonshift=False, lon: Tensor=None):
        self.lonshift = lonshift
        lon = torch.as_tensor(lon)
        new_lon = shift_longitudes(lon)
        self.idx = torch.argsort(new_lon, stable=True)
        self.lon = new_lon[self.idx]

    def __call__(self, sample: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        x, y = sample
        if self.lonshift:
            x = x.index_select(-2, self.idx.to(x.device)) # B, T, H, W, C
            y = y.index_select(-1, self.idx.to(y.device)) # B, T, H, W
        return x, y
class LonUnshift:
    """Convertit les longitudes de [-180, 180] vers [0, 360].

    Les données x et y sont réordonnées le long de la dimension -2 (longitude)
    pour que les longitudes soient croissantes dans [0, 360].
    """

    def __init__(self, lonshift=False, lon: Tensor=None):
        self.lonshift = lonshift
        lon = torch.as_tensor(lon)
        # Longitudes converties : -180..180 -> 0..360
        new_lon = lon % 360
        # Indices de permutation pour trier les longitudes de façon croissante
        self.idx = torch.argsort(new_lon, stable=True)
        self.lon = new_lon[self.idx]  # nouveau tenseur de longitudes

    def __call__(self, sample: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        x, y = sample
        if self.lonshift:
            x = x.index_select(-2, self.idx.to(x.device))
            y = y.index_select(-1, self.idx.to(y.device))
        return x, y


# padding
# inerpol