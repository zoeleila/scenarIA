import numpy as np
from sklearn.linear_model import LinearRegression
import yaml

from scenarIA.src.utils.datautils import weighted_global_mean
from scenarIA.src.data.dataloader import get_dataset, get_dataloaders
from scenarIA.src.utils.settings import CONFIG_DIR, DATASET_DIR
from tests import test

class PatternScaling(object):
    """
    Does pattern scaling. Here we fit one linear model per 
    grid point. The linear model maps a global variable, 
    e.g., cum. CO2 emissions or temperature, to 
    the grid point's local value. 
    This model captures temporal patterns in each grid cell.
    The model is local, i.e., it will be independent of 
    neighboring grid points. The model is linear in time, 
    i.e., it assumes no non-linearly amplifying feedbacks 
    between the global and local variable.
    """
    def __init__(self, deg=1):
        """
        Args:
            deg int: degree of polynomial fit. Default is 1 
                for linear fit.
        """
        self.deg = deg
        self.coeffs = None

    def train(self, in_global, out_local):
        """
        Fits polynomial with degree self.deg from in_global to 
        every location in out_local. Choose deg=1 for linear fit.

        Args:
            in_global np.array((n_t,)): The model input is a 
                global variable, e.g., annual global mean surface 
                temperature anomalies of in °C
            out_local np.array((n_t,n_lat,n_lon)): The model 
                output is a locally-resolved variable. E.g., annual 
                mean surface temperature anomalies at every lat,lon
                in °C
        Sets:
            coeffs np.array((deg+1, n_lat, n_lon))
        """
        n_t, n_lat, n_lon = out_local.shape
        
        # Preprocess data, by flattening in space
        out_local = out_local.reshape(n_t,-1) # (n_t, n_lat*n_lon)

        # Fit linear regression coefficients to every grid point
        self.coeffs = np.polyfit(in_global, out_local, deg=self.deg) # (2, n_lat*n_lon)

        # Reshape coefficients onto locally-resolved grid
        self.coeffs = self.coeffs.reshape(-1, n_lat, n_lon) # (2, n_lat, n_lon)

    def predict(self, in_global):
        """
        Args:
            in_global np.array((n_t,))
        Returns:
            preds np.array((n_t, n_lat, n_lon))
        """
        n_lat = self.coeffs.shape[1]
        n_lon = self.coeffs.shape[2]

        # Predict by applying pattern scaling coefficients on locally-resolved grid
        in_global = np.tile(in_global[:,np.newaxis, np.newaxis], reps=(1,n_lat,n_lon)) # repeat onto local grid to get shape (n_t, n_lat, n_lon)
        preds = np.polyval(self.coeffs, in_global) # (n_t, n_lat, n_lon)

        return preds
    

if __name__ == "__main__":
    
    with open(CONFIG_DIR / 'config.yaml') as file:
        config = yaml.safe_load(file)
    
    config['data']['seq_length'] = 1
    lat = dict(np.load(DATASET_DIR / config['data']['dataset_path'] / 'coords.npz', allow_pickle=True))['lat']
    lon = dict(np.load(DATASET_DIR / config['data']['dataset_path'] / 'coords.npz', allow_pickle=True))['lon']
    
    # Train
    train_dataloader = get_dataloaders(config=config, data_type='train', transforms=True)
    train_in = []
    train_out = []
    for batch in train_dataloader:
        x, y, _, _ = batch
        train_in.append(x)
        train_out.append(y)
    train_out = np.concatenate(train_out, axis=0).squeeze() # shape (n, n_lat, n_lon, channels)
    train_out = np.expand_dims(train_out, axis=-1) # shape (n, n_lat, n_lon, channels=1)
    print("train_out shape before transpose:", train_out.shape)
    train_in = np.concatenate(train_in, axis=0).squeeze() # shape (n, n_lat, n_lon)

    # data shuffle ??
    print("train_in shape:", train_in.shape)
    print("train_out shape:", train_out.shape)

    # Fit global to global
    train_in_global = np.stack([weighted_global_mean(train_in[..., i], lats=lat) for i in range(train_in.shape[-1])], 
                               axis=0).transpose() # n, channels
    train_out_global = np.stack([weighted_global_mean(train_out[..., i], lats=lat) for i in range(train_out.shape[-1])], 
                                axis=0).transpose()
    print("train_in_global shape:", train_in_global.shape)
    print("train_out_global shape:", train_out_global.shape)
    
    linear = LinearRegression()
    linear.fit(train_in_global,
               train_out_global)
    
    # to predi but shuffle feels weird
