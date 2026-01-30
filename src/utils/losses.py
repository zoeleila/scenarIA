import torch
import torch.nn as nn

from scenarIA.src.utils.datautils import compute_weights_from_lats

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

######################################
# Losses from ClimateSET
######################################


class RMSELoss(nn.Module):
    def __init__(self, reduction: str = "none", mask=None):
        super().__init__()
        self.mask = mask

        if reduction == "none":
            self.reduction_fn = None
        elif reduction == "mean":
            self.reduction_fn = torch.mean
        elif reduction == "sum":
            self.reduction_fn = torch.sum
        else:
            raise NotImplementedError

        self.mse = nn.MSELoss(reduction="none")  # mean over all dimensions

    def forward(self, pred, y):
        error = torch.sqrt(self.mse(pred, y))
        if self.mask is not None:
            error = (
                error.mean(dim=1) * self.mask
            ).sum() / self.mask.sum()  # TODO: check

        if self.reduction_fn is not None:
            error = self.reduction_fn(error)

        return error


class NRMSELoss_s_ClimateBench(nn.Module):
    """
    Spatial normalized weighted RMSE taken from Climate Bench.
    Weigting to account for decreasing grid size towards the pole.
    """

    def __init__(self, deg2rad: bool = True):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")

        self.deg2rad = deg2rad

    def forward(self, pred, y):
        # weighting to account for decreasing grid-cell area towards pole
        # lattitude weights
        lat_size = y.shape[-2]
        lats = torch.linspace(-90, 90, lat_size)
        if self.deg2rad:
            weights = torch.cos((torch.pi * lats) / 180)
        else:
            weights = torch.cos(lats)
        weights = weights.to(device)

        # nrmses = sqrt((weights * (x_mean_t -y_mean_n_t)**2))_mean_s / ((weights*y)_mean_s)_mean_t_n
        # TODO: clarify with duncan why not mean over n with x..
        nrmse_s = torch.sqrt(
            self.weighted_global_mean(
                (pred.mean(dim=(0, 1)) - y.mean(dim=(0, 1))) ** 2, weights
            )
        ) / self.weighted_global_mean(y, weights).mean(dim=(0, 1))

        return nrmse_s

    def weighted_global_mean(self, x, weights):
        # weitghs * x summed over lon lat / lon+lat
        x = x.transpose(-1, -2)
        print(weights)
        return torch.mean(x * weights, dim=(-1, -2))


class NRMSELoss_g_ClimateBench(nn.Module):
    """
    Spatial normalized weighted RMSE taken from Climate Bench.
    Weigting to account for decreasing grid size towards the pole.
    """

    def __init__(self, deg2rad: bool = True):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")

        self.deg2rad = deg2rad

    def forward(self, pred, y):
        # weighting to account for decreasing grid-cell area towards pole
        # lattitude weights
        if self.deg2rad:
            weights = torch.cos((torch.pi * torch.arange(y.shape[-2])) / 180)
        else:
            weights = torch.cos(torch.arange(y.shape[-2]))

        weights = weights.to(device)
        

        # nrmseg = sqrt(((x - ( (weights * y_mean_t)_mean_s)**2)_mean_t )  ) / ((weights*y)_mean_s)_mean_t_n
        denom = self.weighted_global_mean(y, weights).mean(dim=(0, 1))

        # TODO: clarify with duncan when to mean over samples for predictions? before or after sqrt?
        nrmse_g = (
            torch.sqrt(
                (
                    self.weighted_global_mean(pred.mean(dim=0), weights)
                    - self.weighted_global_mean(y.mean(dim=0), weights) ** 2
                ).mean(dim=(0))
            )
            / denom
        )

        return nrmse_g

    def weighted_global_mean(self, x, weights):
        # weitghs * x summed over lon lat / lon+lat
        x = x.transpose(-1, -2)
        print(weights)
        return torch.mean(x * weights, dim=(-1, -2))


class NRMSELoss_ClimateBench(nn.Module):
    """
    Combination of global weighted and spatially weighted nrmse.

    """

    def __init__(self, deg2rad: bool = True, alpha: int = 5):
        super().__init__()

        self.nrmse_g = NRMSELoss_g_ClimateBench(deg2rad)
        self.nrmse_s = NRMSELoss_s_ClimateBench(deg2rad)
        self.alpha = alpha

    def forward(self, pred, y):
        nrmseg = self.nrmse_g(pred, y)
        nrmses = self.nrmse_s(pred, y)
        nrmse = nrmses + self.alpha * nrmseg
        return nrmse
    
class LLweighted_MSELoss_Climax(nn.Module):
    """
    Latitude weighted mean squared error taken from ClimaX.
    Mean over all grid point weighted MSE
    
    y : [batch, time, lat, lon]
    pred : [batch, time, lat, lon]
    weights : [lat, lon]

    returns scalar
    """

    def __init__(self, lats, deg2rad: bool = True, mask=None):
        super().__init__()

        self.mse = nn.MSELoss(reduction="none")
        self.deg2rad = deg2rad
        self.mask = mask
        self.lats = lats

    def forward(self, pred, y):
        mse = self.mse(pred, y)
        weights = compute_weights_from_lats(lats=self.lats, 
                                            deg2rad=self.deg2rad)

        weights = weights.to(device)
        if self.mask is not None:
            error = (mse * weights * self.mask).sum() / self.mask.sum()
        else:
            error = (mse * weights).mean()

        return error

######################################
# Losses from ClimateBench (Lutjens et al. 2023) adapted to PyTorch
######################################

class LatWeightedMeanSquaredError(nn.Module):
    """
    Sum over all grid point weighted MSE
    """
    def __init__(self,
                reduction="sum_over_batch_size",
                device='cpu'):
        super().__init__()
        
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction = 'none')
        self.device = device

    def forward(self, y, y_hat):
        # I verified that the above code applies weights correctly with the below snippet:
        # weights_np_repeats = np.repeat(weights_np[None,None,...], repeats=Y_val_all.shape[0], axis=0)
        # weighted_mse_repeats = mse * weights_np_repeats
        # assert torch.all(weighted_mse == weighted_mse_repeats)
        lat_size = y.shape[-2]
        lats = torch.linspace(-90, 90, lat_size)
        weights = torch.cos((torch.pi * lats) / 180)
        weights = weights.expand(lat_size, y.shape[-1])  # expand to lon dimension
        weights = weights / weights.sum()  # normalize weights
        weights = weights.to(self.device)

        mse = self.mse(y, y_hat)
        mse = mse * weights
        mse = mse.sum(axis=torch.arange(len(mse.shape))[1:].tolist()) # reduce over time, lat, lon ????
        if self.reduction == "sum_over_batch_size":
            mse = mse.sum()
        elif self.reduction == "avg_over_batch_size":
            mse = mse.mean()
        return mse

    

if __name__ == "__main__":
    batch_size = 16
    out_time = 10
    lon = 192
    lat = 96
    dummy = torch.rand(size=(batch_size, out_time, lat, lon))

    targets = torch.rand(size=(batch_size, out_time, lat, lon))

    loss_fn = LLweighted_MSELoss_Climax(lats = torch.linspace(-90, 90, lat))
    loss = loss_fn(dummy, targets)
    print(loss)