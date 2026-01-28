"""
Metrics from ClimateSET 
y : [batch, time, lat, lon]
TODO : rewrite, the metrics of climax are prettier but doesn't not adapt to time serie data y : [batch, lat, lon, var]
"""

import torch


def weighted_global_mean(x, weights):
    weights = weights.unsqueeze(-1)  # make broadcastable
    return torch.mean(x * weights, dim=(-1, -2))

def MSE(preds: torch.Tensor, y: torch.Tensor):
    return torch.mean((preds - y) ** 2)


def RMSE(preds: torch.Tensor, y: torch.Tensor):
    return torch.mean(torch.sqrt(MSE(preds, y)))

def NRMSE_s_ClimateBench(preds: torch.Tensor, y: torch.Tensor, deg2rad: bool = True):
    """
    Spatial normalized weighted RMSE taken from Climate Bench.
    Weigting to account for decreasing grid size towards the pole.
    """

    # weighting to account for decreasing grid-cell area towards pole
    # lattitude weights
    lat_size = y.shape[-2]
    lats = torch.linspace(-90, 90, lat_size)
    if deg2rad:
        weights = torch.cos((torch.pi * lats) / 180)
    else:
        weights = torch.cos(lats)

    # nrmses = sqrt((weights * (x_mean_t -y_mean_n_t)**2))_mean_s / ((weights*y)_mean_s)_mean_t_n
    nrmse_s = torch.sqrt(
        weighted_global_mean(
            (preds.mean(axis=(0, 1)) - y.mean(axis=(0, 1))) ** 2, weights
        )
    ) / weighted_global_mean(y, weights).mean(axis=(0, 1))

    return nrmse_s


def NRMSE_g_ClimateBench(preds: torch.Tensor, y: torch.Tensor, deg2rad: bool = True):
    """
    Spatial normalized weighted RMSE taken from Climate Bench.
    Weigting to account for decreasing grid size towards the pole.
    """
    # weighting to account for decreasing grid-cell area towards pole
    # lattitude weights
    lat_size = y.shape[-2]
    lats = torch.linspace(-90, 90, lat_size)
    if deg2rad:
        weights = torch.cos((torch.pi * lats) / 180)
    else:
        weights = torch.cos(lats)


    denom = weighted_global_mean(y, weights).mean(axis=(0, 1))

    # denom is not alowed to be zero!
    if torch.any(preds == 0):
        preds[preds == 0] += 1e-6

    under_sqrt = (
        (
            weighted_global_mean(preds.mean(axis=0), weights)
            - weighted_global_mean(y.mean(axis=0), weights)
        )
        ** 2
    ).mean(axis=0)
    nrmse_g = (
        torch.sqrt(
            (
                weighted_global_mean(preds.mean(axis=0), weights)
                - weighted_global_mean(y.mean(axis=0), weights) ** 2
            ).mean(axis=(0))
        )
        / denom
    )

    return nrmse_g


def NRMSE_ClimateBench(preds: torch.Tensor, y: torch.Tensor, alpha: int = 5):
    """
    Combination of global weighted and spatially weighted nrmse.
    """

    nrmseg = NRMSE_g_ClimateBench(preds, y)
    nrmses = NRMSE_s_ClimateBench(preds, y)
    nrmse = nrmses + alpha * nrmseg
    return nrmse


def LLWeighted_RMSE_WheatherBench(preds: torch.Tensor, y: torch.Tensor):
    """
    Weigthed RMSE taken from Wheather Bench.
    Weighting to account for decreasing grid sizes towards the pole.

    rmse = mean over forecasts and time of torch.sqrt( mean over lon lat L(lat_j)*)MSE(preds, y)
    weights = cos(latitude)/cos(latitude).mean()
    """
    lat_size = y.shape[-2]
    lats = torch.linspace(-90, 90, lat_size)
    
    weights = (torch.cos(lats) / torch.cos(lats)).mean()
    rmse = torch.sqrt(torch.mean(weights * ((preds - y) ** 2), axis=(-1, -2))).mean()

    return rmse


def LLweighted_MSE_Climax(
    preds: torch.Tensor, y: torch.Tensor, deg2rad: bool = True, mask=None
):
    """
    Latitude weighted mean squared error taken from ClimaX.
    Allows to weight the  by the cosine of the latitude to account for gridding differences at equator vs. poles.
    Applied per variable.
    If given a mask, normalized by sum of that.

    """

    # lattitude weights
    lat_size = y.shape[-2]
    lats = torch.linspace(-90, 90, lat_size)
    if deg2rad:
        weights = torch.cos((torch.pi * lats) / 180)
    else:
        weights = torch.cos(lats)
    weights = weights.unsqueeze(-1)
    # they normalize the weights first
    weights = weights / weights.mean()

    if mask is not None:
        error = (((preds - y) ** 2) * weights * mask).sum() / mask.sum()
    else:
        error = (((preds - y) ** 2) * weights).mean()

    return error


def LLweighted_RMSE_Climax(
    preds: torch.Tensor, y: torch.Tensor, deg2rad: bool = True, mask=None
):
    """
    Latitude weighted root mean squared error taken from ClimaX.
    Allows to weight the  by the cosine of the latitude to account for gridding differences at equator vs. poles.
    Applied per variable.
    If given a mask, normalized by sum of that.
    """

    # lattitude weights
    lat_size = y.shape[-2]
    lats = torch.linspace(-90, 90, lat_size)
    if deg2rad:
        weights = torch.cos((torch.pi * lats) / 180)
    else:
        weights = torch.cos(lats)
    weights = weights.unsqueeze(-1)
    # they normalize the weights first
    weights = weights / weights.mean()

    if mask is not None:
        error = (((preds - y) ** 2) * weights * mask).sum() / mask.sum()
    else:
        error = (((preds - y) ** 2) * weights).mean()

    error = torch.sqrt(error)

    return error

if __name__ == "__main__":
    batch_size = 16
    out_time = 10
    lon = 192
    lat = 96
    dummy = torch.randn(batch_size, lat, lon)

    targets = torch.randn(batch_size, lat, lon)

    reduction = "mean"
    mse = MSE(dummy, targets)
    # rmse=RMSE(reduction=reduction)

    nrmse_g = NRMSE_g_ClimateBench(dummy, targets)
    nrmse_s = NRMSE_s_ClimateBench(dummy, targets)
    nrmse = NRMSE_ClimateBench(dummy, targets)

    llrmse_wb = LLWeighted_RMSE_WheatherBench(dummy, targets)

    llmse_cx = LLweighted_MSE_Climax(dummy, targets)
    llrmse_cx = LLweighted_RMSE_Climax(dummy, targets)

    loss = nrmse_g
    print("CB nrmse g loss", loss, loss.shape)

    loss = nrmse_s
    print("CB nrmse s loss", loss, loss.shape)

    loss = nrmse
    print("CB nrmse loss", loss, loss.shape)

    loss = llrmse_wb
    print("WB rmse loss", loss, loss.shape)

    loss = llmse_cx
    print("CX mse loss", loss, loss.shape)

    loss = llrmse_cx
    print("CX nmse loss", loss, loss.shape)