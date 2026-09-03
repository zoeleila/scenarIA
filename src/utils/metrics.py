from locale import normalize
import torch
from scenarIA.src.utils.datautils import compute_weights_from_lats, weighted_global_mean
from torchmetrics import Metric
from scipy import stats



def MSE(y_hat: torch.Tensor, y: torch.Tensor):
    return torch.mean((y_hat - y) ** 2)

def RMSE(y_hat: torch.Tensor, y: torch.Tensor):
    return torch.mean(torch.sqrt(MSE(y_hat, y)))

def SpatialCorr(y_hat: torch.Tensor, y: torch.Tensor):
    if y.dim() == 4: # batch, time, lat, lon
        y = y.mean(dim=0)
        y_hat = y_hat.mean(dim=0)
    corr = [stats.pearsonr(y[i].flatten(), y_hat[i].flatten()).statistic for i in range(y.shape[0])]
    return torch.Tensor(corr).mean()

def TemporalCorr(y_hat: torch.Tensor, y: torch.Tensor):
    if y.dim() == 4: # batch, time, lat, lon
        y = y.mean(dim=0)
        y_hat = y_hat.mean(dim=0)
    # time, lat, lon 
    y = y.flatten(start_dim=1)
    y_hat = y_hat.flatten(start_dim=1)
    corr = [stats.pearsonr(y[:,i].flatten(), y_hat[:,i].flatten()).statistic for i in range(y.shape[-1])]
    return torch.Tensor(corr).mean()



def NRMSE_s_ClimateBench(y_hat: torch.Tensor, y: torch.Tensor, lats: torch.Tensor, normalize=True, weights_normalization=None):
    if y.dim() == 4: # batch, time, lat, lon
        y = y.mean(dim=0)
        y_hat = y_hat.mean(dim=0)

    nrmse_s = torch.sqrt(
        weighted_global_mean(
            (y_hat.mean(axis=0) - y.mean(axis=0)) ** 2, lats, weights_normalization=weights_normalization
        )
    ) 
    if normalize:
        nrmse_s = nrmse_s / torch.abs(weighted_global_mean(y.mean(axis=0), lats, weights_normalization=weights_normalization))

    return nrmse_s


def NRMSE_g_ClimateBench(y_hat: torch.Tensor, y: torch.Tensor, lats: torch.Tensor, normalize=True, weights_normalization=None):
    if y.dim() == 4: # batch, time, lat, lon
        y = y.mean(dim=0)
        y_hat = y_hat.mean(dim=0)

    if torch.any(y_hat == 0):
        y_hat[y_hat == 0] += 1e-6
    nrmse_g = torch.sqrt(
            ((
                weighted_global_mean(y_hat, lats, weights_normalization=weights_normalization)
                - weighted_global_mean(y, lats, weights_normalization=weights_normalization)) ** 2
            ).mean(axis=0)
        )
    if normalize:
        nrmse_g = nrmse_g / torch.abs(weighted_global_mean(y.mean(axis=0), lats, weights_normalization=weights_normalization))

    return nrmse_g


def NRMSE_ClimateBench(y_hat: torch.Tensor, y: torch.Tensor, lats: torch.Tensor, alpha: int = 5):
    if y.dim() == 4: # batch, time, lat, lon
        y = y.mean(dim=0)
        y_hat = y_hat.mean(dim=0)

    nrmseg = NRMSE_g_ClimateBench(y_hat, y, lats, normalize=True, weights_normalization='sum')
    nrmses = NRMSE_s_ClimateBench(y_hat, y, lats, normalize=True, weights_normalization='sum')
    nrmse = nrmses + alpha * nrmseg

    return nrmse


def LLWeighted_RMSE_WheatherBench(y_hat: torch.Tensor, y: torch.Tensor, lats: torch.Tensor):
    weights = (torch.cos(lats) / torch.cos(lats)).mean()
    rmse = torch.sqrt(torch.mean(weights * ((y_hat - y) ** 2), axis=(-1, -2))).mean()
    return rmse


def LLweighted_MSE_Climax(
    y_hat: torch.Tensor, y: torch.Tensor, lats: torch.Tensor, mask=None
):
    weights = compute_weights_from_lats(lats)
    if mask is not None:
        error = (((y_hat - y) ** 2) * weights * mask).sum() / mask.sum()
    else:
        error = (((y_hat - y) ** 2) * weights).mean()

    return error


def LLweighted_RMSE_Climax(
    y_hat: torch.Tensor, y: torch.Tensor, lats: torch.Tensor, mask=None
):
    weights = compute_weights_from_lats(lats)
    if mask is not None:
        error = (((y_hat - y) ** 2) * weights * mask).sum() / mask.sum()
    else:
        error = (((y_hat - y) ** 2) * weights).mean()

    error = torch.sqrt(error)

    return error

class LatWeightedRMSEMetric(Metric):
    """
    Accumulates weighted squared error and computes the LatWeighted RMSE.
    Usage:
        metric = LatWeightedRMSEMetric()
        metric.update(preds, target, lats, mask=opt_mask)
        rmse = metric.compute()
    """

    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("numerator", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("denom", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, lats: torch.Tensor, mask: torch.Tensor = None):
        """
        preds, target: tensors with same shape (e.g. batch, time, lat, lon)
        lats: latitude tensor used to compute weights
        mask: optional mask broadcastable to preds/target (0/1 values)
        """
        weights = compute_weights_from_lats(lats)
        weights = weights.unsqueeze(-1)
        sq = ((preds - target) ** 2) * weights

        if mask is not None:
            num = (sq * mask).sum()
            den = mask.sum()
        else:
            num = sq.sum()
            den = torch.tensor(target.numel(), dtype=num.dtype, device=num.device)

        # ensure tensors are on same device/dtype
        num = num.to(self.numerator.device)
        den = den.to(self.denom.device)

        self.numerator += num
        self.denom += den

    def compute(self):
        # avoid division by zero
        return torch.sqrt(self.numerator / (self.denom + 1e-12))

if __name__ == "__main__":
    batch_size = 16
    out_time = 1
    lon = 192
    lat = 96
    dummy = torch.randn(batch_size, out_time, lat, lon)
    targets = torch.randn(batch_size, out_time, lat, lon)


    lats = torch.linspace(-90, 90, steps=lat)

    reduction = "mean"
    mse = MSE(dummy, targets)
    # rmse=RMSE(reduction=reduction)

    nrmse_g = NRMSE_g_ClimateBench(dummy, targets, lats)
    nrmse_s = NRMSE_s_ClimateBench(dummy, targets, lats)
    nrmse = NRMSE_ClimateBench(dummy, targets, lats)

    llrmse_wb = LLWeighted_RMSE_WheatherBench(dummy, targets, lats)

    llmse_cx = LLweighted_MSE_Climax(dummy, targets, lats)
    llrmse_cx = LLweighted_RMSE_Climax(dummy, targets, lats)

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