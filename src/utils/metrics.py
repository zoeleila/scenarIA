"""
Metrics from ClimateSET 
y : [batch, time, lat, lon]
TODO : rewrite, the metrics of climax are prettier but doesn't not adapt to time serie data y : [batch, lat, lon, var]
"""

import torch

class Metrics:
    def __init__(self,
                 lats : None,
                 deg2rad: bool = True
                 ):
        self.lats = lats
        self.deg2rad = deg2rad

        if self.lats is not None:
            if self.deg2rad:
                weights = torch.cos((torch.pi * self.lats) / 180)
            else:
                weights = torch.cos(self.lats)
            weights = weights.unsqueeze(-1)
            self.weights = weights
        else:
            self.weights = None
    
    def weighted_global_mean(self, x, weights):
        return torch.mean(x * weights, dim=(-1, -2))

    def MSE(self, y_hat: torch.Tensor, y: torch.Tensor):
        return torch.mean((y_hat - y) ** 2)


    def RMSE(self, y_hat: torch.Tensor, y: torch.Tensor):
        return torch.mean(torch.sqrt(self.MSE(y_hat, y)))

    def NRMSE_s_ClimateBench(self, y_hat: torch.Tensor, y: torch.Tensor):
        if y.dim() == 4: # batch, time, lat, lon
            y = y.mean(dim=0)
            y_hat = y_hat.mean(dim=0)

        nrmse_s = torch.sqrt(
            self.weighted_global_mean(
                (y_hat.mean(axis=0) - y.mean(axis=0)) ** 2, self.weights
            )
        ) / self.weighted_global_mean(y, self.weights).mean(axis=0)

        return nrmse_s


    def NRMSE_g_ClimateBench(self, y_hat: torch.Tensor, y: torch.Tensor):
        if y.dim() == 4: # batch, time, lat, lon
            y = y.mean(dim=0)
            y_hat = y_hat.mean(dim=0)

        denom = self.weighted_global_mean(y, self.weights).mean(axis=0)
        if torch.any(y_hat == 0):
            y_hat[y_hat == 0] += 1e-6
        nrmse_g = (
            torch.sqrt(
                (
                    self.weighted_global_mean(y_hat, self.weights)
                    - self.weighted_global_mean(y, self.weights) ** 2
                ).mean(axis=0)
            )
            / denom
        )
        return nrmse_g


    def NRMSE_ClimateBench(self, y_hat: torch.Tensor, y: torch.Tensor, alpha: int = 5):
        if y.dim() == 4: # batch, time, lat, lon
            y = y.mean(dim=0)
            y_hat = y_hat.mean(dim=0)

        nrmseg = self.NRMSE_g_ClimateBench(y_hat, y)
        nrmses = self.NRMSE_s_ClimateBench(y_hat, y)
        nrmse = nrmses + alpha * nrmseg

        return nrmse


    def LLWeighted_RMSE_WheatherBench(self, y_hat: torch.Tensor, y: torch.Tensor):

        weights = (torch.cos(self.lats) / torch.cos(self.lats)).mean()
        rmse = torch.sqrt(torch.mean(weights * ((y_hat - y) ** 2), axis=(-1, -2))).mean()

        return rmse


    def LLweighted_MSE_Climax(
        self, y_hat: torch.Tensor, y: torch.Tensor, mask=None
    ):
        weights_norm = self.weights / self.weights.mean()
        if mask is not None:
            error = (((y_hat - y) ** 2) * weights_norm * mask).sum() / mask.sum()
        else:
            error = (((y_hat - y) ** 2) * weights_norm).mean()

        return error


    def LLweighted_RMSE_Climax(
        self, y_hat: torch.Tensor, y: torch.Tensor, mask=None
    ):
        weights_norm = self.weights / self.weights.mean()
        if mask is not None:
            error = (((y_hat - y) ** 2) * weights_norm * mask).sum() / mask.sum()
        else:
            error = (((y_hat - y) ** 2) * weights_norm).mean()

        error = torch.sqrt(error)

        return error

if __name__ == "__main__":
    batch_size = 16
    out_time = 1
    lon = 192
    lat = 96
    dummy = torch.randn(batch_size, out_time, lat, lon)
    targets = torch.randn(batch_size, out_time, lat, lon)


    lats = torch.linspace(-90, 90, steps=lat)
    m = Metrics(lats)

    reduction = "mean"
    mse = m.MSE(dummy, targets)
    # rmse=RMSE(reduction=reduction)

    nrmse_g = m.NRMSE_g_ClimateBench(dummy, targets)
    nrmse_s = m.NRMSE_s_ClimateBench(dummy, targets)
    nrmse = m.NRMSE_ClimateBench(dummy, targets)

    llrmse_wb = m.LLWeighted_RMSE_WheatherBench(dummy, targets)

    llmse_cx = m.LLweighted_MSE_Climax(dummy, targets)
    llrmse_cx = m.LLweighted_RMSE_Climax(dummy, targets)

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