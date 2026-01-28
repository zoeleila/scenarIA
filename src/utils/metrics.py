import torch
from scenarIA.src.utils.datautils import compute_weights_from_lats, weighted_global_mean



def MSE(y_hat: torch.Tensor, y: torch.Tensor):
    return torch.mean((y_hat - y) ** 2)

def RMSE(y_hat: torch.Tensor, y: torch.Tensor):
    return torch.mean(torch.sqrt(MSE(y_hat, y)))

def NRMSE_s_ClimateBench(y_hat: torch.Tensor, y: torch.Tensor, lats: torch.Tensor):
    if y.dim() == 4: # batch, time, lat, lon
        y = y.mean(dim=0)
        y_hat = y_hat.mean(dim=0)

    nrmse_s = torch.sqrt(
        weighted_global_mean(
            (y_hat.mean(axis=0) - y.mean(axis=0)) ** 2, lats
        )
    ) / torch.abs(weighted_global_mean(y, lats).mean(axis=0))

    return nrmse_s


def NRMSE_g_ClimateBench(y_hat: torch.Tensor, y: torch.Tensor, lats: torch.Tensor):
    if y.dim() == 4: # batch, time, lat, lon
        y = y.mean(dim=0)
        y_hat = y_hat.mean(dim=0)

    if torch.any(y_hat == 0):
        y_hat[y_hat == 0] += 1e-6
    nrmse_g = (
        torch.sqrt(
            ((
                weighted_global_mean(y_hat, lats)
                - weighted_global_mean(y, lats)) ** 2
            ).mean(axis=0)
        )
        / torch.abs(weighted_global_mean(y, lats).mean(axis=0))
    )
    return nrmse_g


def NRMSE_ClimateBench(y_hat: torch.Tensor, y: torch.Tensor, lats: torch.Tensor, alpha: int = 5):
    if y.dim() == 4: # batch, time, lat, lon
        y = y.mean(dim=0)
        y_hat = y_hat.mean(dim=0)

    nrmseg = NRMSE_g_ClimateBench(y_hat, y, lats)
    nrmses = NRMSE_s_ClimateBench(y_hat, y, lats)
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
    weights_norm = weights / weights.mean()
    if mask is not None:
        error = (((y_hat - y) ** 2) * weights_norm * mask).sum() / mask.sum()
    else:
        error = (((y_hat - y) ** 2) * weights_norm).mean()

    return error


def LLweighted_RMSE_Climax(
    y_hat: torch.Tensor, y: torch.Tensor, lats: torch.Tensor, mask=None
):
    weights = compute_weights_from_lats(lats)
    weights_norm = weights / weights.mean()
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