'''
https://github.com/RolnickLab/ClimateSet
'''

from typing import Sequence, Optional, Dict, Union, List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

import segmentation_models_pytorch as smp
import numpy as np

class TimeDistributed(nn.Module):
    "Applies a module over tdim identically for each step"

    def __init__(self, module, low_mem=False, tdim=1):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.low_mem = low_mem
        self.tdim = tdim

    def forward(self, *args, **kwargs):
        "input x with shape:(bs,seq_len,channels,width,height)"
        if self.low_mem or self.tdim != 1:
            return self.low_mem_forward(*args)
        else:
            # only support tdim=1
            inp_shape = args[0].shape
            bs, seq_len = inp_shape[0], inp_shape[1]
            out = self.module(
                *[x.view(bs * seq_len, *x.shape[2:]) for x in args], **kwargs
            )
            out_shape = out.shape
            return out.view(bs, seq_len, *out_shape[1:])

    def low_mem_forward(self, *args, **kwargs):
        "input x with shape:(bs,seq_len,channels,width,height)"
        tlen = args[0].shape[self.tdim]
        args_split = [torch.unbind(x, dim=self.tdim) for x in args]
        out = []
        for i in range(tlen):
            out.append(self.module(*[args[i] for args in args_split]), **kwargs)
        return torch.stack(out, dim=self.tdim)

    def __repr__(self):
        return f"TimeDistributed({self.module})"

class time_UNet(nn.Module):
    """
    ClimateSet
    https://github.com/elena-orlova/SSF-project
    """

    def __init__(
        self,
        in_var_ids: List[str],
        out_var_ids: List[str],
        longitude: int = 32,
        latitude: int = 32,
        activation_function: Union[
            str, callable, None
        ] = None,  # activation after final convolution
        encoder_name="vgg11",
        datamodule_config = None,
        channels_last: bool = True,
        seq_to_seq: bool = True,
        seq_len: int = 1,
        readout: str = "pooling",
        *args,
        **kwargs,
    ):
        super().__init__()

        if datamodule_config is not None:
            if datamodule_config.get("channels_last") is not None:
                self.channels_last = datamodule_config.get("channels_last")
            if datamodule_config.get("lon") is not None:
                self.lon = datamodule_config.get("lon")
            if datamodule_config.get("lat") is not None:
                self.lat = datamodule_config.get("lat")
            if datamodule_config.get("seq_len") is not None:
                self.seq_len = datamodule_config.get("seq_len")
        else:
            self.seq_to_seq = seq_to_seq
            self.lon = longitude
            self.lat = latitude
            self.channels_last = channels_last
            self.seq_len = seq_len
        self.num_output_vars = len(out_var_ids)
        self.num_input_vars = len(in_var_ids)

        # determine padding -> lan and lot must be divisible by 32
        pad_lon = int((np.ceil(self.lon / 32) * 32) - (self.lon / 32) * 32)
        pad_lat = int((np.ceil(self.lat / 32)) * 32 - (self.lat / 32) * 32)

        self.channels_last = channels_last

        # ption 1: linear output layer
        if readout == "linear":
            self.model = torch.nn.Sequential(
                torch.nn.ConstantPad2d(
                    (pad_lat, 0, pad_lon, 0), 0
                ),  # zero padding along lon and lat
                TimeDistributed(
                    smp.Unet(
                        encoder_name=encoder_name,
                        encoder_weights=None,
                        in_channels=self.num_input_vars,
                        classes=self.num_output_vars,
                        activation=activation_function,
                    )
                ),
                torch.nn.Flatten(),
                torch.nn.Linear(
                    in_features=(
                        self.num_output_vars
                        * (self.lon + pad_lon)
                        * (self.lat + pad_lat)
                        * self.seq_len
                    ),
                    out_features=(
                        self.num_output_vars * self.lon * self.lat * self.seq_len
                    ),
                ),  # map back to original size
            )

        elif readout == "pooling":
            self.model = torch.nn.Sequential(
                torch.nn.ConstantPad2d(
                    (pad_lat, 0, pad_lon, 0), 0
                ),  # zero padding along lon and lat
                TimeDistributed(
                    smp.Unet(
                        encoder_name=encoder_name,
                        encoder_weights=None,
                        in_channels=self.num_input_vars,
                        classes=self.num_output_vars,
                        activation=activation_function,
                    )
                ),
                torch.nn.AdaptiveAvgPool3d(
                    output_size=(self.num_output_vars, self.lon, self.lat)
                ),  # map back to original size
            )

        else:
            self.log_text.warn(
                f"Readout {readout} is not supported. Pls choose either 'linear' or 'pooling'"
            )
            raise NotImplementedError

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch_size, sequence_length, lon, lat, in_vars) if channels_last else (batch_size, sequence_lenght, in_vars, lon, lat)
        if self.channels_last:
            x = x.permute(
                (0, 1, 4, 2, 3)
            )  # torch con2d expects channels before height and witdth
        # if images width not divisible by
        x = self.model(x)
        # only effective if linear readout
        x = x.reshape((-1, self.seq_len, self.num_output_vars, self.lon, self.lat))
        if self.channels_last:
            x = x.permute((0, 1, 3, 4, 2))
        x = x.nan_to_num()

        # choosing only last time step if not seq_to_seq task
        if not (self.seq_to_seq):
            x = x[:, -1, :]
            x = torch.unsqueeze(x, 1)

        # returns (batch_size, sequence_length/1, lon, lat, out_vars) if channels_last else (batch_size, sequence_lenght, out_vars, lon, lat)
        return x.permute(0, 1, 3, 2, 4).contiguous()  # lat first

if __name__ == "__main__":
    time_unet = time_UNet(
        in_var_ids=["var1", "var2"],
        out_var_ids=["var3"],
        longitude=192,
        latitude=96,
        activation_function=None,
        datamodule_config=None,
        channels_last=True,
        seq_to_seq=False,
        seq_len=5,
    ).cuda()

    x = torch.randn(16, 5, 96, 192, 2).cuda()  # (batch_size, seq_len, lat, lon, in_vars)
    y_hat = time_unet(x)
    print(y_hat.shape)  # should be (16, 1, 96, 192, 1) since seq_to_seq=False and only last time step is returned