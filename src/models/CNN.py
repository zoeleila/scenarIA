"""
CNN-LSTM adapted from Watson_parris (2023) in Pytorch

TCN : https://github.com/paul-krug/pytorch-tcn.git
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_time_module(time_module_name, input_size, hidden_size=25, **kwargs):
    if time_module_name == 'lstm':
        return nn.LSTM(input_size=input_size, 
                       hidden_size=hidden_size, 
                       batch_first=True)
    
    elif time_module_name == 'gru':
        return nn.GRU(input_size=input_size, 
                      hidden_size=hidden_size, 
                      batch_first=True)

    else:
        raise ValueError(f"Unsupported time module: {time_module_name}")


class CNNBase(nn.Module):
    def __init__(self, slider, height=96, width=144, channels=4, time_module_name='lstm', hidden_size=25,
                 conv_filters=20, conv_kernel=(3, 3), pool_size=2,
                 output_seq_len=1, **kwargs):
        super(CNNBase, self).__init__()
        self.slider = slider
        self.height = height
        self.width = width
        self.channels = channels
        self.conv_filters = conv_filters
        self.conv_kernel = conv_kernel
        self.pool_size = pool_size
        self.time_module_name = time_module_name
        self.output_seq_len = output_seq_len
        self.hidden_size = hidden_size

        # CNN layers
        self.conv = nn.Conv2d(channels, conv_filters, 
                              kernel_size=conv_kernel, 
                              padding='same')
        self.pool = nn.AvgPool2d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Time Module
        self.time_module = get_time_module(time_module_name, 
                                           input_size=conv_filters, 
                                           hidden_size=hidden_size, 
                                           **kwargs)
        self.fc = nn.Linear(hidden_size, height * width)

    def forward(self, x):
        """
        x: tensor of shape (batch, time, height, width, channels)
        """
        batch_size, time_steps, height, width, channels = x.size()

        cnn_features = []
        for t in range(time_steps):
            # Rearrange (batch, height, width, channels) -> (batch, channels, height, width)
            frame = x[:, t].permute(0, 3, 1, 2)
            out = F.relu(self.conv(frame))
            out = self.pool(out)
            out = self.global_avg_pool(out)
            out = out.view(batch_size, -1)  # (batch, conv_filters)
            cnn_features.append(out)

        cnn_features = torch.stack(cnn_features, dim=1)

        time_out, _ = self.time_module(cnn_features)

        if self.output_seq_len == 1:
            time_out = F.relu(time_out[:, -1, :])  # last timestep (see tf LSTM return_sequence)
            out = self.fc(time_out)
            out = out.view(batch_size, 1, self.height, self.width)
        else:
            time_out = F.relu(time_out[:, -self.output_seq_len:, :])  # last output_seq_len timesteps
            out = self.fc(time_out)
            out = out.view(batch_size, self.output_seq_len, self.height, self.width)        
        return out



# Example usage
if __name__ == "__main__":
    print('ok')
    model = CNNBase(slider=5, width=192, height=96, 
                    channels=4, time_module_name='lstm', hidden_size=64,
                    conv_filters=20, conv_kernel=(3, 3), pool_size=2,
                    output_seq_len=1)

    summary(model, input_size=(16, 5, 96, 192, 4))