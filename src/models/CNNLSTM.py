"""
CNN-LSTM adapted from Watson_parris (2023) in Pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNNLSTMModel(nn.Module):
    def __init__(self, slider, height=96, width=144, channels=4,
                 conv_filters=20, conv_kernel=(3, 3), pool_size=2, lstm_units=25,
                 output_seq_len=1):
        super(CNNLSTMModel, self).__init__()
        self.slider = slider
        self.height = height
        self.width = width
        self.channels = channels
        self.conv_filters = conv_filters
        self.conv_kernel = conv_kernel
        self.pool_size = pool_size
        self.lstm_units = lstm_units
        self.output_seq_len = output_seq_len

        # CNN layers
        self.conv = nn.Conv2d(channels, conv_filters, 
                              kernel_size=conv_kernel, 
                              padding='same')
        self.pool = nn.AvgPool2d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # LSTM input = number of filters after CNN pooling
        self.lstm_input_size = conv_filters
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, 
                            hidden_size=lstm_units, 
                            batch_first=True)
        self.fc = nn.Linear(lstm_units, height * width)

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
        # Stack over time → (batch, time, conv_filters)
        cnn_features = torch.stack(cnn_features, dim=1)
        # LSTM
        lstm_out, _ = self.lstm(cnn_features)
        if self.output_seq_len == 1:
            lstm_out = F.relu(lstm_out[:, -1, :])  # last timestep (see tf LSTM return_sequence)
            out = self.fc(lstm_out)
            out = out.view(batch_size, 1, self.height, self.width)
        else:
            lstm_out = F.relu(lstm_out[:, -self.output_seq_len:, :])  # last output_seq_len timesteps
            out = self.fc(lstm_out)  # (batch, output_seq_len, height*width)
            out = out.view(batch_size, self.output_seq_len, self.height, self.width)
        # Dense and reshape to image
        
        return out



# Example usage
if __name__ == "__main__":
    print('ok')
    model = CNNLSTMModel(slider=10, width=192, height=96, channels=4, conv_filters=20, conv_kernel=(3, 3), pool_size=2, lstm_units=25, output_seq_len=1)

    x = torch.randn(16, 10, 96, 192, 4)  # (batch, time, height, width, channels)
    y = model(x)
    print(y.shape)  # Expected: (16, 1, 96, 192)