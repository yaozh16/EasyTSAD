"""
Reimplemented under pytorch framework according to https://github.com/cruiseresearchgroup/TSCP2
"""
import torch.nn as nn
from .TCN import TemporalConvNet
import torch.nn.functional as F


class TCNEncoder(nn.Module):

    def __init__(self, num_dim, window_size, code_size, num_stacks=3, kernel_size=4, dilation_size=4, dropout=0.1):
        super(TCNEncoder, self).__init__()
        self.tcn = TemporalConvNet(num_inputs=num_dim, num_channels=[num_dim] * num_stacks, kernel_size=kernel_size,
                                   dilation_size=dilation_size, dropout=dropout)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(num_dim * window_size, 2 * window_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(2 * window_size, window_size)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(window_size, code_size)
        self.network = nn.Sequential(*[
            TemporalConvNet(num_inputs=num_dim, num_channels=[num_dim] * num_stacks, kernel_size=kernel_size,
                            dilation_size=dilation_size, dropout=dropout),
            nn.Flatten(),
            nn.Linear(num_dim * window_size, 2 * window_size),
            nn.ReLU(),
            nn.Linear(2 * window_size, window_size),
            nn.ReLU(),
            nn.Linear(window_size, code_size)
        ])

    def forward(self, x):

        return F.normalize(self.network(x), p=2, dim=1)
