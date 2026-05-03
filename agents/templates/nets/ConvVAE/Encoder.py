import mlx.nn as nn
import mlx.core as mx
from typing import Tuple

class Encoder(nn.Module):
    def __init__(self, latent_dim: int, input_channels):
        super().__init__()
        self.embedding = nn.Embedding(input_channels, input_channels)
        # 64x64xinput_channels -> 31x31x32
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=0)
        # 31x31x32 -> 14x14x64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        # 14x14x64 -> 6x6x128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0)
        # 6x6x128 -> 2x2x256
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=0)

        # dense -> mu and log_var (2x2x256 = 1024 flattened)
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

        self.rmsNorm1 = nn.RMSNorm(32)
        self.rmsNorm2 = nn.RMSNorm(64)
        self.rmsNorm3 = nn.RMSNorm(128)
        self.rmsNorm4 = nn.RMSNorm(256)

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        x = self.embedding(x)  # Apply identity embedding
        x = nn.relu(self.conv1(x))   # 31x31x32
        x = self.rmsNorm1(x)
        x = nn.relu(self.conv2(x))   # 14x14x64
        x = self.rmsNorm2(x)
        x = nn.relu(self.conv3(x))   # 6x6x128
        x = self.rmsNorm3(x)
        x = nn.relu(self.conv4(x))   # 2x2x256
        x = self.rmsNorm4(x)

        x = x.reshape(x.shape[0], -1)  # flatten -> (,1024)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var