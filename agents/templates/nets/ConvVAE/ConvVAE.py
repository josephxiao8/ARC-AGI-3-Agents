import mlx.nn as nn
import mlx.core as mx
from .Encoder import Encoder
from .Decoder import Decoder
from typing import Tuple

class ConvVAE(nn.Module):
    """
    https://arxiv.org/pdf/1803.10122
    """
    def __init__(self, input_channels, output_channels, latent_dim):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, input_channels)
        self.decoder = Decoder(latent_dim, output_channels, shared_embedding=self.encoder.embedding)

    def reparameterize(self, mu: mx.array, log_var: mx.array) -> mx.array:
        std = mx.exp(0.5 * log_var)
        eps = mx.random.normal(std.shape)
        return mu + std * eps  # z = mu + sigma * N(0,1)

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z)
        return reconstruction, mu, log_var
    