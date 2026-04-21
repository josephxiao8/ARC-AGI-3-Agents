import random
import time
from typing import Any, Tuple

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent


import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


class Encoder(nn.Module):
    def __init__(self, latent_dim: int, input_channels):
        super().__init__()
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

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        x = nn.relu(self.conv1(x))   # 31x31x32
        x = nn.relu(self.conv2(x))   # 14x14x64
        x = nn.relu(self.conv3(x))   # 6x6x128
        x = nn.relu(self.conv4(x))   # 2x2x256

        x = x.reshape(x.shape[0], -1)  # flatten -> (,1024)
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, output_channels: int):
        super().__init__()
        # dense: latent_dim -> 1x1x1024
        self.fc = nn.Linear(latent_dim, 1024)

        # 1x1x1024 -> 5x5x128
        self.deconv1 = nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=1, padding=0)
        # 5x5x128 -> 13x13x64
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=0)
        # 13x13x64 -> 30x30x32
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, padding=0)
        # 30x30x32 -> 64x64xoutput_channels
        self.deconv4 = nn.ConvTranspose2d(32, output_channels, kernel_size=6, stride=2, padding=0)

    def __call__(self, z: mx.array) -> mx.array:
        x = self.fc(z)                           # 1024
        x = x.reshape(x.shape[0], 1, 1, 1024)   # 1x1x1024

        x = nn.relu(self.deconv1(x))             # 5x5x128
        x = nn.relu(self.deconv2(x))             # 13x13x64
        x = nn.relu(self.deconv3(x))             # 30x30x32
        x = mx.sigmoid(self.deconv4(x))          # 64x64xoutput_channels
        return x


class ConvVAE(nn.Module):
    """
    https://arxiv.org/pdf/1803.10122
    """
    def __init__(self, input_channels, output_channels, latent_dim = 32):
        super().__init__()
        self.encoder = Encoder(latent_dim, input_channels)
        self.decoder = Decoder(latent_dim, output_channels)

    def reparameterize(self, mu: mx.array, log_var: mx.array) -> mx.array:
        std = mx.exp(0.5 * log_var)
        eps = mx.random.normal(std.shape)
        return mu + std * eps  # z = mu + sigma * N(0,1)

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z)
        return reconstruction, mu, log_var
    

def vae_loss(reconstruction, x, mu, log_var):
    # Reconstruction loss (BCE)
    recon_loss = nn.losses.binary_cross_entropy(
        reconstruction, x, reduction="sum"
    )
    # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    kl_loss = -0.5 * mx.sum(1 + log_var - mx.square(mu) - mx.exp(log_var))
    return recon_loss + kl_loss



class Random(Agent):
    """An agent that always selects actions at random."""

    MAX_ACTIONS = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)


        # grid info
        self.grid_size = 64
        self.num_coordinates = self.grid_size * self.grid_size
        self.num_colours = 16

        # Model and training
        self.train_frequency = 5
        self.vae = ConvVAE(input_channels=self.num_colours, output_channels=self.num_colours, latent_dim=16)


    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        return any(
            [
                latest_frame.state is GameState.WIN,
                # uncomment to only let the agent play one time
                # latest_frame.state is GameState.GAME_OVER,
            ]
        )

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose which action the Agent should take, fill in any arguments, and return it."""
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            # if game is not started (at init or after GAME_OVER) we need to reset
            # add a small delay before resetting after GAME_OVER to avoid timeout
            action = GameAction.RESET
        else:
            # else choose a random action that isnt reset
            action = random.choice([a for a in GameAction if a is not GameAction.RESET])

        if action.is_simple():
            action.reasoning = f"RNG told me to pick {action.value}"
        elif action.is_complex():
            action.set_data(
                {
                    "x": random.randint(0, 63),
                    "y": random.randint(0, 63),
                }
            )
            action.reasoning = {
                "desired_action": f"{action.value}",
                "my_reason": "RNG said so!",
            }

        if self.action_counter % self.train_frequency == 0:
            self.vae.train()
            self._train_vae_model()


        return action
    
    def _train_vae_model(self):
        # Placeholder for VAE training logic
        # You would typically sample a batch of frames, preprocess them, and then perform a training step here.
        pass
