from collections import deque
from dataclasses import dataclass
import json
import logging
import os
import random
import time
from typing import Any, Tuple

from arcengine import FrameData, GameAction, GameState

from utils import get_environment_directory, setup_experiment_directory, setup_logging_for_experiment

from ..agent import Agent

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

import numpy as np
import numpy.typing as npt

from tensorboardX import SummaryWriter

import hashlib


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


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, output_channels: int, shared_embedding: nn.Embedding = None):
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

        self.rmsNorm1 = nn.RMSNorm(128)
        self.rmsNorm2 = nn.RMSNorm(64)
        self.rmsNorm3 = nn.RMSNorm(32)
        self.rmsNorm4 = nn.RMSNorm(output_channels)

        self.embedding = shared_embedding if shared_embedding is not None else nn.Embedding(output_channels, output_channels)

    def __call__(self, z: mx.array) -> mx.array:
        x = self.fc(z)                           # 1024
        x = x.reshape(x.shape[0], 1, 1, 1024)   # 1x1x1024

        x = nn.relu(self.deconv1(x))             # 5x5x128
        x = self.rmsNorm1(x)
        x = nn.relu(self.deconv2(x))             # 13x13x64
        x = self.rmsNorm2(x)
        x = nn.relu(self.deconv3(x))             # 30x30x32
        x = self.rmsNorm3(x)
        x = self.deconv4(x)                      # 64x64xoutput_channels, logits
        x = self.rmsNorm4(x)
        # project back into embedding space
        x = self.embedding.as_linear(x)

        return x


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
    

def vae_loss(reconstruction, x, mu, log_var, weights, beta=10.0):
    # cross-entropy reconstruction loss
    reconstruction_loss = nn.losses.cross_entropy(reconstruction, x) * weights
    recon_loss_per_example = mx.sum(reconstruction_loss, axis=(1, 2))
    # KL divergence
    kl_loss_per_example = -0.5 * mx.sum(
        1 + log_var - mx.square(mu) - mx.exp(log_var), 
        axis=1
    )

    recon_loss = mx.mean(recon_loss_per_example)
    kl_loss = mx.mean(kl_loss_per_example)

    total_loss = recon_loss * beta + kl_loss
    return total_loss, recon_loss, kl_loss


@dataclass
class Experience:
    state: npt.NDArray[np.int64]  # (64, 64) frame
    # Ravel indices of changed pixels from the previous frame, used for weighting loss.
    diff_ravel_pixel_indices: npt.NDArray[np.uint16] | None


class RandomVAE(Agent):
    """An agent that always selects actions at random."""

    MAX_ACTIONS = 10_000

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
        self.experience_buffer: deque[Experience] = deque(maxlen=200_000)
        self.experience_hashes = set()  # Track unique frames
        self.batch_size = 64

        self._reset_vae_model()

        # setup telemetry
        self.base_dir, log_file = setup_experiment_directory()
        setup_logging_for_experiment(log_file)
        
        # Get environment-specific directory using the real game_id
        env_dir = get_environment_directory(self.base_dir, self.game_id)
        tensorboard_dir = os.path.join(env_dir, 'tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)
        
        self.writer = SummaryWriter(tensorboard_dir)
        self.logger = logging.getLogger(f"ActionAgent_{self.game_id}")


        # need a way to track wheter a new level has started
        self.levels_completed_prev = 0
        self.prev_frame: npt.NDArray[np.int64] | None = None

        self.logger.info(f"Action agent initialized for game_id: {self.game_id}")


    def _reset_vae_model(self) -> None:
        self.vae = ConvVAE(input_channels=self.num_colours, output_channels=self.num_colours, latent_dim=16)
        self.optimizer = optim.AdamW(learning_rate=0.0001)
        self.loss_and_grad_fn = nn.value_and_grad(self.vae, self._loss_fn)


    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        done = any(
            [
                latest_frame.state is GameState.WIN,
                # uncomment to only let the agent play one time
                # latest_frame.state is GameState.GAME_OVER,
            ]
        )

    
        # is_done is always called before choosing an action, so
        # we can save the model just in time before the agent process terminates
        if done or self.action_counter > self.MAX_ACTIONS:
            self._save_model()  # Save model on completion

        return done

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose which action the Agent should take, fill in any arguments, and return it."""

        if latest_frame.levels_completed > self.levels_completed_prev:
            self.levels_completed_prev = latest_frame.levels_completed

            self._save_model(id=f"level_{self.levels_completed_prev}")  # Save model on level up

            # don't reset the model on level up, we want to carry forward past experience to next level
            self.experience_buffer.clear()
            self.experience_hashes.clear()
            self.prev_frame = None

        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            # if game is not started (at init or after GAME_OVER) we need to reset
            # add a small delay before resetting after GAME_OVER to avoid timeout
            action = GameAction.RESET
        else:
            # else choose a random action that isnt reset
            action = random.choice([a for a in GameAction if a is not GameAction.RESET and a.value in latest_frame.available_actions])

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

        
        # store frame examples for training
        # Convert current frame to tensor
        frame_tensors = self._frame_to_tensor(latest_frame)
        
        # If frame processing failed, reset tracking and return random action
        if frame_tensors is None:
            self.logger.error("Cannot parse current game frame.")
            
            action = random.choice(self.action_list[:5])  # Random ACTION1-ACTION5
            action.reasoning = f"Skipped weird frame, random {action.value}"

            self.prev_frame = None  # Reset previous frame tracking on failure

            return action
        
        for ft in frame_tensors:
            ft_hash = self._compute_experience_hash(ft)

            if ft_hash not in self.experience_hashes:

                diff_pixels = None
                if self.prev_frame is not None:
                    diff_pixels = np.flatnonzero(ft != self.prev_frame).astype(np.uint16)

                experience = Experience(
                    state=ft,  # Already numpy bool
                    diff_ravel_pixel_indices=diff_pixels
                )
                self.experience_buffer.append(experience)
                self.experience_hashes.add(ft_hash)


            self.prev_frame = ft


        if self.action_counter % self.train_frequency == 0:
            self.vae.train()
            self._train_vae_model()

        return action
    
    def _train_vae_model(self):
        """Train the VAE model on collected experiences."""
        if len(self.experience_buffer) < self.batch_size:
            return
        
        # Sample batch from experience buffer
        batch_indices = np.random.choice(len(self.experience_buffer), self.batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in batch_indices]
        

        # Prepare batch data - convert numpy arrays to tensors and move to GPU
        states_np = np.stack([exp.state for exp in batch])
        weights_np = np.ones(states_np.shape, dtype=np.float32)
        diff_counts = np.array(
            [
                len(exp.diff_ravel_pixel_indices)
                if exp.diff_ravel_pixel_indices is not None
                else 0
                for exp in batch
            ],
            dtype=np.intp,
        )
        if np.any(diff_counts):
            batch_idx = np.repeat(np.arange(len(batch), dtype=np.intp), diff_counts)
            pixel_idx = np.concatenate(
                [
                    exp.diff_ravel_pixel_indices
                    for exp in batch
                    if exp.diff_ravel_pixel_indices is not None
                    and len(exp.diff_ravel_pixel_indices) > 0
                ]
            ).astype(np.intp, copy=False)
            flat_weights = weights_np.reshape(len(batch), -1)
            flat_weights[batch_idx, pixel_idx] = 64.0

        states = mx.array(states_np)
        weights = mx.array(weights_np)

        _, grads = self.loss_and_grad_fn(states, weights)
        self.optimizer.update(self.vae, grads)
        mx.eval(self.vae.parameters(), self.optimizer.state)

    
    def _compute_experience_hash(self, frame: np.array) -> str:
        """Compute hash for frame to ensure uniqueness."""
        assert frame.shape == (self.grid_size, self.grid_size)
        frame_bytes = frame.tobytes()
        return hashlib.md5(frame_bytes).hexdigest()


    def _frame_to_tensor(self, frame_data: FrameData) -> list[npt.NDArray[np.int64]]:
        """
        Convert frame data to tensor format for the model.
        
        Taken from stochastic goose.
        """
        # Convert frame to numpy array with color indices 0-15
        frame = np.array(frame_data.frame, dtype=np.int64)
        
        assert (frame.shape[-2], frame.shape[-1]) == (self.grid_size, self.grid_size)
        
        # One-hot encode: (64, 64) -> (64, 64, 16)
        # tensor = np.eye(self.num_colours, dtype=np.float32)[frame]
        # return [t for t in tensor]
        return [frame[-1]]
    

    def _loss_fn(
        self,
        states: mx.array,
        weights: mx.array
    ) -> mx.array:
        reconstruction, mu, log_var = self.vae(states)
        total_loss, recon_loss, kl_loss = vae_loss(
            reconstruction,
            states,
            mu,
            log_var,
            weights,
        )
        variance = mx.exp(log_var)

        self.writer.add_scalar('VAE/total_loss', total_loss.item(), self.action_counter)
        self.writer.add_scalar('VAE/reconstruction_loss', recon_loss.item(), self.action_counter)
        self.writer.add_scalar('VAE/kl_loss', kl_loss.item(), self.action_counter)
        self.writer.add_scalar('VAE/latent_variance_mean', variance.mean().item(), self.action_counter)
        self.writer.add_scalar('VAE/latent_variance_min', variance.min().item(), self.action_counter)
        self.writer.add_scalar('VAE/latent_variance_max', variance.max().item(), self.action_counter)
        self.writer.add_histogram('VAE/latent_variance', np.asarray(variance), self.action_counter)
        
        return total_loss
    

    def _save_model(self, id="final") -> None:
        model_path = os.path.join(self.base_dir, f"vae_model_{self.game_id}_{id}.safetensors")
        config_path = os.path.join(self.base_dir, f"vae_model_{self.game_id}_{id}.json")

        self.vae.save_weights(model_path)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "latent_dim": self.vae.latent_dim,
                    "input_channels": self.vae.input_channels,
                    "output_channels": self.vae.output_channels,
                },
                f,
                indent=2,
            )

        self.logger.info(f"Saved VAE weights to {model_path}")
        self.logger.info(f"Saved VAE config to {config_path}")
