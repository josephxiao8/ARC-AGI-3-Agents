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

from .nets import ConvVAE, MLP
from agents.templates.nets.ConvVAE import Encoder

@dataclass
class Experience:
    state: npt.NDArray[np.int64]  # (64, 64) frame
    prev_state: npt.NDArray[np.int64] | None  # (64, 64) previous frame, optional
    # Ravel indices of changed pixels from the previous frame, used for weighting loss.
    action: GameAction | None  # Action taken to get to this state, optional
    diff_ravel_pixel_indices: npt.NDArray[np.uint16] | None



class NextStatePrediction:
    def __init__(self, encoder: Encoder, base_dir: str, writer: SummaryWriter):
        self.encoder = encoder
        self.latent_dim = encoder.latent_dim

        self.writer = writer
        self.base_dir = base_dir
        self.logger = logging.getLogger("NextStatePrediction")
        self.experience_buffer: deque[Experience] = deque(maxlen=200_000)
        self.experience_hashes = set()  # Track unique frames for next state prediction

        self._reset_model()
        self.logger.info("Initialized NextStatePrediction module.")

    def add_experience(self, experience: Experience) -> None:
        assert experience.prev_state is not None, "Previous state is required for next state prediction."

        # Check if (current, previous) state pair is unique using a hash of the current state

        hasher = hashlib.md5()
        hasher.update(experience.prev_state.tobytes())
        hasher.update(experience.state.tobytes())
        combined_hash = hasher.hexdigest()

        if combined_hash not in self.experience_hashes:
            self.experience_hashes.add(combined_hash)
            self.experience_buffer.append(experience)

    def clear_experience(self) -> None:
        self.experience_buffer.clear()
        self.experience_hashes.clear()

        self.logger.info("Cleared experience buffer and hashes for NextStatePrediction.")

    def train_batch(self, action_counter: int, batch_size: int = 64) -> None:
        if len(self.experience_buffer) < batch_size:
            return
        
        self.net.train()
        
        # Sample batch from experience buffer
        batch_indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in batch_indices]

        # Prepare batch data - convert numpy arrays to tensors and move to GPU
        states_np = np.stack([exp.state for exp in batch])
        prev_states_np = np.stack([exp.prev_state for exp in batch])
        # sample uniformly from (0, 1) for each example in the batch
        time_np = np.random.uniform(0, 1, size=(len(batch), 1)).astype(np.float32)
        
        states = mx.array(states_np)
        prev_states = mx.array(prev_states_np)
        time = mx.array(time_np)

        # encode states and prev_states to get latent representations
        self.encoder.eval()
        mu_states, _ = self.encoder(states)
        mu_prev_states, _ = self.encoder(prev_states)
        
        actions = [exp.action for exp in batch]


        _, grads = self.loss_and_grad_fn(mu_states, mu_prev_states, time, actions, action_counter)
        self.optimizer.update(self.net, grads)
        mx.eval(self.net.parameters(), self.optimizer.state)


    def save_model(self, id="final") -> None:
        model_path = os.path.join(self.base_dir, f"next_state_model_{id}.safetensors")
        config_path = os.path.join(self.base_dir, f"next_state_model_{id}.json")

        self.net.save_weights(model_path)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "latent_dim": self.latent_dim,
                    "time_dim": self.net.time_dim,
                    "action_dim": self.net.action_dim,
                    "hidden_dim": self.net.hidden_dim,
                    "flow": "affine",
                    "conditioned_on_action": True,
                },
                f,
                indent=2,
            )

        self.logger.info(f"Saved next-state weights to {model_path}")
        self.logger.info(f"Saved next-state config to {config_path}")

    
    def _reset_model(self) -> None:
        self.net = MLP(
            input_dim=self.latent_dim,
            time_dim=1,
        )
        # play around with LR
        self.optimizer = optim.AdamW(learning_rate=0.001)
        self.loss_and_grad_fn = nn.value_and_grad(self.net, self._loss_fn)


    def _loss_fn(self, states: mx.array, prev_states: mx.array, time: mx.array, actions: list[GameAction], action_counter: int) -> mx.array:
        # affine flow matching loss
        states_t = prev_states * (1.0 - time) + states * time

        velocity_pred = self.net(states_t, time, actions)
        cond_velocity = states - prev_states

        loss = nn.losses.mse_loss(velocity_pred, cond_velocity)

        self.writer.add_scalar('NextStatePrediction/loss', loss.item(), action_counter)
        return loss

class LatentEncoderDecoder(nn.Module):
    def __init__(self, input_channels: int, latent_dim: int, base_dir: str, writer: SummaryWriter):
        super().__init__()
        self.writer = writer
        self.base_dir = base_dir
        self.logger = logging.getLogger("LatentEncoderDecoder")
        self.experience_buffer: deque[Experience] = deque(maxlen=200_000)
        self.experience_hashes = set()  # Track unique frames for next state prediction

        self.input_channels = input_channels
        self.latent_dim = latent_dim

        self._reset_model()

    def add_experience(self, experience: Experience) -> None:
        exp_hash = self._compute_experience_hash(experience.state)

        if exp_hash not in self.experience_hashes:
            self.experience_hashes.add(exp_hash)
            self.experience_buffer.append(experience)


    def clear_experience(self) -> None:
        self.experience_buffer.clear()
        self.experience_hashes.clear()
        self.logger.info("Cleared experience buffer and hashes for LatentEncoderDecoder.")

        
    def train_batch(self, action_counter, batch_size: int = 64) -> None:
        """Train the VAE model on collected experiences."""
        if len(self.experience_buffer) < batch_size:
            return
        
        self.vae.train()
        
        # Sample batch from experience buffer
        batch_indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
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

        _, grads = self.loss_and_grad_fn(states, weights, action_counter)
        self.optimizer.update(self.vae, grads)
        mx.eval(self.vae.parameters(), self.optimizer.state)

    def save_model(self, id="final") -> None:
        model_path = os.path.join(self.base_dir, f"vae_model_{id}.safetensors")
        config_path = os.path.join(self.base_dir, f"vae_model_{id}.json")

        self.vae.save_weights(model_path)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "latent_dim": self.vae.latent_dim,
                    "input_channels": self.vae.input_channels,
                },
                f,
                indent=2,
            )

        self.logger.info(f"Saved VAE weights to {model_path}")
        self.logger.info(f"Saved VAE config to {config_path}")

    def _reset_model(self) -> None:
        self.vae = ConvVAE(input_channels=self.input_channels, latent_dim=self.latent_dim)
        self.optimizer = optim.AdamW(learning_rate=0.0001)
        self.loss_and_grad_fn = nn.value_and_grad(self.vae, self._loss_fn)

    def _compute_experience_hash(self, frame: np.array) -> str:
        frame_bytes = frame.tobytes()
        return hashlib.md5(frame_bytes).hexdigest()

    def _vae_loss(self, reconstruction, x, mu, log_var, weights, beta=10.0):
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

    def _loss_fn(
        self,
        states: mx.array,
        weights: mx.array,
        action_counter: int
    ) -> mx.array:
        reconstruction, mu, log_var = self.vae(states)
        total_loss, recon_loss, kl_loss = self._vae_loss(
            reconstruction,
            states,
            mu,
            log_var,
            weights,
        )
        variance = mx.exp(log_var)

        self.writer.add_scalar('VAE/total_loss', total_loss.item(), action_counter)
        self.writer.add_scalar('VAE/reconstruction_loss', recon_loss.item(), action_counter)
        self.writer.add_scalar('VAE/kl_loss', kl_loss.item(), action_counter)
        self.writer.add_scalar('VAE/latent_variance_mean', variance.mean().item(), action_counter)
        self.writer.add_scalar('VAE/latent_variance_min', variance.min().item(), action_counter)
        self.writer.add_scalar('VAE/latent_variance_max', variance.max().item(), action_counter)
        self.writer.add_histogram('VAE/latent_variance', np.asarray(variance), action_counter)

        return total_loss

class RandomVAENextState(Agent):
    """An agent that always selects actions at random."""

    MAX_ACTIONS = 10_000
    # When to start using the VAE latents as samples for next state prediction
    FILL_BEFORE_NEXT_STATE_PREDICTION = 5_000

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
        self.prev_action: GameAction | None = None

        self._reset_models()

        self.logger.info(f"Action agent initialized for game_id: {self.game_id}")


    def _reset_models(self) -> None:
        self.latent_encoder_decoder = LatentEncoderDecoder(
            input_channels=self.num_colours,
            latent_dim=16,
            base_dir=self.base_dir,
            writer=self.writer,
        )
        
        self.next_state_predictor = NextStatePrediction(
            encoder=self.latent_encoder_decoder.vae.encoder,
            base_dir=self.base_dir,
            writer=self.writer
        )

        self.start_next_state_prediction_training = self.action_counter + self.FILL_BEFORE_NEXT_STATE_PREDICTION 


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
            self._save_models()  # Save model on completion

        return done

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose which action the Agent should take, fill in any arguments, and return it."""

        if latest_frame.levels_completed > self.levels_completed_prev:
            self.levels_completed_prev = latest_frame.levels_completed

            self._save_models(id=f"level_{self.levels_completed_prev}")  # Save model on level up

            # don't reset the model on level up, we want to carry forward past experience to next level
            self.latent_encoder_decoder.clear_experience()
            self.next_state_predictor.clear_experience()

            self.prev_frame = None
            self.prev_action = None

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
            self.prev_action = None

            return action
        
        for ft in frame_tensors:
            diff_pixels = None if self.prev_frame is None else np.flatnonzero(ft != self.prev_frame).astype(np.uint16)
            
            experience = Experience(
                state=ft,  # Already numpy bool
                prev_state=self.prev_frame,
                action=self.prev_action,
                diff_ravel_pixel_indices=diff_pixels,
            )
            self.latent_encoder_decoder.add_experience(experience)

            if self.prev_frame is not None:
                self.next_state_predictor.add_experience(experience)

            self.prev_frame = ft
            self.prev_action = action


        if self.action_counter % self.train_frequency == 0:
            self.latent_encoder_decoder.train_batch(self.action_counter)

            if self.action_counter > self.start_next_state_prediction_training:
                self.next_state_predictor.train_batch(self.action_counter)

        return action


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
    

    def _save_models(self, id="final") -> None:
        self.latent_encoder_decoder.save_model(id=f"{self.game_id}_{id}")
        self.next_state_predictor.save_model(id=f"{self.game_id}_{id}")

        self.logger.info(f"Saved models with id {id}")
