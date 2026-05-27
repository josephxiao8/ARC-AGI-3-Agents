import hashlib
import json
import logging
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import numpy.typing as npt
from arcengine import FrameData, GameAction, GameState
from tensorboardX import SummaryWriter

from agents.templates.nets import ActionModel
from utils import (
    get_environment_directory,
    setup_experiment_directory,
    setup_logging_for_experiment,
)

from ..agent import Agent
from .nets import Layered


@dataclass
class Experience:
    cur_frame: npt.NDArray[np.int64]  # (64, 64) frame
    prev_frame: npt.NDArray[np.int64] | None  # (64, 64) previous frame, optional
    # Ravel indices of changed pixels from the previous frame, used for weighting loss.
    action: GameAction | None  # Action taken to get to this state, optional
    diff_ravel_pixel_indices: npt.NDArray[np.uint16] | None



class NextStatePrediction:
    def __init__(self, base_dir: str, writer: SummaryWriter):
        self.writer = writer
        self.base_dir = base_dir
        self.logger = logging.getLogger("NextStatePrediction")
        self.experience_buffer: deque[Experience] = deque(maxlen=200_000)
        self.experience_hashes = set()  # Track unique frames for next state prediction

        self._reset_model()
        self.logger.info("Initialized NextStatePrediction module.")

    def add_experience(self, experience: Experience) -> None:
        assert experience.prev_frame is not None, "Previous state is required for next state prediction."

        # Check if (current, previous) state pair is unique using a hash of the current state

        hasher = hashlib.md5()
        hasher.update(experience.prev_frame.tobytes())
        hasher.update(experience.cur_frame.tobytes())
        combined_hash = hasher.hexdigest()

        if combined_hash not in self.experience_hashes:
            self.experience_hashes.add(combined_hash)
            self.experience_buffer.append(experience)

    def clear_experience(self) -> None:
        self.experience_buffer.clear()
        self.experience_hashes.clear()

        self.logger.info("Cleared experience buffer and hashes for NextStatePrediction.")

    def train_batch(self, action_counter: int, batch_size: int = 8) -> None:
        if len(self.experience_buffer) < batch_size:
            return
        
        self.net.train()
        
        # Sample batch from experience buffer
        batch_indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in batch_indices]

        # Prepare batch data - convert numpy arrays to tensors and move to GPU
        frames_np = np.stack([exp.cur_frame for exp in batch])
        prev_frames_np = np.stack([exp.prev_frame for exp in batch])
        frames = mx.array(frames_np)
        prev_frames = mx.array(prev_frames_np)
        actions = [exp.action for exp in batch]


        _, grads = self.loss_and_grad_fn(
            frames,
            prev_frames,
            actions,
            action_counter,
        )
        self.optimizer.update(self.net, grads)
        mx.eval(self.net.parameters(), self.optimizer.state)


    def save_model(self, id="final") -> None:
        model_path = os.path.join(self.base_dir, f"layered_next_state_model_{id}.safetensors")
        config_path = os.path.join(self.base_dir, f"layered_next_state_model_{id}.json")

        self.net.save_weights(model_path)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "vocab_size": self.net.vocab_size,
                    "action_dim": self.net.action_dim,
                },
                f,
                indent=2,
            )

        self.logger.info(f"Saved next-state weights to {model_path}")
        self.logger.info(f"Saved next-state config to {config_path}")

    
    def _reset_model(self) -> None:
        self.net = Layered(vocab_size=16, action_dim=16)
        # play around with LR
        self.optimizer = optim.AdamW(learning_rate=0.001)
        self.loss_and_grad_fn = nn.value_and_grad(self.net, self._loss_fn)


    def _loss_fn(
        self,
        frames: mx.array,
        prev_frames: mx.array,
        actions: list[GameAction],
        action_counter: int,
        alpha: float = 1000.0,
        beta: float = 0.1,
        gamma: float = 100.0,
        hole_rate: float = 0.10,
    ) -> mx.array:
        (
            static_layer_logits,
            dynamic_layer_logits,
            dynamic_gate_logits,
            next_dynamic_layer_logits,
            next_dynamic_gate_logits,
        ) = self.net(
            prev_frames,
            actions,
        )

        def reduce_per_pixel_loss(per_pixel_loss: mx.array) -> mx.array:
            """
            Args:
                per_pixel_loss: shape (batch_size, height, width[, channels])
            """
            loss_axes = tuple(range(1, per_pixel_loss.ndim))
            return mx.mean(mx.sum(per_pixel_loss, axis=loss_axes))

        unchanged_pixel_mask = frames == prev_frames
        static_gate_targets = unchanged_pixel_mask[..., None].astype(
            dynamic_gate_logits.dtype
        )

        # Previous-frame reconstruction. The dynamic layer softly gates whether a
        # pixel is reconstructed by static logits or dynamic logits.
        prev_static_ce = nn.losses.cross_entropy(static_layer_logits, prev_frames)
        prev_dynamic_ce = nn.losses.cross_entropy(dynamic_layer_logits, prev_frames)
        prev_dynamic_ce = mx.where(unchanged_pixel_mask, 0.0, prev_dynamic_ce)
        prev_static_gate = mx.sigmoid(dynamic_gate_logits[..., 0])
        prev_mixture_ce = (
            prev_static_gate * prev_static_ce
            + (1.0 - prev_static_gate) * prev_dynamic_ce
        )

        prev_gate_ce = nn.losses.binary_cross_entropy(
            dynamic_gate_logits,
            static_gate_targets,
            with_logits=True,
            reduction="none",
        )

        prev_reconstruction_loss = reduce_per_pixel_loss(prev_mixture_ce)
        prev_gate_loss = reduce_per_pixel_loss(prev_gate_ce)
        total_prev_frame_loss = prev_reconstruction_loss + beta * prev_gate_loss

        hole_probs = hole_rate * prev_static_gate
        hole_probs = hole_rate * prev_static_gate
        inpaint_mask = mx.random.uniform(shape=prev_static_gate.shape) < hole_probs
        inpaint_frames = mx.where(inpaint_mask, mx.zeros_like(prev_frames), prev_frames)
        (
            _,
            _,
            inpaint_static_layer_logits,
            _,
            _,
        ) = self.net.decompose(
            inpaint_frames,
            input_mask=inpaint_mask,
        )
        inpaint_static_ce = nn.losses.cross_entropy(
            inpaint_static_layer_logits,
            prev_frames,
        )
        inpaint_mask_float = inpaint_mask.astype(inpaint_static_ce.dtype)
        inpaint_weights = inpaint_mask_float * prev_static_gate
        inpaint_weight_mass = mx.sum(inpaint_weights) + 1e-6
        inpaint_average_ce = (
            mx.sum(inpaint_static_ce * inpaint_weights) / inpaint_weight_mass
        )
        mean_inpaint_pixels = mx.mean(mx.sum(inpaint_mask_float, axis=(1, 2)))
        background_inpainting_loss = inpaint_average_ce * mean_inpaint_pixels

        # Next-frame prediction. Reuse the previous-frame static layer and let the
        # predicted next dynamic layer decide when to override it.
        next_static_ce = nn.losses.cross_entropy(static_layer_logits, frames)
        next_dynamic_ce = nn.losses.cross_entropy(next_dynamic_layer_logits, frames)
        next_dynamic_ce = mx.where(unchanged_pixel_mask, 0.0, next_dynamic_ce)
        next_static_gate = mx.sigmoid(next_dynamic_gate_logits[..., 0])
        next_mixture_ce = (
            next_static_gate * next_static_ce
            + (1.0 - next_static_gate) * next_dynamic_ce
        )

        next_gate_ce = nn.losses.binary_cross_entropy(
            next_dynamic_gate_logits,
            static_gate_targets,
            with_logits=True,
            reduction="none",
        )

        next_frame_loss = reduce_per_pixel_loss(next_mixture_ce)
        next_gate_loss = reduce_per_pixel_loss(next_gate_ce)
        total_next_frame_loss = next_frame_loss + beta * next_gate_loss

        static_layer_log_probs = nn.log_softmax(static_layer_logits, axis=-1) # (B, H, W, vocab_size)
        static_layer_probs = mx.softmax(static_layer_logits, axis=-1) # (B, H, W, vocab_size)
        mean_dist = mx.mean(static_layer_probs, axis=0, keepdims=True) # (1, H, W, vocab_size)

        # Average KL from mean
        kl_from_mean = nn.losses.kl_div_loss(mean_dist.log(), static_layer_log_probs, reduction='mean')

        total_loss = (
            total_prev_frame_loss
            + total_next_frame_loss
            + gamma * background_inpainting_loss
            + alpha * kl_from_mean
        )

        self.writer.add_scalar(
            'Prev/reconstruction_loss',
            prev_reconstruction_loss.item(),
            action_counter,
        )
        self.writer.add_scalar(
            'Prev/gate_loss',
            prev_gate_loss.item(),
            action_counter,
        )
        self.writer.add_scalar(
            'Prev/total_loss',
            total_prev_frame_loss.item(),
            action_counter,
        )
        self.writer.add_scalar(
            'Next/reconstruction_loss',
            next_frame_loss.item(),
            action_counter,
        )
        self.writer.add_scalar(
            'Next/gate_loss',
            next_gate_loss.item(),
            action_counter,
        )
        self.writer.add_scalar(
            'Next/total_loss',
            total_next_frame_loss.item(),
            action_counter,
        )
        self.writer.add_scalar(
            'Inpaint/background_loss',
            background_inpainting_loss.item(),
            action_counter,
        )
        self.writer.add_scalar(
            'Inpaint/hole_pixels',
            mean_inpaint_pixels.item(),
            action_counter,
        )
        self.writer.add_scalar(
            'KL/kl_from_mean',
            kl_from_mean.item(),
            action_counter,
        )
        self.writer.add_scalar(
            'Combined/total_loss',
            total_loss.item(),
            action_counter,
        )

        # self.writer.add_histogram(
        #     'Actions',
        #     np.array([a.value for a in actions], dtype=np.int64),
        #     action_counter,
        # )

        # self.writer.add_histogram(
        #     'complex action coords x',
        #     np.array([a.action_data.x for a in actions if a.is_complex()], dtype=np.int64),
        #     action_counter,
        # )
        
        # self.writer.add_histogram(
        #     'complex action coords y',
        #     np.array([a.action_data.y for a in actions if a.is_complex()], dtype=np.int64),
        #     action_counter,
        # )

        return total_loss
    
class RLActionModel:
    def __init__(
        self,
        base_dir: str,
        writer: SummaryWriter,
        num_colours: int,
        available_simple_actions: list[GameAction],
        is_coord_action_allowed: bool,
        grid_size: int,
    ) -> None:
        self.base_dir = base_dir
        self.writer = writer
        self.num_colours = num_colours
        self.available_simple_actions = available_simple_actions
        self.is_coord_action_allowed = is_coord_action_allowed
        self.grid_size = grid_size
        self.num_coordinates = grid_size * grid_size
        self.logger = logging.getLogger("RLActionModel")
        self.net = ActionModel(
            num_colors=num_colours,
            num_simple_action_types=len(available_simple_actions),
            is_coord_action_allowed=is_coord_action_allowed,
        )

    def choose_action(
        self,
        frame: npt.NDArray[np.int64],
    ) -> tuple[GameAction, tuple[int, int] | None, int | None, np.ndarray]:
        logits = self.net(mx.array(frame)[None, ...])  # Add batch dimension
        return self._sample_from_combined_output(logits.squeeze(0))

    def _sample_from_combined_output(
        self,
        combined_logits: mx.array,
    ) -> tuple[GameAction, tuple[int, int] | None, int | None, np.ndarray]:
        """
        Sample from combined simple + 64x64 action space.

        Adapted from https://github.com/DriesSmit/ARC3-solution/blob/main/custom_agents/action.py
        """
        num_simple_action_types = len(self.available_simple_actions)
        action_logits = combined_logits[:num_simple_action_types]
        coord_logits = combined_logits[num_simple_action_types:]

        action_probs = mx.sigmoid(action_logits)
        coord_probs_raw = mx.sigmoid(coord_logits)

        # Treat coordinates as one action type by sharing their probability mass.
        coord_probs_scaled = coord_probs_raw / self.num_coordinates

        all_probs_sampling = mx.concatenate([action_probs, coord_probs_scaled], axis=-1)
        all_probs_sampling_np = np.asarray(all_probs_sampling)
        prob_sum = all_probs_sampling_np.sum()
        if not np.isfinite(prob_sum) or prob_sum <= 0:
            raise ValueError(
                "No valid action probabilities available for sampling: "
                f"available_actions={self.available_simple_actions}, prob_sum={prob_sum}"
            )
        all_probs_sampling_np = all_probs_sampling_np / prob_sum

        selected_idx = np.random.choice(len(all_probs_sampling_np), p=all_probs_sampling_np)

        coord_probs_viz = mx.sigmoid(coord_logits)
        all_probs_viz = mx.concatenate([action_probs, coord_probs_viz], axis=-1)
        all_probs_viz_np = np.asarray(all_probs_viz)

        if selected_idx < num_simple_action_types:
            return self.available_simple_actions[selected_idx], None, None, all_probs_viz_np

        coord_idx = selected_idx - num_simple_action_types
        y_idx = coord_idx // self.grid_size
        x_idx = coord_idx % self.grid_size
        return GameAction.ACTION6, (y_idx, x_idx), coord_idx, all_probs_viz_np

    def save_model(self, id="final") -> None:
        model_path = os.path.join(self.base_dir, f"action_model_{id}.safetensors")
        config_path = os.path.join(self.base_dir, f"action_model_{id}.json")

        self.net.save_weights(model_path)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_colors": self.num_colours,
                    "num_simple_action_types": len(self.available_simple_actions),
                    "is_coord_action_allowed": self.is_coord_action_allowed,
                },
                f,
                indent=2,
            )

        self.logger.info(f"Saved action-model weights to {model_path}")
        self.logger.info(f"Saved action-model config to {config_path}")

class RandomLayeredRL(Agent):
    """An agent that always selects actions at random."""

    MAX_ACTIONS = 20_000

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1000000) + hash(self.game_id) % 1000000
        random.seed(seed)

        # grid info
        self.grid_size = 64
        self.num_coordinates = self.grid_size * self.grid_size
        self.num_colours = 16

        # Model and training
        self.train_frequency = 20

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

        self.logger.info(f"Action agent initialized for game_id: {self.game_id}")


    def _reset_models(
        self,
        available_simple_actions: list[GameAction],
        is_coord_action_allowed: bool,
    ) -> None:
        self.next_state_predictor = NextStatePrediction(
            base_dir=self.base_dir,
            writer=self.writer
        )
        self.action_model = RLActionModel(
            base_dir=self.base_dir,
            writer=self.writer,
            num_colours=self.num_colours,
            available_simple_actions=available_simple_actions,
            is_coord_action_allowed=is_coord_action_allowed,
            grid_size=self.grid_size,
        )

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

        if self.action_counter == 0:
            # safe to assume that a game's available actions are the same throughout
            # https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/702079#3461836
            available_simple_actions = [a for a in GameAction if a.is_simple() and a is not GameAction.RESET and a.value in latest_frame.available_actions]
            num_simple_actions = len(available_simple_actions)
            is_coord_action_allowed = GameAction.ACTION6.value in latest_frame.available_actions
            self._reset_models(
                available_simple_actions=available_simple_actions,
                is_coord_action_allowed=is_coord_action_allowed,
            )

            self.logger.info(f"Instantiating models with num_simple_actions={num_simple_actions} and is_coord_action_allowed={is_coord_action_allowed}")

        if latest_frame.levels_completed > self.levels_completed_prev:
            self.levels_completed_prev = latest_frame.levels_completed

            self._save_models(id=f"level_{self.levels_completed_prev}")  # Save model on level up

            # don't reset the model on level up, we want to carry forward past experience to next level
            self.next_state_predictor.clear_experience()

            self.prev_frame = None
            self.prev_action = None

        
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
                cur_frame=ft,  # Already numpy bool
                prev_frame=self.prev_frame,
                action=self.prev_action,
                diff_ravel_pixel_indices=diff_pixels,
            )

            if self.prev_frame is not None and self.prev_action is not None:
                self.next_state_predictor.add_experience(experience)

            self.prev_frame = ft


        if self.action_counter % self.train_frequency == 0:
            self.next_state_predictor.train_batch(self.action_counter)


        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            # if game is not started (at init or after GAME_OVER) we need to reset
            # add a small delay before resetting after GAME_OVER to avoid timeout
            action = GameAction.RESET
        else:
            # else choose a random action that isnt reset

            if latest_frame.levels_completed == 0:
                available_actions = [a for a in GameAction if a is not GameAction.RESET and a.value in latest_frame.available_actions]
                action = random.choice(available_actions)
                if action.is_complex():
                    action.set_data(
                        {
                            "x": random.randint(0, 63),
                            "y": random.randint(0, 63),
                        }
                    )
                action.reasoning = f"Randomly sampled action at level 0: {action.value}"
            else:
                action, coords, _, _ = self.action_model.choose_action(frame_tensors[-1])
                y, x = coords if coords is not None else (None, None)
                if action.is_complex():
                    action.set_data(
                        {
                            "x": x,
                            "y": y,
                        }
                    )


        self.prev_action = action

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
        self.next_state_predictor.save_model(id=f"{self.game_id}_{id}")
        self.action_model.save_model(id=f"{self.game_id}_{id}")

        self.logger.info(f"Saved models with id {id}")
