import hashlib
import json
import logging
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import numpy.typing as npt
from arcengine import FrameData, GameAction, GameState
from tensorboardX import SummaryWriter

from agents.templates.nets import ActionModel, ConnectedComponentConvNet
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
    action: GameAction | None  # Action taken to get to this state, optional


@dataclass
class ActionExperience:
    cur_frame: npt.NDArray[np.int64]
    prev_frame: npt.NDArray[np.int64]
    action_index: int


@dataclass(frozen=True)
class DynamicComponentPrediction:
    frame: npt.NDArray[np.int64]
    dynamic_mask: npt.NDArray[np.bool_]
    component_masks: list[npt.NDArray[np.bool_]]

def _sigmoid_np(logits: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    logits = np.clip(logits, -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def _find_connected_component_masks(
    mask: npt.NDArray[np.bool_],
) -> list[npt.NDArray[np.bool_]]:
    """Returns one boolean mask per connected component."""
    mask = np.asarray(mask, dtype=np.bool_)
    height, width = mask.shape
    visited = np.zeros((height, width), dtype=np.bool_)
    component_masks: list[npt.NDArray[np.bool_]] = []

    for start_y in range(height):
        for start_x in range(width):
            if not mask[start_y, start_x] or visited[start_y, start_x]:
                continue

            component_mask = np.zeros((height, width), dtype=np.bool_)
            queue = deque([(start_y, start_x)])
            visited[start_y, start_x] = True
            component_mask[start_y, start_x] = True

            while queue:
                y, x = queue.popleft()
                for ny, nx in (
                    (y - 1, x),
                    (y - 1, x - 1),
                    (y - 1, x + 1),
                    (y + 1, x),
                    (y + 1, x - 1),
                    (y + 1, x + 1),
                    (y, x - 1),
                    (y, x + 1),
                ):
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        component_mask[ny, nx] = True
                        queue.append((ny, nx))

            component_masks.append(component_mask)

    return component_masks


def _dynamic_gate_logits_to_components(
    frame: npt.NDArray[np.int64],
    dynamic_gate_logits: npt.NDArray[np.float32],
    threshold: float = 0.5,
) -> DynamicComponentPrediction:
    if dynamic_gate_logits.ndim != 2:
        # we need to bfs on a 2d grid
        raise ValueError(f"Expected 2D gate logits, got {dynamic_gate_logits.shape}.")

    gate = _sigmoid_np(dynamic_gate_logits)
    dynamic_mask = gate <= threshold
    return DynamicComponentPrediction(
        frame=frame,
        dynamic_mask=dynamic_mask,
        component_masks=_find_connected_component_masks(dynamic_mask),
    )


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
        assert experience.action is not None, "Action is required for next state prediction."

        # Check if (current, previous, action) state pair is unique using a hash of the current state

        hasher = hashlib.md5()
        hasher.update(experience.prev_frame.tobytes())
        hasher.update(experience.cur_frame.tobytes())
        hasher.update(experience.action.value.to_bytes())
        if experience.action.is_complex() and experience.action.action_data is not None:
            hasher.update(experience.action.action_data.x.to_bytes())
            hasher.update(experience.action.action_data.y.to_bytes())

        combined_hash = hasher.hexdigest()

        if combined_hash not in self.experience_hashes:
            self.experience_hashes.add(combined_hash)
            self.experience_buffer.append(experience)

    def clear_experience(self) -> None:
        self.experience_buffer.clear()
        self.experience_hashes.clear()

        self.logger.info("Cleared experience buffer and hashes for NextStatePrediction.")

    def predict_dynamic_ccs(
        self,
        frames: npt.NDArray[np.int64] | mx.array,
        threshold: float = 0.5,
    ) -> list[DynamicComponentPrediction]:
        """
        Predicts the dynamic connected components for the given frames.

        Returns:
            A list of DynamicComponentPrediction, one for each input frame.
        """
        frames_np = np.asarray(frames, dtype=np.int64)
        if frames_np.ndim != 3:
            raise ValueError(f"Expected frame shape (B, H, W), got {frames_np.shape}.")

        (
            _,
            _,
            _,
            _,
            dynamic_gate_logits,
        ) = self.net.decompose(mx.array(frames_np))

        gate_logits_np = np.asarray(dynamic_gate_logits)
        return [
            _dynamic_gate_logits_to_components(frame, logits.squeeze(-1), threshold=threshold)
            for frame, logits in zip(frames_np, gate_logits_np)
        ]

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
        gamma: float = 50.0,
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


class ConnectedComponentRepresentation:
    """Builds connected-component masks and learns CC embeddings."""

    def __init__(
        self,
        base_dir: str,
        writer: SummaryWriter,
        num_colours: int,
        grid_size: int,
        next_state_predictor: NextStatePrediction,
    ) -> None:
        self.base_dir = base_dir
        self.writer = writer
        self.num_colours = num_colours
        self.grid_size = grid_size
        self.next_state_predictor = next_state_predictor
        self.negative_margin = 0.0
        self.mask_keep_probability = 0.90
        self.logger = logging.getLogger("ConnectedComponentRepresentation")
        self.frame_buffer: deque[npt.NDArray[np.int64]] = deque(maxlen=50_000)
        self.frame_hashes: set[str] = set()

        self.net = ConnectedComponentConvNet(
            num_colors=num_colours,
        )
        self.optimizer = optim.AdamW(learning_rate=0.0005) # TODO: tune
        self.loss_and_grad_fn = nn.value_and_grad(self.net, self._loss_fn)

    def save_model(self, id: str = "final") -> None:
        model_path = os.path.join(self.base_dir, f"cc_representation_model_{id}.safetensors")
        config_path = os.path.join(self.base_dir, f"cc_representation_model_{id}.json")

        self.net.save_weights(model_path)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_colours": self.num_colours,
                    "color_embeddings_dim": self.net.color_embedding_dim,
                    "hidden_dim": self.net.hidden_dim,
                    "component_embedding_dim": self.net.component_embedding_dim,
                    "position_feature_dim": self.net.position_feature_dim,
                },
                f,
                indent=2,
            )

        self.logger.info(f"Saved connected-component representation weights to {model_path}")
        self.logger.info(f"Saved connected-component representation config to {config_path}")

    def add_frame(self, frame: npt.NDArray[np.int64]) -> None:
        assert frame.shape == (self.grid_size, self.grid_size), f"Expected frame shape ({self.grid_size}, {self.grid_size}), got {frame.shape}."

        frame_hash = hashlib.md5(frame.tobytes()).hexdigest()
        if frame_hash in self.frame_hashes:
            return

        self.frame_hashes.add(frame_hash)
        self.frame_buffer.append(frame)

    def train_batch(self, action_counter: int, batch_size: int = 8) -> None:
        if len(self.frame_buffer) < batch_size:
            return

        frame_indices = np.random.choice(
            len(self.frame_buffer),
            batch_size,
            replace=False,
        )
        frames_np = np.stack([self.frame_buffer[int(idx)] for idx in frame_indices])
        frame_components = self.frames_to_components(frames_np)

        component_sizes = [
            len(frame_component.component_masks)
            for frame_component in frame_components
        ]

        self.writer.add_histogram(
            "CCRepresentation/num_components_per_frame",
            np.array(component_sizes, dtype=np.int64),
            action_counter,
        )

        self.net.train()

        frames, dynamic_masks, component_masks = self._components_to_arrays(
            frame_components,
        )

        component_mask_arrays = [
            (
                np.stack(component_mask).astype(np.bool_)
                if component_mask
                else np.zeros(
                    (0, self.grid_size, self.grid_size),
                    dtype=np.bool_,
                )
            )
            for component_mask in component_masks
        ]
        if not any(component_mask_array.shape[0] > 1 for component_mask_array in component_mask_arrays):
            return

        loss, grads = self.loss_and_grad_fn(
            mx.array(frames),
            mx.array(dynamic_masks),
            [mx.array(component_mask) for component_mask in component_mask_arrays],
        )
        self.optimizer.update(self.net, grads)
        mx.eval(self.net.parameters(), self.optimizer.state)

        self.writer.add_scalar(
            "CCRepresentation/pair_loss",
            float(loss.item()),
            action_counter,
        )

    def frame_to_components(self, frame: npt.NDArray[np.int64]) -> DynamicComponentPrediction:
        return self.frames_to_components(frame)[0]

    def frames_to_components(
        self,
        frames: npt.NDArray[np.int64],
    ) -> list[DynamicComponentPrediction]:
        frames_np = np.asarray(frames, dtype=np.int64)
        if frames_np.ndim == 2:
            frames_np = frames_np[None, :, :]
        if frames_np.ndim != 3:
            raise ValueError(f"Expected frames with shape (B, H, W), got {frames_np.shape}.")
        if frames_np.shape[1:] != (self.grid_size, self.grid_size):
            raise ValueError(
                f"Expected frame shape ({self.grid_size}, {self.grid_size}), got {frames_np.shape[1:]}."
            )

        return self.next_state_predictor.predict_dynamic_ccs(frames=frames_np)
       

    def _components_to_arrays(
        self,
        components: list[DynamicComponentPrediction],
    ) -> tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.bool_],
        list[npt.NDArray[np.bool_]],
    ]:
        frames = np.stack([component.frame for component in components]).astype(np.int64)
        dynamic_mask = np.stack([component.dynamic_mask for component in components])
        component_masks = [component.component_masks for component in components]
        return frames, dynamic_mask, component_masks

    def _loss_fn(
        self,
        frames: mx.array,
        dynamic_masks: mx.array,
        component_masks: list[mx.array],
    ) -> mx.array:
        frame_features = self.net(
            frames,
            dynamic_masks,
        )

        frame_cc_embeddings = []
        for frame_idx, cc_masks_for_frame in enumerate(component_masks):
            component_count = cc_masks_for_frame.shape[0]
            if component_count == 0:
                continue
            repeated_frame_features = mx.repeat(
                frame_features[frame_idx : frame_idx + 1],
                component_count,
                axis=0,
            )
            frame_cc_embeddings.append(
                self.net.cc_embedding_from_features(
                    repeated_frame_features,
                    cc_masks_for_frame,
                )
            )

        frame_cc_similarity_matrices = [
            self._normalize_mx(cc_embeddings) @ self._normalize_mx(cc_embeddings).T
            for cc_embeddings in frame_cc_embeddings
        ]

        frame_losses = [
            self.loss_from_similarity_matrix(similarity_matrix)
            for similarity_matrix in frame_cc_similarity_matrices
        ]

        return mx.mean(mx.stack(frame_losses))


    def loss_from_similarity_matrix(self, similarity_matrix: mx.array) -> mx.array:
        num_components = similarity_matrix.shape[0]
        if num_components <= 1:
            return mx.array(0.0)

        off_diagonal_mask = 1.0 - mx.eye(num_components)
        negative_penalty = mx.maximum(
            similarity_matrix - self.negative_margin,
            0.0,
        )
        negative_penalty = negative_penalty * negative_penalty * off_diagonal_mask
        return mx.sum(negative_penalty) / (num_components * (num_components - 1))


    def _normalize_mx(self, embeddings: mx.array) -> mx.array:
        norm = mx.sqrt(mx.sum(embeddings * embeddings, axis=-1, keepdims=True))
        return embeddings / mx.maximum(norm, 1e-6)


class FrameSimilarityScorer:
    """Scores dynamic frame geometry with component and relation histograms."""

    def __init__(
        self,
        component_representation: ConnectedComponentRepresentation,
        coarse_grid_size: int = 8,
    ) -> None:
        self.component_representation = component_representation
        self.coarse_grid_size = coarse_grid_size
        self.angle_bins = 8
        self.distance_bins = 4
        self.size_ratio_bins = 4
        self.area_bins = np.array(
            [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
            dtype=np.float32,
        )
        self.weights = {
            "dynamic_pixels": 0.35,
            "components": 0.25,
            "relations": 0.40,
        }

    def score_frames_against_references(
        self,
        frames: npt.NDArray[np.int64],
        reference_frames: npt.NDArray[np.int64] | list[npt.NDArray[np.int64]],
    ) -> npt.NDArray[np.float32]:
        frames_np = np.asarray(frames, dtype=np.int64)
        is_single = frames_np.ndim == 2
        if is_single:
            frames_np = frames_np[None, :, :]
        if frames_np.ndim != 3:
            raise ValueError(f"Expected frames with shape (B, H, W), got {frames_np.shape}.")

        reference_frames_np = np.asarray(reference_frames, dtype=np.int64)
        if reference_frames_np.ndim == 2:
            reference_frames_np = reference_frames_np[None, :, :]
        if reference_frames_np.ndim != 3:
            raise ValueError(
                f"Expected reference frames with shape (B, H, W), got {reference_frames_np.shape}."
            )

        candidate_components = self.component_representation.frames_to_components(frames_np)
        reference_components = self.component_representation.frames_to_components(
            reference_frames_np
        )
        reference_descriptors = [
            self._frame_descriptor(components)
            for components in reference_components
        ]

        rewards = [
            self._score_against_reference_descriptors(
                components,
                reference_descriptors,
            )
            for components in candidate_components
        ]

        rewards_np = np.clip(np.asarray(rewards, dtype=np.float32), 0.0, 1.0)
        return rewards_np[:1] if is_single else rewards_np

    def _score_against_reference_descriptors(
        self,
        frame_components: DynamicComponentPrediction,
        reference_descriptors: list[npt.NDArray[np.float32]],
    ) -> float:
        if not reference_descriptors:
            return 0.0

        best_score = 0.0
        for transformed_components in self._symmetry_variants(frame_components):
            descriptor = self._frame_descriptor(transformed_components)
            for reference_descriptor in reference_descriptors:
                best_score = max(
                    best_score,
                    float(np.dot(descriptor, reference_descriptor)),
                )
        return best_score

    def _frame_descriptor(
        self,
        frame_components: DynamicComponentPrediction,
    ) -> npt.NDArray[np.float32]:
        frame = np.asarray(frame_components.frame, dtype=np.int64)
        dynamic_mask = np.asarray(frame_components.dynamic_mask, dtype=np.bool_)
        component_masks = [
            np.asarray(component_mask, dtype=np.bool_)
            for component_mask in frame_components.component_masks
        ]
        component_stats = self._component_stats(frame, component_masks)

        parts = [
            (
                self.weights["dynamic_pixels"],
                self._dynamic_pixel_descriptor(frame, dynamic_mask),
            ),
            (
                self.weights["components"],
                self._component_descriptor(component_stats),
            ),
            (
                self.weights["relations"],
                self._relation_descriptor(component_stats),
            ),
        ]

        descriptor_parts = [
            np.sqrt(weight) * self._normalize_np(part)
            for weight, part in parts
        ]
        return self._normalize_np(np.concatenate(descriptor_parts).astype(np.float32))

    def _dynamic_pixel_descriptor(
        self,
        frame: npt.NDArray[np.int64],
        dynamic_mask: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.float32]:
        num_colours = self.component_representation.num_colours
        grid = self.coarse_grid_size
        hist_size = num_colours * grid * grid
        if not np.any(dynamic_mask):
            return np.zeros(hist_size + grid * grid, dtype=np.float32)

        height, width = frame.shape
        ys, xs = np.nonzero(dynamic_mask)
        colours = np.clip(frame[ys, xs], 0, num_colours - 1)
        cell_y = np.minimum((ys * grid) // height, grid - 1)
        cell_x = np.minimum((xs * grid) // width, grid - 1)
        cell_idx = cell_y * grid + cell_x
        colour_position_idx = colours * grid * grid + cell_idx

        colour_position_hist = np.bincount(
            colour_position_idx,
            minlength=hist_size,
        ).astype(np.float32)
        mask_hist = np.bincount(
            cell_idx,
            minlength=grid * grid,
        ).astype(np.float32)
        return np.concatenate([colour_position_hist, mask_hist]).astype(np.float32)

    def _component_stats(
        self,
        frame: npt.NDArray[np.int64],
        component_masks: list[npt.NDArray[np.bool_]],
    ) -> list[dict[str, float]]:
        height, width = frame.shape
        stats: list[dict[str, float]] = []

        for component_mask in component_masks:
            if not np.any(component_mask):
                continue

            ys, xs = np.nonzero(component_mask)
            colours = np.clip(
                frame[component_mask],
                0,
                self.component_representation.num_colours - 1,
            )
            colour_hist = np.bincount(
                colours,
                minlength=self.component_representation.num_colours,
            )
            dominant_colour = int(np.argmax(colour_hist))
            area = float(len(ys))
            min_y, max_y = int(np.min(ys)), int(np.max(ys))
            min_x, max_x = int(np.min(xs)), int(np.max(xs))
            bbox_height = float(max_y - min_y + 1)
            bbox_width = float(max_x - min_x + 1)
            centroid_y = float(np.mean(ys) / max(height - 1, 1) * 2.0 - 1.0)
            centroid_x = float(np.mean(xs) / max(width - 1, 1) * 2.0 - 1.0)

            stats.append(
                {
                    "area": area,
                    "dominant_colour": float(dominant_colour),
                    "centroid_y": centroid_y,
                    "centroid_x": centroid_x,
                    "bbox_height": bbox_height,
                    "bbox_width": bbox_width,
                }
            )

        return stats

    def _component_descriptor(
        self,
        component_stats: list[dict[str, float]],
    ) -> npt.NDArray[np.float32]:
        num_colours = self.component_representation.num_colours
        grid = self.coarse_grid_size
        colour_hist = np.zeros(num_colours, dtype=np.float32)
        colour_position_hist = np.zeros(num_colours * grid * grid, dtype=np.float32)
        area_hist = np.zeros(len(self.area_bins) + 1, dtype=np.float32)
        bbox_hist = np.zeros(grid * grid, dtype=np.float32)

        for stats in component_stats:
            colour = int(stats["dominant_colour"])
            area = float(stats["area"])
            centroid_y = float(stats["centroid_y"])
            centroid_x = float(stats["centroid_x"])
            cell_y = self._coarse_coord_bin(centroid_y)
            cell_x = self._coarse_coord_bin(centroid_x)
            cell_idx = cell_y * grid + cell_x

            colour_hist[colour] += area
            colour_position_hist[colour * grid * grid + cell_idx] += area
            area_hist[np.searchsorted(self.area_bins, area, side="right")] += 1.0

            bbox_y_bin = min(
                int(stats["bbox_height"] / self.component_representation.grid_size * grid),
                grid - 1,
            )
            bbox_x_bin = min(
                int(stats["bbox_width"] / self.component_representation.grid_size * grid),
                grid - 1,
            )
            bbox_hist[bbox_y_bin * grid + bbox_x_bin] += 1.0

        return np.concatenate(
            [
                colour_hist,
                colour_position_hist,
                area_hist,
                bbox_hist,
            ]
        ).astype(np.float32)

    def _relation_descriptor(
        self,
        component_stats: list[dict[str, float]],
    ) -> npt.NDArray[np.float32]:
        num_colours = self.component_representation.num_colours
        hist_size = (
            num_colours
            * num_colours
            * self.angle_bins
            * self.distance_bins
            * self.size_ratio_bins
        )
        relation_hist = np.zeros(hist_size, dtype=np.float32)
        if len(component_stats) < 2:
            return relation_hist

        for left_idx in range(len(component_stats)):
            for right_idx in range(left_idx + 1, len(component_stats)):
                first, second = self._ordered_relation_pair(
                    component_stats[left_idx],
                    component_stats[right_idx],
                )
                colour_a = int(first["dominant_colour"])
                colour_b = int(second["dominant_colour"])
                dy = float(second["centroid_y"] - first["centroid_y"])
                dx = float(second["centroid_x"] - first["centroid_x"])
                angle = (np.arctan2(dy, dx) + np.pi) / (2.0 * np.pi)
                angle_bin = min(int(angle * self.angle_bins), self.angle_bins - 1)
                distance = np.sqrt(dx * dx + dy * dy) / np.sqrt(8.0)
                distance_bin = min(
                    int(distance * self.distance_bins),
                    self.distance_bins - 1,
                )
                size_ratio = min(first["area"], second["area"]) / max(
                    first["area"],
                    second["area"],
                    1e-6,
                )
                size_ratio_bin = min(
                    int(size_ratio * self.size_ratio_bins),
                    self.size_ratio_bins - 1,
                )
                relation_idx = ((((
                    colour_a * num_colours + colour_b
                ) * self.angle_bins + angle_bin
                ) * self.distance_bins + distance_bin
                ) * self.size_ratio_bins + size_ratio_bin)
                relation_hist[relation_idx] += 1.0

        return relation_hist

    def _ordered_relation_pair(
        self,
        first: dict[str, float],
        second: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, float]]:
        first_key = (
            int(first["dominant_colour"]),
            -float(first["area"]),
            float(first["centroid_y"]),
            float(first["centroid_x"]),
        )
        second_key = (
            int(second["dominant_colour"]),
            -float(second["area"]),
            float(second["centroid_y"]),
            float(second["centroid_x"]),
        )
        return (first, second) if first_key <= second_key else (second, first)

    def _coarse_coord_bin(self, coord: float) -> int:
        normalized = np.clip((coord + 1.0) * 0.5, 0.0, 1.0 - 1e-6)
        return int(normalized * self.coarse_grid_size)

    def _symmetry_variants(
        self,
        frame_components: DynamicComponentPrediction,
    ) -> list[DynamicComponentPrediction]:
        return [
            DynamicComponentPrediction(
                frame=self._transform_grid(frame_components.frame, transform_idx),
                dynamic_mask=self._transform_grid(
                    frame_components.dynamic_mask,
                    transform_idx,
                ),
                component_masks=[
                    self._transform_grid(component_mask, transform_idx)
                    for component_mask in frame_components.component_masks
                ],
            )
            for transform_idx in range(8)
        ]

    def _transform_grid(
        self,
        grid: npt.NDArray[Any],
        transform_idx: int,
    ) -> npt.NDArray[Any]:
        if transform_idx == 0:
            return np.asarray(grid)
        if transform_idx == 1:
            return np.rot90(grid, 1)
        if transform_idx == 2:
            return np.rot90(grid, 2)
        if transform_idx == 3:
            return np.rot90(grid, 3)
        if transform_idx == 4:
            return np.fliplr(grid)
        if transform_idx == 5:
            return np.flipud(grid)
        if transform_idx == 6:
            return np.transpose(grid)
        return np.fliplr(np.rot90(grid, 1))

    def _normalize_np(
        self,
        vector: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        vector_np = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector_np))
        if norm <= 1e-6:
            return np.zeros_like(vector_np, dtype=np.float32)
        return (vector_np / norm).astype(np.float32)


class RLActionModel:
    def __init__(
        self,
        base_dir: str,
        writer: SummaryWriter,
        num_colours: int,
        available_simple_actions: list[GameAction],
        is_coord_action_allowed: bool,
        grid_size: int,
        frame_similarity_scorer: FrameSimilarityScorer,
        get_ccs_fn: Callable[[npt.NDArray[np.int64]], npt.NDArray[np.int64]] | None = None,
    ) -> None:
        self.base_dir = base_dir
        self.writer = writer
        self.num_colours = num_colours
        self.available_simple_actions = available_simple_actions
        self.is_coord_action_allowed = is_coord_action_allowed
        self.grid_size = grid_size
        self.num_coordinates = grid_size * grid_size
        self.frame_similarity_scorer = frame_similarity_scorer
        self.logger = logging.getLogger("RLActionModel")
        self.experience_buffer: deque[ActionExperience] = deque(maxlen=100_000)
        self.experience_hashes = set()
        self.winning_frames: npt.NDArray[np.int64] | None = None

        self.get_ccs_fn = get_ccs_fn

        self._reset_model()
        self.logger.info("Initialized RLActionModel module")

    def _reset_model(self) -> None:
        self.net = ActionModel(
            num_colors=self.num_colours,
            num_simple_action_types=len(self.available_simple_actions),
            is_coord_action_allowed=self.is_coord_action_allowed,
        )
        self.optimizer = optim.AdamW(learning_rate=0.001) # TODO: tune
        self.loss_and_grad_fn = nn.value_and_grad(self.net, self._loss_fn)

    def add_experience(self, experience: Experience) -> None:
        assert experience.prev_frame is not None, "Previous state is required for action prediction."
        assert experience.action is not None, "Action is required for action prediction."
        if experience.action is GameAction.RESET:
            return

        # Check if (current, previous, action) state pair is unique using a hash of the current state
        action_index = self._action_to_index(experience.action)

        hasher = hashlib.md5()
        hasher.update(experience.prev_frame.tobytes())
        hasher.update(experience.cur_frame.tobytes())
        hasher.update(int(action_index).to_bytes(2))

        combined_hash = hasher.hexdigest()

        if combined_hash not in self.experience_hashes:
            self.experience_hashes.add(combined_hash)
            self.experience_buffer.append(
                ActionExperience(
                    cur_frame=experience.cur_frame,
                    prev_frame=experience.prev_frame,
                    action_index=action_index,
                )
            )

    def _action_to_index(self, action: GameAction) -> int:
        if action in self.available_simple_actions:
            return self.available_simple_actions.index(action)

        if action.is_complex() and self.is_coord_action_allowed:
            return (
                len(self.available_simple_actions)
                + action.action_data.y * self.grid_size
                + action.action_data.x
            )

        raise ValueError(f"Action {action} is not available to this action model.")

    def clear_experience(self) -> None:
        self.experience_buffer.clear()
        self.experience_hashes.clear()

        self.logger.info("Cleared experience buffer for RLActionModel.")

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
        policy_logits = self._policy_logits(combined_logits)
        policy_probs = mx.softmax(policy_logits, axis=-1)
        policy_probs_np = np.asarray(policy_probs)
        prob_sum = policy_probs_np.sum()
        if not np.isfinite(prob_sum) or prob_sum <= 0:
            raise ValueError(
                "No valid action probabilities available for sampling: "
                f"available_actions={self.available_simple_actions}, prob_sum={prob_sum}"
            )
        policy_probs_np = policy_probs_np / prob_sum

        selected_idx = np.random.choice(len(policy_probs_np), p=policy_probs_np)

        if selected_idx < num_simple_action_types:
            return self.available_simple_actions[selected_idx], None, None, policy_probs_np

        coord_idx = selected_idx - num_simple_action_types
        y_idx = coord_idx // self.grid_size
        x_idx = coord_idx % self.grid_size
        return GameAction.ACTION6, (y_idx, x_idx), coord_idx, policy_probs_np

    def _policy_logits(self, combined_logits: mx.array) -> mx.array:
        num_simple_action_types = len(self.available_simple_actions)
        if not self.is_coord_action_allowed:
            return combined_logits

        action_logits = combined_logits[..., :num_simple_action_types]
        coord_logits = combined_logits[..., num_simple_action_types:] - np.log(
            self.num_coordinates
        )
        return mx.concatenate([action_logits, coord_logits], axis=-1)

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


    def set_winning_frame(
        self,
        winning_frames: npt.NDArray[np.int64],
    ) -> None:
        self.winning_frames = np.asarray(winning_frames, dtype=np.int64)


    def train_batch(self, action_counter: int, batch_size: int = 8) -> None:
        assert self.winning_frames is not None, "Winning frames must be set before training."
        
        if len(self.experience_buffer) < batch_size:
            return
        
        self.net.train()

        # Sample batch from experience buffer
        batch_indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in batch_indices]

        # Prepare batch data - convert numpy arrays to tensors and move to GPU
        frames_np = np.stack([exp.cur_frame for exp in batch])
        prev_frames_np = np.stack([exp.prev_frame for exp in batch])
        prev_frames = mx.array(prev_frames_np)
        action_indices = mx.array(
            np.array([exp.action_index for exp in batch], dtype=np.int64)
        )
        rewards_np = self.frame_similarity_scorer.score_frames_against_references(
            frames_np,
            self.winning_frames,
        ).astype(np.float32)
        rewards = mx.array(rewards_np[:, None])

        _, grads = self.loss_and_grad_fn(
            prev_frames,
            action_indices,
            rewards,
            action_counter,
        )

        self.optimizer.update(self.net, grads)
        mx.eval(self.net.parameters(), self.optimizer.state)

        self.writer.add_scalar(
            "RLAction/reward_mean",
            float(np.mean(rewards_np)),
            action_counter,
        )
        self.writer.add_scalar(
            "RLAction/reward_max",
            float(np.max(rewards_np)),
            action_counter,
        )

    def _loss_fn(
        self,
        frames: mx.array,
        action_indices: mx.array,
        rewards: mx.array,
        action_counter: int,
    ) -> mx.array:
        combined_logits = self.net(frames)  # (batch, # actions)
        policy_logits = self._policy_logits(combined_logits)
        action_indices = mx.expand_dims(action_indices.astype(mx.int64), axis=1)

        rewards = rewards.astype(mx.float32)
        log_probs = nn.log_softmax(policy_logits, axis=1)
        selected_log_probs = mx.take_along_axis(log_probs, action_indices, axis=1)
        main_loss = -mx.mean(selected_log_probs * rewards)

        policy_probs = mx.softmax(policy_logits, axis=1)
        entropy = -mx.mean(mx.sum(policy_probs * log_probs, axis=-1))

        entropy_coeff = 0.001
        total_loss = main_loss - entropy_coeff * entropy

        self.writer.add_scalar(
            "RLAction/main_loss",
            float(main_loss.item()),
            action_counter,
        )
        self.writer.add_scalar(
            "RLAction/selected_logit_abs_mean",
            float(
                mx.mean(
                    mx.abs(mx.take_along_axis(policy_logits, action_indices, axis=1))
                ).item()
            ),
            action_counter,
        )
        self.writer.add_histogram(
            "RLAction/selected_log_prob",
            np.asarray(selected_log_probs, dtype=np.float32),
            action_counter,
        )
        self.writer.add_scalar(
            "RLAction/entropy",
            float(entropy.item()),
            action_counter,
        )
        
        return total_loss


class RandomLayeredRL(Agent):
    """An agent that always selects actions at random."""

    MAX_ACTIONS = 10_000
    START_CC_COMPARISION_TRAINING = 100
    START_RL_TRAINING = sys.maxsize


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

        # store recent frames for each level to use as positive examples of winning frames in the frame similarity scorer
        self.level_frame_history: deque[npt.NDArray[np.int64]] = deque(maxlen=4)

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
        self.component_representation = ConnectedComponentRepresentation(
            base_dir=self.base_dir,
            writer=self.writer,
            num_colours=self.num_colours,
            grid_size=self.grid_size,
            next_state_predictor=self.next_state_predictor,
        )
        self.frame_similarity_scorer = FrameSimilarityScorer(
            component_representation=self.component_representation,
        )
        self.action_model = RLActionModel(
            base_dir=self.base_dir,
            writer=self.writer,
            num_colours=self.num_colours,
            available_simple_actions=available_simple_actions,
            is_coord_action_allowed=is_coord_action_allowed,
            grid_size=self.grid_size,
            frame_similarity_scorer=self.frame_similarity_scorer,
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
            assert len(self.level_frame_history), "No frames stored for completed level, cannot set winning frame for frame similarity scorer."

            winning_frames = np.stack(list(self.level_frame_history))
            self.levels_completed_prev = latest_frame.levels_completed

            self._save_models(id=f"level_{self.levels_completed_prev}")  # Save model on level up

            # don't reset the model on level up, we want to carry forward past experience to next level
            self.next_state_predictor.clear_experience()
            self.action_model.clear_experience()
            self.action_model.set_winning_frame(winning_frames)
            self.level_frame_history.clear()

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
            experience = Experience(
                cur_frame=ft,  # Already numpy bool
                prev_frame=self.prev_frame,
                action=self.prev_action,
            )

            if self.prev_frame is not None and self.prev_action is not None:
                self.next_state_predictor.add_experience(experience)
                self.component_representation.add_frame(ft)

                if latest_frame.levels_completed > 0:
                    self.action_model.add_experience(experience)

            self.prev_frame = ft
            self.level_frame_history.append(ft)


        if self.action_counter % self.train_frequency == 0:
            self.next_state_predictor.train_batch(self.action_counter)
            if self.action_counter > self.START_CC_COMPARISION_TRAINING:
                self.component_representation.train_batch(self.action_counter)
            if latest_frame.levels_completed > 0 and self.action_counter > self.START_RL_TRAINING:
                self.action_model.train_batch(self.action_counter)


        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            # if game is not started (at init or after GAME_OVER) we need to reset
            # add a small delay before resetting after GAME_OVER to avoid timeout
            action = GameAction.RESET
            self.level_frame_history.clear()
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
        self.component_representation.save_model(id=f"{self.game_id}_{id}")
        self.action_model.save_model(id=f"{self.game_id}_{id}")

        self.logger.info(f"Saved models with id {id}")
