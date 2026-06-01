from __future__ import annotations

import argparse
import base64
import io
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import arc_agi
import mlx.core as mx
import numpy as np
from arcengine import FrameDataRaw, GameAction
from flask import Flask, jsonify, render_template_string, request
from PIL import Image, ImageDraw

from agents.recorder import RECORDING_SUFFIX
from agents.templates.nets import Layered
from view_utils import create_grid_image, hex_to_rgb, key_colors


WINNING_REFERENCE_HISTORY_LENGTH = 4


@dataclass(frozen=True)
class ModelPaths:
    config_path: Path
    weights_path: Path


@dataclass(frozen=True)
class RecordingFrame:
    index: int
    timestamp: str
    grid: list[list[int]]
    state: str
    levels_completed: int
    win_levels: int
    available_actions: list[int]
    action_input: dict[str, Any] | None
    game_id: str


@dataclass
class GameSnapshot:
    image: str | None
    state: str
    levels_completed: int
    win_levels: int
    available_actions: list[int]
    action_input: dict[str, Any] | None
    step_count: int
    game_id: str


def list_model_paths(runs_dir: Path) -> list[ModelPaths]:
    model_paths: list[ModelPaths] = []
    if not runs_dir.exists():
        return model_paths

    for config_path in sorted(runs_dir.rglob("layered_next_state_model_*.json")):
        weights_path = config_path.with_suffix(".safetensors")
        if weights_path.exists():
            model_paths.append(
                ModelPaths(config_path=config_path, weights_path=weights_path)
            )
    return model_paths


def list_cc_model_paths(runs_dir: Path) -> list[ModelPaths]:
    model_paths: list[ModelPaths] = []
    if not runs_dir.exists():
        return model_paths

    for config_path in sorted(runs_dir.rglob("cc_representation_model_*.json")):
        weights_path = config_path.with_suffix(".safetensors")
        if weights_path.exists():
            model_paths.append(
                ModelPaths(config_path=config_path, weights_path=weights_path)
            )
    return model_paths


def list_recording_paths(recordings_dir: Path) -> list[Path]:
    if not recordings_dir.exists():
        return []
    return sorted(recordings_dir.glob(f"*{RECORDING_SUFFIX}"))


def int_action_id(action_id: Any) -> int:
    if hasattr(action_id, "value"):
        return int(action_id.value)
    return int(action_id)


def action_label(action_id: int, data: dict[str, Any] | None = None) -> str:
    label = "RESET" if action_id == 0 else f"ACTION{action_id}"
    if data and "x" in data and "y" in data:
        return f"{label} ({data['x']}, {data['y']})"
    if data:
        return f"{label} {data}"
    return label


def action_from_id(action_id: int, data: dict[str, Any] | None = None) -> GameAction:
    action = GameAction.from_id(action_id)
    if action.is_complex() and data:
        action.set_data({"x": int(data.get("x", 0)), "y": int(data.get("y", 0))})
    return action


def parse_action_input(raw_action: Any) -> dict[str, Any] | None:
    if raw_action is None:
        return None
    if isinstance(raw_action, dict):
        raw_id = raw_action.get("id", 0)
        action_id = int_action_id(raw_id)
        data = dict(raw_action.get("data") or {})
        return {
            "id": action_id,
            "label": action_label(action_id, data),
            "data": data,
            "reasoning": raw_action.get("reasoning"),
        }

    action_id = int_action_id(raw_action)
    return {
        "id": action_id,
        "label": action_label(action_id),
        "data": {},
        "reasoning": None,
    }


def state_label(state: Any) -> str:
    if hasattr(state, "name"):
        return str(state.name)
    return str(state)


def load_recording_frames(recording_path: Path) -> list[RecordingFrame]:
    frames: list[RecordingFrame] = []
    with recording_path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            data = event.get("data", {})
            frame = data.get("frame")
            if not frame or not isinstance(frame, list):
                continue

            grid = frame[-1]
            if len(grid) != 64 or len(grid[0]) != 64:
                raise ValueError(
                    f"Expected a 64x64 grid, got {len(grid)}x{len(grid[0])} "
                    f"in {recording_path} at frame index {len(frames)}."
                )

            available_actions = [
                int_action_id(action_id)
                for action_id in data.get("available_actions", [])
            ]
            action_input = parse_action_input(data.get("action_input"))
            frames.append(
                RecordingFrame(
                    index=len(frames),
                    timestamp=str(event.get("timestamp", "")),
                    grid=grid,
                    state=state_label(data.get("state", "")),
                    levels_completed=int(data.get("levels_completed", 0)),
                    win_levels=int(data.get("win_levels", 0)),
                    available_actions=available_actions,
                    action_input=action_input,
                    game_id=str(data.get("game_id", "")),
                )
            )
    return frames


def frame_to_grid(frame_data: FrameDataRaw | None) -> np.ndarray | None:
    if frame_data is None or not frame_data.frame:
        return None

    grid = np.asarray(frame_data.frame[-1], dtype=np.int64)
    if grid.shape != (64, 64):
        return None
    return grid


def action_input_dict(frame_data: FrameDataRaw | None) -> dict[str, Any] | None:
    if frame_data is None or frame_data.action_input is None:
        return None

    action = frame_data.action_input.id
    action_id = int_action_id(action)
    data = dict(frame_data.action_input.data or {})
    return {
        "id": action_id,
        "label": action_label(action_id, data),
        "data": data,
        "reasoning": frame_data.action_input.reasoning,
    }


def snapshot_from_frame(
    frame_data: FrameDataRaw | None,
    step_count: int,
    fallback_game_id: str = "",
) -> GameSnapshot:
    grid = frame_to_grid(frame_data)
    available_actions = []
    if frame_data is not None:
        available_actions = [
            int_action_id(action_id) for action_id in frame_data.available_actions
        ]

    return GameSnapshot(
        image=grid_to_data_url(grid),
        state=state_label(frame_data.state) if frame_data is not None else "NOT_STARTED",
        levels_completed=int(frame_data.levels_completed) if frame_data is not None else 0,
        win_levels=int(frame_data.win_levels) if frame_data is not None else 0,
        available_actions=available_actions,
        action_input=action_input_dict(frame_data),
        step_count=step_count,
        game_id=str(frame_data.game_id) if frame_data is not None else fallback_game_id,
    )


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    probs = np.exp(shifted)
    return probs / np.clip(probs.sum(axis=-1, keepdims=True), 1e-8, None)


def decode_logits(logits: np.ndarray, mode: str) -> np.ndarray:
    if mode == "sample":
        probs = softmax_np(logits)
        flat_probs = probs.reshape(-1, probs.shape[-1])
        cdf = np.cumsum(flat_probs, axis=-1)
        cdf[:, -1] = 1.0
        draws = np.random.random((flat_probs.shape[0], 1))
        sampled = np.sum(cdf < draws, axis=-1)
        return sampled.astype(np.int64).reshape(logits.shape[:-1])

    return np.argmax(logits, axis=-1).astype(np.int64)


def create_layer_image(
    grid: np.ndarray,
    cell_size: int = 8,
    border_width: int = 1,
    static_gate_token_id: int | None = None,
) -> Image.Image:
    grid = np.asarray(grid, dtype=np.int64)
    height, width = grid.shape
    img_width = width * cell_size + (width + 1) * border_width
    img_height = height * cell_size + (height + 1) * border_width

    image = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(image)

    for y in range(height):
        for x in range(width):
            token = int(grid[y, x])
            left = x * (cell_size + border_width) + border_width
            top = y * (cell_size + border_width) + border_width
            right = left + cell_size
            bottom = top + cell_size

            if static_gate_token_id is not None and token == static_gate_token_id:
                draw.rectangle([left, top, right, bottom], fill=(242, 242, 238))
                draw.line([left, bottom, right, top], fill=(206, 206, 198))
            else:
                color = hex_to_rgb(key_colors.get(token, "#FFFFFF"))
                draw.rectangle([left, top, right, bottom], fill=color)

    return image


def create_gate_image(
    gate: np.ndarray,
    cell_size: int = 8,
    border_width: int = 1,
) -> Image.Image:
    gate = np.asarray(gate, dtype=np.float32).squeeze()
    if gate.ndim != 2:
        raise ValueError(f"Expected a 2D gate array, got {gate.shape}.")

    gate = np.clip(gate, 0.0, 1.0)
    height, width = gate.shape
    img_width = width * cell_size + (width + 1) * border_width
    img_height = height * cell_size + (height + 1) * border_width

    image = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(image)

    dynamic_rgb = np.array((42, 84, 102), dtype=np.float32)
    static_rgb = np.array((244, 214, 94), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            value = gate[y, x]
            color = tuple(
                np.round(dynamic_rgb * (1.0 - value) + static_rgb * value)
                .astype(np.uint8)
                .tolist()
            )
            left = x * (cell_size + border_width) + border_width
            top = y * (cell_size + border_width) + border_width
            right = left + cell_size
            bottom = top + cell_size
            draw.rectangle([left, top, right, bottom], fill=color)

    return image


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def grid_to_data_url(grid: np.ndarray | None) -> str | None:
    if grid is None:
        return None
    return image_to_data_url(create_grid_image(grid, cell_size=8, border_width=1))


def layer_grid_to_data_url(
    grid: np.ndarray | None,
    static_gate_token_id: int | None = None,
) -> str | None:
    if grid is None:
        return None
    return image_to_data_url(
        create_layer_image(
            grid,
            cell_size=8,
            border_width=1,
            static_gate_token_id=static_gate_token_id,
        )
    )


def gate_to_data_url(gate: np.ndarray | None) -> str | None:
    if gate is None:
        return None
    return image_to_data_url(create_gate_image(gate, cell_size=8, border_width=1))


def gate_to_values(gate: np.ndarray | None) -> list[list[float]] | None:
    if gate is None:
        return None
    gate = np.asarray(gate, dtype=np.float32).squeeze()
    if gate.ndim != 2:
        raise ValueError(f"Expected a 2D gate array, got {gate.shape}.")
    return np.round(gate, 6).astype(float).tolist()


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(logits, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-logits))


def grid_similarity(
    grid: np.ndarray,
    reference_grid: np.ndarray,
) -> float:
    grid = np.asarray(grid, dtype=np.int64)
    reference_grid = np.asarray(reference_grid, dtype=np.int64)
    if grid.shape != reference_grid.shape:
        raise ValueError(
            f"Expected grids with matching shapes, got {grid.shape} and {reference_grid.shape}."
        )
    return float(np.mean(grid == reference_grid))


def max_similarity_to_references(
    grid: np.ndarray | None,
    reference_grids: list[np.ndarray],
) -> float | None:
    if grid is None or not reference_grids:
        return None
    return max(grid_similarity(grid, reference_grid) for reference_grid in reference_grids)


def find_connected_component_masks(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=np.bool_)
    height, width = mask.shape
    visited = np.zeros((height, width), dtype=np.bool_)
    component_masks: list[np.ndarray] = []

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


def previous_score_delta(
    current_score: float | None,
    previous_score: float | None,
) -> float | None:
    if current_score is None or previous_score is None:
        return None
    return current_score - previous_score


def similarity_payload_from_scores(
    current_score: float | None,
    previous_score: float | None,
    reference_count: int,
) -> dict[str, Any]:
    return {
        "score": current_score,
        "delta": previous_score_delta(current_score, previous_score),
        "reference_count": reference_count,
    }


def recording_similarity_payload(
    frames: list[RecordingFrame],
    frame_index: int,
    layered_visualizer: LayeredVisualizer,
    cc_scorer: ConnectedComponentSimilarityScorer | None,
) -> dict[str, Any]:
    reference_grids: list[np.ndarray] = []
    level_history: deque[np.ndarray] = deque(maxlen=WINNING_REFERENCE_HISTORY_LENGTH)
    current_level = frames[0].levels_completed if frames else 0

    for frame in frames[:frame_index]:
        grid = np.asarray(frame.grid, dtype=np.int64)
        if frame.levels_completed > current_level:
            reference_grids.extend(grid.copy() for grid in level_history)
            level_history.clear()
            current_level = frame.levels_completed

        level_history.append(grid.copy())

    current_grid = np.asarray(frames[frame_index].grid, dtype=np.int64)
    previous_grid = (
        np.asarray(frames[frame_index - 1].grid, dtype=np.int64)
        if frame_index > 0
        else None
    )
    current_score = (
        cc_scorer.score_frame_against_references(
            layered_visualizer,
            current_grid,
            reference_grids,
        )
        if cc_scorer is not None
        else None
    )
    previous_score = (
        cc_scorer.score_frame_against_references(
            layered_visualizer,
            previous_grid,
            reference_grids,
        )
        if cc_scorer is not None and previous_grid is not None
        else None
    )

    return similarity_payload_from_scores(
        current_score=current_score,
        previous_score=previous_score,
        reference_count=len(reference_grids),
    )


def live_similarity_payload(
    current_grid: np.ndarray | None,
    previous_grid: np.ndarray | None,
    reference_grids: list[np.ndarray],
    layered_visualizer: LayeredVisualizer,
    cc_scorer: ConnectedComponentSimilarityScorer | None,
) -> dict[str, Any]:
    current_score = (
        cc_scorer.score_frame_against_references(
            layered_visualizer,
            current_grid,
            reference_grids,
        )
        if cc_scorer is not None
        else None
    )
    previous_score = (
        cc_scorer.score_frame_against_references(
            layered_visualizer,
            previous_grid,
            reference_grids,
        )
        if cc_scorer is not None and previous_grid is not None
        else None
    )
    return similarity_payload_from_scores(
        current_score=current_score,
        previous_score=previous_score,
        reference_count=len(reference_grids),
    )


def isolate_cc(gate: np.ndarray) -> np.ndarray:
    """
    Args:
    - gate: A 2D array of shape (height, width) representing the sigmoid gate values. 0 if fully dynamic, 1 if fully static.
    """
    return gate <= 0.5

class LayeredVisualizer:
    def __init__(self, config_path: Path, weights_path: Path) -> None:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.config_path = config_path
        self.weights_path = weights_path
        self.vocab_size = int(config.get("vocab_size", 16))
        self.action_dim = int(config.get("action_dim", 16))
        self.static_gate_token_id = -1
        self.net: Layered | None = None
        self._load_lock = Lock()

    def _get_net(self) -> Layered:
        if self.net is None:
            with self._load_lock:
                if self.net is None:
                    net = Layered(
                        vocab_size=self.vocab_size,
                        action_dim=self.action_dim,
                    )
                    net.load_weights(str(self.weights_path))
                    net.eval()
                    self.net = net
        return self.net

    def forward(
        self,
        grid: list[list[int]] | np.ndarray,
        action: GameAction,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        frame = np.asarray(grid, dtype=np.int64)
        if frame.shape != (64, 64):
            raise ValueError(f"Expected a 64x64 grid, got {frame.shape}.")

        net = self._get_net()
        (
            static_logits,
            dynamic_logits,
            dynamic_gate_logits,
            next_dynamic_logits,
            next_dynamic_gate_logits,
        ) = net(
            mx.array(frame[None, :, :]),
            [action],
        )
        return (
            np.asarray(static_logits)[0],
            np.asarray(dynamic_logits)[0],
            np.asarray(dynamic_gate_logits)[0],
            np.asarray(next_dynamic_logits)[0],
            np.asarray(next_dynamic_gate_logits)[0],
        )

    def render_layers(
        self,
        grid: list[list[int]] | np.ndarray,
        action: GameAction,
        mode: str,
    ) -> dict[str, Any]:
        if mode not in {"argmax", "sample"}:
            raise ValueError("mode must be 'argmax' or 'sample'.")

        frame = np.asarray(grid, dtype=np.int64)
        (
            static_logits,
            dynamic_logits,
            dynamic_gate_logits,
            next_dynamic_logits,
            next_dynamic_gate_logits,
        ) = self.forward(frame, action)

        static_grid = decode_logits(static_logits, mode)
        dynamic_grid = decode_logits(dynamic_logits, mode)
        dynamic_gate = sigmoid_np(dynamic_gate_logits[..., 0])
        dynamic_gate_cc = isolate_cc(dynamic_gate)
        dynamic_static_mask = dynamic_gate >= 0.5
        dynamic_masked_grid = np.where(
            dynamic_static_mask,
            self.static_gate_token_id,
            dynamic_grid,
        )
        reconstruction_grid = np.where(
            dynamic_static_mask,
            static_grid,
            dynamic_grid,
        )

        next_dynamic_grid = decode_logits(next_dynamic_logits, mode)
        next_dynamic_gate = sigmoid_np(next_dynamic_gate_logits[..., 0])
        next_dynamic_static_mask = next_dynamic_gate >= 0.5
        next_dynamic_masked_grid = np.where(
            next_dynamic_static_mask,
            self.static_gate_token_id,
            next_dynamic_grid,
        )
        next_reconstruction_grid = np.where(
            next_dynamic_static_mask,
            static_grid,
            next_dynamic_grid,
        )

        static_confidence = np.max(softmax_np(static_logits), axis=-1)

        return {
            "input": grid_to_data_url(frame),
            "static": layer_grid_to_data_url(static_grid),
            "dynamic": layer_grid_to_data_url(
                dynamic_grid,
                static_gate_token_id=self.static_gate_token_id,
            ),
            "dynamic_masked": layer_grid_to_data_url(
                dynamic_masked_grid,
                static_gate_token_id=self.static_gate_token_id,
            ),
            "dynamic_gate": gate_to_data_url(dynamic_gate),
            "dynamic_gate_cc": gate_to_data_url(dynamic_gate_cc),
            "reconstruction": grid_to_data_url(reconstruction_grid),
            "next_dynamic": layer_grid_to_data_url(
                next_dynamic_grid,
                static_gate_token_id=self.static_gate_token_id,
            ),
            "next_dynamic_masked": layer_grid_to_data_url(
                next_dynamic_masked_grid,
                static_gate_token_id=self.static_gate_token_id,
            ),
            "next_dynamic_gate": gate_to_data_url(next_dynamic_gate),
            "gate_values": {
                "dynamic_gate": gate_to_values(dynamic_gate),
                "dynamic_gate_cc": gate_to_values(dynamic_gate_cc),
                "next_dynamic_gate": gate_to_values(next_dynamic_gate),
            },
            "next_reconstruction": grid_to_data_url(next_reconstruction_grid),
            "metrics": {
                "dynamic_gate_mean": float(dynamic_gate.mean()),
                "dynamic_static_gate_fraction": float(
                    np.mean(dynamic_static_mask)
                ),
                "next_gate_mean": float(next_dynamic_gate.mean()),
                "next_static_gate_fraction": float(
                    np.mean(next_dynamic_static_mask)
                ),
                "static_confidence_mean": float(static_confidence.mean()),
            },
        }

    def dynamic_component_masks(
        self,
        grid: list[list[int]] | np.ndarray,
        threshold: float = 0.5,
    ) -> list[np.ndarray]:
        frame = np.asarray(grid, dtype=np.int64)
        if frame.shape != (64, 64):
            raise ValueError(f"Expected a 64x64 grid, got {frame.shape}.")

        *_, dynamic_gate_logits = self._get_net().decompose(mx.array(frame[None, :, :]))
        dynamic_gate = sigmoid_np(np.asarray(dynamic_gate_logits)[0, ..., 0])
        return find_connected_component_masks(dynamic_gate <= threshold)


class ConnectedComponentSimilarityScorer:
    def __init__(self, config_path: Path, weights_path: Path) -> None:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.config_path = config_path
        self.weights_path = weights_path
        self.num_colours = int(config.get("num_colours", 16))
        self.color_embedding_dim = int(config.get("color_embeddings_dim", 16))
        self.hidden_dim = int(config.get("hidden_dim", 64))
        self.component_embedding_dim = int(config.get("component_embedding_dim", 48))
        self.position_feature_dim = int(config.get("position_feature_dim", 0))
        self.net: ConnectedComponentConvNet | None = None
        self._load_lock = Lock()

    def _get_net(self) -> ConnectedComponentConvNet:
        if self.net is None:
            with self._load_lock:
                if self.net is None:
                    net = ConnectedComponentConvNet(
                        num_colors=self.num_colours,
                        color_embedding_dim=self.color_embedding_dim,
                        hidden_dim=self.hidden_dim,
                        component_embedding_dim=self.component_embedding_dim,
                        position_feature_dim=self.position_feature_dim,
                    )
                    net.load_weights(str(self.weights_path))
                    net.eval()
                    self.net = net
        return self.net

    def score_frame_against_references(
        self,
        layered_visualizer: LayeredVisualizer,
        grid: np.ndarray | None,
        reference_grids: list[np.ndarray],
    ) -> float | None:
        if grid is None or not reference_grids:
            return None

        candidate_embeddings = self._frame_component_embeddings(layered_visualizer, grid)
        if candidate_embeddings.shape[0] == 0:
            return None

        scores = []
        for reference_grid in reference_grids:
            reference_embeddings = self._frame_component_embeddings(
                layered_visualizer,
                reference_grid,
            )
            if reference_embeddings.shape[0] == 0:
                continue
            pairwise_scores = candidate_embeddings @ reference_embeddings.T
            scores.append(self._max_matching_score(pairwise_scores))

        return max(scores) if scores else None

    def _frame_component_embeddings(
        self,
        layered_visualizer: LayeredVisualizer,
        grid: np.ndarray,
    ) -> np.ndarray:
        frame = np.asarray(grid, dtype=np.int64)
        if frame.shape != (64, 64):
            raise ValueError(f"Expected a 64x64 grid, got {frame.shape}.")

        component_masks = layered_visualizer.dynamic_component_masks(frame)
        if not component_masks:
            return np.zeros((0, self.component_embedding_dim), dtype=np.float32)

        frames = np.repeat(frame[None, :, :], len(component_masks), axis=0)
        masks = np.stack(component_masks)
        dynamic_masks = np.repeat(np.any(masks, axis=0)[None, :, :], len(component_masks), axis=0)
        component_features = self._get_net()(mx.array(frames), mx.array(dynamic_masks))
        embeddings = self._get_net().cc_embedding_from_features(
            component_features,
            mx.array(masks),
        )
        embeddings = self._normalize_mx(embeddings)
        return np.asarray(embeddings, dtype=np.float32)

    def _normalize_mx(self, embeddings: mx.array) -> mx.array:
        norm = mx.sqrt(mx.sum(embeddings * embeddings, axis=-1, keepdims=True))
        return embeddings / mx.maximum(norm, 1e-6)

    def _max_matching_score(self, pairwise_scores: np.ndarray) -> float:
        num_candidate_components, num_reference_components = pairwise_scores.shape
        if num_candidate_components == 0 or num_reference_components == 0:
            return 0.0

        scores = pairwise_scores.astype(np.float64)
        if scores.shape[0] > scores.shape[1]:
            scores = scores.T

        assignment = self._max_weight_assignment(scores)
        total_score = sum(
            float(scores[row_idx, col_idx])
            for row_idx, col_idx in enumerate(assignment)
        )
        return total_score / max(num_candidate_components, num_reference_components)

    def _max_weight_assignment(self, scores: np.ndarray) -> np.ndarray:
        num_rows, num_cols = scores.shape
        costs = -scores
        potentials_rows = np.zeros(num_rows + 1, dtype=np.float64)
        potentials_cols = np.zeros(num_cols + 1, dtype=np.float64)
        matched_row_by_col = np.zeros(num_cols + 1, dtype=np.int64)
        previous_col = np.zeros(num_cols + 1, dtype=np.int64)

        for row in range(1, num_rows + 1):
            matched_row_by_col[0] = row
            current_col = 0
            min_cost_by_col = np.full(num_cols + 1, np.inf, dtype=np.float64)
            used_cols = np.zeros(num_cols + 1, dtype=np.bool_)

            while True:
                used_cols[current_col] = True
                current_row = matched_row_by_col[current_col]
                delta = np.inf
                next_col = 0

                for col in range(1, num_cols + 1):
                    if used_cols[col]:
                        continue
                    current_cost = (
                        costs[current_row - 1, col - 1]
                        - potentials_rows[current_row]
                        - potentials_cols[col]
                    )
                    if current_cost < min_cost_by_col[col]:
                        min_cost_by_col[col] = current_cost
                        previous_col[col] = current_col
                    if min_cost_by_col[col] < delta:
                        delta = min_cost_by_col[col]
                        next_col = col

                for col in range(num_cols + 1):
                    if used_cols[col]:
                        potentials_rows[matched_row_by_col[col]] += delta
                        potentials_cols[col] -= delta
                    else:
                        min_cost_by_col[col] -= delta

                current_col = next_col
                if matched_row_by_col[current_col] == 0:
                    break

            while True:
                next_col = previous_col[current_col]
                matched_row_by_col[current_col] = matched_row_by_col[next_col]
                current_col = next_col
                if current_col == 0:
                    break

        assignment = np.full(num_rows, -1, dtype=np.int64)
        for col in range(1, num_cols + 1):
            row = matched_row_by_col[col]
            if row != 0:
                assignment[row - 1] = col - 1

        return assignment


class GameController:
    def __init__(
        self,
        environments_dir: Path,
        recordings_dir: Path,
        operation_mode: arc_agi.OperationMode,
    ) -> None:
        self.arc = arc_agi.Arcade(
            operation_mode=operation_mode,
            environments_dir=str(environments_dir),
            recordings_dir=str(recordings_dir),
        )
        self.lock = Lock()
        self.env: arc_agi.EnvironmentWrapper | None = None
        self.current_frame: FrameDataRaw | None = None
        self.current_game_id = ""
        self.step_count = 0
        self.winning_reference_grids: list[np.ndarray] = []
        self.level_frame_history: deque[np.ndarray] = deque(
            maxlen=WINNING_REFERENCE_HISTORY_LENGTH
        )
        self.previous_grid: np.ndarray | None = None

    def games(self) -> list[dict[str, Any]]:
        games = []
        for env_info in sorted(self.arc.get_environments(), key=lambda env: env.game_id):
            games.append(
                {
                    "game_id": env_info.game_id,
                    "title": env_info.title or env_info.game_id,
                    "tags": env_info.tags or [],
                    "default_fps": env_info.default_fps,
                }
            )
        return games

    def start(self, game_id: str, seed: int = 0) -> GameSnapshot:
        with self.lock:
            env = self.arc.make(
                game_id,
                seed=seed,
                save_recording=False,
                include_frame_data=True,
                render_mode=None,
            )
            if env is None:
                raise ValueError(f"Failed to create environment: {game_id}")

            self.env = env
            self.current_frame = env.observation_space
            self.current_game_id = game_id
            self.step_count = 0
            self._reset_similarity_tracking(frame_to_grid(self.current_frame))
            return snapshot_from_frame(self.current_frame, self.step_count, game_id)

    def reset(self) -> GameSnapshot:
        with self.lock:
            if self.env is None:
                raise ValueError("Start a game first.")

            self.current_frame = self.env.reset()
            self.step_count = 0
            self._reset_similarity_tracking(frame_to_grid(self.current_frame))
            return snapshot_from_frame(
                self.current_frame,
                self.step_count,
                self.current_game_id,
            )

    def current_snapshot(self) -> GameSnapshot:
        with self.lock:
            return snapshot_from_frame(
                self.current_frame,
                self.step_count,
                self.current_game_id,
            )

    def current_grid(self) -> np.ndarray | None:
        with self.lock:
            return frame_to_grid(self.current_frame)

    def similarity_payload(
        self,
        layered_visualizer: LayeredVisualizer,
        cc_scorer: ConnectedComponentSimilarityScorer | None,
    ) -> dict[str, Any]:
        with self.lock:
            return live_similarity_payload(
                current_grid=frame_to_grid(self.current_frame),
                previous_grid=self.previous_grid,
                reference_grids=self.winning_reference_grids,
                layered_visualizer=layered_visualizer,
                cc_scorer=cc_scorer,
            )

    def step(
        self,
        action_id: int,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        with self.lock:
            if self.env is None or self.current_frame is None:
                raise ValueError("Start a game first.")

            available_actions = {
                int_action_id(action_id)
                for action_id in (self.current_frame.available_actions or [])
            }
            if action_id not in available_actions:
                raise InvalidActionError(
                    f"{action_label(action_id, data)} is not available."
                )

            action = action_from_id(action_id, data)
            action_data = data if action.is_complex() else {}
            if action.is_complex():
                if "x" not in action_data or "y" not in action_data:
                    raise InvalidActionError(
                        f"{action_label(action_id)} needs a board position."
                    )
                action_data = {
                    "x": int(action_data["x"]),
                    "y": int(action_data["y"]),
                }
                action.set_data(action_data)

            before_frame = self.current_frame
            before_grid = frame_to_grid(before_frame)
            before_snapshot = snapshot_from_frame(
                before_frame,
                self.step_count,
                self.current_game_id,
            )

            next_frame = self.env.step(
                action,
                data=action_data,
                reasoning={"source": "layered_next_state_viz_server"},
            )
            if next_frame is None:
                raise RuntimeError("Environment returned no frame.")

            self.current_frame = next_frame
            self.step_count += 1
            current_grid = frame_to_grid(self.current_frame)
            if self.current_frame.levels_completed > before_snapshot.levels_completed:
                self.winning_reference_grids.extend(
                    grid.copy() for grid in self.level_frame_history
                )
                self.level_frame_history.clear()
            if current_grid is not None:
                self.level_frame_history.append(current_grid.copy())
            self.previous_grid = before_grid.copy() if before_grid is not None else None
            return {
                "before": before_snapshot.__dict__,
                "current": snapshot_from_frame(
                    self.current_frame,
                    self.step_count,
                    self.current_game_id,
                ).__dict__,
                "before_grid": before_grid,
                "current_grid": current_grid,
                "action": {
                    "id": action_id,
                    "label": action_label(action_id, action_data),
                    "data": action_data,
                },
            }

    def _reset_similarity_tracking(self, grid: np.ndarray | None) -> None:
        self.winning_reference_grids.clear()
        self.level_frame_history.clear()
        self.previous_grid = None
        if grid is not None:
            self.level_frame_history.append(grid.copy())


class InvalidActionError(ValueError):
    pass


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Layered Next-State Visualizer</title>
    <style>
      body {
        margin: 18px;
        color: #191919;
        background: #f7f7f4;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }

      h1,
      h2 {
        margin: 0 0 10px;
        font-weight: 650;
      }

      h1 {
        font-size: 22px;
      }

      h2 {
        font-size: 14px;
      }

      label {
        display: block;
        margin-bottom: 6px;
        font-size: 12px;
        color: #444;
      }

      select,
      input,
      button {
        font: inherit;
      }

      button {
        min-height: 34px;
        padding: 7px 10px;
        border: 1px solid #bdbdb7;
        background: #fff;
        cursor: pointer;
      }

      button.is-muted {
        color: #777;
        background: #ededeb;
      }

      .layout {
        display: flex;
        align-items: flex-start;
        gap: 20px;
      }

      .controls {
        width: 320px;
        flex: 0 0 auto;
      }

      .control-group {
        margin-bottom: 16px;
      }

      .select,
      .number-input {
        width: 100%;
        box-sizing: border-box;
        padding: 6px 8px;
      }

      .mode-toggle,
      .source-toggle {
        display: grid;
        gap: 6px;
      }

      .mode-toggle label,
      .source-toggle label {
        display: flex;
        align-items: center;
        gap: 7px;
        margin: 0;
        color: #222;
      }

      .scrubber-row,
      .button-row,
      .action-grid {
        display: grid;
        gap: 8px;
      }

      .scrubber-row {
        grid-template-columns: 1fr 42px;
        align-items: center;
      }

      .button-row {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }

      .action-grid {
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }

      .meta,
      .status-text {
        margin-top: 6px;
        font-size: 12px;
        color: #555;
      }

      .preview {
        min-width: 0;
        flex: 1 1 auto;
      }

      .score-strip {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 12px;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid #d9d9d4;
      }

      .score-label {
        margin-bottom: 4px;
        color: #666;
        font-size: 11px;
        letter-spacing: 0;
        text-transform: uppercase;
      }

	      .score-value {
	        min-height: 18px;
	        overflow-wrap: anywhere;
	        font-size: 13px;
	      }

	      .similarity-panel {
	        margin: 0 0 18px;
	        padding: 12px 0 14px;
	        border-bottom: 1px solid #d9d9d4;
	      }

	      .similarity-header {
	        display: grid;
	        grid-template-columns: repeat(3, minmax(0, 1fr));
	        gap: 12px;
	        margin-bottom: 10px;
	      }

	      .similarity-value {
	        min-height: 20px;
	        font-size: 16px;
	        font-variant-numeric: tabular-nums;
	      }

	      .similarity-value.is-up {
	        color: #167a4a;
	      }

	      .similarity-value.is-down {
	        color: #aa2f2f;
	      }

	      .similarity-plot {
	        display: block;
	        width: 100%;
	        height: 130px;
	        border: 1px solid #d5d5d0;
	        background: #fff;
	      }

      .preview-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(190px, 1fr));
        gap: 18px;
        align-items: start;
      }

      .preview-panel {
        min-width: 0;
      }

      img {
        display: block;
        width: 100%;
        height: auto;
        border: 1px solid #d5d5d0;
        background: #fff;
        image-rendering: pixelated;
      }

      #input-image.is-clickable-board {
        cursor: crosshair;
      }

      .gate-image {
        cursor: crosshair;
      }

      .gate-tooltip {
        position: fixed;
        z-index: 10;
        min-width: 118px;
        padding: 6px 8px;
        color: #fff;
        background: rgba(32, 32, 32, 0.94);
        border: 1px solid rgba(255, 255, 255, 0.14);
        font-size: 12px;
        font-variant-numeric: tabular-nums;
        line-height: 1.35;
        pointer-events: none;
        transform: translate(12px, 12px);
      }

      .metrics {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 10px;
        margin: 18px 0 0;
        padding-top: 12px;
        border-top: 1px solid #d9d9d4;
      }

      .metric {
        min-width: 0;
      }

      .metric-label {
        margin-bottom: 3px;
        color: #666;
        font-size: 11px;
      }

      .metric-value {
        font-size: 13px;
        font-variant-numeric: tabular-nums;
      }

      .toast {
        position: fixed;
        left: 50%;
        bottom: 24px;
        transform: translateX(-50%);
        max-width: min(560px, calc(100vw - 32px));
        padding: 10px 14px;
        color: #fff;
        background: #242424;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
        opacity: 0;
        pointer-events: none;
        transition: opacity 140ms ease;
      }

      .toast.is-visible {
        opacity: 1;
      }

      [hidden] {
        display: none !important;
      }

      @media (max-width: 1180px) {
        .preview-grid {
          grid-template-columns: repeat(2, minmax(220px, 1fr));
        }

        .metrics {
          grid-template-columns: repeat(3, minmax(0, 1fr));
        }
      }

      @media (max-width: 860px) {
        body {
          margin: 14px;
        }

        .layout {
          flex-direction: column;
        }

        .controls {
          width: 100%;
        }

	        .score-strip,
	        .similarity-header,
	        .preview-grid,
	        .metrics {
	          grid-template-columns: 1fr;
        }
      }
    </style>
  </head>
  <body>
    <div class="layout">
      <section class="controls">
        <h1>Layered Model</h1>

	        <div class="control-group">
	          <label for="model-select">Model</label>
	          <select id="model-select" class="select"></select>
	          <div id="model-status" class="status-text"></div>
	        </div>

	        <div class="control-group">
	          <label for="cc-model-select">CC Embedding Model</label>
	          <select id="cc-model-select" class="select"></select>
	          <div id="cc-model-status" class="status-text"></div>
	        </div>

        <div class="control-group source-toggle">
          <label><input type="radio" name="source-mode" value="recording" checked> Recording</label>
          <label><input type="radio" name="source-mode" value="live"> Live game</label>
        </div>

        <div class="control-group mode-toggle">
          <label><input type="radio" name="decode-mode" value="argmax" checked> Argmax</label>
          <label><input type="radio" name="decode-mode" value="sample"> Sample</label>
        </div>

        <div id="recording-controls">
          <div class="control-group">
            <label for="recording-select">Recording</label>
            <select id="recording-select" class="select"></select>
            <div id="recording-status" class="status-text"></div>
          </div>

          <div class="control-group">
            <label for="frame-scrubber">Frame</label>
            <div class="scrubber-row">
              <input id="frame-scrubber" type="range" min="0" max="0" step="1" value="0">
              <span id="scrubber-value" class="meta">0</span>
            </div>
            <div id="scrubber-meta" class="meta">No recording selected.</div>
          </div>

          <div class="control-group button-row">
            <button id="play-button">Play</button>
            <button id="reload-recording-button">Reload</button>
          </div>
        </div>

        <div id="live-controls" hidden>
          <div class="control-group">
            <label for="game-select">Game</label>
            <select id="game-select" class="select"></select>
          </div>

          <div class="control-group">
            <label for="seed-input">Seed</label>
            <input id="seed-input" class="number-input" type="number" value="0">
          </div>

          <div class="control-group button-row">
            <button id="start-button">Start</button>
            <button id="reset-button">Reset</button>
          </div>

          <div class="control-group action-grid" id="action-buttons"></div>
        </div>
      </section>

      <section class="preview">
        <div class="score-strip">
          <div>
            <div class="score-label">Source</div>
            <div id="source-value" class="score-value">-</div>
          </div>
          <div>
            <div class="score-label">State</div>
            <div id="state-value" class="score-value">-</div>
          </div>
          <div>
            <div class="score-label">Level</div>
            <div id="level-value" class="score-value">-</div>
          </div>
          <div>
            <div class="score-label">Step</div>
            <div id="step-value" class="score-value">-</div>
          </div>
          <div>
            <div class="score-label">Action</div>
            <div id="action-value" class="score-value">-</div>
          </div>
	        </div>
	
	        <div class="similarity-panel">
	          <h2>Learned CC Similarity To Previous Winning Frames</h2>
	          <div class="similarity-header">
	            <div>
	              <div class="score-label">Current</div>
	              <div id="similarity-value" class="similarity-value">-</div>
	            </div>
	            <div>
	              <div class="score-label">Change</div>
	              <div id="similarity-delta-value" class="similarity-value">-</div>
	            </div>
	            <div>
	              <div class="score-label">References</div>
	              <div id="similarity-reference-value" class="similarity-value">-</div>
	            </div>
	          </div>
	          <canvas id="similarity-plot" class="similarity-plot"></canvas>
	        </div>
	
	        <div class="preview-grid">
          <div class="preview-panel">
            <h2>Input Frame</h2>
            <img id="input-image" alt="Input frame">
          </div>
          <div class="preview-panel">
            <h2>Static Layer</h2>
            <img id="static-image" alt="Static layer">
          </div>
          <div class="preview-panel">
            <h2>Dynamic Layer</h2>
            <img id="dynamic-image" alt="Dynamic layer">
          </div>
          <div class="preview-panel">
            <h2>Dynamic Masked</h2>
            <img id="dynamic-masked-image" alt="Dynamic layer with static-gated cells masked">
          </div>
          <div class="preview-panel">
            <h2>Dynamic Gate</h2>
            <img id="dynamic-gate-image" class="gate-image" alt="Dynamic sigmoid gate">
          </div>
          <div class="preview-panel">
            <h2>Dynamic Gate CC</h2>
            <img id="dynamic-gate-cc-image" class="gate-image" alt="Dynamic gate connected components">
          </div>
          <div class="preview-panel">
            <h2>Reconstruction</h2>
            <img id="reconstruction-image" alt="Layered reconstruction">
          </div>
          <div class="preview-panel">
            <h2>Predicted Next</h2>
            <img id="next-image" alt="Predicted next frame">
          </div>
          <div class="preview-panel">
            <h2>Actual Next</h2>
            <img id="actual-next-image" alt="Actual next frame">
          </div>
          <div class="preview-panel">
            <h2>Next Dynamic</h2>
            <img id="next-dynamic-image" alt="Predicted next dynamic layer">
          </div>
          <div class="preview-panel">
            <h2>Next Dynamic Masked</h2>
            <img id="next-dynamic-masked-image" alt="Predicted next dynamic layer with static-gated cells masked">
          </div>
          <div class="preview-panel">
            <h2>Next Gate</h2>
            <img id="next-gate-image" class="gate-image" alt="Predicted next sigmoid gate">
          </div>
        </div>

        <div class="metrics">
          <div class="metric">
            <div class="metric-label">Static conf</div>
            <div id="static-confidence-value" class="metric-value">-</div>
          </div>
          <div class="metric">
            <div class="metric-label">Dyn gate</div>
            <div id="dynamic-gate-value" class="metric-value">-</div>
          </div>
          <div class="metric">
            <div class="metric-label">Dyn static %</div>
            <div id="dynamic-static-frac-value" class="metric-value">-</div>
          </div>
          <div class="metric">
            <div class="metric-label">Next gate</div>
            <div id="next-gate-value" class="metric-value">-</div>
          </div>
          <div class="metric">
            <div class="metric-label">Next static %</div>
            <div id="next-static-frac-value" class="metric-value">-</div>
          </div>
        </div>
      </section>
    </div>

    <div id="toast" class="toast"></div>
    <div id="gate-tooltip" class="gate-tooltip" hidden></div>

    <script>
	      const games = {{ games|tojson }};
	      const models = {{ models|tojson }};
	      const ccModels = {{ cc_models|tojson }};
	      const recordings = {{ recordings|tojson }};
	      const initialGameId = {{ selected_game_id|tojson }};
	      const initialModelKey = {{ selected_model_key|tojson }};
	      const initialCcModelKey = {{ selected_cc_model_key|tojson }};
	
	      const modelSelect = document.getElementById("model-select");
	      const ccModelSelect = document.getElementById("cc-model-select");
	      const ccModelStatus = document.getElementById("cc-model-status");
	      const modelStatus = document.getElementById("model-status");
      const recordingSelect = document.getElementById("recording-select");
      const recordingStatus = document.getElementById("recording-status");
      const frameScrubber = document.getElementById("frame-scrubber");
      const scrubberValue = document.getElementById("scrubber-value");
      const scrubberMeta = document.getElementById("scrubber-meta");
      const playButton = document.getElementById("play-button");
      const reloadRecordingButton = document.getElementById("reload-recording-button");
      const recordingControls = document.getElementById("recording-controls");
      const liveControls = document.getElementById("live-controls");
      const gameSelect = document.getElementById("game-select");
      const seedInput = document.getElementById("seed-input");
      const startButton = document.getElementById("start-button");
      const resetButton = document.getElementById("reset-button");
      const actionButtons = document.getElementById("action-buttons");
      const sourceValue = document.getElementById("source-value");
      const stateValue = document.getElementById("state-value");
	      const levelValue = document.getElementById("level-value");
	      const stepValue = document.getElementById("step-value");
	      const actionValue = document.getElementById("action-value");
	      const similarityValue = document.getElementById("similarity-value");
	      const similarityDeltaValue = document.getElementById("similarity-delta-value");
	      const similarityReferenceValue = document.getElementById("similarity-reference-value");
	      const similarityPlot = document.getElementById("similarity-plot");
	      const inputImage = document.getElementById("input-image");
      const staticImage = document.getElementById("static-image");
      const dynamicImage = document.getElementById("dynamic-image");
      const dynamicMaskedImage = document.getElementById("dynamic-masked-image");
      const dynamicGateImage = document.getElementById("dynamic-gate-image");
      const dynamicGateCcImage = document.getElementById("dynamic-gate-cc-image");
      const reconstructionImage = document.getElementById("reconstruction-image");
      const nextImage = document.getElementById("next-image");
      const actualNextImage = document.getElementById("actual-next-image");
      const nextDynamicImage = document.getElementById("next-dynamic-image");
      const nextDynamicMaskedImage = document.getElementById("next-dynamic-masked-image");
      const nextGateImage = document.getElementById("next-gate-image");
      const staticConfidenceValue = document.getElementById("static-confidence-value");
      const dynamicGateValue = document.getElementById("dynamic-gate-value");
      const dynamicStaticFracValue = document.getElementById("dynamic-static-frac-value");
      const nextGateValue = document.getElementById("next-gate-value");
      const nextStaticFracValue = document.getElementById("next-static-frac-value");
      const toast = document.getElementById("toast");
      const gateTooltip = document.getElementById("gate-tooltip");
      const modeInputs = [...document.querySelectorAll('input[name="decode-mode"]')];
      const sourceInputs = [...document.querySelectorAll('input[name="source-mode"]')];

      const actionDefinitions = [
        {id: 1, label: "A1", keys: ["ArrowUp", "w", "W"]},
        {id: 2, label: "A2", keys: ["ArrowDown", "s", "S"]},
        {id: 3, label: "A3", keys: ["ArrowLeft", "a", "A"]},
        {id: 4, label: "A4", keys: ["ArrowRight", "d", "D"]},
        {id: 5, label: "A5", keys: [" "]},
        {id: 6, label: "A6", keys: []},
        {id: 7, label: "A7", keys: ["Enter"]},
      ];

      let currentFrames = [];
      let currentSnapshot = null;
      let recordingRequestId = 0;
	      let renderRequestId = 0;
	      let playTimer = null;
	      let toastTimer = null;
	      let currentGateValues = {};
	      const similarityHistoryLimit = 80;
	      let similarityHistory = [];
	      let similarityHistoryKey = "";

      function getDecodeMode() {
        const selected = modeInputs.find((input) => input.checked);
        return selected ? selected.value : "argmax";
      }

      function getSourceMode() {
        const selected = sourceInputs.find((input) => input.checked);
        return selected ? selected.value : "recording";
      }

	      function selectedModel() {
	        return models.find((model) => model.key === modelSelect.value);
	      }

	      function selectedCcModel() {
	        return ccModels.find((model) => model.key === ccModelSelect.value);
	      }

      function showToast(message) {
        toast.textContent = message;
        toast.classList.add("is-visible");
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => {
          toast.classList.remove("is-visible");
        }, 2200);
      }

      function setImage(image, src) {
        if (src) {
          image.src = src;
        } else {
          image.removeAttribute("src");
        }
      }

      function formatNumber(value) {
        return Number.isFinite(value) ? value.toFixed(4) : "-";
      }

	      function formatPercent(value) {
	        return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "-";
	      }

	      function formatSimilarity(value) {
	        return Number.isFinite(value) ? value.toFixed(4) : "-";
	      }

	      function setTrendClass(element, value) {
	        element.classList.toggle("is-up", Number.isFinite(value) && value > 0);
	        element.classList.toggle("is-down", Number.isFinite(value) && value < 0);
	      }

		      function currentSimilarityHistoryKey() {
		        const sourceMode = getSourceMode();
		        const sourceKey = sourceMode === "recording"
		          ? recordingSelect.value
		          : `${gameSelect.value}:${seedInput.value || 0}`;
		        return [
		          sourceMode,
		          sourceKey,
		          modelSelect.value,
		          ccModelSelect.value,
		          getDecodeMode(),
		        ].join("|");
		      }

		      function resetSimilarityHistory() {
		        similarityHistory = [];
		        similarityHistoryKey = currentSimilarityHistoryKey();
		      }

		      function previousSimilarityPoint(step) {
		        return similarityHistory
		          .filter((point) => point.step < step)
		          .sort((a, b) => b.step - a.step)[0] || null;
		      }

		      function rememberSimilarityPoint(step, score, referenceCount) {
		        const key = currentSimilarityHistoryKey();
		        if (key !== similarityHistoryKey) {
		          similarityHistory = [];
		          similarityHistoryKey = key;
		        }
		        if (!Number.isFinite(step) || !Number.isFinite(score)) {
		          return null;
		        }

		        const previousPoint = previousSimilarityPoint(step);
		        const existingIndex = similarityHistory.findIndex((point) => point.step === step);
		        const point = {step, score, referenceCount};
		        if (existingIndex >= 0) {
		          similarityHistory[existingIndex] = point;
		        } else {
		          similarityHistory.push(point);
		        }
		        while (similarityHistory.length > similarityHistoryLimit) {
		          similarityHistory.shift();
		        }
		        return previousPoint;
		      }

		      function renderSimilarity(similarity, step) {
		        if (!similarity) {
		          resetSimilarityHistory();
		          similarityValue.textContent = "-";
		          similarityDeltaValue.textContent = "-";
		          similarityReferenceValue.textContent = "0";
		          setTrendClass(similarityValue, NaN);
		          setTrendClass(similarityDeltaValue, NaN);
		          drawSimilarityPlot(0, NaN, 0);
		          return;
		        }
		        const score = Number(similarity.score);
		        const delta = Number(similarity.delta);
		        const referenceCount = Number(similarity.reference_count || 0);
		        const pointStep = Number.isFinite(Number(step)) ? Number(step) : similarityHistory.length;
		        const previousPoint = rememberSimilarityPoint(pointStep, score, referenceCount);
		        const displayDelta = Number.isFinite(delta)
		          ? delta
		          : (previousPoint ? score - previousPoint.score : NaN);

		        similarityValue.textContent = formatSimilarity(score);
		        similarityDeltaValue.textContent = Number.isFinite(displayDelta)
		          ? `${displayDelta >= 0 ? "+" : ""}${formatSimilarity(displayDelta)}`
		          : "-";
		        similarityReferenceValue.textContent = referenceCount ? String(referenceCount) : "0";
		        setTrendClass(similarityValue, displayDelta);
		        setTrendClass(similarityDeltaValue, displayDelta);
		        drawSimilarityPlot(pointStep, displayDelta, referenceCount);
		      }

		      function drawSimilarityPlot(currentStep, delta, referenceCount) {
		        const rect = similarityPlot.getBoundingClientRect();
		        const dpr = window.devicePixelRatio || 1;
		        const width = Math.max(240, Math.floor(rect.width || 640));
	        const height = Math.max(110, Math.floor(rect.height || 130));
	        similarityPlot.width = Math.floor(width * dpr);
	        similarityPlot.height = Math.floor(height * dpr);
	        const ctx = similarityPlot.getContext("2d");
	        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
	        ctx.clearRect(0, 0, width, height);
	        ctx.fillStyle = "#ffffff";
	        ctx.fillRect(0, 0, width, height);

	        const padding = {left: 34, right: 12, top: 12, bottom: 24};
	        const plotWidth = width - padding.left - padding.right;
	        const plotHeight = height - padding.top - padding.bottom;
	        ctx.strokeStyle = "#e1e1dc";
	        ctx.lineWidth = 1;
	        for (const yValue of [-1, 0, 1]) {
	          const y = padding.top + ((1 - yValue) / 2) * plotHeight;
	          ctx.beginPath();
	          ctx.moveTo(padding.left, y);
	          ctx.lineTo(width - padding.right, y);
	          ctx.stroke();
	        }

	        ctx.fillStyle = "#777";
	        ctx.font = "11px system-ui, sans-serif";
	        ctx.fillText("1.0", 5, padding.top + 4);
	        ctx.fillText("0.0", 5, padding.top + plotHeight * 0.5 + 4);
	        ctx.fillText("-1.0", 5, padding.top + plotHeight + 4);

		        const points = similarityHistory
		          .filter((point) => Number.isFinite(point.score))
		          .sort((a, b) => a.step - b.step);
		        if (!points.length) {
		          ctx.fillStyle = "#777";
		          ctx.fillText("No previous winning-frame references yet.", padding.left, padding.top + 22);
		          return;
		        }

		        const toY = (score) => padding.top + ((1 - Math.max(-1, Math.min(1, score))) / 2) * plotHeight;
		        const minStep = points[0].step;
		        const maxStep = points[points.length - 1].step;
		        const stepSpan = Math.max(1, maxStep - minStep);
		        const toX = (step) => padding.left + ((step - minStep) / stepSpan) * plotWidth;

		        ctx.strokeStyle = "#56616f";
		        ctx.lineWidth = 2;
		        ctx.beginPath();
		        points.forEach((point, index) => {
		          const x = toX(point.step);
		          const y = toY(point.score);
		          if (index === 0) {
		            ctx.moveTo(x, y);
		          } else {
		            ctx.lineTo(x, y);
		          }
		        });
		        ctx.stroke();

		        if (points.length > 1) {
		          const currentIndex = points.findIndex((point) => point.step === currentStep);
		          const index = currentIndex >= 0 ? currentIndex : points.length - 1;
		          const currentPoint = points[index];
		          const previousPoint = points[index - 1] || null;
		          const segmentDelta = previousPoint ? currentPoint.score - previousPoint.score : delta;
		          if (previousPoint) {
		            ctx.strokeStyle = segmentDelta >= 0 ? "#1f9d62" : "#c64646";
		            ctx.lineWidth = 3;
		            ctx.beginPath();
		            ctx.moveTo(toX(previousPoint.step), toY(previousPoint.score));
		            ctx.lineTo(toX(currentPoint.step), toY(currentPoint.score));
		            ctx.stroke();
		          }
		        }

		        for (const point of points) {
		          ctx.fillStyle = point.step === currentStep ? "#111" : "#9aa0a6";
		          ctx.beginPath();
		          ctx.arc(toX(point.step), toY(point.score), point.step === currentStep ? 4 : 2.5, 0, Math.PI * 2);
		          ctx.fill();
		        }

		        const currentPoint = points.find((point) => point.step === currentStep) || points[points.length - 1];
		        if (currentPoint) {
		          ctx.fillStyle = Number.isFinite(delta) && delta < 0 ? "#c64646" : "#1f9d62";
		          ctx.beginPath();
		          ctx.arc(toX(currentPoint.step), toY(currentPoint.score), 5, 0, Math.PI * 2);
		          ctx.fill();
		        }

		        ctx.fillStyle = "#555";
		        ctx.fillText(`${referenceCount} refs`, padding.left, height - 7);
		        if (points.length > 1) {
		          ctx.fillText(`${minStep}..${maxStep}`, width - padding.right - 58, height - 7);
		        }
		      }
	
	      function renderMetrics(metrics) {
        metrics = metrics || {};
        staticConfidenceValue.textContent = formatNumber(metrics.static_confidence_mean);
        dynamicGateValue.textContent = formatNumber(metrics.dynamic_gate_mean);
        dynamicStaticFracValue.textContent = formatPercent(metrics.dynamic_static_gate_fraction);
        nextGateValue.textContent = formatNumber(metrics.next_gate_mean);
        nextStaticFracValue.textContent = formatPercent(metrics.next_static_gate_fraction);
      }

      function setGateValues(values) {
        currentGateValues = values || {};
        hideGateTooltip();
      }

      function hideGateTooltip() {
        gateTooltip.hidden = true;
      }

      function gateHoverPosition(event, image, values) {
        const height = values.length;
        const width = values[0]?.length || 0;
        if (!height || !width) {
          return null;
        }

        const rect = image.getBoundingClientRect();
        const relX = (event.clientX - rect.left) / rect.width;
        const relY = (event.clientY - rect.top) / rect.height;
        if (relX < 0 || relX >= 1 || relY < 0 || relY >= 1) {
          return null;
        }

        return {
          x: Math.max(0, Math.min(width - 1, Math.floor(relX * width))),
          y: Math.max(0, Math.min(height - 1, Math.floor(relY * height))),
        };
      }

      function showGateTooltip(event, image, key, label) {
        const values = currentGateValues[key];
        if (!values) {
          hideGateTooltip();
          return;
        }

        const position = gateHoverPosition(event, image, values);
        if (!position) {
          hideGateTooltip();
          return;
        }

        const value = values[position.y]?.[position.x];
        if (!Number.isFinite(value)) {
          hideGateTooltip();
          return;
        }

        gateTooltip.innerHTML = `${label}<br>x ${position.x}, y ${position.y}<br>value ${value.toFixed(6)}`;
        gateTooltip.style.left = `${event.clientX}px`;
        gateTooltip.style.top = `${event.clientY}px`;
        gateTooltip.hidden = false;
      }

      function renderLayerImages(layers) {
        if (!layers) {
          for (const image of [
            inputImage,
            staticImage,
            dynamicImage,
            dynamicMaskedImage,
            dynamicGateImage,
            dynamicGateCcImage,
            reconstructionImage,
            nextImage,
            nextDynamicImage,
            nextDynamicMaskedImage,
            nextGateImage,
          ]) {
            setImage(image, null);
          }
          renderMetrics(null);
          setGateValues(null);
          return;
        }
        setImage(inputImage, layers.input);
        setImage(staticImage, layers.static);
        setImage(dynamicImage, layers.dynamic);
        setImage(dynamicMaskedImage, layers.dynamic_masked);
        setImage(dynamicGateImage, layers.dynamic_gate);
        setImage(dynamicGateCcImage, layers.dynamic_gate_cc);
        setImage(reconstructionImage, layers.reconstruction);
        setImage(nextImage, layers.next_reconstruction);
        setImage(nextDynamicImage, layers.next_dynamic);
        setImage(nextDynamicMaskedImage, layers.next_dynamic_masked);
        setImage(nextGateImage, layers.next_dynamic_gate);
        renderMetrics(layers.metrics);
        setGateValues(layers.gate_values);
      }

      function renderMeta(payload) {
        sourceValue.textContent = payload.source || "-";
        stateValue.textContent = payload.state || "-";
        levelValue.textContent = payload.win_levels === undefined
          ? "-"
          : `${payload.levels_completed}/${payload.win_levels}`;
        stepValue.textContent = payload.step === undefined ? "-" : String(payload.step);
        actionValue.textContent = payload.action?.label || "-";
      }

	      function updateModelStatus() {
	        const model = selectedModel();
	        if (!model) {
	          modelStatus.textContent = models.length ? "No model selected." : "No layered models found.";
	          return;
	        }
	        modelStatus.textContent = model.label;
	      }

	      function updateCcModelStatus() {
	        const model = selectedCcModel();
	        if (!model) {
	          ccModelStatus.textContent = ccModels.length ? "No CC model selected." : "No CC embedding models found.";
	          return;
	        }
	        ccModelStatus.textContent = model.label;
	      }

	      function buildModelOptions() {
        if (!models.length) {
          const option = document.createElement("option");
          option.value = "";
          option.textContent = "No models found";
          modelSelect.appendChild(option);
          updateModelStatus();
          return;
        }

        for (const model of models) {
          const option = document.createElement("option");
          option.value = model.key;
          option.textContent = model.label;
          modelSelect.appendChild(option);
        }
        modelSelect.value = initialModelKey || models[0].key;
	        updateModelStatus();
	      }

	      function buildCcModelOptions() {
	        if (!ccModels.length) {
	          const option = document.createElement("option");
	          option.value = "";
	          option.textContent = "No CC models found";
	          ccModelSelect.appendChild(option);
	          updateCcModelStatus();
	          return;
	        }

	        for (const model of ccModels) {
	          const option = document.createElement("option");
	          option.value = model.key;
	          option.textContent = model.label;
	          ccModelSelect.appendChild(option);
	        }
	        ccModelSelect.value = initialCcModelKey || ccModels[0].key;
	        updateCcModelStatus();
	      }

      function buildRecordingOptions() {
        const emptyOption = document.createElement("option");
        emptyOption.value = "";
        emptyOption.textContent = "None";
        recordingSelect.appendChild(emptyOption);

        for (const recording of recordings) {
          const option = document.createElement("option");
          option.value = recording.key;
          option.textContent = recording.label;
          recordingSelect.appendChild(option);
        }
        if (recordings.length) {
          recordingSelect.value = recordings[0].key;
        }
      }

      function buildGameOptions() {
        for (const game of games) {
          const option = document.createElement("option");
          option.value = game.game_id;
          option.textContent = `${game.title || game.game_id} (${game.game_id})`;
          gameSelect.appendChild(option);
        }
        gameSelect.value = initialGameId || games[0]?.game_id || "";
      }

      function buildActionButtons() {
        for (const action of actionDefinitions) {
          const button = document.createElement("button");
          button.type = "button";
          button.textContent = action.label;
          button.dataset.actionId = String(action.id);
          button.addEventListener("click", () => sendAction(action.id, {}));
          actionButtons.appendChild(button);
        }
      }

      function isActionAvailable(actionId) {
        return Boolean(currentSnapshot?.available_actions?.includes(actionId));
      }

      function updateActionButtons() {
        for (const button of actionButtons.querySelectorAll("button")) {
          const actionId = Number(button.dataset.actionId);
          button.classList.toggle("is-muted", !isActionAvailable(actionId));
        }
        inputImage.classList.toggle(
          "is-clickable-board",
          getSourceMode() === "live" && isActionAvailable(6),
        );
      }

	      function setSourceMode() {
	        const mode = getSourceMode();
	        recordingControls.hidden = mode !== "recording";
	        liveControls.hidden = mode !== "live";
	        resetSimilarityHistory();
	        if (mode === "recording") {
	          renderSelectedRecordingFrame();
	        } else {
	          stopPlayback();
	          renderSimilarity(null);
	        }
	        updateActionButtons();
	      }

      async function getJson(url) {
        const response = await fetch(url);
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Request failed.");
        }
        return data;
      }

      async function postJson(url, payload) {
        const response = await fetch(url, {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Request failed.");
        }
        return data;
      }

      async function loadRecordingFrames() {
        const requestId = recordingRequestId + 1;
        recordingRequestId = requestId;
        currentFrames = [];
        frameScrubber.min = "0";
        frameScrubber.max = "0";
        frameScrubber.value = "0";
        scrubberValue.textContent = "0";
        scrubberMeta.textContent = "No recording selected.";
        recordingStatus.textContent = "";
        stopPlayback();

	        if (!recordingSelect.value) {
	          renderLayerImages(null);
		          renderSimilarity(null);
		          setImage(actualNextImage, null);
		          return;
		        }

        recordingStatus.textContent = "Loading...";
        try {
          const payload = await getJson(`/api/recordings/${encodeURIComponent(recordingSelect.value)}`);
          if (requestId !== recordingRequestId) {
            return;
          }
	          currentFrames = payload.frames;
	          resetSimilarityHistory();
	          frameScrubber.max = String(Math.max(0, currentFrames.length - 1));
          updateScrubberMeta();
          recordingStatus.textContent = `Loaded ${currentFrames.length} frame${currentFrames.length === 1 ? "" : "s"}.`;
          renderSelectedRecordingFrame();
        } catch (error) {
          if (requestId !== recordingRequestId) {
            return;
          }
          recordingStatus.textContent = error instanceof Error ? error.message : "Failed to load recording.";
        }
      }

      function updateScrubberMeta() {
        scrubberValue.textContent = String(Number(frameScrubber.value) + 1);
        if (!currentFrames.length) {
          scrubberMeta.textContent = "No recording selected.";
          return;
        }
        const frame = currentFrames[Number(frameScrubber.value)];
        scrubberMeta.textContent = `Frame ${frame.index + 1}/${currentFrames.length}  ${frame.timestamp}`;
      }

      async function renderSelectedRecordingFrame() {
        if (getSourceMode() !== "recording" || !recordingSelect.value || !modelSelect.value) {
          return;
        }

        const requestId = renderRequestId + 1;
        renderRequestId = requestId;
        updateScrubberMeta();
        const params = new URLSearchParams();
	        params.set("model", modelSelect.value);
	        params.set("cc_model", ccModelSelect.value);
	        params.set("recording", recordingSelect.value);
        params.set("frame_index", frameScrubber.value);
        params.set("mode", getDecodeMode());

        try {
          const payload = await getJson(`/api/recording/render?${params.toString()}`);
          if (requestId !== renderRequestId) {
            return;
          }
	          renderMeta(payload);
	          renderLayerImages(payload.layers);
		          renderSimilarity(payload.similarity, payload.step);
	          setImage(actualNextImage, payload.actual_next_image);
        } catch (error) {
          showToast(error instanceof Error ? error.message : "Failed to render frame.");
        }
      }

      function stopPlayback() {
        if (playTimer) {
          clearInterval(playTimer);
          playTimer = null;
        }
        playButton.textContent = "Play";
      }

      function togglePlayback() {
        if (playTimer) {
          stopPlayback();
          return;
        }
        if (!currentFrames.length) {
          showToast("Load a recording first.");
          return;
        }
        playButton.textContent = "Pause";
        playTimer = setInterval(() => {
          const nextValue = Number(frameScrubber.value) + 1;
          if (nextValue > Number(frameScrubber.max)) {
            stopPlayback();
            return;
          }
          frameScrubber.value = String(nextValue);
          renderSelectedRecordingFrame();
        }, 350);
      }

      async function startGame() {
        try {
	          const payload = await postJson("/api/live/start", {
	            game_id: gameSelect.value,
	            seed: Number(seedInput.value || 0),
	            model: modelSelect.value,
	            cc_model: ccModelSelect.value,
	            mode: getDecodeMode(),
	          });
		          currentSnapshot = payload.current;
		          resetSimilarityHistory();
		          renderMeta(payload);
		          renderLayerImages(payload.layers);
		          renderSimilarity(payload.similarity, payload.step);
	          setImage(actualNextImage, null);
          updateActionButtons();
        } catch (error) {
          showToast(error instanceof Error ? error.message : "Failed to start game.");
        }
      }

      async function resetGame() {
        try {
	          const payload = await postJson("/api/live/reset", {
	            model: modelSelect.value,
	            cc_model: ccModelSelect.value,
	            mode: getDecodeMode(),
	          });
		          currentSnapshot = payload.current;
		          resetSimilarityHistory();
		          renderMeta(payload);
		          renderLayerImages(payload.layers);
		          renderSimilarity(payload.similarity, payload.step);
	          setImage(actualNextImage, null);
          updateActionButtons();
        } catch (error) {
          showToast(error instanceof Error ? error.message : "Failed to reset game.");
        }
      }

      async function renderLiveCurrent() {
        if (!currentSnapshot) {
          return;
        }

        try {
	          const payload = await postJson("/api/live/current_render", {
	            model: modelSelect.value,
	            cc_model: ccModelSelect.value,
	            mode: getDecodeMode(),
	          });
	          currentSnapshot = payload.current;
	          renderMeta(payload);
	          renderLayerImages(payload.layers);
		          renderSimilarity(payload.similarity, payload.step);
	          setImage(actualNextImage, null);
          updateActionButtons();
        } catch (error) {
          showToast(error instanceof Error ? error.message : "Failed to render current frame.");
        }
      }

      async function sendAction(actionId, data) {
        if (!currentSnapshot) {
          showToast("Start a game first.");
          return;
        }
        if (!isActionAvailable(actionId)) {
          showToast(`ACTION${actionId} is not available.`);
          return;
        }

        try {
          const payload = await postJson("/api/live/action", {
	            action_id: actionId,
	            data,
	            model: modelSelect.value,
	            cc_model: ccModelSelect.value,
	            mode: getDecodeMode(),
	          });
	          currentSnapshot = payload.current;
	          renderMeta(payload);
	          renderLayerImages(payload.current_layers);
		          renderSimilarity(payload.similarity, payload.step);
	          setImage(nextImage, payload.predicted_next_image);
          setImage(nextDynamicImage, payload.predicted_next_dynamic_image);
          setImage(nextDynamicMaskedImage, payload.predicted_next_dynamic_masked_image);
          setImage(actualNextImage, payload.current.image);
          updateActionButtons();
          if (payload.current.state === "WIN") {
            showToast("Game won.");
          } else if (payload.current.state === "GAME_OVER") {
            showToast("Game over.");
          }
        } catch (error) {
          showToast(error instanceof Error ? error.message : "Action failed.");
        }
      }

      function handleInputBoardClick(event) {
        if (getSourceMode() !== "live" || !isActionAvailable(6)) {
          return;
        }
        if (!currentSnapshot) {
          showToast("Start a game first.");
          return;
        }
        const rect = inputImage.getBoundingClientRect();
        const x = Math.max(0, Math.min(63, Math.floor(((event.clientX - rect.left) / rect.width) * 64)));
        const y = Math.max(0, Math.min(63, Math.floor(((event.clientY - rect.top) / rect.height) * 64)));
        sendAction(6, {x, y});
      }

      function handleKeyboard(event) {
        if (getSourceMode() !== "live") {
          return;
        }
        if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) {
          return;
        }
        if (event.key === "r" || event.key === "R") {
          event.preventDefault();
          resetGame();
          return;
        }

        const action = actionDefinitions.find((definition) => definition.keys.includes(event.key));
        if (!action) {
          return;
        }
        event.preventDefault();
        sendAction(action.id, {});
      }

	      modelSelect.addEventListener("change", () => {
	        updateModelStatus();
	        if (getSourceMode() === "recording") {
	          renderSelectedRecordingFrame();
	        } else if (currentSnapshot) {
	          renderLiveCurrent();
	        }
	      });
	      ccModelSelect.addEventListener("change", () => {
	        updateCcModelStatus();
	        if (getSourceMode() === "recording") {
	          renderSelectedRecordingFrame();
	        } else if (currentSnapshot) {
	          renderLiveCurrent();
	        }
	      });
      for (const input of modeInputs) {
        input.addEventListener("change", () => {
          if (getSourceMode() === "recording") {
            renderSelectedRecordingFrame();
          } else if (currentSnapshot) {
            renderLiveCurrent();
          }
        });
      }
      for (const input of sourceInputs) {
        input.addEventListener("change", setSourceMode);
      }
      recordingSelect.addEventListener("change", loadRecordingFrames);
      reloadRecordingButton.addEventListener("click", loadRecordingFrames);
      frameScrubber.addEventListener("input", renderSelectedRecordingFrame);
      playButton.addEventListener("click", togglePlayback);
      startButton.addEventListener("click", startGame);
      resetButton.addEventListener("click", resetGame);
	      gameSelect.addEventListener("change", () => {
	        currentSnapshot = null;
	        resetSimilarityHistory();
	        renderSimilarity(null);
	        updateActionButtons();
	      });
      inputImage.addEventListener("click", handleInputBoardClick);
      dynamicGateImage.addEventListener("mousemove", (event) => {
        showGateTooltip(event, dynamicGateImage, "dynamic_gate", "Dynamic gate");
      });
      dynamicGateImage.addEventListener("mouseleave", hideGateTooltip);
      dynamicGateCcImage.addEventListener("mousemove", (event) => {
        showGateTooltip(event, dynamicGateCcImage, "dynamic_gate_cc", "Dynamic gate CC");
      });
      dynamicGateCcImage.addEventListener("mouseleave", hideGateTooltip);
      nextGateImage.addEventListener("mousemove", (event) => {
        showGateTooltip(event, nextGateImage, "next_dynamic_gate", "Next gate");
      });
      nextGateImage.addEventListener("mouseleave", hideGateTooltip);
      window.addEventListener("keydown", handleKeyboard);

	      buildModelOptions();
	      buildCcModelOptions();
	      buildRecordingOptions();
      buildGameOptions();
      buildActionButtons();
      updateActionButtons();
      setSourceMode();
      loadRecordingFrames();
    </script>
  </body>
</html>
"""


def create_app(
    controller: GameController,
    visualizers: dict[str, LayeredVisualizer],
    cc_scorers: dict[str, ConnectedComponentSimilarityScorer],
    model_options: list[dict[str, Any]],
    cc_model_options: list[dict[str, Any]],
    selected_model_key: str,
    selected_cc_model_key: str,
    recordings_dir: Path,
) -> Flask:
    app = Flask(__name__)
    recording_paths_by_key: dict[str, Path] = {}
    recording_frames_by_key: dict[str, list[RecordingFrame]] = {}
    recording_frames_lock = Lock()
    recording_options: list[dict[str, str]] = []

    for recording_path in list_recording_paths(recordings_dir):
        key = recording_path.name
        recording_paths_by_key[key] = recording_path
        recording_options.append({"key": key, "label": key})

    def get_visualizer(model_key: str | None = None) -> LayeredVisualizer:
        key = model_key or selected_model_key
        visualizer = visualizers.get(key)
        if visualizer is None:
            raise ValueError(f"Unknown model: {key}")
        return visualizer

    def get_cc_scorer(model_key: str | None = None) -> ConnectedComponentSimilarityScorer | None:
        key = model_key or selected_cc_model_key
        if not key:
            return None
        scorer = cc_scorers.get(key)
        if scorer is None:
            raise ValueError(f"Unknown CC model: {key}")
        return scorer

    def get_recording_frames(recording_key: str) -> list[RecordingFrame]:
        frames = recording_frames_by_key.get(recording_key)
        if frames is not None:
            return frames

        recording_path = recording_paths_by_key.get(recording_key)
        if recording_path is None:
            raise ValueError(f"Unknown recording: {recording_key}")

        with recording_frames_lock:
            frames = recording_frames_by_key.get(recording_key)
            if frames is None:
                frames = load_recording_frames(recording_path)
                recording_frames_by_key[recording_key] = frames
        return frames

    def get_recording_frame(recording_key: str, frame_index: int) -> RecordingFrame:
        frames = get_recording_frames(recording_key)
        if frame_index < 0 or frame_index >= len(frames):
            raise ValueError(f"Frame index out of range: {frame_index}")
        return frames[frame_index]

    def render_grid_layers(
        model_key: str,
        grid: np.ndarray | list[list[int]],
        action: GameAction,
        mode: str,
    ) -> dict[str, Any]:
        visualizer = get_visualizer(model_key)
        return visualizer.render_layers(grid, action=action, mode=mode)

    @app.get("/")
    def index() -> str:
        games = controller.games()
        selected_game_id = "ls20-9607627b"
        if selected_game_id not in {game["game_id"] for game in games}:
            selected_game_id = games[0]["game_id"] if games else ""
        return render_template_string(
            HTML_TEMPLATE,
            games=games,
            models=model_options,
            cc_models=cc_model_options,
            recordings=recording_options,
            selected_game_id=selected_game_id,
            selected_model_key=selected_model_key,
            selected_cc_model_key=selected_cc_model_key,
        )

    @app.get("/api/models")
    def models_json() -> Any:
        return jsonify({"models": model_options, "selected_model_key": selected_model_key})

    @app.get("/api/games")
    def games_json() -> Any:
        return jsonify({"games": controller.games()})

    @app.get("/api/recordings")
    def recordings_json() -> Any:
        return jsonify({"recordings": recording_options})

    @app.get("/api/recordings/<recording_key>")
    def recording_frames_json(recording_key: str) -> Any:
        if recording_key not in recording_paths_by_key:
            return jsonify({"error": f"Unknown recording: {recording_key}"}), 404
        frames = get_recording_frames(recording_key)
        return jsonify(
            {
                "frames": [
                    {
                        "index": frame.index,
                        "timestamp": frame.timestamp,
                        "state": frame.state,
                        "action": frame.action_input,
                    }
                    for frame in frames
                ]
            }
        )

    @app.get("/api/recording/render")
    def recording_render_json() -> Any:
        model_key = str(request.args.get("model") or selected_model_key)
        cc_model_key = str(request.args.get("cc_model") or selected_cc_model_key)
        recording_key = str(request.args.get("recording") or "")
        mode = str(request.args.get("mode") or "argmax")
        frame_index = int(request.args.get("frame_index") or 0)
        if not recording_key:
            return jsonify({"error": "Missing recording."}), 400

        try:
            frames = get_recording_frames(recording_key)
            frame = get_recording_frame(recording_key, frame_index)
            next_frame = frames[frame_index + 1] if frame_index + 1 < len(frames) else None
            action_input = next_frame.action_input if next_frame else frame.action_input
            action_id = int(action_input["id"]) if action_input else 0
            action_data = dict(action_input.get("data") or {}) if action_input else {}
            action = action_from_id(action_id, action_data)
            visualizer = get_visualizer(model_key)
            cc_scorer = get_cc_scorer(cc_model_key)
            layers = visualizer.render_layers(frame.grid, action=action, mode=mode)

            return jsonify(
                {
                    "source": "Recording",
                    "state": frame.state,
                    "levels_completed": frame.levels_completed,
                    "win_levels": frame.win_levels,
                    "step": frame.index,
                    "action": {
                        "id": action_id,
                        "label": action_label(action_id, action_data),
                        "data": action_data,
                    },
                    "layers": layers,
                    "similarity": recording_similarity_payload(
                        frames,
                        frame_index,
                        visualizer,
                        cc_scorer,
                    ),
                    "actual_next_image": (
                        grid_to_data_url(np.asarray(next_frame.grid, dtype=np.int64))
                        if next_frame is not None
                        else None
                    ),
                }
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/live/start")
    def live_start_json() -> Any:
        payload = request.get_json(force=True) or {}
        game_id = str(payload.get("game_id") or "")
        seed = int(payload.get("seed") or 0)
        model_key = str(payload.get("model") or selected_model_key)
        cc_model_key = str(payload.get("cc_model") or selected_cc_model_key)
        mode = str(payload.get("mode") or "argmax")
        if not game_id:
            return jsonify({"error": "Missing game_id."}), 400

        try:
            snapshot = controller.start(game_id, seed)
            grid = controller.current_grid()
            action = action_from_id(0, {})
            layers = render_grid_layers(model_key, grid, action=action, mode=mode) if grid is not None else None
            visualizer = get_visualizer(model_key)
            return jsonify(
                {
                    "source": "Live",
                    "state": snapshot.state,
                    "levels_completed": snapshot.levels_completed,
                    "win_levels": snapshot.win_levels,
                    "step": snapshot.step_count,
                    "action": {"id": 0, "label": "RESET", "data": {}},
                    "current": snapshot.__dict__,
                    "layers": layers,
                    "similarity": controller.similarity_payload(
                        visualizer,
                        get_cc_scorer(cc_model_key),
                    ),
                }
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/live/reset")
    def live_reset_json() -> Any:
        payload = request.get_json(force=True) or {}
        model_key = str(payload.get("model") or selected_model_key)
        cc_model_key = str(payload.get("cc_model") or selected_cc_model_key)
        mode = str(payload.get("mode") or "argmax")

        try:
            snapshot = controller.reset()
            grid = controller.current_grid()
            action = action_from_id(0, {})
            layers = render_grid_layers(model_key, grid, action=action, mode=mode) if grid is not None else None
            visualizer = get_visualizer(model_key)
            return jsonify(
                {
                    "source": "Live",
                    "state": snapshot.state,
                    "levels_completed": snapshot.levels_completed,
                    "win_levels": snapshot.win_levels,
                    "step": snapshot.step_count,
                    "action": {"id": 0, "label": "RESET", "data": {}},
                    "current": snapshot.__dict__,
                    "layers": layers,
                    "similarity": controller.similarity_payload(
                        visualizer,
                        get_cc_scorer(cc_model_key),
                    ),
                }
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/live/current_render")
    def live_current_render_json() -> Any:
        payload = request.get_json(force=True) or {}
        model_key = str(payload.get("model") or selected_model_key)
        cc_model_key = str(payload.get("cc_model") or selected_cc_model_key)
        mode = str(payload.get("mode") or "argmax")

        try:
            snapshot = controller.current_snapshot()
            grid = controller.current_grid()
            action_input = snapshot.action_input or {"id": 0, "data": {}}
            action_id = int(action_input.get("id", 0))
            action_data = dict(action_input.get("data") or {})
            action = action_from_id(action_id, action_data)
            layers = (
                render_grid_layers(model_key, grid, action=action, mode=mode)
                if grid is not None
                else None
            )
            visualizer = get_visualizer(model_key)
            return jsonify(
                {
                    "source": "Live",
                    "state": snapshot.state,
                    "levels_completed": snapshot.levels_completed,
                    "win_levels": snapshot.win_levels,
                    "step": snapshot.step_count,
                    "action": {
                        "id": action_id,
                        "label": action_label(action_id, action_data),
                        "data": action_data,
                    },
                    "current": snapshot.__dict__,
                    "layers": layers,
                    "similarity": controller.similarity_payload(
                        visualizer,
                        get_cc_scorer(cc_model_key),
                    ),
                }
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/live/action")
    def live_action_json() -> Any:
        payload = request.get_json(force=True) or {}
        model_key = str(payload.get("model") or selected_model_key)
        cc_model_key = str(payload.get("cc_model") or selected_cc_model_key)
        mode = str(payload.get("mode") or "argmax")
        action_id = int(payload.get("action_id"))
        action_data = dict(payload.get("data") or {})

        try:
            action = action_from_id(action_id, action_data)
            before_grid = controller.current_grid()
            before_layers = (
                render_grid_layers(model_key, before_grid, action=action, mode=mode)
                if before_grid is not None
                else None
            )
            step_payload = controller.step(action_id=action_id, data=action_data)
            current_grid = step_payload["current_grid"]
            current_layers = (
                render_grid_layers(model_key, current_grid, action=action, mode=mode)
                if current_grid is not None
                else None
            )
            current = step_payload["current"]
            visualizer = get_visualizer(model_key)

            return jsonify(
                {
                    "source": "Live",
                    "state": current["state"],
                    "levels_completed": current["levels_completed"],
                    "win_levels": current["win_levels"],
                    "step": current["step_count"],
                    "action": step_payload["action"],
                    "before": step_payload["before"],
                    "current": current,
                    "current_layers": current_layers,
                    "similarity": controller.similarity_payload(
                        visualizer,
                        get_cc_scorer(cc_model_key),
                    ),
                    "predicted_next_image": before_layers["next_reconstruction"] if before_layers else None,
                    "predicted_next_dynamic_image": before_layers["next_dynamic"] if before_layers else None,
                    "predicted_next_dynamic_masked_image": before_layers["next_dynamic_masked"] if before_layers else None,
                }
            )
        except InvalidActionError as exc:
            return jsonify({"error": str(exc)}), 409
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    return app


def build_model_options(
    runs_dir: Path,
    explicit_config: str | None = None,
    explicit_weights: str | None = None,
) -> tuple[dict[str, LayeredVisualizer], list[dict[str, Any]], str]:
    if explicit_config or explicit_weights:
        if not explicit_config or not explicit_weights:
            raise ValueError("Pass both --config and --weights together.")
        model_paths_list = [
            ModelPaths(
                config_path=Path(explicit_config),
                weights_path=Path(explicit_weights),
            )
        ]
    else:
        model_paths_list = list_model_paths(runs_dir)

    visualizers: dict[str, LayeredVisualizer] = {}
    model_options: list[dict[str, Any]] = []
    selected_model_key = ""

    if model_paths_list:
        selected_model = max(
            model_paths_list,
            key=lambda paths: paths.config_path.stat().st_mtime,
        )
        selected_model_key = str(selected_model.weights_path)

    for model_paths in model_paths_list:
        key = str(model_paths.weights_path)
        visualizer = LayeredVisualizer(
            config_path=model_paths.config_path,
            weights_path=model_paths.weights_path,
        )
        visualizers[key] = visualizer
        model_options.append(
            {
                "key": key,
                "label": key,
                "vocab_size": visualizer.vocab_size,
                "action_dim": visualizer.action_dim,
            }
        )

    return visualizers, model_options, selected_model_key


def build_cc_model_options(
    runs_dir: Path,
) -> tuple[
    dict[str, ConnectedComponentSimilarityScorer],
    list[dict[str, Any]],
    str,
]:
    model_paths_list = list_cc_model_paths(runs_dir)
    scorers: dict[str, ConnectedComponentSimilarityScorer] = {}
    model_options: list[dict[str, Any]] = []
    selected_model_key = ""

    if model_paths_list:
        selected_model = max(
            model_paths_list,
            key=lambda paths: paths.config_path.stat().st_mtime,
        )
        selected_model_key = str(selected_model.weights_path)

    for model_paths in model_paths_list:
        key = str(model_paths.weights_path)
        scorer = ConnectedComponentSimilarityScorer(
            config_path=model_paths.config_path,
            weights_path=model_paths.weights_path,
        )
        scorers[key] = scorer
        model_options.append(
            {
                "key": key,
                "label": key,
                "component_embedding_dim": scorer.component_embedding_dim,
            }
        )

    return scorers, model_options, selected_model_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize layered next-state model reconstructions and predictions."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", default=5004, type=int, help="Port to listen on.")
    parser.add_argument("--runs-dir", default="runs", help="Directory to scan for layered models.")
    parser.add_argument("--recordings-dir", default="recordings", help="Directory to scan for recordings.")
    parser.add_argument("--environments-dir", default="environment_files", help="Directory with local ARC-AGI game files.")
    parser.add_argument("--config", help="Path to a layered model config JSON.")
    parser.add_argument("--weights", help="Path to layered model weights.")
    parser.add_argument(
        "--operation-mode",
        choices=["offline", "normal", "online"],
        default="offline",
        help="ARC-AGI operation mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    controller = GameController(
        environments_dir=Path(args.environments_dir),
        recordings_dir=Path(args.recordings_dir),
        operation_mode=arc_agi.OperationMode(args.operation_mode),
    )
    visualizers, model_options, selected_model_key = build_model_options(
        runs_dir=Path(args.runs_dir),
        explicit_config=args.config,
        explicit_weights=args.weights,
    )
    cc_scorers, cc_model_options, selected_cc_model_key = build_cc_model_options(
        runs_dir=Path(args.runs_dir),
    )
    app = create_app(
        controller=controller,
        visualizers=visualizers,
        cc_scorers=cc_scorers,
        model_options=model_options,
        cc_model_options=cc_model_options,
        selected_model_key=selected_model_key,
        selected_cc_model_key=selected_cc_model_key,
        recordings_dir=Path(args.recordings_dir),
    )
    app.run(
        host=args.host,
        port=args.port,
        debug=True,
        threaded=False,
        use_reloader=True,
    )


if __name__ == "__main__":
    main()
