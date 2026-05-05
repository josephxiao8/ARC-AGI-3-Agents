from __future__ import annotations

import argparse
import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from arcengine import FrameDataRaw, GameAction
from flask import Flask, jsonify, render_template_string, request
import arc_agi
import mlx.core as mx
import numpy as np

from agents.templates.nets import ConvVAE, MLP
from view_utils import create_grid_image


@dataclass(frozen=True)
class ModelPaths:
    vae_config_path: Path
    vae_weights_path: Path
    next_config_path: Path | None
    next_weights_path: Path | None


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
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        for vae_config_path in sorted(run_dir.glob("vae_model_*.json")):
            vae_weights_path = vae_config_path.with_suffix(".safetensors")
            if not vae_weights_path.exists():
                continue

            suffix = vae_config_path.name.removeprefix("vae_model_")
            next_config_path = run_dir / f"next_state_model_{suffix}"
            next_weights_path = next_config_path.with_suffix(".safetensors")
            if not next_config_path.exists() or not next_weights_path.exists():
                next_config_path = None
                next_weights_path = None

            model_paths.append(
                ModelPaths(
                    vae_config_path=vae_config_path,
                    vae_weights_path=vae_weights_path,
                    next_config_path=next_config_path,
                    next_weights_path=next_weights_path,
                )
            )
    return model_paths


class NextStateVisualizer:
    def __init__(self, model_paths: ModelPaths) -> None:
        config = json.loads(model_paths.vae_config_path.read_text(encoding="utf-8"))
        self.vae_config_path = model_paths.vae_config_path
        self.vae_weights_path = model_paths.vae_weights_path
        self.next_config_path = model_paths.next_config_path
        self.next_weights_path = model_paths.next_weights_path
        self.latent_dim = int(config["latent_dim"])
        self.input_channels = int(config["input_channels"])
        self.vae: ConvVAE | None = None
        self.next_net: MLP | None = None
        self._load_lock = Lock()

    @property
    def has_next_state_model(self) -> bool:
        return self.next_config_path is not None and self.next_weights_path is not None

    def _get_vae(self) -> ConvVAE:
        if self.vae is None:
            with self._load_lock:
                if self.vae is None:
                    vae = ConvVAE(
                        input_channels=self.input_channels,
                        latent_dim=self.latent_dim,
                    )
                    vae.load_weights(str(self.vae_weights_path))
                    vae.eval()
                    self.vae = vae
        return self.vae

    def _get_next_net(self) -> MLP:
        if self.next_config_path is None or self.next_weights_path is None:
            raise ValueError("Selected model has no saved next-state weights.")

        if self.next_net is None:
            with self._load_lock:
                if self.next_net is None:
                    config = json.loads(self.next_config_path.read_text(encoding="utf-8"))
                    net = MLP(
                        input_dim=int(config.get("latent_dim", self.latent_dim)),
                        time_dim=int(config.get("time_dim", 1)),
                        hidden_dim=int(config.get("hidden_dim", 128)),
                    )
                    net.load_weights(str(self.next_weights_path))
                    net.eval()
                    self.next_net = net
        return self.next_net

    def encode_grid(self, grid: list[list[int]]) -> np.ndarray:
        vae = self._get_vae()
        frame = np.asarray(grid, dtype=np.int64)
        if frame.shape != (64, 64):
            raise ValueError(f"Expected a 64x64 grid, got {frame.shape}.")

        mu, _ = vae.encoder(mx.array(frame[None, :, :]))
        return np.asarray(mu)[0].astype(np.float32)

    def decode_latent(self, latent: np.ndarray, mode: str) -> np.ndarray:
        vae = self._get_vae()
        if latent.shape != (self.latent_dim,):
            raise ValueError(f"Expected latent shape {(self.latent_dim,)}, got {latent.shape}.")

        decoded = vae.decoder(mx.array(latent[None, :]))
        decoded_np = np.asarray(decoded)[0]

        if mode == "sample":
            shifted = decoded_np - np.max(decoded_np, axis=-1, keepdims=True)
            probs = np.exp(shifted)
            probs = probs / np.clip(probs.sum(axis=-1, keepdims=True), 1e-8, None)
            flat_probs = probs.reshape(-1, probs.shape[-1])
            sampled = []
            for row in flat_probs:
                normalized = np.asarray(row, dtype=np.float64)
                normalized_sum = float(normalized.sum())
                if normalized_sum <= 0:
                    normalized = np.full_like(normalized, 1.0 / normalized.shape[0])
                else:
                    normalized = normalized / normalized_sum
                normalized[-1] = 1.0 - normalized[:-1].sum()
                normalized = np.clip(normalized, 0.0, 1.0)
                normalized = normalized / normalized.sum()
                sampled.append(np.random.choice(normalized.shape[0], p=normalized))
            return np.asarray(sampled, dtype=np.int64).reshape(64, 64)

        return np.argmax(decoded_np, axis=-1).astype(np.int64)

    def predict_next_grid(self, grid: list[list[int]], mode: str, steps: int) -> np.ndarray:
        net = self._get_next_net()
        latent = self.encode_grid(grid)
        z_t = mx.array(latent[None, :])
        step_count = max(1, int(steps))
        dt = 1.0 / step_count

        # simple forward Euler integration for the latent velocity model
        for step in range(step_count):
            time = mx.array([[step / step_count]], dtype=mx.float32)
            velocity = net(z_t, time)
            z_t = z_t + velocity * dt

        return self.decode_latent(np.asarray(z_t)[0].astype(np.float32), mode=mode)


def state_label(state: Any) -> str:
    if hasattr(state, "name"):
        return str(state.name)
    return str(state)


def action_label(action_id: int, data: dict[str, Any] | None = None) -> str:
    label = "RESET" if action_id == 0 else f"ACTION{action_id}"
    if data and "x" in data and "y" in data:
        return f"{label} ({data['x']}, {data['y']})"
    if data:
        return f"{label} {data}"
    return label


def frame_to_grid(frame_data: FrameDataRaw | None) -> np.ndarray | None:
    if frame_data is None or not frame_data.frame:
        return None

    grid = np.asarray(frame_data.frame[-1], dtype=np.int64)
    if grid.shape != (64, 64):
        return None
    return grid


def grid_to_data_url(grid: np.ndarray | None) -> str | None:
    if grid is None:
        return None

    image = create_grid_image(grid, cell_size=8, border_width=1)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def action_input_dict(frame_data: FrameDataRaw | None) -> dict[str, Any] | None:
    if frame_data is None or frame_data.action_input is None:
        return None

    action = frame_data.action_input.id
    action_id = int(action.value) if hasattr(action, "value") else int(action)
    return {
        "id": action_id,
        "label": action_label(action_id, frame_data.action_input.data),
        "data": frame_data.action_input.data,
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
        available_actions = [int(action_id) for action_id in frame_data.available_actions]

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
            return snapshot_from_frame(self.current_frame, self.step_count, game_id)

    def reset(self) -> GameSnapshot:
        with self.lock:
            if self.env is None:
                raise ValueError("Start a game first.")

            self.current_frame = self.env.reset()
            self.step_count = 0
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

    def step(
        self,
        action_id: int,
        data: dict[str, Any],
        prediction_model: NextStateVisualizer | None,
        decode_mode: str,
        flow_steps: int,
    ) -> dict[str, Any]:
        with self.lock:
            if self.env is None or self.current_frame is None:
                raise ValueError("Start a game first.")

            available_actions = set(self.current_frame.available_actions or [])
            if action_id not in available_actions:
                raise InvalidActionError(f"{action_label(action_id, data)} is not available.")

            action = GameAction.from_id(action_id)
            action_data = data if action.is_complex() else {}
            if action.is_complex():
                if "x" not in action_data or "y" not in action_data:
                    raise InvalidActionError(f"{action_label(action_id)} needs a board position.")
                action_data = {
                    "x": int(action_data["x"]),
                    "y": int(action_data["y"]),
                }

            before_frame = self.current_frame
            before_grid = frame_to_grid(before_frame)
            prediction = {
                "available": False,
                "image": None,
                "status": "No next-state model selected.",
            }

            if prediction_model is not None and prediction_model.has_next_state_model and before_grid is not None:
                try:
                    predicted_grid = prediction_model.predict_next_grid(
                        before_grid.tolist(),
                        mode=decode_mode,
                        steps=flow_steps,
                    )
                    prediction = {
                        "available": True,
                        "image": grid_to_data_url(predicted_grid),
                        "status": "Predicted.",
                    }
                except Exception as exc:
                    prediction = {
                        "available": False,
                        "image": None,
                        "status": f"Prediction failed: {exc}",
                    }
            elif prediction_model is not None and not prediction_model.has_next_state_model:
                prediction["status"] = "Selected run has no saved next-state weights."

            next_frame = self.env.step(
                action,
                data=action_data,
                reasoning={"source": "interactive_next_state_pred_viz_server"},
            )
            if next_frame is None:
                raise RuntimeError("Environment returned no frame.")

            self.current_frame = next_frame
            self.step_count += 1

            return {
                "before": snapshot_from_frame(before_frame, self.step_count - 1, self.current_game_id).__dict__,
                "current": snapshot_from_frame(self.current_frame, self.step_count, self.current_game_id).__dict__,
                "prediction": prediction,
                "action": {
                    "id": action_id,
                    "label": action_label(action_id, action_data),
                    "data": action_data,
                },
            }


class InvalidActionError(ValueError):
    pass


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>ARC-AGI Next State Player</title>
    <style>
      body {
        margin: 24px;
        font-family: sans-serif;
        color: #151515;
        background: #f7f7f5;
      }

      h1,
      h2 {
        margin-top: 0;
        margin-bottom: 12px;
        font-weight: 600;
      }

      h1 {
        font-size: 24px;
      }

      h2 {
        font-size: 16px;
      }

      label {
        display: block;
        margin-bottom: 6px;
        font-size: 13px;
      }

      select,
      button,
      input {
        font: inherit;
      }

      button {
        min-height: 36px;
        padding: 8px 12px;
        border: 1px solid #bdbdb8;
        background: #fff;
        cursor: pointer;
      }

      button.is-muted {
        color: #767676;
        background: #ececea;
      }

      .layout {
        display: flex;
        align-items: flex-start;
        gap: 24px;
      }

      .controls {
        width: 300px;
        flex: 0 0 auto;
      }

      .control-group {
        margin-bottom: 18px;
      }

      .select,
      .number-input {
        width: 100%;
        box-sizing: border-box;
        padding: 6px 8px;
      }

      .button-row {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 8px;
      }

      .action-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 8px;
      }

      .mode-toggle label {
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 6px;
      }

      .steps-row {
        display: grid;
        grid-template-columns: 1fr 42px;
        align-items: center;
        gap: 10px;
      }

      .steps-value,
      .meta,
      .status-text {
        font-size: 12px;
        color: #555;
      }

      .status-text {
        margin-top: 6px;
      }

      .preview {
        min-width: 0;
        flex: 1 1 auto;
      }

      .score-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin-bottom: 18px;
        padding-bottom: 14px;
        border-bottom: 1px solid #ddddda;
      }

      .score-label {
        margin-bottom: 4px;
        font-size: 11px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0;
      }

      .score-value {
        min-height: 19px;
        overflow-wrap: anywhere;
        font-size: 13px;
      }

      .preview-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(210px, 1fr));
        gap: 22px;
        align-items: start;
      }

      .preview-panel {
        min-width: 0;
      }

      img {
        display: block;
        width: 100%;
        height: auto;
        image-rendering: pixelated;
        border: 1px solid #d7d7d2;
        background: #fff;
      }

      #current-image {
        cursor: crosshair;
      }

      .toast {
        position: fixed;
        left: 50%;
        bottom: 24px;
        transform: translateX(-50%);
        max-width: min(520px, calc(100vw - 32px));
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

      @media (max-width: 1080px) {
        .preview-grid {
          grid-template-columns: repeat(2, minmax(220px, 1fr));
        }
      }

      @media (max-width: 820px) {
        body {
          margin: 16px;
        }

        .layout {
          flex-direction: column;
        }

        .controls {
          width: 100%;
        }

        .score-strip,
        .preview-grid {
          grid-template-columns: 1fr;
        }
      }
    </style>
  </head>
  <body>
    <div class="layout">
      <section class="controls">
        <h1>ARC-AGI Player</h1>

        <div class="control-group">
          <label for="game-select">Game</label>
          <select id="game-select" class="select"></select>
        </div>

        <div class="control-group">
          <label for="model-select">Model</label>
          <select id="model-select" class="select"></select>
          <div id="model-status" class="status-text"></div>
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

        <div class="control-group mode-toggle">
          <label><input type="radio" name="decode-mode" value="argmax" checked> Argmax</label>
          <label><input type="radio" name="decode-mode" value="sample"> Sample</label>
        </div>

        <div class="control-group">
          <label for="step-slider">Flow steps</label>
          <div class="steps-row">
            <input id="step-slider" type="range" min="1" max="64" step="1" value="16">
            <span id="step-value" class="steps-value">16</span>
          </div>
        </div>
      </section>

      <section class="preview">
        <div class="score-strip">
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
            <div id="step-count-value" class="score-value">-</div>
          </div>
          <div>
            <div class="score-label">Action</div>
            <div id="action-value" class="score-value">-</div>
          </div>
        </div>

        <div class="preview-grid">
          <div class="preview-panel">
            <h2>Current Frame</h2>
            <img id="current-image" alt="Current game frame">
          </div>
          <div class="preview-panel">
            <h2>Predicted Next</h2>
            <img id="prediction-image" alt="Predicted next frame">
            <div id="prediction-status" class="status-text"></div>
          </div>
        </div>
      </section>
    </div>

    <div id="toast" class="toast"></div>

    <script>
      const games = {{ games|tojson }};
      const models = {{ models|tojson }};
      const initialGameId = {{ selected_game_id|tojson }};
      const initialModelKey = {{ selected_model_key|tojson }};

      const gameSelect = document.getElementById("game-select");
      const modelSelect = document.getElementById("model-select");
      const modelStatus = document.getElementById("model-status");
      const seedInput = document.getElementById("seed-input");
      const startButton = document.getElementById("start-button");
      const resetButton = document.getElementById("reset-button");
      const actionButtons = document.getElementById("action-buttons");
      const stepSlider = document.getElementById("step-slider");
      const stepValue = document.getElementById("step-value");
      const stateValue = document.getElementById("state-value");
      const levelValue = document.getElementById("level-value");
      const stepCountValue = document.getElementById("step-count-value");
      const actionValue = document.getElementById("action-value");
      const currentImage = document.getElementById("current-image");
      const predictionImage = document.getElementById("prediction-image");
      const predictionStatus = document.getElementById("prediction-status");
      const toast = document.getElementById("toast");
      const modeInputs = [...document.querySelectorAll('input[name="decode-mode"]')];

      let currentSnapshot = null;
      let toastTimer = null;

      const actionDefinitions = [
        {id: 1, label: "A1", keys: ["ArrowUp", "w", "W"]},
        {id: 2, label: "A2", keys: ["ArrowDown", "s", "S"]},
        {id: 3, label: "A3", keys: ["ArrowLeft", "a", "A"]},
        {id: 4, label: "A4", keys: ["ArrowRight", "d", "D"]},
        {id: 5, label: "A5", keys: [" "]},
        {id: 6, label: "A6", keys: []},
        {id: 7, label: "A7", keys: ["Enter"]},
      ];

      function getDecodeMode() {
        const selected = modeInputs.find((input) => input.checked);
        return selected ? selected.value : "argmax";
      }

      function selectedModel() {
        return models.find((model) => model.key === modelSelect.value);
      }

      function showToast(message) {
        toast.textContent = message;
        toast.classList.add("is-visible");
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => {
          toast.classList.remove("is-visible");
        }, 2200);
      }

      function isActionAvailable(actionId) {
        return Boolean(currentSnapshot?.available_actions?.includes(actionId));
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

      function buildModelOptions() {
        const noneOption = document.createElement("option");
        noneOption.value = "";
        noneOption.textContent = "None";
        modelSelect.appendChild(noneOption);

        for (const model of models) {
          const option = document.createElement("option");
          option.value = model.key;
          option.textContent = model.label;
          modelSelect.appendChild(option);
        }
        modelSelect.value = initialModelKey || "";
        updateModelStatus();
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

      function updateModelStatus() {
        const model = selectedModel();
        if (!model) {
          modelStatus.textContent = "No model selected.";
          return;
        }
        modelStatus.textContent = model.has_next_state
          ? "Next-state weights found."
          : "No next-state weights found.";
      }

      function updateActionButtons() {
        for (const button of actionButtons.querySelectorAll("button")) {
          const actionId = Number(button.dataset.actionId);
          button.classList.toggle("is-muted", !isActionAvailable(actionId));
        }
      }

      function setImage(image, src) {
        if (src) {
          image.src = src;
        } else {
          image.removeAttribute("src");
        }
      }

      function renderSnapshot(snapshot, actionLabel = "-") {
        currentSnapshot = snapshot;
        stateValue.textContent = snapshot.state;
        levelValue.textContent = `${snapshot.levels_completed}/${snapshot.win_levels}`;
        stepCountValue.textContent = String(snapshot.step_count);
        actionValue.textContent = actionLabel;
        setImage(currentImage, snapshot.image);
        updateActionButtons();
      }

      function clearGameState() {
        currentSnapshot = null;
        stateValue.textContent = "-";
        levelValue.textContent = "-";
        stepCountValue.textContent = "-";
        actionValue.textContent = "-";
        predictionStatus.textContent = "";
        setImage(currentImage, null);
        setImage(predictionImage, null);
        updateActionButtons();
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

      async function startGame() {
        try {
          const payload = await postJson("/api/start", {
            game_id: gameSelect.value,
            seed: Number(seedInput.value || 0),
          });
          renderSnapshot(payload.current, "RESET");
          predictionStatus.textContent = "";
          setImage(predictionImage, null);
        } catch (error) {
          showToast(error instanceof Error ? error.message : "Failed to start game.");
        }
      }

      async function resetGame() {
        try {
          const payload = await postJson("/api/reset", {});
          renderSnapshot(payload.current, "RESET");
          predictionStatus.textContent = "";
          setImage(predictionImage, null);
        } catch (error) {
          showToast(error instanceof Error ? error.message : "Failed to reset game.");
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
          const payload = await postJson("/api/action", {
            action_id: actionId,
            data,
            model: modelSelect.value,
            mode: getDecodeMode(),
            steps: Number(stepSlider.value),
          });
          renderSnapshot(payload.current, payload.action.label);
          setImage(predictionImage, payload.prediction.image);
          predictionStatus.textContent = payload.prediction.status;
          if (payload.current.state === "WIN") {
            showToast("Game won.");
          } else if (payload.current.state === "GAME_OVER") {
            showToast("Game over.");
          }
        } catch (error) {
          showToast(error instanceof Error ? error.message : "Action failed.");
        }
      }

      function handleBoardClick(event) {
        if (!currentSnapshot) {
          showToast("Start a game first.");
          return;
        }
        const rect = currentImage.getBoundingClientRect();
        const x = Math.max(0, Math.min(63, Math.floor(((event.clientX - rect.left) / rect.width) * 64)));
        const y = Math.max(0, Math.min(63, Math.floor(((event.clientY - rect.top) / rect.height) * 64)));
        sendAction(6, {x, y});
      }

      function handleKeyboard(event) {
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

      startButton.addEventListener("click", startGame);
      resetButton.addEventListener("click", resetGame);
      gameSelect.addEventListener("change", clearGameState);
      modelSelect.addEventListener("change", updateModelStatus);
      stepSlider.addEventListener("input", () => {
        stepValue.textContent = stepSlider.value;
      });
      currentImage.addEventListener("click", handleBoardClick);
      window.addEventListener("keydown", handleKeyboard);

      buildGameOptions();
      buildModelOptions();
      buildActionButtons();
      updateActionButtons();
    </script>
  </body>
</html>
"""


def create_app(
    controller: GameController,
    visualizers: dict[str, NextStateVisualizer],
    model_options: list[dict[str, Any]],
    selected_model_key: str,
) -> Flask:
    app = Flask(__name__)

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
            selected_game_id=selected_game_id,
            selected_model_key=selected_model_key,
        )

    @app.get("/api/games")
    def games_json() -> Any:
        return jsonify({"games": controller.games()})

    @app.get("/api/models")
    def models_json() -> Any:
        return jsonify({"models": model_options, "selected_model_key": selected_model_key})

    @app.get("/api/current")
    def current_json() -> Any:
        return jsonify({"current": controller.current_snapshot().__dict__})

    @app.post("/api/start")
    def start_json() -> Any:
        payload = request.get_json(force=True) or {}
        game_id = str(payload.get("game_id") or "")
        seed = int(payload.get("seed") or 0)
        if not game_id:
            return jsonify({"error": "Missing game_id."}), 400
        try:
            return jsonify({"current": controller.start(game_id, seed).__dict__})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/reset")
    def reset_json() -> Any:
        try:
            return jsonify({"current": controller.reset().__dict__})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/api/action")
    def action_json() -> Any:
        payload = request.get_json(force=True) or {}
        model_key = str(payload.get("model") or "")
        prediction_model = visualizers.get(model_key) if model_key else None
        mode = str(payload.get("mode") or "argmax")
        if mode not in {"argmax", "sample"}:
            return jsonify({"error": "mode must be 'argmax' or 'sample'."}), 400
        try:
            response = controller.step(
                action_id=int(payload.get("action_id")),
                data=dict(payload.get("data") or {}),
                prediction_model=prediction_model,
                decode_mode=mode,
                flow_steps=int(payload.get("steps") or 16),
            )
            return jsonify(response)
        except InvalidActionError as exc:
            return jsonify({"error": str(exc)}), 409
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    return app


def build_model_options(runs_dir: Path) -> tuple[dict[str, NextStateVisualizer], list[dict[str, Any]], str]:
    visualizers: dict[str, NextStateVisualizer] = {}
    model_options: list[dict[str, Any]] = []

    model_paths_list = list_model_paths(runs_dir)
    selected_model_key = ""
    if model_paths_list:
        selected_model = max(model_paths_list, key=lambda paths: paths.vae_config_path.stat().st_mtime)
        selected_model_key = str(selected_model.vae_weights_path)

    for model_paths in model_paths_list:
        key = str(model_paths.vae_weights_path)
        visualizer = NextStateVisualizer(model_paths)
        visualizers[key] = visualizer
        model_options.append(
            {
                "key": key,
                "label": key,
                "has_next_state": visualizer.has_next_state_model,
                "latent_dim": visualizer.latent_dim,
            }
        )

    return visualizers, model_options, selected_model_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play ARC-AGI games with next-state model previews.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", default=5003, type=int, help="Port to listen on.")
    parser.add_argument("--runs-dir", default="runs", help="Directory to scan for saved VAE/next-state models.")
    parser.add_argument("--environments-dir", default="environment_files", help="Directory with local ARC-AGI game files.")
    parser.add_argument("--recordings-dir", default="recordings", help="Directory for ARC-AGI scorecard/recording data.")
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
    visualizers, model_options, selected_model_key = build_model_options(Path(args.runs_dir))
    app = create_app(controller, visualizers, model_options, selected_model_key)
    app.run(
        host=args.host,
        port=args.port,
        debug=False,
        threaded=False,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
