from __future__ import annotations

import argparse
import io
import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from flask import Flask, jsonify, render_template_string, request, send_file
import mlx.core as mx
import numpy as np

from agents.templates.nets import ConvVAE
from agents.recorder import RECORDING_SUFFIX
from view_utils import create_grid_image


@dataclass(frozen=True)
class ModelPaths:
    config_path: Path
    weights_path: Path


@dataclass(frozen=True)
class RecordingFrame:
    index: int
    timestamp: str
    grid: list[list[int]]


def list_model_paths(runs_dir: Path) -> list[ModelPaths]:
    model_paths: list[ModelPaths] = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        for config_path in sorted(run_dir.glob("vae_model_*.json")):
            weights_path = config_path.with_suffix(".safetensors")
            if weights_path.exists():
                model_paths.append(ModelPaths(config_path=config_path, weights_path=weights_path))
    return model_paths


def find_latest_model_paths(runs_dir: Path) -> ModelPaths:
    model_paths = list_model_paths(runs_dir)
    if not model_paths:
        raise FileNotFoundError(
            f"No VAE config files found under {runs_dir}. "
            "Pass --config and --weights explicitly."
        )

    return max(model_paths, key=lambda paths: paths.config_path.stat().st_mtime)


def list_recording_paths(recordings_dir: Path) -> list[Path]:
    return sorted(recordings_dir.glob(f"*{RECORDING_SUFFIX}"))


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
                continue
            frames.append(
                RecordingFrame(
                    index=len(frames),
                    timestamp=str(event.get("timestamp", "")),
                    grid=grid,
                )
            )
    return frames


class VAEVisualizer:
    def __init__(self, config_path: Path, weights_path: Path) -> None:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.config_path = config_path
        self.weights_path = weights_path
        self.latent_dim = int(config["latent_dim"])
        self.input_channels = int(config["input_channels"])
        self.vae: ConvVAE | None = None
        self._load_lock = Lock()

    def _get_vae(self) -> ConvVAE:
        if self.vae is None:
            with self._load_lock:
                if self.vae is None:
                    vae = ConvVAE(
                        input_channels=self.input_channels,
                        latent_dim=self.latent_dim,
                    )
                    vae.load_weights(str(self.weights_path))
                    vae.eval()
                    self.vae = vae
        return self.vae

    def decode_grid(self, latent_values: list[float], mode: str) -> np.ndarray:
        vae = self._get_vae()
        latent = np.asarray(latent_values, dtype=np.float32)
        if latent.shape != (self.latent_dim,):
            raise ValueError(
                f"Expected {self.latent_dim} latent values, got {latent.shape[0]}."
            )

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

    def encode_grid(self, grid: list[list[int]]) -> list[float]:
        vae = self._get_vae()
        frame = np.asarray(grid, dtype=np.int64)
        if frame.shape != (64, 64):
            raise ValueError(f"Expected a 64x64 grid, got {frame.shape}.")

        mu, _ = vae.encoder(mx.array(frame[None, :, :]))
        return np.asarray(mu)[0].astype(np.float32).tolist()


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>ConvVAE Latent Visualizer</title>
    <style>
      body {
        margin: 24px;
        font-family: sans-serif;
      }

      h1, h2 {
        margin-top: 0;
        margin-bottom: 16px;
        font-weight: 600;
      }

      .layout {
        display: flex;
        align-items: flex-start;
        gap: 24px;
      }

      .controls {
        width: 220px;
      }

      .model-select {
        width: 100%;
        margin-bottom: 16px;
        padding: 6px 8px;
        font: inherit;
      }

      .input-controls {
        margin-bottom: 16px;
      }

      .input-controls label {
        display: block;
        margin-bottom: 6px;
        font-size: 13px;
      }

      .mode-toggle {
        margin-bottom: 16px;
        font-size: 13px;
      }

      .mode-toggle label {
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 6px;
      }

      .scrubber {
        width: 100%;
      }

      .scrubber-meta {
        margin-top: 6px;
        font-size: 12px;
        color: #444;
      }

      .slider-list {
        display: grid;
        gap: 10px;
        margin-bottom: 20px;
      }

      .slider-row {
        display: grid;
        grid-template-columns: 48px 1fr 52px;
        align-items: center;
        gap: 10px;
      }

      input[type="range"] {
        width: 100%;
      }

      .slider-range,
      .slider-value {
        font-size: 12px;
        color: #444;
      }

      .slider-value {
        text-align: right;
        font-variant-numeric: tabular-nums;
      }

      button {
        padding: 8px 14px;
        font: inherit;
        cursor: pointer;
      }

      .preview {
        display: flex;
        flex-direction: column;
        flex: 1 1 0;
        min-width: 0;
      }

      .preview-grid {
        display: flex;
        gap: 24px;
        align-items: flex-start;
        width: 100%;
      }

      .preview-panel {
        flex: 1 1 0;
      }

      .preview-panel h2 {
        margin-bottom: 8px;
      }

      .status-text {
        margin-top: 8px;
        font-size: 12px;
        color: #444;
      }

      .status-text.is-loading {
        color: #8a5a00;
      }

      .delta-log {
        margin-top: 24px;
        border-top: 1px solid #ddd;
        padding-top: 16px;
      }

      .delta-log h2 {
        margin-bottom: 10px;
      }

      .delta-log-grid {
        display: grid;
        gap: 6px;
        overflow-x: auto;
      }

      .delta-log-row {
        display: grid;
        gap: 6px;
      }

      .delta-log-cell {
        min-width: 52px;
        padding: 6px 4px;
        font-size: 11px;
        text-align: center;
        border: 1px solid #ddd;
        border-radius: 4px;
        font-variant-numeric: tabular-nums;
        background: #fff;
      }

      .delta-log-header .delta-log-cell {
        font-weight: 600;
        background: #f5f5f5;
      }

      .delta-log-empty {
        margin: 0;
        font-size: 12px;
        color: #666;
      }

      img {
        display: block;
        width: 100%;
        height: auto;
        image-rendering: pixelated;
      }

      @media (max-width: 900px) {
        .layout {
          flex-direction: column;
        }

        .controls {
          width: 100%;
        }

        img {
          width: 100%;
          max-width: 640px;
        }
      }
    </style>
  </head>
  <body>
    <div class="layout">
      <section class="controls">
        <h1>Z</h1>
        <select id="model-select" class="model-select"></select>
        <div class="input-controls">
          <label for="recording-select">Initial state recording</label>
          <select id="recording-select" class="model-select"></select>
          <input id="frame-scrubber" class="scrubber" type="range" min="0" max="0" step="1" value="0">
          <div id="scrubber-meta" class="scrubber-meta">No recording selected.</div>
          <div id="recording-status" class="status-text"></div>
        </div>
        <div class="mode-toggle">
          <label><input type="radio" name="decode-mode" value="argmax" checked> Argmax</label>
          <label><input type="radio" name="decode-mode" value="sample"> Sample</label>
        </div>
        <div class="slider-list" id="sliders"></div>
        <button id="random">Randomize Z</button>
      </section>

      <section class="preview">
        <div class="preview-grid">
          <div class="preview-panel">
            <h2>Initial State</h2>
            <img id="input-image" alt="Selected recorded frame">
          </div>
          <div class="preview-panel">
            <h2>Reconstruction</h2>
            <img id="grid-image" alt="Decoded ARC grid">
            <div id="encode-status" class="status-text"></div>
          </div>
        </div>
        <section class="delta-log">
          <h2>Latent Delta Log</h2>
          <p id="delta-log-empty" class="delta-log-empty">Scrub between frames to log latent changes.</p>
          <div id="delta-log" class="delta-log-grid" hidden></div>
        </section>
      </section>
    </div>

    <script>
      const modelOptions = {{ model_options|tojson }};
      const initialModelKey = {{ selected_model_key|tojson }};
      const recordingOptions = {{ recording_options|tojson }};
      const sliderContainer = document.getElementById("sliders");
      const gridImage = document.getElementById("grid-image");
      const inputImage = document.getElementById("input-image");
      const encodeStatus = document.getElementById("encode-status");
      const modelSelect = document.getElementById("model-select");
      const recordingSelect = document.getElementById("recording-select");
      const frameScrubber = document.getElementById("frame-scrubber");
      const scrubberMeta = document.getElementById("scrubber-meta");
      const recordingStatus = document.getElementById("recording-status");
      const deltaLog = document.getElementById("delta-log");
      const deltaLogEmpty = document.getElementById("delta-log-empty");
      const modeInputs = [...document.querySelectorAll('input[name="decode-mode"]')];
      let currentLatentDim = 0;
      let currentFrames = [];
      let lastEncodedLatent = null;
      let lastEncodedFrameMeta = null;
      let deltaLogEntries = [];
      let recordingLoadRequestId = 0;

      function getSelectedModel() {
        return modelOptions.find((model) => model.key === modelSelect.value);
      }

      function buildModelOptions() {
        for (const model of modelOptions) {
          const option = document.createElement("option");
          option.value = model.key;
          option.textContent = model.label;
          modelSelect.appendChild(option);
        }
        modelSelect.value = initialModelKey;
      }

      function buildRecordingOptions() {
        const emptyOption = document.createElement("option");
        emptyOption.value = "";
        emptyOption.textContent = "None";
        recordingSelect.appendChild(emptyOption);

        for (const recording of recordingOptions) {
          const option = document.createElement("option");
          option.value = recording.key;
          option.textContent = recording.label;
          recordingSelect.appendChild(option);
        }
      }

      function getDecodeMode() {
        const selected = modeInputs.find((input) => input.checked);
        return selected ? selected.value : "argmax";
      }

      function buildSlider(index) {
        const row = document.createElement("div");
        row.className = "slider-row";

        const range = document.createElement("span");
        range.className = "slider-range";
        range.textContent = "[-3, 3]";

        const input = document.createElement("input");
        input.type = "range";
        input.min = "-3";
        input.max = "3";
        input.step = "0.01";
        input.value = "0";
        input.dataset.index = String(index);
        input.addEventListener("input", () => {
          value.textContent = Number(input.value).toFixed(2);
          render();
        });

        const value = document.createElement("span");
        value.className = "slider-value";
        value.textContent = "0.00";
        value.dataset.valueFor = String(index);

        row.appendChild(range);
        row.appendChild(input);
        row.appendChild(value);
        sliderContainer.appendChild(row);
      }

      function rebuildSliders(latentDim) {
        sliderContainer.replaceChildren();
        currentLatentDim = latentDim;
        for (let i = 0; i < latentDim; i += 1) {
          buildSlider(i);
        }
      }

      function getLatents() {
        return [...document.querySelectorAll('input[type="range"]')].map(
          (input) => input.id === "frame-scrubber" ? null : Number(input.value)
        ).filter((value) => value !== null);
      }

      function setLatents(latents) {
        for (const [index, value] of latents.entries()) {
          const input = document.querySelector(`input[data-index="${index}"]`);
          if (!input) {
            continue;
          }
          const nextValue = Number(value).toFixed(2);
          input.value = nextValue;
          document.querySelector(`[data-value-for="${index}"]`).textContent = nextValue;
        }
      }

      function maxLatentDelta(nextLatent) {
        if (!lastEncodedLatent || lastEncodedLatent.length !== nextLatent.length) {
          return null;
        }
        let maxDelta = 0;
        for (let i = 0; i < nextLatent.length; i += 1) {
          maxDelta = Math.max(maxDelta, Math.abs(nextLatent[i] - lastEncodedLatent[i]));
        }
        return maxDelta;
      }

      function buildDeltaLogHeader() {
        const headerRow = document.createElement("div");
        headerRow.className = "delta-log-row delta-log-header";
        headerRow.style.gridTemplateColumns = `repeat(${currentLatentDim}, minmax(52px, 1fr))`;

        for (let i = 0; i < currentLatentDim; i += 1) {
          const cell = document.createElement("div");
          cell.className = "delta-log-cell";
          cell.textContent = `z${i}`;
          headerRow.appendChild(cell);
        }

        return headerRow;
      }

      function renderDeltaLog() {
        if (!deltaLogEntries.length) {
          deltaLog.hidden = true;
          deltaLogEmpty.hidden = false;
          return;
        }

        deltaLog.hidden = false;
        deltaLogEmpty.hidden = true;
        deltaLog.replaceChildren();
        deltaLog.appendChild(buildDeltaLogHeader());

        for (const entry of deltaLogEntries) {
          const row = document.createElement("div");
          row.className = "delta-log-row";
          row.style.gridTemplateColumns = `repeat(${entry.deltas.length}, minmax(52px, 1fr))`;

          for (const delta of entry.deltas) {
            const cell = document.createElement("div");
            cell.className = "delta-log-cell";
            cell.textContent = `${delta >= 0 ? "+" : ""}${delta.toFixed(2)}`;
            row.appendChild(cell);
          }

          deltaLog.appendChild(row);
        }
      }

      function resetDeltaLog() {
        deltaLogEntries = [];
        renderDeltaLog();
      }

      function setRecordingLoading(isLoading, message = "") {
        recordingStatus.textContent = message;
        recordingStatus.classList.toggle("is-loading", isLoading);
        recordingSelect.disabled = isLoading;
        frameScrubber.disabled = isLoading || !currentFrames.length;
      }

      function appendDeltaLog(previousLatent, nextLatent, previousFrameMeta, nextFrameMeta) {
        if (!previousFrameMeta || !nextFrameMeta) {
          return;
        }
        deltaLogEntries.unshift({
          deltas: nextLatent.map((value, index) => value - previousLatent[index]),
        });
        deltaLogEntries = deltaLogEntries.slice(0, 5);
        renderDeltaLog();
      }

      async function render() {
        const params = new URLSearchParams();
        params.set("model", modelSelect.value);
        params.set("mode", getDecodeMode());
        params.set("latent", JSON.stringify(getLatents()));
        gridImage.src = `/sample.png?${params.toString()}&t=${Date.now()}`;
      }

      function renderFromSelectedFrame() {
        if (!currentFrames.length) {
          render();
          return;
        }
        const params = new URLSearchParams();
        params.set("model", modelSelect.value);
        params.set("mode", getDecodeMode());
        params.set("recording", recordingSelect.value);
        params.set("frame_index", frameScrubber.value);
        gridImage.src = `/reconstruct_state.png?${params.toString()}&t=${Date.now()}`;
      }

      async function loadRecordingFrames() {
        const requestId = recordingLoadRequestId + 1;
        recordingLoadRequestId = requestId;

        if (!recordingSelect.value) {
          currentFrames = [];
          lastEncodedLatent = null;
          lastEncodedFrameMeta = null;
          frameScrubber.min = "0";
          frameScrubber.max = "0";
          frameScrubber.value = "0";
          scrubberMeta.textContent = "No recording selected.";
          setRecordingLoading(false, "");
          encodeStatus.textContent = "";
          inputImage.removeAttribute("src");
          resetDeltaLog();
          render();
          return;
        }

        lastEncodedLatent = null;
        lastEncodedFrameMeta = null;
        resetDeltaLog();
        const recordingLabel = recordingSelect.selectedOptions[0]?.textContent || recordingSelect.value;
        currentFrames = [];
        frameScrubber.min = "0";
        frameScrubber.max = "0";
        frameScrubber.value = "0";
        scrubberMeta.textContent = "Loading recording frames...";
        setRecordingLoading(true, `Loading ${recordingLabel}...`);

        try {
          const response = await fetch(`/api/recordings/${encodeURIComponent(recordingSelect.value)}`);
          const payload = await response.json();
          if (requestId !== recordingLoadRequestId) {
            return;
          }
          if (!response.ok) {
            throw new Error(payload.error || "Failed to load recording.");
          }
          currentFrames = payload.frames;
          frameScrubber.min = "0";
          frameScrubber.max = String(Math.max(0, currentFrames.length - 1));
          frameScrubber.value = "0";
          updateScrubberMeta();
          setRecordingLoading(false, `Loaded ${currentFrames.length} frame${currentFrames.length === 1 ? "" : "s"}.`);
          await encodeSelectedFrame();
        } catch (error) {
          if (requestId !== recordingLoadRequestId) {
            return;
          }
          currentFrames = [];
          frameScrubber.min = "0";
          frameScrubber.max = "0";
          frameScrubber.value = "0";
          scrubberMeta.textContent = "Failed to load recording.";
          inputImage.removeAttribute("src");
          encodeStatus.textContent = "";
          setRecordingLoading(false, error instanceof Error ? error.message : "Failed to load recording.");
        }
      }

      function updateScrubberMeta() {
        if (!currentFrames.length) {
          scrubberMeta.textContent = "No recording selected.";
          return;
        }
        const frame = currentFrames[Number(frameScrubber.value)];
        scrubberMeta.textContent = `Frame ${frame.index + 1}/${currentFrames.length}  ${frame.timestamp}`;
        const params = new URLSearchParams();
        params.set("recording", recordingSelect.value);
        params.set("frame_index", frameScrubber.value);
        inputImage.src = `/recording_frame.png?${params.toString()}&t=${Date.now()}`;
      }

      async function encodeSelectedFrame() {
        if (!currentFrames.length) {
          return;
        }

        const params = new URLSearchParams();
        params.set("model", modelSelect.value);
        params.set("recording", recordingSelect.value);
        params.set("frame_index", frameScrubber.value);
        const response = await fetch(`/api/encode_state?${params.toString()}`);
        const payload = await response.json();
        const delta = maxLatentDelta(payload.latent);
        const previousLatent = lastEncodedLatent;
        const previousFrameMeta = lastEncodedFrameMeta;
        const nextFrameMeta = {
          frameIndex: payload.frame_index,
          timestamp: payload.timestamp,
        };
        setLatents(payload.latent);
        lastEncodedLatent = payload.latent;
        lastEncodedFrameMeta = nextFrameMeta;
        if (delta === null) {
          encodeStatus.textContent = `Encoded selected frame at ${payload.timestamp}.`;
        } else if (delta < 1e-6) {
          encodeStatus.textContent = "Encoded latent unchanged for this selected frame.";
        } else {
          encodeStatus.textContent = `Encoded latent changed. Max delta: ${delta.toFixed(4)}.`;
          appendDeltaLog(previousLatent, payload.latent, previousFrameMeta, nextFrameMeta);
        }
      }

      document.getElementById("random").addEventListener("click", () => {
        for (const input of document.querySelectorAll('input[type="range"]')) {
          if (input.id === "frame-scrubber") {
            continue;
          }
          const u1 = Math.max(Math.random(), 1e-12);
          const u2 = Math.random();
          const standardNormal = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
          const nextValue = Math.max(-3, Math.min(3, standardNormal)).toFixed(2);
          input.value = nextValue;
          document.querySelector(`[data-value-for="${input.dataset.index}"]`).textContent = nextValue;
        }
        encodeStatus.textContent = "Using randomized latent.";
        render();
      });

      modelSelect.addEventListener("change", () => {
        const model = getSelectedModel();
        lastEncodedLatent = null;
        lastEncodedFrameMeta = null;
        resetDeltaLog();
        if (!model || model.latent_dim === currentLatentDim) {
          if (currentFrames.length) {
            renderFromSelectedFrame();
            encodeSelectedFrame();
          } else {
            render();
          }
          return;
        }
        rebuildSliders(model.latent_dim);
        if (currentFrames.length) {
          renderFromSelectedFrame();
          encodeSelectedFrame();
        } else {
          render();
        }
      });

      recordingSelect.addEventListener("change", loadRecordingFrames);
      frameScrubber.addEventListener("input", () => {
        updateScrubberMeta();
        renderFromSelectedFrame();
        encodeSelectedFrame();
      });
      for (const input of modeInputs) {
        input.addEventListener("change", () => {
          if (currentFrames.length) {
            renderFromSelectedFrame();
          } else {
            render();
          }
        });
      }

      buildModelOptions();
      buildRecordingOptions();
      rebuildSliders(getSelectedModel().latent_dim);
      loadRecordingFrames();
    </script>
  </body>
</html>
"""


def parse_latent_arg(raw_latent: str | None, latent_dim: int) -> list[float]:
    if raw_latent is None:
        return [0.0] * latent_dim

    values = json.loads(raw_latent)
    if not isinstance(values, list):
        raise ValueError("latent must be a JSON array")
    if len(values) != latent_dim:
        raise ValueError(f"latent must contain exactly {latent_dim} values")
    return [float(value) for value in values]


def create_app(
    model_paths_list: list[ModelPaths],
    selected_model_key: str,
    recordings_dir: Path,
) -> Flask:
    app = Flask(__name__)
    visualizers: dict[str, VAEVisualizer] = {}
    model_options: list[dict[str, Any]] = []
    recording_paths_by_key: dict[str, Path] = {}
    recording_frames_by_key: dict[str, list[RecordingFrame]] = {}
    recording_frames_lock = Lock()
    recording_options: list[dict[str, str]] = []

    for model_paths in model_paths_list:
        model_key = str(model_paths.weights_path)
        visualizer = VAEVisualizer(
            config_path=model_paths.config_path,
            weights_path=model_paths.weights_path,
        )
        visualizers[model_key] = visualizer
        model_options.append(
            {
                "key": model_key,
                "label": str(model_paths.weights_path),
                "latent_dim": visualizer.latent_dim,
            }
        )

    for recording_path in list_recording_paths(recordings_dir):
        key = recording_path.name
        recording_paths_by_key[key] = recording_path
        recording_options.append({"key": key, "label": key})

    if selected_model_key not in visualizers:
        raise ValueError(f"Unknown selected model: {selected_model_key}")

    @app.get("/")
    def index() -> str:
        return render_template_string(
            HTML_TEMPLATE,
            model_options=model_options,
            selected_model_key=selected_model_key,
            recording_options=recording_options,
        )

    def get_visualizer() -> VAEVisualizer:
        model_key = request.args.get("model", selected_model_key)
        visualizer = visualizers.get(model_key)
        if visualizer is None:
            raise ValueError(f"Unknown model: {model_key}")
        return visualizer

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

    def get_recording_frame() -> RecordingFrame:
        recording_key = request.args.get("recording")
        if not recording_key:
            raise ValueError("Missing recording.")
        frames = get_recording_frames(recording_key)
        frame_index = int(request.args.get("frame_index", "0"))
        if frame_index < 0 or frame_index >= len(frames):
            raise ValueError(f"Frame index out of range: {frame_index}")
        return frames[frame_index]

    @app.get("/api/sample")
    def sample_json() -> Any:
        visualizer = get_visualizer()
        latent_values = parse_latent_arg(request.args.get("latent"), visualizer.latent_dim)
        mode = request.args.get("mode", "argmax")
        if mode not in {"argmax", "sample"}:
            return jsonify({"error": "mode must be 'argmax' or 'sample'"}), 400

        grid = visualizer.decode_grid(latent_values, mode=mode)
        return jsonify(
            {
                "latent": latent_values,
                "mode": mode,
                "grid": grid.tolist(),
            }
        )

    @app.get("/sample.png")
    def sample_png() -> Any:
        visualizer = get_visualizer()
        latent_values = parse_latent_arg(request.args.get("latent"), visualizer.latent_dim)
        mode = request.args.get("mode", "argmax")
        if mode not in {"argmax", "sample"}:
            return jsonify({"error": "mode must be 'argmax' or 'sample'"}), 400

        grid = visualizer.decode_grid(latent_values, mode=mode)
        image = create_grid_image(grid, cell_size=8, border_width=1)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return send_file(buffer, mimetype="image/png")

    @app.get("/recording_frame.png")
    def recording_frame_png() -> Any:
        frame = get_recording_frame()
        image = create_grid_image(np.asarray(frame.grid, dtype=np.int64), cell_size=8, border_width=1)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return send_file(buffer, mimetype="image/png")

    @app.get("/reconstruct_state.png")
    def reconstruct_state_png() -> Any:
        visualizer = get_visualizer()
        frame = get_recording_frame()
        mode = request.args.get("mode", "argmax")
        if mode not in {"argmax", "sample"}:
            return jsonify({"error": "mode must be 'argmax' or 'sample'"}), 400
        latent = visualizer.encode_grid(frame.grid)
        grid = visualizer.decode_grid(latent, mode=mode)
        image = create_grid_image(grid, cell_size=8, border_width=1)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return send_file(buffer, mimetype="image/png")

    @app.get("/api/models")
    def models_json() -> Any:
        return jsonify(
            {
                "models": model_options,
                "selected_model_key": selected_model_key,
            }
        )

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
                    {"index": frame.index, "timestamp": frame.timestamp}
                    for frame in frames
                ]
            }
        )

    @app.get("/api/encode_state")
    def encode_state() -> Any:
        visualizer = get_visualizer()
        frame = get_recording_frame()
        return jsonify(
            {
                "latent": visualizer.encode_grid(frame.grid),
                "timestamp": frame.timestamp,
                "frame_index": frame.index,
            }
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ConvVAE latent dimensions.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", default=5001, type=int, help="Port to listen on.")
    parser.add_argument("--runs-dir", default="runs", help="Directory to scan for saved VAE runs.")
    parser.add_argument("--recordings-dir", default="recordings", help="Directory to scan for saved recordings.")
    parser.add_argument("--config", help="Path to the VAE config JSON.")
    parser.add_argument("--weights", help="Path to the VAE weights safetensors file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.config or args.weights:
        if not args.config or not args.weights:
            raise ValueError("Pass both --config and --weights together.")
        model_paths_list = [
            ModelPaths(
                config_path=Path(args.config),
                weights_path=Path(args.weights),
            )
        ]
        selected_model = model_paths_list[0]
    else:
        runs_dir = Path(args.runs_dir)
        model_paths_list = list_model_paths(runs_dir)
        if not model_paths_list:
            raise FileNotFoundError(
                f"No VAE config files found under {runs_dir}. "
                "Pass --config and --weights explicitly."
            )
        selected_model = max(model_paths_list, key=lambda paths: paths.config_path.stat().st_mtime)

    app = create_app(
        model_paths_list=model_paths_list,
        selected_model_key=str(selected_model.weights_path),
        recordings_dir=Path(args.recordings_dir),
    )
    app.run(
        host=args.host,
        port=args.port,
        debug=False,
        threaded=False,
        use_reloader=False,
    )
    


if __name__ == "__main__":
    main()
