"""
Live Neural Network Core
NEXUS-01 — Simulated differentiable policy for decision-making
Architecture ready for PyTorch / RL replacement
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Any
from config.settings import SimulationConfig, DECISION_LABELS
from src.neural.activations import relu, softmax


class NeuralCore:
    def __init__(self, config: SimulationConfig = None, seed: int = 42):
        self.cfg = config or SimulationConfig()
        self.rng = np.random.default_rng(seed)
        self.input_dim = self.cfg.INPUT_DIM
        self.hidden_dims = list(self.cfg.HIDDEN_DIMS)
        self.output_dim = self.cfg.OUTPUT_DIM
        self.layer_sizes = [self.input_dim] + self.hidden_dims + [self.output_dim]
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self._init_weights()
        self.activations: List[np.ndarray] = [np.zeros(s) for s in self.layer_sizes]
        self.prev_activations: List[np.ndarray] = [np.zeros(s) for s in self.layer_sizes]
        self.last_inference_ms: float = 0.0
        self.last_confidence: float = 0.0
        self.last_decision_idx: int = 0
        self.active_connections: int = 0
        self.total_inferences: int = 0

    def _init_weights(self):
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            w = self.rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)
            if i == 0:
                w[0, :] += 0.4
                w[1, :] += 0.25
                w[3, :] += 0.35
                w[7, :] += 0.3
            b = self.rng.uniform(-0.1, 0.1, size=(fan_out,)).astype(np.float32)
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x: np.ndarray) -> Tuple[int, float, np.ndarray]:
        import time
        t0 = time.perf_counter()
        self.prev_activations = [a.copy() for a in self.activations]
        a = x.astype(np.float32).ravel()
        self.activations[0] = a.copy()
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            if i < len(self.weights) - 1:
                a = relu(z)
                a += self.rng.normal(0, self.cfg.ACTIVATION_NOISE, size=a.shape).astype(np.float32)
                a = np.clip(a, 0, None)
            else:
                a = z
            self.activations[i + 1] = a.copy()
        probs = softmax(a)
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        self.last_decision_idx = idx
        self.last_confidence = conf
        self.last_inference_ms = (time.perf_counter() - t0) * 1000.0
        self.total_inferences += 1
        self.active_connections = 0
        for i, w in enumerate(self.weights):
            act = self.activations[i]
            for j in range(w.shape[0]):
                if act[j] > 0.15:
                    self.active_connections += int(np.sum(np.abs(w[j]) > 0.08))
        return idx, conf, probs

    def get_visualization_data(self) -> Dict[str, Any]:
        layers = []
        for i, size in enumerate(self.layer_sizes):
            acts = self.activations[i]
            layers.append({"size": size, "activations": acts.tolist(), "mean_act": float(np.mean(acts)), "max_act": float(np.max(acts))})
        edges = []
        for li, w in enumerate(self.weights):
            src_act = self.activations[li]
            for si in range(min(w.shape[0], 12)):
                for di in range(min(w.shape[1], 10)):
                    strength = float(abs(w[si, di]) * (0.3 + src_act[si]))
                    if strength > 0.05:
                        edges.append({"layer": li, "src": si, "dst": di, "weight": float(w[si, di]), "intensity": min(1.0, strength)})
        return {
            "layers": layers,
            "edges": edges[:180],
            "decision": DECISION_LABELS[self.last_decision_idx],
            "confidence": self.last_confidence,
            "inference_ms": self.last_inference_ms,
            "active_connections": self.active_connections,
            "total_inferences": self.total_inferences,
        }
