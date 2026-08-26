"""
Neural Inference Pipeline
NEXUS-01 — bridges sensors → NeuralCore → decisions
"""

from __future__ import annotations
import time
import numpy as np
from typing import Tuple, Dict, Any, List
from src.neural.network import NeuralCore
from config.settings import DECISION_LABELS


class InferenceEngine:
    """
    High-level inference facade.
    Designed so the underlying NeuralCore can later be swapped for a PyTorch module.
    """

    def __init__(self, neural_core: NeuralCore = None, seed: int = 42):
        self.core = neural_core or NeuralCore(seed=seed)
        self.last_latency_ms: float = 0.0
        self.history: List[Dict[str, Any]] = []

    def infer(self, features: np.ndarray) -> Tuple[int, float, np.ndarray, str]:
        """
        Run forward pass.
        Returns: (decision_idx, confidence, probabilities, decision_label)
        """
        t0 = time.perf_counter()
        idx, conf, probs = self.core.forward(features)
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        label = DECISION_LABELS[idx]
        self.history.append({
            "decision": label,
            "confidence": conf,
            "latency_ms": self.last_latency_ms,
        })
        if len(self.history) > 200:
            self.history = self.history[-200:]
        return idx, conf, probs, label

    def batch_infer(self, batch: np.ndarray) -> List[Tuple[int, float, str]]:
        results = []
        for feat in batch:
            idx, conf, _, label = self.infer(feat)
            results.append((idx, conf, label))
        return results

    def get_core_visualization(self) -> Dict[str, Any]:
        return self.core.get_visualization_data()

    @property
    def total_inferences(self) -> int:
        return self.core.total_inferences
