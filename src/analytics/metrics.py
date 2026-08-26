"""
Real-time Metrics & System Efficiency Index
NEXUS-01
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any
from src.simulation.robot import RobotState


class MetricsCollector:
    def __init__(self):
        self.task_completions = 0
        self.failures = 0
        self.recoveries = 0
        self._prev_transporting = 0

    def compute(self, fleet, env, events, neural) -> Dict[str, Any]:
        stats = fleet.stats()
        n = len(fleet.robots)
        active = fleet.active_count()
        avg_bat = fleet.average_battery()
        avg_health = fleet.average_health()
        avg_temp = fleet.average_temp()
        transporting = stats.get("TRANSPORTING", 0)
        if transporting < self._prev_transporting:
            self.task_completions += self._prev_transporting - transporting
        self._prev_transporting = transporting
        utilization = active / max(n, 1)
        charging_demand = stats.get("CHARGING", 0) / max(n, 1)
        degraded = stats.get("DEGRADED", 0) + stats.get("ALERT", 0)
        sei = (
            0.25 * (avg_bat / 100) +
            0.20 * (avg_health / 100) +
            0.20 * utilization +
            0.15 * (1.0 - min(1.0, degraded / max(n, 1))) +
            0.10 * (1.0 - min(1.0, env.event_intensity)) +
            0.10 * neural.last_confidence
        ) * 100
        recent_critical = sum(1 for e in events.events[-20:] if e.severity == "critical")
        return {
            "fleet": {
                "active": active,
                "charging": stats.get("CHARGING", 0),
                "maintenance": stats.get("MAINTENANCE", 0),
                "degraded": stats.get("DEGRADED", 0),
                "idle": stats.get("IDLE", 0),
                "alert": stats.get("ALERT", 0),
                "moving": stats.get("MOVING", 0) + stats.get("TRANSPORTING", 0) + stats.get("PICKING", 0),
            },
            "operations": {
                "tasks_completed": self.task_completions,
                "throughput_proxy": round(self.task_completions / max(1, neural.total_inferences / 50), 2),
                "utilization": round(utilization * 100, 1),
                "avg_wait_proxy": round(stats.get("WAITING", 0) / max(n, 1) * 10, 2),
            },
            "energy": {
                "avg_battery": round(avg_bat, 1),
                "charging_demand": round(charging_demand * 100, 1),
                "avg_temp": round(avg_temp, 1),
            },
            "intelligence": {
                "neural_activation": round(float(np.mean(neural.activations[-1])), 3) if len(neural.activations[-1]) else 0,
                "decision_confidence": round(neural.last_confidence, 3),
                "inference_ms": round(neural.last_inference_ms, 3),
                "active_connections": neural.active_connections,
                "total_inferences": neural.total_inferences,
            },
            "reliability": {
                "failures": recent_critical,
                "anomalies": sum(1 for r in fleet.robots if r.anomaly),
                "system_health": round(avg_health, 1),
                "recovery_rate": round(self.recoveries / max(1, self.failures + self.recoveries), 2),
            },
            "sei": round(sei, 1),
            "demand": round(env.demand, 2),
            "event_intensity": round(env.event_intensity, 2),
        }
