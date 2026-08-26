"""
Simulation Engine — Core Loop
NEXUS-01
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
from config.settings import SimulationConfig
from src.simulation.environment import Environment
from src.simulation.events import EventEngine
from src.agents.fleet import Fleet
from src.analytics.metrics import MetricsCollector
from src.analytics.anomalies import AnomalyDetector


class SimulationEngine:
    def __init__(self, seed: int = 42, config: SimulationConfig = None):
        self.cfg = config or SimulationConfig()
        self.seed = seed
        self.env = Environment(self.cfg, seed=seed)
        self.fleet = Fleet(self.cfg, seed=seed)
        self.events = EventEngine(seed=seed)
        self.metrics = MetricsCollector()
        self.anomaly = AnomalyDetector()
        self.timestep = 0
        self.sim_time = 0.0
        self.running = False
        self.speed = 1.0
        self.history_metrics: List[Dict] = []

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.seed = seed
        self.env = Environment(self.cfg, seed=self.seed)
        self.fleet = Fleet(self.cfg, seed=self.seed)
        self.events = EventEngine(seed=self.seed)
        self.metrics = MetricsCollector()
        self.anomaly = AnomalyDetector()
        self.timestep = 0
        self.sim_time = 0.0
        self.history_metrics.clear()

    def step(self) -> Dict[str, Any]:
        dt = self.cfg.TIMESTEP * self.speed
        self.fleet.step(self.env, dt)
        new_event = self.events.spawn(self.fleet.robots, self.env)
        self.events.recover_stations(self.env)
        anomalies = self.anomaly.detect(self.fleet.robots)
        m = self.metrics.compute(self.fleet, self.env, self.events, self.fleet.neural)
        self.history_metrics.append(m)
        if len(self.history_metrics) > self.cfg.METRICS_WINDOW:
            self.history_metrics = self.history_metrics[-self.cfg.METRICS_WINDOW:]
        self.timestep += 1
        self.sim_time += dt
        events_feed = self.events.recent(12)
        for rlog in getattr(self.fleet, "reassignment_log", [])[-3:]:
            events_feed.insert(0, {
                "id": f"REA-{self.timestep}",
                "type": "TASK_REASSIGNED",
                "severity": "info",
                "message": rlog.get("message", "Task reassigned"),
                "robot_id": rlog.get("to"),
                "timestamp": self.sim_time,
                "active": True,
            })
        return {
            "timestep": self.timestep,
            "sim_time": round(self.sim_time, 2),
            "robots": self.fleet.get_telemetry(),
            "stations": self.env.get_station_positions(),
            "congestion": self.env.congestion_map.copy(),
            "events": events_feed[:15],
            "new_event": new_event.to_dict() if new_event else None,
            "metrics": m,
            "neural": self.fleet.neural.get_visualization_data(),
            "anomalies": anomalies,
            "fleet_stats": self.fleet.stats(),
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            "timestep": self.timestep,
            "sim_time": self.sim_time,
            "running": self.running,
            "speed": self.speed,
            "seed": self.seed,
        }
