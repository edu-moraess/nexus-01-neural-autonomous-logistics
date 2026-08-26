"""
Individual Robot Agent
NEXUS-01 Neural Autonomous Logistics System
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid


class RobotState(Enum):
    IDLE = "IDLE"
    MOVING = "MOVING"
    PICKING = "PICKING"
    TRANSPORTING = "TRANSPORTING"
    CHARGING = "CHARGING"
    MAINTENANCE = "MAINTENANCE"
    WAITING = "WAITING"
    ALERT = "ALERT"
    DEGRADED = "DEGRADED"


@dataclass
class Robot:
    id: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    direction: float = 0.0
    battery: float = 100.0
    temperature: float = 35.0
    load: float = 0.0
    capacity: float = 100.0
    health: float = 100.0
    communication: float = 100.0
    state: RobotState = RobotState.IDLE
    task: Optional[str] = None
    target_x: Optional[float] = None
    target_y: Optional[float] = None
    priority: float = 0.5
    risk: float = 0.0
    decision: str = "WAIT"
    decision_confidence: float = 0.5
    history: List[Dict[str, Any]] = field(default_factory=list)
    anomaly: bool = False
    anomaly_score: float = 0.0
    last_decision_time: float = 0.0
    task_progress: float = 0.0
    speed: float = 1.5
    max_speed: float = 2.5

    def __post_init__(self):
        if not self.id:
            self.id = f"R{uuid.uuid4().hex[:4].upper()}"

    @property
    def position(self) -> tuple:
        return (self.x, self.y)

    @property
    def is_operational(self) -> bool:
        return self.state not in (RobotState.MAINTENANCE, RobotState.ALERT) and self.health > 15

    @property
    def needs_charge(self) -> bool:
        return self.battery < 22.0

    @property
    def needs_maintenance(self) -> bool:
        return self.health < 30.0 or self.temperature > 70.0

    def sense(self, env_snapshot: Dict[str, Any]) -> np.ndarray:
        """Generate sensor vector for neural input (16 dims)."""
        density = env_snapshot.get("local_density", 0.3)
        congestion = env_snapshot.get("congestion", 0.2)
        nearest_charge = env_snapshot.get("nearest_charge_dist", 20.0)
        nearest_pick = env_snapshot.get("nearest_pick_dist", 15.0)
        nearest_deliver = env_snapshot.get("nearest_deliver_dist", 15.0)
        station_avail = env_snapshot.get("station_availability", 0.7)
        demand = env_snapshot.get("demand", 0.5)
        event_intensity = env_snapshot.get("event_intensity", 0.0)

        features = np.array([
            self.battery / 100.0,
            self.temperature / 100.0,
            self.load / self.capacity,
            self.health / 100.0,
            self.communication / 100.0,
            self.speed / self.max_speed,
            self.priority,
            self.risk,
            min(nearest_charge / 50.0, 1.0),
            min(nearest_pick / 50.0, 1.0),
            min(nearest_deliver / 50.0, 1.0),
            density,
            congestion,
            station_avail,
            demand,
            event_intensity,
        ], dtype=np.float32)
        return np.clip(features, 0.0, 1.0)

    def apply_decision(self, decision_idx: int, confidence: float, env: Dict[str, Any]):
        from config.settings import DECISION_LABELS
        self.decision = DECISION_LABELS[decision_idx]
        self.decision_confidence = float(confidence)

        if self.decision == "RECHARGE" or (self.needs_charge and self.state != RobotState.CHARGING):
            self.state = RobotState.CHARGING
            self.target_x, self.target_y = env.get("nearest_charge_pos", (self.x, self.y))
        elif self.decision == "MAINTENANCE" or self.needs_maintenance:
            self.state = RobotState.MAINTENANCE
            self.target_x, self.target_y = env.get("maint_pos", (60.0, 40.0))
        elif self.decision == "EXECUTE_TASK":
            if self.load > 5:
                self.state = RobotState.TRANSPORTING
                self.target_x, self.target_y = env.get("nearest_deliver_pos", (self.x + 5, self.y))
            else:
                self.state = RobotState.PICKING
                self.target_x, self.target_y = env.get("nearest_pick_pos", (self.x + 5, self.y))
        elif self.decision == "WAIT":
            self.state = RobotState.WAITING
            self.vx = self.vy = 0.0
        elif self.decision == "REDUCE_SPEED":
            self.speed = max(0.6, self.speed * 0.7)
            if self.state in (RobotState.MOVING, RobotState.TRANSPORTING, RobotState.PICKING):
                self.state = RobotState.MOVING
        elif self.decision == "AVOID_ZONE":
            self.target_x = self.x + np.random.uniform(-8, 8)
            self.target_y = self.y + np.random.uniform(-8, 8)
            self.state = RobotState.MOVING
        elif self.decision == "ASSIST_PEER":
            peer = env.get("nearest_degraded_pos")
            if peer:
                self.target_x, self.target_y = peer
                self.state = RobotState.MOVING
                self.priority = min(1.0, self.priority + 0.2)
        elif self.decision == "PRIORITIZE_LOAD":
            self.priority = min(1.0, self.priority + 0.3)
            self.state = RobotState.PICKING
        elif self.decision == "SWITCH_TASK":
            self.task = None
            self.load = 0.0
            self.state = RobotState.IDLE

    def update_physics(self, dt: float, obstacles: List = None):
        if self.state in (RobotState.WAITING, RobotState.ALERT):
            self.vx = self.vy = 0.0
            return

        if self.state == RobotState.CHARGING:
            self.battery = min(100.0, self.battery + 1.2 * dt * 10)
            self.temperature = max(30.0, self.temperature - 0.5 * dt * 10)
            self.vx = self.vy = 0.0
            if self.battery > 92:
                self.state = RobotState.IDLE
            return

        if self.state == RobotState.MAINTENANCE:
            self.health = min(100.0, self.health + 0.8 * dt * 10)
            self.temperature = max(32.0, self.temperature - 0.8 * dt * 10)
            self.vx = self.vy = 0.0
            if self.health > 85 and self.temperature < 45:
                self.state = RobotState.IDLE
            return

        if self.target_x is not None and self.target_y is not None:
            dx = self.target_x - self.x
            dy = self.target_y - self.y
            dist = np.hypot(dx, dy)
            if dist < 1.2:
                self.x, self.y = self.target_x, self.target_y
                self.vx = self.vy = 0.0
                if self.state == RobotState.PICKING:
                    self.load = min(self.capacity, self.load + 40 + np.random.uniform(0, 20))
                    self.state = RobotState.TRANSPORTING
                    self.task_progress = 0.0
                elif self.state == RobotState.TRANSPORTING:
                    self.load = 0.0
                    self.state = RobotState.IDLE
                    self.task = None
                return

            speed = self.speed * (0.6 if self.state == RobotState.DEGRADED else 1.0)
            if self.decision == "REDUCE_SPEED":
                speed *= 0.65
            self.direction = np.arctan2(dy, dx)
            self.vx = np.cos(self.direction) * speed
            self.vy = np.sin(self.direction) * speed
            self.x += self.vx * dt * 8
            self.y += self.vy * dt * 8
            self.state = RobotState.MOVING if self.state not in (RobotState.TRANSPORTING, RobotState.PICKING) else self.state

        activity = 0.4 if self.state in (RobotState.MOVING, RobotState.TRANSPORTING, RobotState.PICKING) else 0.1
        self.battery = max(0.0, self.battery - 0.08 * activity * dt * 10)
        self.temperature += (activity * 0.6 - 0.15) * dt * 10
        self.temperature = np.clip(self.temperature, 28.0, 95.0)

        if self.battery < 5:
            self.state = RobotState.ALERT
            self.risk = 0.9
        if self.temperature > 78:
            self.state = RobotState.DEGRADED
            self.risk = max(self.risk, 0.7)
            self.health = max(10.0, self.health - 0.3 * dt * 10)

        self.x = np.clip(self.x, 1.0, 79.0)
        self.y = np.clip(self.y, 1.0, 49.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "x": round(float(self.x), 2),
            "y": round(float(self.y), 2),
            "battery": round(float(self.battery), 1),
            "temp": round(float(self.temperature), 1),
            "load": round(float(self.load), 1),
            "health": round(float(self.health), 1),
            "state": self.state.value,
            "decision": self.decision,
            "confidence": round(float(self.decision_confidence), 2),
            "risk": round(float(self.risk), 2),
            "anomaly": self.anomaly,
            "comm": round(float(self.communication), 1),
        }
