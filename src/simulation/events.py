"""
Dynamic Event Engine
NEXUS-01
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import time


@dataclass
class Event:
    id: str
    type: str
    severity: str
    message: str
    robot_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    active: bool = True
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "severity": self.severity,
            "message": self.message,
            "robot_id": self.robot_id,
            "timestamp": self.timestamp,
            "active": self.active,
        }


class EventEngine:
    EVENT_TYPES = [
        ("battery_degradation", 0.18, "warning"),
        ("sensor_failure", 0.08, "critical"),
        ("motor_degradation", 0.12, "warning"),
        ("communication_loss", 0.10, "critical"),
        ("congestion", 0.20, "info"),
        ("priority_task", 0.15, "info"),
        ("station_unavailable", 0.07, "warning"),
        ("overheating", 0.12, "critical"),
        ("unexpected_obstacle", 0.10, "warning"),
        ("robot_offline", 0.05, "critical"),
    ]

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.events: List[Event] = []
        self.counter = 0
        self.prob = 0.035

    def _make_id(self) -> str:
        self.counter += 1
        return f"EVT-{self.counter:04d}"

    def spawn(self, robots: List, env) -> Optional[Event]:
        if self.rng.random() > self.prob:
            return None
        types, probs, sevs = zip(*[(t[0], t[1], t[2]) for t in self.EVENT_TYPES])
        probs = np.array(probs)
        probs /= probs.sum()
        etype = self.rng.choice(types, p=probs)
        severity = dict(zip(types, sevs))[etype]
        robot = self.rng.choice(robots) if robots else None
        rid = robot.id if robot else None
        msg_map = {
            "battery_degradation": f"Battery degradation detected on {rid}. Capacity reduced.",
            "sensor_failure": f"Primary lidar anomaly on {rid}. Perception degraded.",
            "motor_degradation": f"Drive motor efficiency drop on {rid}.",
            "communication_loss": f"Link degradation on {rid}. Packet loss rising.",
            "congestion": "Congestion spike detected in sector. Rerouting recommended.",
            "priority_task": f"High-priority logistics task injected. Target robot {rid}.",
            "station_unavailable": "Charging station offline for maintenance cycle.",
            "overheating": f"Thermal threshold exceeded on {rid}.",
            "unexpected_obstacle": "Dynamic obstacle detected in primary corridor.",
            "robot_offline": f"{rid} entered offline state. Task redistribution required.",
        }
        event = Event(id=self._make_id(), type=etype, severity=severity, message=msg_map.get(etype, f"Event {etype}"), robot_id=rid)
        self.events.append(event)
        if len(self.events) > 80:
            self.events = self.events[-80:]
        self._apply(event, robots, env)
        return event

    def _apply(self, event: Event, robots: List, env):
        rid = event.robot_id
        robot = next((r for r in robots if r.id == rid), None)
        from src.simulation.robot import RobotState
        if event.type == "battery_degradation" and robot:
            robot.battery = max(5.0, robot.battery - self.rng.uniform(12, 25))
            robot.risk = min(1.0, robot.risk + 0.25)
            robot.anomaly = True
            robot.anomaly_score = 0.7
        elif event.type == "sensor_failure" and robot:
            robot.communication = max(20.0, robot.communication - 30)
            robot.health = max(30.0, robot.health - 10)
            robot.anomaly = True
        elif event.type == "motor_degradation" and robot:
            robot.speed = max(0.5, robot.speed * 0.7)
            robot.health = max(25.0, robot.health - 15)
            if robot.health < 40:
                robot.state = RobotState.DEGRADED
            robot.risk = min(1.0, robot.risk + 0.3)
        elif event.type == "communication_loss" and robot:
            robot.communication = max(5.0, robot.communication - 50)
            robot.anomaly = True
            robot.anomaly_score = 0.85
        elif event.type == "congestion":
            env.event_intensity = min(1.0, env.event_intensity + 0.4)
            cy, cx = self.rng.integers(10, 40), self.rng.integers(15, 65)
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < env.height and 0 <= nx < env.width:
                        env.congestion_map[ny, nx] = min(1.0, env.congestion_map[ny, nx] + 0.35)
        elif event.type == "priority_task" and robot:
            robot.priority = 1.0
            robot.state = RobotState.PICKING
        elif event.type == "station_unavailable":
            charges = [s for s in env.stations if s.type == "charge"]
            if charges:
                s = self.rng.choice(charges)
                s.available = False
        elif event.type == "overheating" and robot:
            robot.temperature = min(92.0, robot.temperature + self.rng.uniform(15, 25))
            robot.risk = min(1.0, robot.risk + 0.4)
            robot.state = RobotState.DEGRADED
            robot.anomaly = True
        elif event.type == "unexpected_obstacle":
            env.event_intensity = min(1.0, env.event_intensity + 0.3)
        elif event.type == "robot_offline" and robot:
            robot.state = RobotState.ALERT
            robot.health = max(5.0, robot.health - 40)
            robot.communication = 0.0
            robot.anomaly = True
            robot.anomaly_score = 1.0
        env.event_intensity = min(1.0, env.event_intensity + 0.15)

    def recover_stations(self, env):
        for s in env.stations:
            if not s.available and self.rng.random() < 0.02:
                s.available = True

    def recent(self, n: int = 12) -> List[Dict]:
        return [e.to_dict() for e in reversed(self.events[-n:])]
