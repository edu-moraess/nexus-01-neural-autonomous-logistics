"""
Warehouse Digital Twin Environment
NEXUS-01
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from config.settings import SimulationConfig, ZONES


@dataclass
class Station:
    id: str
    x: float
    y: float
    type: str
    capacity: int = 2
    occupied: int = 0
    available: bool = True

    @property
    def free_slots(self) -> int:
        return max(0, self.capacity - self.occupied)


class Environment:
    def __init__(self, config: SimulationConfig = None, seed: int = 42):
        self.cfg = config or SimulationConfig()
        self.rng = np.random.default_rng(seed)
        self.width = self.cfg.MAP_WIDTH
        self.height = self.cfg.MAP_HEIGHT
        self.stations: List[Station] = []
        self.congestion_map = np.zeros((self.height, self.width), dtype=np.float32)
        self.restricted = np.zeros((self.height, self.width), dtype=bool)
        self.demand = 0.5
        self.event_intensity = 0.0
        self._init_stations()
        self._init_zones()

    def _init_stations(self):
        for i, (x, y) in enumerate([(58, 8), (65, 8), (72, 8), (58, 14)]):
            self.stations.append(Station(f"CHG-{i+1}", x, y, "charge", capacity=2))
        for i, (x, y) in enumerate([(8, 35), (12, 40), (18, 33), (8, 42), (20, 42)]):
            self.stations.append(Station(f"PKP-{i+1}", x, y, "pickup", capacity=3))
        for i, (x, y) in enumerate([(35, 35), (42, 38), (48, 33), (35, 42), (45, 42)]):
            self.stations.append(Station(f"DLV-{i+1}", x, y, "delivery", capacity=3))
        for i, (x, y) in enumerate([(60, 40), (70, 42)]):
            self.stations.append(Station(f"MNT-{i+1}", x, y, "maintenance", capacity=1))

    def _init_zones(self):
        for y in range(6, 14):
            for x in range(32, 48):
                if 0 <= y < self.height and 0 <= x < self.width:
                    self.restricted[y, x] = True

    def update_congestion(self, robots: List):
        self.congestion_map *= 0.85
        for r in robots:
            ix, iy = int(np.clip(r.x, 0, self.width - 1)), int(np.clip(r.y, 0, self.height - 1))
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    ny, nx = iy + dy, ix + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        dist = max(1, abs(dx) + abs(dy))
                        self.congestion_map[ny, nx] += 0.25 / dist
        self.congestion_map = np.clip(self.congestion_map, 0, 1.0)

    def get_local_density(self, x: float, y: float, robots: List, radius: float = 6.0) -> float:
        count = sum(1 for r in robots if np.hypot(r.x - x, r.y - y) < radius)
        return min(1.0, count / 6.0)

    def nearest_station(self, x: float, y: float, stype: str) -> Tuple[float, float, float, Station]:
        candidates = [s for s in self.stations if s.type == stype and s.available]
        if not candidates:
            candidates = [s for s in self.stations if s.type == stype]
        if not candidates:
            return x, y, 999.0, None
        best = min(candidates, key=lambda s: np.hypot(s.x - x, s.y - y))
        dist = np.hypot(best.x - x, best.y - y)
        return best.x, best.y, dist, best

    def snapshot_for_robot(self, robot, robots: List) -> Dict[str, Any]:
        cx, cy, cdist, _ = self.nearest_station(robot.x, robot.y, "charge")
        px, py, pdist, _ = self.nearest_station(robot.x, robot.y, "pickup")
        dx, dy, ddist, _ = self.nearest_station(robot.x, robot.y, "delivery")
        mx, my, _, _ = self.nearest_station(robot.x, robot.y, "maintenance")
        degraded = [r for r in robots if r.state.value in ("DEGRADED", "ALERT") and r.id != robot.id]
        nearest_deg = None
        if degraded:
            peer = min(degraded, key=lambda r: np.hypot(r.x - robot.x, r.y - robot.y))
            nearest_deg = (peer.x, peer.y)
        avail = np.mean([1.0 if s.available and s.free_slots > 0 else 0.3 for s in self.stations])
        return {
            "local_density": self.get_local_density(robot.x, robot.y, robots),
            "congestion": float(self.congestion_map[int(np.clip(robot.y, 0, self.height-1)), int(np.clip(robot.x, 0, self.width-1))]),
            "nearest_charge_dist": cdist,
            "nearest_pick_dist": pdist,
            "nearest_deliver_dist": ddist,
            "nearest_charge_pos": (cx, cy),
            "nearest_pick_pos": (px, py),
            "nearest_deliver_pos": (dx, dy),
            "maint_pos": (mx, my),
            "station_availability": avail,
            "demand": self.demand,
            "event_intensity": self.event_intensity,
            "nearest_degraded_pos": nearest_deg,
        }

    def update_demand(self, dt: float):
        self.demand += self.rng.normal(0, 0.02) * dt * 5
        self.demand = float(np.clip(self.demand, 0.15, 0.95))
        self.event_intensity *= 0.92

    def get_station_positions(self) -> Dict[str, List[Tuple]]:
        out = {"charge": [], "pickup": [], "delivery": [], "maintenance": []}
        for s in self.stations:
            out[s.type].append((s.x, s.y, s.id, s.available))
        return out
