"""
NEXUS-01 Configuration
Neural Autonomous Logistics System
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

@dataclass
class SimulationConfig:
    # World
    MAP_WIDTH: int = 80
    MAP_HEIGHT: int = 50
    GRID_RESOLUTION: float = 1.0

    # Fleet
    NUM_ROBOTS: int = 18
    ROBOT_CAPACITY: float = 100.0
    MAX_SPEED: float = 2.5
    BATTERY_CAPACITY: float = 100.0
    BATTERY_DRAIN_RATE: float = 0.08
    BATTERY_CHARGE_RATE: float = 1.2
    TEMP_NOMINAL: float = 35.0
    TEMP_CRITICAL: float = 75.0

    # Stations
    NUM_CHARGING_STATIONS: int = 4
    NUM_PICKUP_STATIONS: int = 5
    NUM_DELIVERY_STATIONS: int = 5
    NUM_MAINTENANCE_BAYS: int = 2

    # Simulation
    TIMESTEP: float = 0.15
    DEFAULT_SEED: int = 42
    MAX_EVENTS_HISTORY: int = 80
    EVENT_PROBABILITY: float = 0.035

    # Neural Network
    INPUT_DIM: int = 16
    HIDDEN_DIMS: List[int] = field(default_factory=lambda: [32, 24, 16])
    OUTPUT_DIM: int = 9  # decision classes
    ACTIVATION_NOISE: float = 0.08

    # Performance
    TARGET_FPS: int = 8
    METRICS_WINDOW: int = 60

# Decision mapping
DECISION_LABELS = [
    "EXECUTE_TASK",
    "WAIT",
    "RECHARGE",
    "REDUCE_SPEED",
    "SWITCH_TASK",
    "MAINTENANCE",
    "AVOID_ZONE",
    "ASSIST_PEER",
    "PRIORITIZE_LOAD"
]

ROBOT_STATES = [
    "IDLE",
    "MOVING",
    "PICKING",
    "TRANSPORTING",
    "CHARGING",
    "MAINTENANCE",
    "WAITING",
    "ALERT",
    "DEGRADED"
]

# Colors (dark tactical palette)
COLORS = {
    "bg": "#0a0e14",
    "panel": "#111820",
    "border": "#1e2a38",
    "text": "#c5d1de",
    "accent": "#3d8bfd",
    "success": "#3dd68c",
    "warning": "#f5a524",
    "danger": "#f31260",
    "muted": "#6b7c93",
    "robot_idle": "#6b7c93",
    "robot_moving": "#3d8bfd",
    "robot_transport": "#3dd68c",
    "robot_charging": "#f5a524",
    "robot_alert": "#f31260",
    "robot_maint": "#a78bfa",
    "station_charge": "#f5a524",
    "station_pick": "#3d8bfd",
    "station_deliver": "#3dd68c",
    "zone_restricted": "#3a1c28",
    "congestion": "#4a2c1a",
}

ZONES = {
    "storage": (5, 5, 25, 20),
    "charging": (55, 5, 75, 18),
    "maintenance": (55, 35, 75, 48),
    "pickup": (5, 30, 25, 45),
    "delivery": (30, 30, 50, 45),
    "ops_center": (30, 5, 50, 15),
}
